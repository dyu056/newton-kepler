"""
Pure-attention model for testing the RG predictions.

Matches `PureAttentionNoLN` in
r2_research/rg_kepler_testbed/compute_rg_parameters.py, which is the architecture
the RG derivation in RG_predictions_kepler_testbed.md actually assumes.

WHAT WAS REMOVED relative to model_cv.py / model_noln.py, and why
------------------------------------------------------------------
1. **Residual connections** (`x = x + attn(x)`, `x = x + mlp(x)`) -- REMOVED.
   The RG theory defines G as the map from embedding to attention input and builds
   D = sqrt(d) * G * Gamma * G. A residual path makes the effective map G + I, which
   changes D's whole spectrum. This is the main reason the theory's mu values do not
   match a residual network.

2. **MLP block** (c_fc 128->512 -> SiLU -> c_proj 512->128) -- REMOVED.
   The derivation assumes df_i/d(attn_out)_i = W_out in one step
   (compute_rg_parameters.py:227). With an MLP it becomes W_out @ J_MLP, and SiLU's
   Jacobian varies per sample, so Gamma = dL/dS is no longer the object analysed.

3. **LayerNorm** (ln_1, ln_2, ln_f) -- REMOVED (not just set to Identity).
   Measured separately: with LN the origin is not a critical point of the gradient
   flow, so the softmax expansion about uniform attention has no clean eps scaling.

KEPT, identical to the reference and to model_cv.py
---------------------------------------------------
input_embedding (2->d), wpe position embedding, single-head causal attention with
1/sqrt(d) scaling, c_attn QKV projection, c_proj, output_head (d->2), MSE loss.

INITIALIZATION
--------------
Default `sigma_init = (0.1 / (3 d))^{1/4}`, the reference's calibration
(compute_rg_parameters.py:140) chosen so std(S) ~ 0.1 -- large enough that softmax
has a non-trivial gradient, small enough that the Taylor expansion stays valid.
For d=128 that is 0.127, versus the 0.02 hardcoded in model_cv.py. c_proj gets
sigma_init/sqrt(2), mirroring the reference.

USAGE NOTES
-----------
* Requires `loss_mask: last` semantics: the RG rank(Gamma)=1 structure holds only for
  a single target position. With block_size < 100 the chop path already yields one
  target per sequence, so `block_sizes: [50]` achieves this.
* `probe.py` hooks `block.mlp` and `block.mlp.silu`; those layers do not exist here.
  Use `setup_pure_attn_hooks` below instead of probe.setup_activation_hooks.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def calibrated_sigma(d, target_std_S=0.1):
    """Reference calibration: std(S) ~ 3 d sigma^4  =>  sigma = (target/(3d))^{1/4}."""
    return (target_std_S / (3.0 * d)) ** 0.25


class GPTConfigPureAttn:
    """Plain config object; no dataclass decorator so attributes can be set freely."""

    def __init__(self, block_size=50, n_embd=128, n_head=1, input_dim=2,
                 bias=True, dropout=0.0, sigma_init=None, target_std_S=0.1):
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.input_dim = input_dim
        self.bias = bias
        self.dropout = dropout
        self.target_std_S = target_std_S
        self.sigma_init = (calibrated_sigma(n_embd, target_std_S)
                           if sigma_init is None else float(sigma_init))
        # present so shared observation code that reads these does not break
        self.n_layer = 1
        self.attention_alpha = 0.0
        self.varcov_enabled = False
        self.varcov_target_std = 0.1


class PureAttention(nn.Module):
    """Single-head causal attention. No LayerNorm, no MLP, no residual."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.dropout = config.dropout

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # observation hooks (mirrors CausalSelfAttention in model_cv.py)
        self.capture_attention_stats = False
        self.last_attention_stats = None
        self.compute_entropy = False
        self.last_entropy = None
        # cache for RG diagnostics; filled only when requested
        self.last_scores = None
        self.last_weights = None
        self.cache_qkv = False
        self.last_qkv = None

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size))

    @torch.no_grad()
    def _record_attention_stats(self, probabilities):
        if not self.capture_attention_stats:
            self.last_attention_stats = None
            return
        probs = probabilities.detach().float().clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1)
        per_head = entropy.mean(dim=(0, 2))
        sequence_length = probabilities.size(-2)
        if sequence_length > 1:
            visible = torch.arange(2, sequence_length + 1,
                                   device=entropy.device, dtype=entropy.dtype)
            normalized = entropy[:, :, 1:] / visible.log().view(1, 1, -1)
            normalized_per_head = normalized.mean(dim=(0, 2))
            normalized_mean = normalized.mean()
        else:
            normalized_per_head = torch.zeros_like(per_head)
            normalized_mean = entropy.new_tensor(0.0)
        self.last_attention_stats = {
            "mean_entropy": float(entropy.mean().item()),
            "normalized_mean_entropy": float(normalized_mean.item()),
            "entropy_by_head": per_head.cpu().tolist(),
            "normalized_entropy_by_head": normalized_per_head.cpu().tolist(),
            "sequence_length": int(sequence_length),
        }

    def forward(self, x):
        B, T, C = x.size()
        self.last_attention_stats = None
        self.last_entropy = None
        self.last_scores = None
        self.last_weights = None
        self.last_qkv = None

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        nh = self.n_head
        k = k.view(B, T, nh, C // nh).transpose(1, 2)
        q = q.view(B, T, nh, C // nh).transpose(1, 2)
        v = v.view(B, T, nh, C // nh).transpose(1, 2)

        # always the manual path: the RG diagnostics need S and A explicitly
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        weights = F.softmax(att, dim=-1)

        if self.capture_attention_stats:
            self._record_attention_stats(weights)
            self.last_scores = att.detach()
            self.last_weights = weights.detach()
        if self.cache_qkv:
            self.last_qkv = (q.detach(), k.detach(), v.detach())
        if self.compute_entropy:
            self.last_entropy = -(weights * torch.log(weights + 1e-12)).sum(-1).mean()

        y = self.attn_dropout(weights) @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class _AttnOnlyBlock(nn.Module):
    """A `Block` with only `attn` -- no ln_1/ln_2, no mlp, and NO residual add.

    Exists so that `model.transformer.h[0].attn` resolves the same way it does in
    model_cv.py, keeping external code and hooks familiar.
    """

    def __init__(self, config):
        super().__init__()
        self.attn = PureAttention(config)

    def forward(self, x):
        return x + self.attn(x)      # residual connection added (experiment)


class GPTPureAttn(nn.Module):
    """embed -> +pos -> attention -> output_head.  No LN, no MLP, no residual."""

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        # kept under `transformer` / `h` so external code that walks
        # model.transformer.h[i].attn keeps working
        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([_AttnOnlyBlock(config)]),
        ))
        self.input_embedding = nn.Linear(config.input_dim, config.n_embd,
                                        bias=config.bias)
        self.output_head = nn.Linear(config.n_embd, config.input_dim,
                                     bias=config.bias)

        self.sigma_init = config.sigma_init
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0.0, std=self.sigma_init / math.sqrt(2))

        print("PureAttn: params=%.6fM  sigma_init=%.4f  (d=%d, T=%d)" % (
            self.get_num_params() / 1e6, self.sigma_init,
            config.n_embd, config.block_size))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.sigma_init)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.sigma_init)

    def get_num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def embed_continuous(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.input_embedding(x)

    def forward(self, x, targets=None, compute_entropy=False):
        attn = self.transformer.h[0].attn
        attn.compute_entropy = compute_entropy
        device = x.device

        if x.dim() == 2:
            x = x.unsqueeze(-1)
        b, t, d = x.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        assert d == self.config.input_dim, \
            f"Input dimension {d} does not match config.input_dim {self.config.input_dim}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        h = self.embed_continuous(x) + self.transformer.wpe(pos)
        h = self.transformer.drop(h)

        # NO residual: the attention output replaces h rather than adding to it
        h = self.transformer.h[0](h)
        predictions = self.output_head(h)

        loss = None
        if targets is not None:
            if targets.dim() == 2:
                targets = targets.unsqueeze(-1)
            if targets.size(1) < predictions.size(1):
                predictions = predictions[:, -targets.size(1):, :]
            elif targets.size(1) > predictions.size(1):
                raise ValueError(
                    f"Target sequence length {targets.size(1)} > "
                    f"prediction length {predictions.size(1)}")
            loss = F.mse_loss(predictions.reshape(-1), targets.reshape(-1))
        return predictions, loss

    # ── RG diagnostics ──

    def forward_with_internals(self, x):
        """Return (pred, q, k, v, S, A) like the reference's return_qk=True."""
        attn = self.transformer.h[0].attn
        prev_stats, prev_qkv = attn.capture_attention_stats, attn.cache_qkv
        attn.capture_attention_stats, attn.cache_qkv = True, True
        try:
            pred, _ = self.forward(x, None)
            q, k, v = attn.last_qkv
            return pred, q, k, v, attn.last_scores, attn.last_weights
        finally:
            attn.capture_attention_stats, attn.cache_qkv = prev_stats, prev_qkv

    # ── parity with model_cv API, so shared code does not crash ──

    def variance_covariance_penalties(self):
        return None, None

    def variance_covariance_stats(self):
        return {}

    def attention_entropy_penalty(self):
        e = self.transformer.h[0].attn.last_entropy
        return e if e is not None else None

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size])
        attn = self.transformer.h[0].attn
        attn.bias = attn.bias[:, :, :block_size, :block_size]

    @torch.no_grad()
    def generate(self, x, max_new_tokens):
        input_was_2d = x.dim() == 2
        if input_was_2d:
            x = x.unsqueeze(-1)
        for _ in range(max_new_tokens):
            x_cond = (x if x.size(1) <= self.config.block_size
                      else x[:, -self.config.block_size:])
            predictions, _ = self(x_cond)
            x = torch.cat((x, predictions[:, -1, :].unsqueeze(1)), dim=1)
        if input_was_2d:
            x = x.squeeze(-1)
        return x


# ─────────────────────── activation hooks ───────────────────────

class _ActivationStore(dict):
    capture_enabled = False


def setup_pure_attn_hooks(model, verbose=False):
    """Replacement for probe.setup_activation_hooks for this architecture.

    probe.py hooks `block.mlp`, `block.mlp.silu` and `transformer.ln_f`, none of
    which exist here. This registers the subset that is meaningful:

        input_embed          input_embedding output
        after_pos_emb        embedding + position embedding (transformer.drop)
        block_0_attn_output  attention output = the full residual stream
        after_ln_f           alias of the above, so downstream code that keys on
                             'after_ln_f' still finds the final representation

    Returns (hooks, activation_dict) with the same contract as probe.py.
    """
    activation_dict = _ActivationStore()
    hooks = []

    def save(name, value):
        if activation_dict.capture_enabled:
            if isinstance(value, tuple):
                value = value[0]
            activation_dict[name] = value.detach()

    def input_embed_hook(module, inputs, output):
        save("input_embed", output)

    def after_pos_emb_hook(module, inputs, output):
        save("after_pos_emb", output)

    def attn_output_hook(module, inputs, output):
        save("block_0_attn_output", output)
        save("after_ln_f", output)      # final representation under this architecture

    hooks.append(model.input_embedding.register_forward_hook(input_embed_hook))
    hooks.append(model.transformer.drop.register_forward_hook(after_pos_emb_hook))
    hooks.append(model.transformer.h[0].attn.register_forward_hook(attn_output_hook))

    if verbose:
        print(f"PureAttn hooks registered: {len(hooks)}")
    return hooks, activation_dict


def build_pure_attn_model(block_size=50, n_embd=128, n_head=1, input_dim=2,
                          bias=True, dropout=0.0, sigma_init=None,
                          target_std_S=0.1, device=None):
    """Convenience constructor mirroring the reference's calibrated setup."""
    cfg = GPTConfigPureAttn(block_size=block_size, n_embd=n_embd, n_head=n_head,
                            input_dim=input_dim, bias=bias, dropout=dropout,
                            sigma_init=sigma_init, target_std_S=target_std_S)
    model = GPTPureAttn(cfg)
    if device is not None:
        model = model.to(device)
    return model
