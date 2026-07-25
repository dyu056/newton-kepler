"""
MLP-only model for continuous variables — no attention.

Each block:  x = x + MLP2(LayerNorm(x))

MLP2 has two hidden layers:
    Linear(n_embd → mlp_mult*n_embd) → SiLU
    → Linear(mlp_mult*n_embd → mlp_mult*n_embd) → SiLU
    → Linear(mlp_mult*n_embd → n_embd) → Dropout

Same GPTConfigCV / GPTCV interface as model_cv.py, so train.py probes
and observe.py hooks continue to work (with hasattr guards for attn).
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from loss import compute_varcov_penalty


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ── MLP (single hidden layer, same as model_cv) ─────────────────────────
class MLP(nn.Module):

    def __init__(self, config, layer_name):
        super().__init__()
        self.layer_name = layer_name
        mult = config.mlp_mult
        self.varcov_enabled = bool(config.varcov_enabled)
        self.varcov_target_std = float(config.varcov_target_std)

        self.c_fc   = nn.Linear(config.n_embd, mult * config.n_embd, bias=config.bias)
        self.silu   = nn.SiLU()
        self.c_proj = nn.Linear(mult * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.last_variance_loss = None
        self.last_covariance_loss = None
        self.last_mean_std = None

    def forward(self, x):
        x = self.c_fc(x)
        x = self.silu(x)
        projected = self.c_proj(x)

        self.last_variance_loss = None
        self.last_covariance_loss = None
        self.last_mean_std = None
        if self.varcov_enabled:
            (
                self.last_variance_loss,
                self.last_covariance_loss,
                self.last_mean_std,
            ) = compute_varcov_penalty(projected, target_std=self.varcov_target_std)

        return self.dropout(projected)


# ── MLP-only block (no attention) ────────────────────────────────────────
class MLPBlock(nn.Module):

    def __init__(self, config, layer_name):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config, layer_name=layer_name)

    def forward(self, x):
        x = x + self.mlp(self.ln(x))
        return x


# ── Config ───────────────────────────────────────────────────────────────
@dataclass
class GPTConfigCV:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12          # kept for compat, unused
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    input_dim: int = 1
    attention_alpha: float = 0.0   # kept for compat, unused
    varcov_enabled: bool = False
    varcov_target_std: float = 0.1
    mlp_mult: int = 4              # hidden dim = mlp_mult * n_embd


# ── Main model ───────────────────────────────────────────────────────────
class GPTCV(nn.Module):
    """
    GPT-style model for continuous variables, with every attention block
    replaced by a pure MLP block (MLPBlock).
    """

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                MLPBlock(config, layer_name=f"block_{index}")
                for index in range(config.n_layer)
            ]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.input_embedding = nn.Linear(config.input_dim, config.n_embd, bias=config.bias)
        self.output_head = nn.Linear(config.n_embd, config.input_dim, bias=config.bias)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.6fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def embed_continuous(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.input_embedding(x)

    def forward(self, x, targets=None, compute_entropy=False):
        # compute_entropy is a no-op in MLP model (no attention → no entropy)
        device = x.device

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        b, t, d = x.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert d == self.config.input_dim, \
            f"Input dimension {d} does not match config.input_dim {self.config.input_dim}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        x_emb = self.embed_continuous(x)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(x_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            predictions = self.output_head(x)

            if targets.dim() == 2:
                targets = targets.unsqueeze(-1)

            if targets.size(1) < predictions.size(1):
                predictions = predictions[:, -targets.size(1):, :]
            elif targets.size(1) > predictions.size(1):
                raise ValueError(
                    f"Target sequence length {targets.size(1)} > prediction length {predictions.size(1)}"
                )

            predictions_flat = predictions.reshape(-1)
            targets_flat = targets.reshape(-1)
            loss = F.mse_loss(predictions_flat, targets_flat)
        else:
            predictions = self.output_head(x)
            loss = None

        return predictions, loss

    def variance_covariance_penalties(self):
        variance_losses, covariance_losses = [], []
        for block in self.transformer.h:
            if block.mlp.last_variance_loss is not None:
                variance_losses.append(block.mlp.last_variance_loss)
            if block.mlp.last_covariance_loss is not None:
                covariance_losses.append(block.mlp.last_covariance_loss)
        variance = torch.stack(variance_losses).mean() if variance_losses else None
        covariance = torch.stack(covariance_losses).mean() if covariance_losses else None
        return variance, covariance

    def variance_covariance_stats(self):
        results = {}
        for index, block in enumerate(self.transformer.h):
            mlp = block.mlp
            if mlp.last_mean_std is None:
                continue
            results[f"block_{index}"] = {
                "mean_std": float(mlp.last_mean_std.detach().item()),
                "variance_loss": float(mlp.last_variance_loss.detach().item()),
                "covariance_loss": float(mlp.last_covariance_loss.detach().item()),
            }
        return results

    def attention_entropy_penalty(self):
        # No attention → no entropy.  Returning a zero scalar so training
        # loops that unconditionally call this method don't crash.
        return torch.tensor(0.0)

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        flops_per_token = 6 * N
        flops_per_fwdbwd = flops_per_token * cfg.block_size
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        return flops_achieved / flops_promised

    @torch.no_grad()
    def generate(self, x, max_new_tokens):
        input_was_2d = x.dim() == 2
        if input_was_2d:
            x = x.unsqueeze(-1)

        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= self.config.block_size else x[:, -self.config.block_size:]
            predictions, _ = self(x_cond)
            x_next = predictions[:, -1, :]
            x = torch.cat((x, x_next.unsqueeze(1)), dim=1)

        if input_was_2d:
            x = x.squeeze(-1)
        return x
