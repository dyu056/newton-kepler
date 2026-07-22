"""
MLP-only model for continuous variables — no attention.
Replaces every Transformer block with a pure 2-hidden-layer MLP block.

Each MLP block:
    x = x + MLP(LayerNorm(x))

where MLP has two hidden layers (instead of the original single hidden layer),
each matching the original MLP's hidden-layer settings:
    Linear(n_embd -> 4*n_embd) -> SiLU -> Linear(4*n_embd -> 4*n_embd) -> SiLU -> Linear(4*n_embd -> n_embd)

Stack n_layer such blocks (default 2, matching the branch name "harry_2mlp").

Based on model_cv.py; shares the same GPTConfigCV and training interface so
kepler_cv.py / kepler_cv_blocksize.py can import and use it directly.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Shared building blocks (identical to model_cv.py)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


# ---------------------------------------------------------------------------
# Two-hidden-layer MLP  (the key difference vs model_cv.py)
# ---------------------------------------------------------------------------

class MLP2(nn.Module):
    """
    MLP with **two** hidden layers instead of one.
    Hidden dim = config.mlp_mult * n_embd, activation = SiLU.
    """

    def __init__(self, config):
        super().__init__()
        mult = config.mlp_mult
        self.c_fc   = nn.Linear(config.n_embd, mult * config.n_embd, bias=config.bias)
        self.c_fc2  = nn.Linear(mult * config.n_embd, mult * config.n_embd, bias=config.bias)
        self.silu   = nn.SiLU()
        self.c_proj = nn.Linear(mult * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.silu(x)
        x = self.c_fc2(x)
        x = self.silu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# MLP-only Block  (no attention — the defining change)
# ---------------------------------------------------------------------------

class MLPBlock(nn.Module):
    """
    A block that contains ONLY an MLP with residual connection.
    No attention, no second branch.
    """

    def __init__(self, config):
        super().__init__()
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP2(config)

    def forward(self, x):
        x = x + self.mlp(self.ln(x))
        return x


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class GPTConfigCV:
    block_size: int = 1024
    n_layer: int = 12          
    n_head: int = 12          # kept for compat but unused
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    input_dim: int = 1
    # attention_alpha is no longer meaningful — kept for config compat
    attention_alpha: float = 0.0
    mlp_mult: int = 4           # hidden dim = mlp_mult * n_embd


# ---------------------------------------------------------------------------
# GPTCV  —  same interface, MLP-only internals
# ---------------------------------------------------------------------------

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
            h=nn.ModuleList([MLPBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.input_embedding = nn.Linear(config.input_dim, config.n_embd, bias=config.bias)
        self.output_head = nn.Linear(config.n_embd, config.input_dim, bias=config.bias)

        # init all weights
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
        b, t, d = x.size()
        assert d == self.config.input_dim, \
            f"Input dimension {d} does not match config.input_dim {self.config.input_dim}"
        return self.input_embedding(x)

    def forward(self, x, targets=None):
        device = x.device

        if x.dim() == 2:
            x = x.unsqueeze(-1)

        b, t, d = x.size()
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert d == self.config.input_dim, \
            f"Input dimension {d} does not match config.input_dim {self.config.input_dim}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # embed
        x_emb = self.embed_continuous(x)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(x_emb + pos_emb)

        # MLP-only blocks  (no attention anywhere)
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

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

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
