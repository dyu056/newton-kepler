"""
Attention-only model for continuous variables — no MLP.

Each block:  x = x + CausalSelfAttention(LayerNorm(x))

Same GPTConfigCV / GPTCV interface as model_cv.py.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.alpha = config.attention_alpha

        self.capture_attention_stats = False
        self.last_attention_stats = None
        self.compute_entropy = False
        self.last_entropy = None

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
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
            visible_keys = torch.arange(2, sequence_length + 1, device=entropy.device, dtype=entropy.dtype)
            normalized = entropy[:, :, 1:] / visible_keys.log().view(1, 1, -1)
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

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        need_weights = self.capture_attention_stats or self.compute_entropy or (self.alpha != 1.0)

        if self.flash and not need_weights:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            if self.alpha != 1.0:
                positions = torch.arange(1, T + 1, dtype=att.dtype, device=att.device)
                position_scale = torch.log(positions) ** self.alpha
                att = att * position_scale.view(1, 1, T, 1)
            att = F.softmax(att, dim=-1)

            if self.capture_attention_stats:
                self._record_attention_stats(att)

            if self.compute_entropy:
                entropy = -torch.sum(att * torch.log(att + 1e-12), dim=-1)
                self.last_entropy = entropy.mean()
            else:
                self.last_entropy = None

            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ── Attention-only block (no MLP) ───────────────────────────────────────
class AttBlock(nn.Module):

    def __init__(self, config, layer_name):
        super().__init__()
        self.layer_name = layer_name
        self.ln = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)

    def forward(self, x):
        x = x + self.attn(self.ln(x))
        return x


# ── Config ───────────────────────────────────────────────────────────────
@dataclass
class GPTConfigCV:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    input_dim: int = 1
    attention_alpha: float = 0.0
    varcov_enabled: bool = False
    varcov_target_std: float = 0.1


# ── Main model ───────────────────────────────────────────────────────────
class GPTCV(nn.Module):
    """GPT-style model for continuous variables, attention-only blocks."""

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([
                AttBlock(config, layer_name=f"block_{index}")
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
        for block in self.transformer.h:
            block.attn.compute_entropy = compute_entropy
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
                    f"Target sequence length {targets.size(1)} > prediction length {predictions.size(1)}")

            predictions_flat = predictions.reshape(-1)
            targets_flat = targets.reshape(-1)
            loss = F.mse_loss(predictions_flat, targets_flat)
        else:
            predictions = self.output_head(x)
            loss = None

        return predictions, loss

    def variance_covariance_penalties(self):
        return None, None

    def variance_covariance_stats(self):
        return {}

    def attention_entropy_penalty(self):
        entropies = []
        for block in self.transformer.h:
            if block.attn.last_entropy is not None:
                entropies.append(block.attn.last_entropy)
        if not entropies:
            return None
        return torch.stack(entropies).mean()

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

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
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
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
