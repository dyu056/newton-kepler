"""
Static/dynamic decoupled continuous trajectory model.

The model predicts next 2D positions while separating its latent space into:
  - z_static: one trajectory-level vector, shared across all timesteps
  - z_dyn[t]: a causal, timestep-specific dynamic vector
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model_cv import Block, LayerNorm


@dataclass
class StaticDynamicConfig:
    block_size: int = 100
    input_dim: int = 2
    static_window: int = 20
    static_dim: int = 16
    dynamic_dim: int = 32
    n_layer_static: int = 1
    n_layer_dynamic: int = 2
    n_head: int = 1
    dropout: float = 0.0
    bias: bool = True
    attention_alpha: float = 0.0


class StaticDynamicKeplerModel(nn.Module):
    """Causal predictor with explicit static and dynamic latent streams."""

    def __init__(self, config: StaticDynamicConfig):
        super().__init__()
        self.config = config

        self.static_input = nn.Linear(config.input_dim, config.static_dim, bias=config.bias)
        self.static_pos = nn.Embedding(config.static_window, config.static_dim)
        static_block_config = self._block_config(config.static_dim, config.n_layer_static)
        self.static_blocks = nn.ModuleList([Block(static_block_config) for _ in range(config.n_layer_static)])
        self.static_ln = LayerNorm(config.static_dim, bias=config.bias)

        self.dynamic_input = nn.Linear(config.input_dim, config.dynamic_dim, bias=config.bias)
        self.dynamic_pos = nn.Embedding(config.block_size, config.dynamic_dim)
        dynamic_block_config = self._block_config(config.dynamic_dim, config.n_layer_dynamic)
        self.dynamic_blocks = nn.ModuleList([Block(dynamic_block_config) for _ in range(config.n_layer_dynamic)])
        self.dynamic_ln = LayerNorm(config.dynamic_dim, bias=config.bias)

        decoder_dim = config.static_dim + config.dynamic_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_dim, 4 * decoder_dim, bias=config.bias),
            nn.SiLU(),
            nn.Linear(4 * decoder_dim, config.input_dim, bias=config.bias),
        )

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / (2 * config.n_layer_dynamic) ** 0.5)

        print("number of parameters: %.6fM" % (self.get_num_params() / 1e6,))

    def _block_config(self, n_embd, n_layer):
        cfg = type("BlockConfig", (), {})()
        cfg.block_size = self.config.block_size
        cfg.n_embd = n_embd
        cfg.n_head = self.config.n_head
        cfg.n_layer = n_layer
        cfg.dropout = self.config.dropout
        cfg.bias = self.config.bias
        cfg.attention_alpha = self.config.attention_alpha
        return cfg

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        return sum(param.numel() for param in self.parameters())

    def encode_static(self, x, start=0):
        """Encode one fixed-size trajectory window into one static vector."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size, time_steps, _ = x.shape
        window = min(self.config.static_window, time_steps - start)
        if window <= 0:
            raise ValueError("Static window start is outside the input sequence.")

        x_window = x[:, start:start + window, :]
        pos = torch.arange(window, device=x.device)
        h = self.static_input(x_window) + self.static_pos(pos)
        for block in self.static_blocks:
            h = block(h)
        h = self.static_ln(h)
        return h.mean(dim=1)

    def encode_dynamic(self, x):
        """Encode the causal dynamic state at every timestep."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _, time_steps, _ = x.shape
        if time_steps > self.config.block_size:
            raise ValueError(f"Sequence length {time_steps} exceeds block size {self.config.block_size}.")

        pos = torch.arange(time_steps, device=x.device)
        h = self.dynamic_input(x) + self.dynamic_pos(pos)
        for block in self.dynamic_blocks:
            h = block(h)
        return self.dynamic_ln(h)

    def forward(self, x, targets=None, return_latents=False, static_consistency_weight=0.0):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        z_static = self.encode_static(x, start=0)
        z_dyn = self.encode_dynamic(x)
        z_static_expanded = z_static.unsqueeze(1).expand(-1, z_dyn.size(1), -1)
        decoder_input = torch.cat([z_static_expanded, z_dyn], dim=-1)
        predictions = self.decoder(decoder_input)

        loss = None
        loss_terms = {}
        if targets is not None:
            if targets.dim() == 2:
                targets = targets.unsqueeze(-1)
            pred_loss = F.mse_loss(predictions.reshape(-1), targets.reshape(-1))
            loss = pred_loss
            loss_terms["prediction"] = pred_loss.detach()

            if static_consistency_weight > 0.0 and x.size(1) >= 2 * self.config.static_window:
                late_start = x.size(1) - self.config.static_window
                z_static_late = self.encode_static(x, start=late_start)
                consistency = F.mse_loss(z_static, z_static_late)
                loss = loss + static_consistency_weight * consistency
                loss_terms["static_consistency"] = consistency.detach()

        if return_latents:
            return predictions, loss, {
                "z_static": z_static,
                "z_dyn": z_dyn,
                "z_static_expanded": z_static_expanded,
                "loss_terms": loss_terms,
            }
        return predictions, loss

    @torch.no_grad()
    def generate(self, x, max_new_tokens):
        input_was_2d = x.dim() == 2
        if input_was_2d:
            x = x.unsqueeze(-1)

        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= self.config.block_size else x[:, -self.config.block_size:]
            predictions, _ = self.forward(x_cond)
            x_next = predictions[:, -1, :]
            x = torch.cat((x, x_next.unsqueeze(1)), dim=1)

        if input_was_2d:
            x = x.squeeze(-1)
        return x
