"""
Full definition of a GPT Language Model for continuous variables, all of it in this single file.
This version works with continuous x coordinates instead of discrete tokens.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from loss import compute_varcov_penalty

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

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
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.alpha = config.attention_alpha  # Position-dependent scaling parameter
        # Disabled during normal training. Evaluation code can enable this flag
        # to cache only aggregated entropy statistics, never the full matrix.
        self.capture_attention_stats = False
        self.last_attention_stats = None
        self.compute_entropy = False      # 是否计算可微熵
        self.last_entropy = None          # 存储当前 batch 的熵值（标量 tensor，可微）
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        # Always register bias buffer (needed for manual attention when alpha != 1.0)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))

    @torch.no_grad()
    def _record_attention_stats(self, probabilities):
        if not self.capture_attention_stats:
            self.last_attention_stats = None
            return

        probs = probabilities.detach().float().clamp_min(1e-12)
        entropy = -(probs * probs.log()).sum(dim=-1)  # (B, H, T)
        per_head = entropy.mean(dim=(0, 2))

        sequence_length = probabilities.size(-2)
        if sequence_length > 1:
            visible_keys = torch.arange(
                2, sequence_length + 1,
                device=entropy.device,
                dtype=entropy.dtype,
            )
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
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        self.last_attention_stats = None
        self.last_entropy = None  # 新增：重置熵缓存

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        need_weights = self.capture_attention_stats or self.compute_entropy or (self.alpha != 1.0)

        if self.flash and not need_weights:
            # Flash attention (no weights)
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # Manual attention (weights available)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            if self.alpha != 1.0:
                positions = torch.arange(1, T + 1, dtype=att.dtype, device=att.device)
                log_positions = torch.log(positions)
                position_scale = log_positions ** self.alpha
                att = att * position_scale.view(1, 1, T, 1)
            att = F.softmax(att, dim=-1)

            # Record stats for observation (no gradient)
            if self.capture_attention_stats:
                self._record_attention_stats(att)

            # Compute differentiable entropy if requested
            if self.compute_entropy:
                entropy = -torch.sum(att * torch.log(att + 1e-12), dim=-1)  # (B, nh, T)
                self.last_entropy = entropy.mean()  # scalar, differentiable
            else:
                self.last_entropy = None

            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config, layer_name):
        super().__init__()
        self.layer_name = layer_name
        self.varcov_enabled = bool(config.varcov_enabled)
        self.varcov_target_std = float(config.varcov_target_std)
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        #self.gelu    = nn.GELU()
        self.silu    = nn.SiLU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
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
            ) = compute_varcov_penalty(
                projected,
                target_std=self.varcov_target_std,
            )

        return self.dropout(projected)

class Block(nn.Module):

    def __init__(self, config, layer_name):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config, layer_name=layer_name)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfigCV:
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_dim: int = 1 # Dimension of input continuous variables (e.g., 1 for 1D, 2 for 2D coordinates)
    attention_alpha: float = 0.0 # Position-dependent scaling parameter: (log N)^alpha applied before softmax
    varcov_enabled: bool = False
    varcov_target_std: float = 0.1

class GPTCV(nn.Module):
    """
    GPT model for continuous variables.
    Input and output are continuous x coordinates instead of discrete tokens.
    """

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([
                Block(config, layer_name=f"block_{index}")
                for index in range(config.n_layer)
            ]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        # Learnable input embedding layer: maps input_dim to n_embd
        self.input_embedding = nn.Linear(config.input_dim, config.n_embd, bias=config.bias)
        # Output head: predict continuous values of same dimension as input
        self.output_head = nn.Linear(config.n_embd, config.input_dim, bias=config.bias)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.6fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
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
        """
        Embed continuous variable x into n_embd dimension vector using a learnable linear layer.
        
        Input: x of shape (b, t) or (b, t, d) where d is the number of dimensions
               - If (b, t): treated as 1D input
               - If (b, t, d): d-dimensional input (e.g., d=2 for x,y coordinates)
        Output: embeddings of shape (b, t, n_embd) using learnable linear transformation
        """
        if x.dim() == 2:
            # x is (b, t), add dimension to make (b, t, 1)
            x = x.unsqueeze(-1)
        
        b, t, d = x.size()
        assert d == self.config.input_dim, \
            f"Input dimension {d} does not match config.input_dim {self.config.input_dim}"
        
        # Use learnable linear embedding: (b, t, d) -> (b, t, n_embd)
        x_emb = self.input_embedding(x)
        
        return x_emb

    def forward(self, x, targets=None,compute_entropy=False):
        """
        Forward pass for continuous variables.
        
        Args:
            x: continuous input of shape (b, t) or (b, t, d) where:
               - b is batch size
               - t is sequence length
               - d is number of dimensions (defaults to config.input_dim)
            targets: optional continuous targets of shape (b, t) or (b, t, d) for computing loss
        
        Returns:
            predictions: predicted continuous values of shape (b, t, d)
            loss: MSE loss if targets are provided, None otherwise
        """
        for block in self.transformer.h:
            block.attn.compute_entropy = compute_entropy
        device = x.device
        
        # Handle input shape: if 2D, add dimension
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (b, t) -> (b, t, 1)
        
        b, t, d = x.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert d == self.config.input_dim, \
            f"Input dimension {d} does not match config.input_dim {self.config.input_dim}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Embed continuous variable x into n_embd dimensions using learnable linear layer
        x_emb = self.embed_continuous(x) # shape (b, t, n_embd)
        
        # Add position embeddings
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(x_emb + pos_emb)
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            predictions = self.output_head(x) # shape (b, t, d)
            
            # Handle target shape: if 2D, add dimension
            if targets.dim() == 2:
                targets = targets.unsqueeze(-1)  # (b, t) -> (b, t, 1)
            
            # Handle case where targets might have different sequence length than predictions
            # This can happen when we only want to predict the last position
            if targets.size(1) < predictions.size(1):
                # Only use the last predictions that match target length
                predictions = predictions[:, -targets.size(1):, :]  # Take last t_target positions
            elif targets.size(1) > predictions.size(1):
                # Pad predictions or truncate targets (shouldn't happen normally)
                raise ValueError(f"Target sequence length {targets.size(1)} > prediction length {predictions.size(1)}")
            
            # Reshape for loss computation
            predictions_flat = predictions.reshape(-1) # shape (b*t*d,)
            targets_flat = targets.reshape(-1) # shape (b*t*d,)
            # Use MSE loss for continuous regression
            loss = F.mse_loss(predictions_flat, targets_flat)
        else:
            # inference-time: predict all positions
            predictions = self.output_head(x) # shape (b, t, d)
            loss = None

        return predictions, loss

    def variance_covariance_penalties(self):
        """Return layer-averaged penalties from the latest forward pass."""
        variance_losses = []
        covariance_losses = []
        for block in self.transformer.h:
            if block.mlp.last_variance_loss is not None:
                variance_losses.append(block.mlp.last_variance_loss)
            if block.mlp.last_covariance_loss is not None:
                covariance_losses.append(block.mlp.last_covariance_loss)

        variance = (
            torch.stack(variance_losses).mean()
            if variance_losses else None
        )
        covariance = (
            torch.stack(covariance_losses).mean()
            if covariance_losses else None
        )
        return variance, covariance

    def variance_covariance_stats(self):
        """Return detached per-layer statistics from the latest forward pass."""
        results = {}
        for index, block in enumerate(self.transformer.h):
            mlp = block.mlp
            if mlp.last_mean_std is None:
                continue
            results[f"block_{index}"] = {
                "mean_std": float(mlp.last_mean_std.detach().item()),
                "variance_loss": float(
                    mlp.last_variance_loss.detach().item()
                ),
                "covariance_loss": float(
                    mlp.last_covariance_loss.detach().item()
                ),
            }
        return results
    def attention_entropy_penalty(self):
        entropies = []
        for block in self.transformer.h:
            if block.attn.last_entropy is not None:
                entropies.append(block.attn.last_entropy)
        if not entropies:
            return None
        return torch.stack(entropies).mean()
        
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, x, max_new_tokens):
        """
        Generate new continuous values given a conditioning sequence.
        Take a conditioning sequence of continuous values x and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        
        Args:
            x: input sequence of shape (b, t) or (b, t, d)
            max_new_tokens: number of new tokens to generate
            temperature: temperature for sampling (higher = more random)
        
        Returns:
            x: extended sequence of shape (b, t + max_new_tokens, d) or (b, t + max_new_tokens) if input was 2D
        """
        # Handle input shape
        input_was_2d = x.dim() == 2
        if input_was_2d:
            x = x.unsqueeze(-1)  # (b, t) -> (b, t, 1)
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            x_cond = x if x.size(1) <= self.config.block_size else x[:, -self.config.block_size:]
            # forward the model to get the predictions
            predictions, _ = self(x_cond)
            # pluck the prediction at the final step: shape (b, d)
            x_next = predictions[:, -1, :] # shape (b, d)
            
            # append predicted value to the running sequence
            x = torch.cat((x, x_next.unsqueeze(1)), dim=1)  # (b, t, d) -> (b, t+1, d)
        
        # Return in same format as input
        if input_was_2d:
            x = x.squeeze(-1)  # (b, t, 1) -> (b, t)
        
        return x

