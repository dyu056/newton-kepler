"""Activation collection and linear probes for physical and orbital quantities."""

import numpy as np
import torch
# ── NumPy-only replacements for sklearn (no network on H200) ──


class __LinearRegression:
    """Thin wrapper around np.linalg.lstsq, sklearn-compatible API."""
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X_aug = np.column_stack([X, np.ones(X.shape[0])])
        coef, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
        self.coef_ = coef[:-1]
        self.intercept_ = coef[-1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


def _r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

from model_cv import GPTConfigCV


class _ActivationStore(dict):
    """Dictionary used by hooks, with capture disabled outside probe forwards."""

    capture_enabled: bool = False


def setup_activation_hooks(model, verbose=False):
    """Register reusable hooks while keeping them dormant during normal training."""
    activation_dict = _ActivationStore()
    hooks = []
    n_layer = GPTConfigCV.n_layer

    def extract_tensor(value):
        if isinstance(value, tuple):
            value = value[0]
        return value.detach()

    def save(name, value):
        if activation_dict.capture_enabled:
            activation_dict[name] = extract_tensor(value)

    def make_attn_output_hook(block_idx):
        def hook(module, inputs, output):
            save(f"block_{block_idx}_attn_output", output)
        return hook

    def make_after_attn_merge_hook(block_idx):
        def hook(module, inputs):
            save(f"block_{block_idx}_after_attn_merge", inputs)
        return hook

    def make_mlp_output_hook(block_idx):
        def hook(module, inputs, output):
            save(f"block_{block_idx}_mlp_output", output)
        return hook

    def make_after_mlp_merge_hook(block_idx):
        def hook(module, inputs, output):
            save(f"block_{block_idx}_after_mlp_merge", output)
        return hook

    def make_mlp_hidden_hook(block_idx):
        def hook(module, inputs, output):
            save(f"block_{block_idx}_mlp_hidden", output)
        return hook

    for block_idx in range(n_layer):
        block = model.transformer.h[block_idx]
        hooks.append(block.attn.register_forward_hook(make_attn_output_hook(block_idx)))
        hooks.append(block.mlp.register_forward_pre_hook(make_after_attn_merge_hook(block_idx)))
        hooks.append(block.mlp.register_forward_hook(make_mlp_output_hook(block_idx)))
        hooks.append(block.register_forward_hook(make_after_mlp_merge_hook(block_idx)))
        hooks.append(block.mlp.silu.register_forward_hook(make_mlp_hidden_hook(block_idx)))

    def input_embed_hook(module, inputs, output):
        save("input_embed", output)

    def after_pos_emb_hook(module, inputs, output):
        save("after_pos_emb", output)

    def after_ln_f_hook(module, inputs, output):
        save("after_ln_f", output)

    hooks.append(model.input_embedding.register_forward_hook(input_embed_hook))
    hooks.append(model.transformer.drop.register_forward_hook(after_pos_emb_hook))
    hooks.append(model.transformer.ln_f.register_forward_hook(after_ln_f_hook))

    if verbose:
        print(f"Hooks registered for {n_layer} transformer blocks")
        print(f"Total hooks: {len(hooks)}")

    return hooks, activation_dict


def compute_force_probes(positions):
    """1D SHM probe targets: position x and finite-difference velocity dx.

    dx = (x_{t+1} - x_t) / Δt   where Δt = 1/20 fps = 0.05 s.

    dx is NOT in training data — model must learn it implicitly from positions.
    """
    if torch.is_tensor(positions):
        x = positions[..., 0]
        dx = (x[:, 1:] - x[:, :-1]) / 0.05
        dx_padded = torch.cat([dx, dx[:, -1:]], dim=1)
        return {"x": x.reshape(-1), "dx": dx_padded.reshape(-1)}
    x = np.asarray(positions)[..., 0]
    dx = (x[:, 1:] - x[:, :-1]) / 0.05
    dx_padded = np.concatenate([dx, dx[:, -1:]], axis=1)
    return {"x": x.flatten(), "dx": dx_padded.flatten()}


def collect_activations(model, inputs, targets, activation_dict, verbose=False):
    """Run one eval forward, capturing activations only for this call."""
    del targets  # Kept in the public interface for backward compatibility.


def collect_activations(model, inputs, targets, activation_dict, verbose=False):
    """Run one eval forward, capturing activations only for this call."""
    del targets  # Kept in the public interface for backward compatibility.
    was_training = model.training
    model.eval()
    activation_dict.clear()
    activation_dict.capture_enabled = True
    try:
        with torch.no_grad():
            predictions, _ = model(inputs, None)
            force_probes = compute_force_probes(inputs)
    finally:
        activation_dict.capture_enabled = False
        model.train(was_training)

    if not activation_dict:
        raise RuntimeError("No activations were captured")

    if verbose:
        print("Activations collected:")
        for name, value in activation_dict.items():
            print(f"  {name}: {tuple(value.shape)}")

    return predictions, force_probes


def initialize_probe_indices(train_size, eval_size, max_samples, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1000)
    train_samples = min(max_samples, train_size)
    eval_samples = min(max_samples, eval_size)
    return (
        torch.randperm(train_size, generator=generator)[:train_samples],
        torch.randperm(eval_size, generator=generator)[:eval_samples],
    )


def snapshot_activations(activation_dict):
    """Clone activations without forcing a GPU-to-CPU transfer."""
    if not activation_dict:
        raise RuntimeError("No activations were captured")
    return {name: value.detach().clone() for name, value in activation_dict.items()}


def safe_r2_score(target, prediction):
    """Original NumPy/sklearn R² helper retained for geometry probes."""
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError(f"R² shape mismatch: {target.shape} versus {prediction.shape}")
    if target.size < 2 or np.var(target) <= 1e-12:
        return float("nan")
    return float(_r2_score(target, prediction))


def reshape_token_target(value, batch_size, time_steps, name):
    """Backward-compatible NumPy target reshaping helper."""
    value = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if value.shape == (batch_size, time_steps):
        target = value
    elif value.shape == (batch_size, time_steps, 1):
        target = value[..., 0]
    elif value.shape == (batch_size * time_steps,):
        target = value.reshape(batch_size, time_steps)
    else:
        raise ValueError(f"{name}: unsupported target shape {value.shape}")
    if not np.isfinite(target).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return target


def _reshape_token_target_torch(value, batch_size, time_steps, name, device):
    value = torch.as_tensor(value, device=device)
    if value.shape == (batch_size, time_steps):
        target = value
    elif value.shape == (batch_size, time_steps, 1):
        target = value[..., 0]
    elif value.shape == (batch_size * time_steps,):
        target = value.reshape(batch_size, time_steps)
    else:
        raise ValueError(f"{name}: unsupported target shape {tuple(value.shape)}")
    if not torch.isfinite(target).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return target.to(dtype=torch.float32)


@torch.no_grad()
def _fit_single_target_probe(train_x, train_y, eval_x):
    """Fit one independent OLS probe with an intercept on the tensor device."""
    device = train_x.device
    train_x = train_x.to(device=device, dtype=torch.float32)
    eval_x = eval_x.to(device=device, dtype=torch.float32)
    train_y = train_y.to(device=device, dtype=torch.float32).reshape(-1, 1)

    x_mean = train_x.mean(dim=0, keepdim=True)
    y_mean = train_y.mean(dim=0, keepdim=True)
    centered_x = train_x - x_mean
    centered_y = train_y - y_mean

    try:
        weight = torch.linalg.lstsq(centered_x, centered_y).solution
    except RuntimeError:
        weight = torch.linalg.pinv(centered_x) @ centered_y

    train_prediction = ((train_x - x_mean) @ weight + y_mean).reshape(-1)
    eval_prediction = ((eval_x - x_mean) @ weight + y_mean).reshape(-1)
    return train_prediction, eval_prediction


@torch.no_grad()
def _safe_r2_score_torch(target, prediction):
    target = target.reshape(-1).to(dtype=torch.float32)
    prediction = prediction.reshape(-1).to(device=target.device, dtype=torch.float32)
    if target.shape != prediction.shape:
        raise ValueError(f"R² shape mismatch: {target.shape} versus {prediction.shape}")
    if target.numel() < 2:
        return float("nan")
    centered = target - target.mean()
    total = centered.square().sum()
    if total <= 1e-12:
        return float("nan")
    residual = (target - prediction).square().sum()
    return float((1.0 - residual / total).item())


def run_linear_probes(train_activation_dict, train_force_probes,
                      eval_activation_dict, eval_force_probes,
                      verbose=False):
    """Fit every target as an independent linear probe on GPU when available."""
    if set(train_activation_dict) != set(eval_activation_dict):
        raise ValueError("Train/eval activation layers do not match")

    # Use whatever keys the force function returned (12 for 2D Kepler, 2 for 1D Spring)
    probe_targets = sorted(train_force_probes.keys())
    probe_results = {}

    for layer_name, train_activations in train_activation_dict.items():
        eval_activations = eval_activation_dict[layer_name]
        device = train_activations.device
        eval_activations = eval_activations.to(device)

        batch_size, time_steps, _ = train_activations.shape
        eval_batch_size, eval_time_steps, _ = eval_activations.shape
        train_act_flat = train_activations.reshape(-1, train_activations.shape[-1])
        eval_act_flat = eval_activations.reshape(-1, eval_activations.shape[-1])
        train_act_last = train_activations[:, -1, :]
        eval_act_last = eval_activations[:, -1, :]

        probes = {}
        for probe_name in probe_targets:
            train_force = _reshape_token_target_torch(
                train_force_probes[probe_name], batch_size, time_steps,
                f"{probe_name} train", device,
            )
            eval_force = _reshape_token_target_torch(
                eval_force_probes[probe_name], eval_batch_size, eval_time_steps,
                f"{probe_name} eval", device,
            )
            train_force_flat = train_force.reshape(-1)
            eval_force_flat = eval_force.reshape(-1)
            train_force_last = train_force[:, -1]
            eval_force_last = eval_force[:, -1]

            train_pred_all, eval_pred_all = _fit_single_target_probe(
                train_act_flat, train_force_flat, eval_act_flat,
            )
            train_r2_all = _safe_r2_score_torch(train_force_flat, train_pred_all)
            eval_r2_all = _safe_r2_score_torch(eval_force_flat, eval_pred_all)

            train_pred_last, eval_pred_last = _fit_single_target_probe(
                train_act_last, train_force_last, eval_act_last,
            )
            train_r2_last = _safe_r2_score_torch(train_force_last, train_pred_last)
            eval_r2_last = _safe_r2_score_torch(eval_force_last, eval_pred_last)

            probes[probe_name] = {
                "train_r2_all": train_r2_all,
                "eval_r2_all": eval_r2_all,
                "generalization_gap_all": train_r2_all - eval_r2_all,
                "train_r2_sequence_last": train_r2_last,
                "eval_r2_sequence_last": eval_r2_last,
                "generalization_gap_sequence_last": train_r2_last - eval_r2_last,
            }
        probe_results[layer_name] = probes

    if verbose:
        print("\nLinear Probe Results (R² scores):")
        print("=" * 80)
        for layer_name, probes in probe_results.items():
            print(f"\n{layer_name}:")
            for probe_name, result in probes.items():
                print(
                    f"  {probe_name:20s}: all train/eval = "
                    f"{result['train_r2_all']:.4f}/{result['eval_r2_all']:.4f}, "
                    f"sequence-last train/eval = "
                    f"{result['train_r2_sequence_last']:.4f}/"
                    f"{result['eval_r2_sequence_last']:.4f}"
                )

    return probe_results
