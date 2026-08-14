"""Activation collection and linear probes for physical and orbital quantities.

Also hosts the attention-entropy capture used during training, so the
measurement frequency can be controlled from the config yaml.
"""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from model_att import setup_pure_attn_hooks


class _ActivationStore(dict):
    """Dictionary used by hooks, with capture disabled outside probe forwards."""

    capture_enabled: bool = False


def setup_activation_hooks(model, verbose=False):
    """Register hooks compatible with the pure-attention architecture."""
    return setup_pure_attn_hooks(model, verbose=verbose)


# ─────────────────────── attention entropy capture ───────────────────────

def set_attention_entropy_capture(model, enabled):
    """Enable or disable entropy collection on every transformer block."""
    for block in model.transformer.h:
        block.attn.capture_attention_stats = bool(enabled)
        if not enabled:
            block.attn.last_attention_stats = None


def collect_attention_entropy(model):
    """Snapshot the entropy statistics produced by the latest forward pass.

    Returns a per-block dict of the stats recorded by
    ``PureAttention._record_attention_stats``:
    mean_entropy, normalized_mean_entropy, entropy_by_head,
    normalized_entropy_by_head, sequence_length.
    """
    results = {}
    for block_index, block in enumerate(model.transformer.h):
        stats = block.attn.last_attention_stats
        if stats is None:
            raise RuntimeError(
                f"No attention statistics captured for block {block_index}. "
                "Enable capture before the forward pass."
            )
        results[f"block_{block_index}"] = dict(stats)
    return results


def compute_gravitational_force(positions):
    """Compute the original twelve force/position probe targets."""
    if torch.is_tensor(positions):
        x = positions[..., 0]
        y = positions[..., 1]
        r = torch.sqrt(x.square() + y.square())
        r3 = r.pow(3).clamp_min(1e-10)
        fx = -x / r3
        fy = -y / r3
        magnitude = torch.sqrt(fx.square() + fy.square())
        safe_magnitude = magnitude.clamp_min(1e-10)
        safe_r = r.clamp_min(1e-10)
        safe_r2 = r.square().clamp_min(1e-10)
        safe_r3 = r.pow(3).clamp_min(1e-10)
        return {
            "Fx": fx.reshape(-1),
            "Fy": fy.reshape(-1),
            "F_magnitude": magnitude.reshape(-1),
            "F_direction_x": (fx / safe_magnitude).reshape(-1),
            "F_direction_y": (fy / safe_magnitude).reshape(-1),
            "r": r.reshape(-1),
            "inv_r": (1.0 / safe_r).reshape(-1),
            "r_squared": r.square().reshape(-1),
            "inv_r_squared": (1.0 / safe_r2).reshape(-1),
            "inv_r_cubed": (1.0 / safe_r3).reshape(-1),
            "x": x.reshape(-1),
            "y": y.reshape(-1),
        }

    positions = np.asarray(positions)
    r = np.sqrt(positions[:, :, 0] ** 2 + positions[:, :, 1] ** 2)
    r3 = np.where(r ** 3 < 1e-10, 1e-10, r ** 3)
    fx = -positions[:, :, 0] / r3
    fy = -positions[:, :, 1] / r3
    magnitude = np.sqrt(fx ** 2 + fy ** 2)
    safe_magnitude = np.where(magnitude < 1e-10, 1e-10, magnitude)
    return {
        "Fx": fx.flatten(),
        "Fy": fy.flatten(),
        "F_magnitude": magnitude.flatten(),
        "F_direction_x": (fx / safe_magnitude).flatten(),
        "F_direction_y": (fy / safe_magnitude).flatten(),
        "r": r.flatten(),
        "inv_r": (1.0 / np.where(r < 1e-10, 1e-10, r)).flatten(),
        "r_squared": (r ** 2).flatten(),
        "inv_r_squared": (1.0 / np.where(r ** 2 < 1e-10, 1e-10, r ** 2)).flatten(),
        "inv_r_cubed": (1.0 / np.where(r ** 3 < 1e-10, 1e-10, r ** 3)).flatten(),
        "x": positions[:, :, 0].flatten(),
        "y": positions[:, :, 1].flatten(),
    }


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
            gravitational_force = compute_gravitational_force(inputs)
    finally:
        activation_dict.capture_enabled = False
        model.train(was_training)

    if not activation_dict:
        raise RuntimeError("No activations were captured")

    if verbose:
        print("Activations collected:")
        for name, value in activation_dict.items():
            print(f"  {name}: {tuple(value.shape)}")

    return predictions, gravitational_force


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
    return float(r2_score(target, prediction))


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


def run_linear_probes(train_activation_dict, train_gravitational_force,
                      eval_activation_dict, eval_gravitational_force,
                      verbose=False):
    """Fit every target as an independent linear probe on GPU when available."""
    if set(train_activation_dict) != set(eval_activation_dict):
        raise ValueError("Train/eval activation layers do not match")

    probe_targets = [
        "F_magnitude", "F_direction_x", "F_direction_y", "Fx", "Fy",
        "r", "inv_r", "r_squared", "inv_r_squared", "inv_r_cubed", "x", "y",
    ]
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
                train_gravitational_force[probe_name], batch_size, time_steps,
                f"{probe_name} train", device,
            )
            eval_force = _reshape_token_target_torch(
                eval_gravitational_force[probe_name], eval_batch_size, eval_time_steps,
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


def run_geometry_probes(train_activation_dict, train_orbital_params, train_trajectory_ids,
                        eval_activation_dict, eval_orbital_params, eval_trajectory_ids):
    """Fit geometry probes on train sequences and score held-out eval sequences."""
    base_targets = [
        "e", "a", "b", "c", "average_radius", "LRL_x", "LRL_y",
        "LRL_magnitude", "LRL_angle", "n_x", "n_y",
    ]
    probe_targets = base_targets + ["1/a", "1/a^2", "1/b", "1/b^2"]

    def values(params, trajectory_ids):
        if (
            len(trajectory_ids) == 0
            or np.min(trajectory_ids) < 0
            or np.max(trajectory_ids) >= len(params)
        ):
            raise ValueError("Geometry trajectory IDs are outside the orbital-parameter split")
        result = {
            name: np.asarray([params[int(i)][name] for i in trajectory_ids])
            for name in base_targets
        }
        result["1/a"] = 1.0 / result["a"]
        result["1/a^2"] = 1.0 / result["a"] ** 2
        result["1/b"] = 1.0 / result["b"]
        result["1/b^2"] = 1.0 / result["b"] ** 2
        return result

    train_values = values(train_orbital_params, train_trajectory_ids)
    eval_values = values(eval_orbital_params, eval_trajectory_ids)
    results = {}
    for layer_name, train_activations in train_activation_dict.items():
        eval_activations = eval_activation_dict[layer_name]
        train_all = train_activations.reshape(-1, train_activations.shape[-1]).cpu().numpy()
        eval_all = eval_activations.reshape(-1, eval_activations.shape[-1]).cpu().numpy()
        train_last = train_activations[:, -1, :].cpu().numpy()
        eval_last = eval_activations[:, -1, :].cpu().numpy()
        layer_results = {}
        for name in probe_targets:
            train_target_all = np.repeat(train_values[name], train_activations.shape[1])
            eval_target_all = np.repeat(eval_values[name], eval_activations.shape[1])
            probe_all = LinearRegression().fit(train_all, train_target_all)
            probe_last = LinearRegression().fit(train_last, train_values[name])
            train_r2_all = safe_r2_score(train_target_all, probe_all.predict(train_all))
            eval_r2_all = safe_r2_score(eval_target_all, probe_all.predict(eval_all))
            train_r2_last = safe_r2_score(train_values[name], probe_last.predict(train_last))
            eval_r2_last = safe_r2_score(eval_values[name], probe_last.predict(eval_last))
            layer_results[name] = {
                "train_r2_all": train_r2_all,
                "eval_r2_all": eval_r2_all,
                "train_r2_sequence_last": train_r2_last,
                "eval_r2_sequence_last": eval_r2_last,
            }
        results[layer_name] = layer_results
    return results