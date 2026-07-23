import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _to_python(value):
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.shape == ():
                return value.item()
            return value.tolist()
        return value
    return value


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def load_result_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {key: _to_python(data[key]) for key in data.files}


def parse_log(log_path):
    """Parse old verbose logs for per-layer Fx/Fy probe R2 curves."""
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    layers = {}
    current_step = None
    in_probe = False
    current_layer = None

    step_pattern = re.compile(r"Evaluating at step (\d+)")
    layer_pattern = re.compile(r"^([a-zA-Z0-9_]+):\s*$")
    fx_pattern = re.compile(
        r"^\s+Fx\s+:\s+all train/eval = ([\d.-]+)/([\d.-]+),\s+"
        r"sequence-last train/eval = ([\d.-]+)/([\d.-]+)"
    )
    fy_pattern = re.compile(
        r"^\s+Fy\s+:\s+all train/eval = ([\d.-]+)/([\d.-]+),\s+"
        r"sequence-last train/eval = ([\d.-]+)/([\d.-]+)"
    )

    for line in lines:
        m_step = step_pattern.search(line)
        if m_step:
            current_step = int(m_step.group(1))
            in_probe = False
            continue

        if "Linear Probe Results" in line:
            in_probe = True
            continue

        if not in_probe or current_step is None:
            continue

        m_layer = layer_pattern.match(line)
        if m_layer:
            current_layer = m_layer.group(1)
            layers.setdefault(current_layer, _empty_probe_layer())
            continue

        if current_layer is None:
            continue

        for target, pattern in (("Fx", fx_pattern), ("Fy", fy_pattern)):
            match = pattern.match(line)
            if not match:
                continue
            all_train, all_eval, seq_train, seq_eval = map(float, match.groups())
            series = layers[current_layer][target]
            series["steps"].append(current_step)
            series["all_train"].append(all_train)
            series["all_eval"].append(all_eval)
            series["seq_train"].append(seq_train)
            series["seq_eval"].append(seq_eval)
            break

    return {
        layer: data
        for layer, data in layers.items()
        if data["Fx"]["steps"] or data["Fy"]["steps"]
    }


def _empty_probe_layer():
    return {
        "Fx": {"steps": [], "all_train": [], "all_eval": [], "seq_train": [], "seq_eval": []},
        "Fy": {"steps": [], "all_train": [], "all_eval": [], "seq_train": [], "seq_eval": []},
    }


def extract_probe_r2_from_npz(result, targets=("Fx", "Fy")):
    eval_results = result.get("eval_results") or []
    layers = {}

    for idx, eval_result in enumerate(eval_results):
        if not isinstance(eval_result, dict):
            continue
        probe_results = eval_result.get("probe_results")
        if not probe_results:
            continue
        step = eval_result.get("step")
        if step is None:
            eval_steps = _as_list(result.get("eval_steps"))
            step = eval_steps[idx] if idx < len(eval_steps) else idx

        for layer_name, layer_probe in probe_results.items():
            layer_series = layers.setdefault(layer_name, _empty_probe_layer())
            for target in targets:
                metrics = layer_probe.get(target) if isinstance(layer_probe, dict) else None
                if not metrics:
                    continue
                series = layer_series[target]
                series["steps"].append(int(step))
                series["all_train"].append(float(metrics.get("train_r2_all", np.nan)))
                series["all_eval"].append(float(metrics.get("eval_r2_all", np.nan)))
                series["seq_train"].append(float(metrics.get("train_r2_sequence_last", np.nan)))
                series["seq_eval"].append(float(metrics.get("eval_r2_sequence_last", np.nan)))

    return layers


def plot_losses(result, output_png):
    train_losses = np.asarray(result.get("train_losses", []), dtype=float)
    test_losses = np.asarray(result.get("test_losses", []), dtype=float)
    total_losses = np.asarray(result.get("total_losses", []), dtype=float)

    if train_losses.size == 0 and test_losses.size == 0 and total_losses.size == 0:
        print("No loss data found.")
        return False

    plt.figure(figsize=(8, 5))
    if train_losses.size:
        plt.plot(np.arange(1, train_losses.size + 1), train_losses, label="train MSE")
    if test_losses.size:
        plt.plot(np.arange(1, test_losses.size + 1), test_losses, label="test MSE")
    if total_losses.size:
        same_as_train = (
            train_losses.size == total_losses.size
            and np.allclose(total_losses, train_losses, equal_nan=True)
        )
        if not same_as_train:
            plt.plot(np.arange(1, total_losses.size + 1), total_losses, label="total loss", alpha=0.8)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Loss plot saved to {output_png}")
    return True


def plot_layer_r2(layers, output_png):
    """Plot per-layer Fx/Fy R2 curves from either old logs or structured npz."""
    layers = {
        name: data
        for name, data in layers.items()
        if data.get("Fx", {}).get("steps") or data.get("Fy", {}).get("steps")
    }
    if not layers:
        print("No layer R2 data found.")
        return False

    n_layers = len(layers)
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    for idx, (layer_name, data) in enumerate(layers.items()):
        ax = axes[idx]
        if data["Fx"]["steps"]:
            steps = data["Fx"]["steps"]
            ax.plot(steps, data["Fx"]["all_train"], "b-", label="Fx all train")
            ax.plot(steps, data["Fx"]["all_eval"], "b--", label="Fx all eval")
            ax.plot(steps, data["Fx"]["seq_train"], "r-", label="Fx seq train")
            ax.plot(steps, data["Fx"]["seq_eval"], "r--", label="Fx seq eval")
        if data["Fy"]["steps"]:
            steps = data["Fy"]["steps"]
            ax.plot(steps, data["Fy"]["all_train"], "g-", label="Fy all train")
            ax.plot(steps, data["Fy"]["all_eval"], "g--", label="Fy all eval")
            ax.plot(steps, data["Fy"]["seq_train"], "m-", label="Fy seq train")
            ax.plot(steps, data["Fy"]["seq_eval"], "m--", label="Fy seq eval")
        ax.set_title(layer_name)
        ax.set_xlabel("Step")
        ax.set_ylabel("R2")
        ax.legend(loc="best", fontsize=6)
        ax.grid(True, alpha=0.3)

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"R2 plot saved to {output_png}")
    return True


def _mean_layer_metric(split_payload, metric):
    if not isinstance(split_payload, dict):
        return np.nan
    values = []
    for layer_stats in split_payload.values():
        if isinstance(layer_stats, dict) and metric in layer_stats:
            values.append(float(layer_stats[metric]))
    return float(np.nanmean(values)) if values else np.nan


def _collect_observable_series(result):
    series = {}
    eval_results = result.get("eval_results") or []
    eval_steps = _as_list(result.get("eval_steps"))

    def add(name, step, value):
        if np.isfinite(value):
            bucket = series.setdefault(name, {"steps": [], "values": []})
            bucket["steps"].append(int(step))
            bucket["values"].append(float(value))

    for idx, eval_result in enumerate(eval_results):
        if not isinstance(eval_result, dict):
            continue
        step = eval_result.get("step")
        if step is None:
            step = eval_steps[idx] if idx < len(eval_steps) else idx

        attention = eval_result.get("attention_entropy")
        if isinstance(attention, dict):
            for split in ("train", "eval"):
                add(
                    f"attention_entropy/{split}/normalized_mean",
                    step,
                    _mean_layer_metric(attention.get(split), "normalized_mean_entropy"),
                )
                add(
                    f"attention_entropy/{split}/mean",
                    step,
                    _mean_layer_metric(attention.get(split), "mean_entropy"),
                )

        varcov = eval_result.get("variance_covariance")
        if isinstance(varcov, dict):
            for split in ("train", "eval"):
                for metric in ("mean_std", "variance_loss", "covariance_loss"):
                    add(f"variance_covariance/{split}/{metric}", step, _mean_layer_metric(varcov.get(split), metric))

        rank = eval_result.get("activation_rank")
        if isinstance(rank, dict):
            for split in ("train", "eval"):
                for metric in ("effective_rank", "normalized_effective_rank", "numerical_rank"):
                    add(f"activation_rank/{split}/{metric}", step, _mean_layer_metric(rank.get(split), metric))

        activation_sv = eval_result.get("activation_singular_values")
        if isinstance(activation_sv, dict):
            for split in ("train", "eval"):
                for metric in ("effective_rank", "numerical_rank", "max", "mean"):
                    add(f"activation_singular_values/{split}/{metric}", step, _mean_layer_metric(activation_sv.get(split), metric))

        weight_sv = eval_result.get("weight_singular_values")
        if isinstance(weight_sv, dict):
            for metric in ("effective_rank", "numerical_rank", "max", "mean"):
                add(f"weight_singular_values/{metric}", step, _mean_layer_metric(weight_sv, metric))

    return series


def plot_observables(result, output_png):
    series = _collect_observable_series(result)
    if not series:
        print("No observable data found.")
        return False

    groups = {}
    for name, values in series.items():
        group = name.split("/", 1)[0]
        groups.setdefault(group, {})[name] = values

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(9, max(3, 3 * n_groups)), squeeze=False)
    axes = axes.reshape(-1)

    for ax, (group, group_series) in zip(axes, groups.items()):
        for name, values in sorted(group_series.items()):
            label = name.split("/", 1)[1] if "/" in name else name
            ax.plot(values["steps"], values["values"], marker="o", markersize=2, label=label)
        ax.set_title(group)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"Observable plot saved to {output_png}")
    return True


def plot_npz(npz_path, output_dir=None):
    npz_path = Path(npz_path)
    output_dir = Path(output_dir) if output_dir is not None else npz_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / npz_path.stem

    result = load_result_npz(npz_path)
    plot_losses(result, f"{prefix}_loss.png")
    plot_layer_r2(extract_probe_r2_from_npz(result), f"{prefix}_r2.png")
    plot_observables(result, f"{prefix}_observables.png")


def plot_path(input_path, output_dir=None):
    input_path = Path(input_path)
    if input_path.is_dir():
        result_files = sorted((input_path / "results").glob("*.npz"))
        if not result_files:
            result_files = sorted(input_path.glob("*.npz"))
        if result_files:
            out = Path(output_dir) if output_dir is not None else input_path / "plots"
            for result_file in result_files:
                plot_npz(result_file, out)
            return
        log_path = input_path / "train.log"
        if log_path.exists():
            out = Path(output_dir) if output_dir is not None else input_path / "plots"
            out.mkdir(parents=True, exist_ok=True)
            plot_layer_r2(parse_log(log_path), out / "log_r2.png")
            return
        raise FileNotFoundError(f"No .npz results or train.log found in {input_path}")

    suffix = input_path.suffix.lower()
    if suffix == ".npz":
        plot_npz(input_path, output_dir)
    else:
        out_png = output_dir if output_dir is not None else "layer_r2_curves.png"
        plot_layer_r2(parse_log(input_path), out_png)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plotting.py <run_dir|result.npz|train.log> [output_dir_or_png]")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Input {path} not found.")
        sys.exit(1)

    plot_path(path, sys.argv[2] if len(sys.argv) > 2 else None)
