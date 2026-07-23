import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Publication-style visual theme ─────────────────────────────────────
FX_COLORS = {
    "all_train": "#87CEEB",
    "all_eval": "#5B9BD5",
    "seq_train": "#2166AC",
    "seq_eval": "#08306B",
}
FY_COLORS = {
    "all_train": "#F6B26B",
    "all_eval": "#E67E22",
    "seq_train": "#C44E52",
    "seq_eval": "#8C2D2D",
}
LOSS_COLORS = {
    "train": "#2166AC",
    "test": "#C44E52",
    "total": "#6A51A3",
}
OBSERVABLE_COLORS = [
    "#08306B", "#2166AC", "#5B9BD5", "#87CEEB",
    "#8C2D2D", "#C44E52", "#E67E22", "#F6B26B",
    "#6A51A3", "#4C956C",
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.7,
    "legend.fontsize": 8,
    "grid.linestyle": "--",
    "grid.alpha": 0.28,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="in")
    ax.grid(True, linestyle="--", alpha=0.28)


def _markevery(length, target_markers=9):
    return max(1, int(length) // target_markers)


def _positive_for_log(values):
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values) | (values <= 0)] = np.nan
    return values


def _pretty_layer_name(name):
    fixed = {
        "input_embed": "Input embedding",
        "after_pos_emb": "After positional embedding",
        "after_ln_f": "Final layer norm",
    }
    if name in fixed:
        return fixed[name]

    match = re.match(r"block_(\d+)_(.*)", name)
    if not match:
        return name.replace("_", " ")

    block_idx, suffix = match.groups()
    suffix_labels = {
        "attn_output": "Attention output",
        "after_attn_merge": "After attention residual",
        "mlp_fc1": "MLP linear 1 output",
        "mlp_hidden1": "MLP SiLU 1 output",
        "mlp_fc2": "MLP linear 2 output",
        "mlp_hidden2": "MLP SiLU 2 output",
        "mlp_hidden": "MLP hidden activation",
        "mlp_output": "MLP output",
        "after_mlp_merge": "After MLP residual",
    }
    label = suffix_labels.get(suffix, suffix.replace("_", " "))
    return f"Block {block_idx}\n{label}"


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
    """Plot train/test losses using the shared publication-style palette."""
    train_losses = np.asarray(result.get("train_losses", []), dtype=float)
    test_losses = np.asarray(result.get("test_losses", []), dtype=float)
    total_losses = np.asarray(result.get("total_losses", []), dtype=float)

    if train_losses.size == 0 and test_losses.size == 0 and total_losses.size == 0:
        print("No loss data found.")
        return False

    has_train_panel = train_losses.size > 0 or total_losses.size > 0
    has_test_panel = test_losses.size > 0
    n_panels = int(has_train_panel) + int(has_test_panel)

    fig, axes = plt.subplots(
        1,
        max(1, n_panels),
        figsize=(7.4 * max(1, n_panels), 5.2),
        squeeze=False,
    )
    axes = axes.reshape(-1)
    panel_idx = 0

    if has_train_panel:
        ax = axes[panel_idx]
        panel_idx += 1

        if train_losses.size:
            train_steps = np.arange(1, train_losses.size + 1)
            ax.plot(
                train_steps,
                _positive_for_log(train_losses),
                color=LOSS_COLORS["train"],
                label="Train MSE",
                linewidth=1.6,
                alpha=0.92,
            )

        if total_losses.size:
            same_as_train = (
                train_losses.size == total_losses.size
                and np.allclose(total_losses, train_losses, equal_nan=True)
            )
            if not same_as_train:
                total_steps = np.arange(1, total_losses.size + 1)
                ax.plot(
                    total_steps,
                    _positive_for_log(total_losses),
                    color=LOSS_COLORS["total"],
                    label="Total loss",
                    linewidth=1.5,
                    alpha=0.86,
                )

        ax.set_title("Training Loss")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend(frameon=False, loc="best")
        _style_axis(ax)

    if has_test_panel:
        ax = axes[panel_idx]
        test_steps = np.arange(1, test_losses.size + 1)
        ax.plot(
            test_steps,
            _positive_for_log(test_losses),
            color=LOSS_COLORS["test"],
            label="Test MSE",
            linewidth=1.6,
            alpha=0.92,
        )
        ax.set_title("Test Loss")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        ax.legend(frameon=False, loc="best")
        _style_axis(ax)

    fig.suptitle("Loss Curves", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    print(f"Loss plot saved to {output_png}")
    return True


def plot_layer_r2(layers, output_png):
    """Plot per-layer Fx/Fy R² curves from either old logs or structured NPZ."""
    layers = {
        name: data
        for name, data in layers.items()
        if data.get("Fx", {}).get("steps") or data.get("Fy", {}).get("steps")
    }
    if not layers:
        print("No layer R² data found.")
        return False

    n_layers = len(layers)
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.3 * n_rows),
        squeeze=False,
    )
    axes = axes.reshape(-1)

    curve_specs = [
        ("Fx", "all_train", "all_train", r"$F_x$ all · train", "-", None),
        ("Fx", "all_eval", "all_eval", r"$F_x$ all · eval", "--", None),
        ("Fx", "seq_train", "seq_train", r"$F_x$ last · train", "-", "o"),
        ("Fx", "seq_eval", "seq_eval", r"$F_x$ last · eval", "--", "o"),
        ("Fy", "all_train", "all_train", r"$F_y$ all · train", "-", None),
        ("Fy", "all_eval", "all_eval", r"$F_y$ all · eval", "--", None),
        ("Fy", "seq_train", "seq_train", r"$F_y$ last · train", "-", "s"),
        ("Fy", "seq_eval", "seq_eval", r"$F_y$ last · eval", "--", "s"),
    ]

    shared_handles = []
    shared_labels = []

    for idx, (layer_name, data) in enumerate(layers.items()):
        ax = axes[idx]

        for target, series_key, color_key, label, linestyle, marker in curve_specs:
            target_data = data.get(target, {})
            steps = target_data.get("steps", [])
            values = target_data.get(series_key, [])
            if not steps or not values:
                continue

            palette = FX_COLORS if target == "Fx" else FY_COLORS
            line, = ax.plot(
                steps,
                values,
                color=palette[color_key],
                linestyle=linestyle,
                marker=marker,
                markersize=3.3 if marker else 0,
                markevery=_markevery(len(steps)) if marker else None,
                markerfacecolor="white" if marker else None,
                markeredgewidth=0.9 if marker else None,
                linewidth=1.55,
                alpha=0.95,
                label=label,
            )
            if label not in shared_labels:
                shared_handles.append(line)
                shared_labels.append(label)

        ax.set_title(_pretty_layer_name(layer_name))
        ax.set_xlabel("Training step")
        ax.set_ylabel(r"$R^2$")
        ax.set_ylim(-0.05, 1.05)
        _style_axis(ax)

    for j in range(n_layers, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        r"Linear Probe Evolution: $F_x$ and $F_y$",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    if shared_handles:
        fig.legend(
            shared_handles,
            shared_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=4,
            frameon=False,
            columnspacing=1.5,
            handlelength=2.6,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output_png)
    plt.close(fig)
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
    """Plot enabled observables with a consistent categorical palette."""
    series = _collect_observable_series(result)
    if not series:
        print("No observable data found.")
        return False

    groups = {}
    for name, values in series.items():
        group = name.split("/", 1)[0]
        groups.setdefault(group, {})[name] = values

    n_groups = len(groups)
    fig, axes = plt.subplots(
        n_groups,
        1,
        figsize=(10, max(3.8, 3.7 * n_groups)),
        squeeze=False,
    )
    axes = axes.reshape(-1)

    for ax, (group, group_series) in zip(axes, groups.items()):
        for idx, (name, values) in enumerate(sorted(group_series.items())):
            raw_label = name.split("/", 1)[1] if "/" in name else name
            label = raw_label.replace("/", " · ").replace("_", " ")
            is_eval = "/eval/" in name
            marker = "s" if is_eval else "o"

            ax.plot(
                values["steps"],
                values["values"],
                color=OBSERVABLE_COLORS[idx % len(OBSERVABLE_COLORS)],
                linestyle="--" if is_eval else "-",
                marker=marker,
                markersize=3.2,
                markevery=_markevery(len(values["steps"])),
                markerfacecolor="white",
                markeredgewidth=0.85,
                label=label,
                linewidth=1.55,
                alpha=0.95,
            )

        ax.set_title(group.replace("_", " ").title())
        ax.set_xlabel("Training step")
        ax.set_ylabel("Value")
        ax.legend(
            fontsize=8,
            frameon=False,
            ncol=min(3, max(1, len(group_series))),
            loc="best",
        )
        _style_axis(ax)

    fig.suptitle(
        "Observable Evolution During Training",
        fontsize=15,
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
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