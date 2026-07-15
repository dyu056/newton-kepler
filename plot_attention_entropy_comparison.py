"""Plot baseline and entropy-regularized attention-entropy training curves."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results/kepler_cv_blocksize")
PLOTS_DIR = Path("plots")
BASELINE_RESULT = RESULTS_DIR / (
    "results_block_size_100_num_trajectories_10000_noise_scale_0.1_"
    "loss_mask_all_n_steps_20000.npz"
)
BASELINE_LOG = Path(
    "run_logs/kepler_cv_blocksize_baseline_entropy_tracked_20000_steps.log"
)
REGULARIZED_RESULT = RESULTS_DIR / (
    "results_block_size_100_num_trajectories_10000_noise_scale_0.1_"
    "loss_mask_all_attention_entropy_reg_0.001_start_0_end_none_"
    "n_steps_20000.npz"
)
OUTPUT = PLOTS_DIR / "attention_entropy_curve_baseline_vs_1e-3_20000_steps.png"


def load_baseline_entropy():
    """Prefer the result archive, but support the current partially saved log."""
    if BASELINE_RESULT.exists():
        result = np.load(BASELINE_RESULT, allow_pickle=True)
        if "attention_entropies" in result:
            entropy = np.asarray(result["attention_entropies"], dtype=float)
            finite = np.isfinite(entropy)
            if finite.any():
                return np.arange(1, len(entropy) + 1)[finite], entropy[finite], "result"

    text = BASELINE_LOG.read_text()
    matches = re.findall(
        r"Step (\d+).*?Attention Entropy: ([0-9]+(?:\.[0-9]+)?)", text
    )
    if not matches:
        raise ValueError(f"No baseline attention entropy found in {BASELINE_LOG}")
    steps, entropy = zip(*((int(step), float(value)) for step, value in matches))
    return np.asarray(steps), np.asarray(entropy), "log"


def main():
    baseline_steps, baseline_entropy, baseline_source = load_baseline_entropy()
    regularized = np.load(REGULARIZED_RESULT, allow_pickle=True)
    regularized_entropy = np.asarray(regularized["attention_entropies"], dtype=float)
    regularized_steps = np.arange(1, len(regularized_entropy) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        baseline_steps,
        baseline_entropy,
        color="tab:blue",
        linewidth=2.2,
        label="Baseline ($\\lambda=0$)",
    )
    ax.plot(
        regularized_steps,
        regularized_entropy,
        color="tab:orange",
        linewidth=1.2,
        alpha=0.9,
        label="Entropy reg. ($\\lambda=10^{-3}$)",
    )
    ax.set(
        title="Attention entropy during training: baseline vs $\\lambda=10^{-3}$",
        xlabel="Training step",
        ylabel="Normalized attention entropy",
        xlim=(0, 20_000),
        ylim=(0, 1.02),
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=True)

    if baseline_source == "log" and baseline_steps[-1] < 20_000:
        ax.axvline(baseline_steps[-1], color="tab:blue", linestyle=":", alpha=0.7)
        ax.annotate(
            f"baseline log ends at step {baseline_steps[-1]:,}",
            xy=(baseline_steps[-1], baseline_entropy[-1]),
            xytext=(baseline_steps[-1] + 700, min(0.98, baseline_entropy[-1] + 0.06)),
            arrowprops={"arrowstyle": "->", "color": "tab:blue"},
            color="tab:blue",
        )

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
