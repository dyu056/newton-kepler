"""
Plot loss curves and attention entropy from the training results npz.

Usage:
    python plot_training_curves.py --npz <results.npz>
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    train_losses = np.asarray(d["train_losses"], dtype=float)
    test_losses = np.asarray(d["test_losses"], dtype=float)

    # attention entropy from the independent every-N capture
    ent_steps = np.asarray(d["attention_entropy_steps"], dtype=int)
    ent_stats = d["attention_entropy_stats"]
    entropy = np.array([
        np.mean([b["normalized_mean_entropy"] for b in s.values()])
        for s in ent_stats
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curves
    ax = axes[0]
    ax.plot(np.arange(1, len(train_losses) + 1), train_losses,
            color="#08306B", lw=1.5, label="train")
    ax.plot(np.linspace(1, len(train_losses), len(test_losses)),
            test_losses, color="#4C956C", lw=1.2, alpha=0.8, label="test")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("MSE")
    ax.set_title(f"Loss (final train {train_losses[-1]:.5f})")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.3)

    # Attention entropy
    ax = axes[1]
    ax.plot(ent_steps, entropy, color="#6A51A3", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("normalized attention entropy")
    ax.set_title(f"Attention entropy (min {entropy.min():.4f} @step {ent_steps[entropy.argmin()]})")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("pure_attn lr0.01 lossall_shifted (every-position supervision)",
                 fontweight="bold")
    fig.tight_layout()
    out = Path(args.npz).with_name("training_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
