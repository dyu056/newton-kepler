"""
Plot everything from the d256 final-step-only experiment:
- loss curve (train/test)
- final attention matrix (100x100 grayscale)
- weight singular-value spectra
- activation singular-value spectra (train/eval per layer)
- final evaluation summary (entropy, effective rank)

Usage:
    python plot_final_snapshot.py --npz <results.npz>
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
    out_dir = Path(args.npz).parent

    train_losses = np.asarray(d["train_losses"], dtype=float)
    test_losses = np.asarray(d["test_losses"], dtype=float)

    # ── Fig 1: loss curve ──
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(1, len(train_losses) + 1), train_losses,
            color="#0072B2", lw=1.5, label="train")
    ax.plot(np.linspace(1, len(train_losses), len(test_losses)),
            test_losses, color="#E69F00", lw=1.2, alpha=0.85, label="test")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("MSE")
    ax.set_title(f"Loss curve (d256, SGD full-batch, 50000 steps) — "
                 f"final train {train_losses[-1]:.5f}")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    # ── Fig 2: final attention matrix ──
    mat = d["attention_matrices"][0]  # (100, 100)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    im = ax.imshow(mat, cmap="Greys", aspect="auto",
                   interpolation="nearest")
    ax.set_xlabel("Key position"); ax.set_ylabel("Query position")
    ax.set_title(f"Final attention matrix @step {int(d['attention_matrix_steps'][0])} "
                 f"(100 x 100, averaged over batch)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "attention_matrix_final.png", dpi=150)
    plt.close(fig)

    # ── Fig 3: weight singular values ──
    final_eval = d["eval_results"][-1]
    wsv = final_eval["weight_singular_values"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, st in wsv.items():
        if isinstance(st, dict) and "singular_values" in st:
            sv = np.asarray(st["singular_values"], dtype=float)
            ax.plot(np.arange(1, len(sv) + 1), sv, marker="o", ms=3,
                    lw=1.2, label=f"{name} (er={st.get('effective_rank', 0):.1f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("singular value index"); ax.set_ylabel("magnitude")
    ax.set_title("Weight singular-value spectra @final step")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "weight_singular_values.png", dpi=150)
    plt.close(fig)

    # ── Fig 4: activation singular values (train) ──
    asv = final_eval["activation_singular_values"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for split, ax in zip(["train", "eval"], axes):
        for layer, st in asv[split].items():
            if isinstance(st, dict) and "singular_values" in st:
                sv = np.asarray(st["singular_values"], dtype=float)
                ax.plot(np.arange(1, len(sv) + 1), sv, marker="o", ms=3,
                        lw=1.2, label=f"{layer} (er={st.get('effective_rank', 0):.1f})")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("singular value index"); ax.set_ylabel("magnitude")
        ax.set_title(f"Activation singular values ({split}) @final step")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "activation_singular_values.png", dpi=150)
    plt.close(fig)

    # ── Summary text ──
    ar = final_eval["activation_rank"]
    ae = final_eval["attention_entropy"]
    print("=" * 60)
    print("FINAL EVALUATION SUMMARY (step 50000)")
    print("=" * 60)
    print(f"final train loss: {train_losses[-1]:.6f}")
    print(f"final test loss : {test_losses[-1]:.6f}")
    for split in ["train", "eval"]:
        print(f"--- {split} ---")
        for layer, st in ar[split].items():
            print(f"  {layer:22s} effective_rank={st['effective_rank']:.2f} "
                  f"numerical_rank={st['numerical_rank']}")
        ent = np.mean([b["normalized_mean_entropy"]
                       for b in ae[split].values()])
        print(f"  normalized attn entropy: {ent:.4f}")
    print(f"\nPlots saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
