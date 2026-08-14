"""
Render the attention-matrix evolution as a grayscale GIF.

Loads the training results npz, extracts the attention matrices captured
every N steps (averaged over batch and heads, shape T x T), and renders
each as a grayscale heatmap frame. Frames are assembled into an animated
GIF with imageio.

Usage:
    python make_attn_gif.py --npz <results.npz> --out attention_evolution.gif
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def load_matrices(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    steps = np.asarray(d["attention_matrix_steps"], dtype=int)
    matrices = d["attention_matrices"]
    return steps, matrices


def render_gif(steps, matrices, out_path, fps=10, dpi=110):
    out_path = Path(out_path)
    frames_dir = out_path.with_suffix("") / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    vmin = min(float(np.nanmin(m)) for m in matrices)
    vmax = max(float(np.nanmax(m)) for m in matrices)

    frame_paths = []
    for step, matrix in zip(steps, matrices):
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        im = ax.imshow(
            matrix, cmap="Greys", aspect="auto",
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(f"Step {step}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Key position", fontsize=10)
        ax.set_ylabel("Query position", fontsize=10)
        ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        frame_path = frames_dir / f"attn_{step:06d}.png"
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        frame_paths.append(frame_path)

    images = [imageio.imread(str(p)) for p in frame_paths]
    imageio.mimsave(str(out_path), images, fps=fps, loop=0)
    print(f"Saved {len(images)} frames -> {out_path.resolve()}")
    print(f"Frames kept in {frames_dir.resolve()} (delete if not needed)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True,
                        help="path to the training results npz")
    parser.add_argument("--out", default="attention_evolution.gif")
    parser.add_argument("--fps", type=float, default=10)
    args = parser.parse_args()

    steps, matrices = load_matrices(args.npz)
    if len(matrices) == 0:
        raise SystemExit(
            "No attention matrices found in npz — was "
            "observe.attention_matrix.enabled true in the config?"
        )
    print(f"Loaded {len(matrices)} matrices (steps {steps[0]}..{steps[-1]})")
    render_gif(steps, matrices, args.out, fps=args.fps)


if __name__ == "__main__":
    main()
