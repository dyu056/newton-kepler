"""
Plot real vs model-predicted Kepler orbit trajectories.

Loads the trained model weights (.pt saved next to the results npz) and a
trajectory from the test split, conditions on the first few points, then
generates the rest autoregressively. Draws real (blue) and predicted (red)
orbits on the same axes.

Usage:
    python plot_orbit.py --weights <model_state.pt> --npz <results.npz>
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from model_att import GPTConfigPureAttn, GPTPureAttn
from data_utils import load_trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="path to model_state.pt")
    parser.add_argument("--npz", required=True,
                        help="path to results npz (for hyperparameters)")
    parser.add_argument("--traj_index", type=int, default=None,
                        help="test trajectory index (random if omitted)")
    parser.add_argument("--cond_points", type=int, default=20,
                        help="number of conditioning points")
    parser.add_argument("--out", default="orbit_real_vs_pred.png")
    args = parser.parse_args()

    # Hyperparameters from the npz
    d = np.load(args.npz, allow_pickle=True)
    block_size = int(d["block_size"])
    n_embd = int(d["n_embd"])
    n_head = int(d["n_head"])
    train_size = int(d["train_size"])
    test_size = int(d["test_size"])
    num_trajectories = train_size + test_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild model and load weights
    cfg = GPTConfigPureAttn(block_size=block_size, n_embd=n_embd, n_head=n_head)
    model = GPTPureAttn(cfg).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and any(k.startswith("transformer") or k.startswith("input_embedding") for k in state):
        model.load_state_dict(state)
    else:
        # Saved inside a results dict wrapper
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
    model.eval()

    # Pick a test trajectory
    trajectories = load_trajectories("data_cv", num_trajectories_needed=num_trajectories)
    test_traj = trajectories[train_size:]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(test_traj)) if args.traj_index is None else args.traj_index
    real = test_traj[idx]  # (100, 2)

    # Condition on first cond_points, generate the rest
    with torch.no_grad():
        cond = torch.from_numpy(real[:args.cond_points]).float().unsqueeze(0).to(device)
        gen = model.generate(cond, max_new_tokens=100 - args.cond_points)
        pred_full = gen.squeeze(0).cpu().numpy()

    # Plot — colorblind-safe (Okabe-Ito palette) + shape/linestyle double encoding
    fig, ax = plt.subplots(figsize=(7, 7))

    # Find the orbit period: smallest P such that x_t ~ x_{t-P}
    # The full 100-point trajectory wraps the ellipse ~6 times; we draw ONE
    # clean revolution to avoid the multi-loop chord tangle.
    diffs = [
        (P, np.linalg.norm(real[P:] - real[:-P], axis=1).mean())
        for P in range(5, 40)
    ]
    period = min(diffs, key=lambda pd: pd[1])[0]
    one_rev = real[:period + 1]

    # Real orbit: solid blue line over ONE revolution (a clean ellipse),
    # with an arrow showing the direction of motion
    ax.plot(one_rev[:, 0], one_rev[:, 1], "-", color="#0072B2", lw=2.2,
            label=f"Real orbit (1 revolution = {period} points)")
    ax.annotate("", xy=one_rev[5], xytext=one_rev[3],
                arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.8))

    # Conditioning points: thin black polyline in temporal order, with
    # "start" and "end" labels only (the window spans >1 revolution since
    # the orbit period is 16 points, so per-point labels would overlap)
    cond_pts = real[:args.cond_points]
    ax.plot(cond_pts[:, 0], cond_pts[:, 1], "-", color="black", lw=1.0,
            alpha=0.55, label=f"Conditioning path (first {args.cond_points} points)")
    ax.plot(cond_pts[0, 0], cond_pts[0, 1], "s", color="black",
            ms=8, mfc="none", mew=1.8)
    ax.annotate("start", (cond_pts[0, 0], cond_pts[0, 1]),
                textcoords="offset points", xytext=(8, -12),
                fontsize=9, color="black", fontweight="bold")
    ax.plot(cond_pts[-1, 0], cond_pts[-1, 1], "s", color="black",
            ms=8, mfc="none", mew=1.8)
    ax.annotate("end", (cond_pts[-1, 0], cond_pts[-1, 1]),
                textcoords="offset points", xytext=(8, 6),
                fontsize=9, color="black", fontweight="bold")

    # Model prediction: dashed orange line with open circles
    pred_pts = pred_full[args.cond_points:]
    ax.plot(pred_pts[:, 0], pred_pts[:, 1], "--", color="#E69F00", lw=1.8,
            label=f"Model prediction ({len(pred_pts)} generated points)")
    ax.plot(pred_pts[:, 0], pred_pts[:, 1], "o", color="#E69F00",
            ms=4, mfc="none", mew=1.0)
    # Mark the prediction start
    ax.plot(pred_pts[0, 0], pred_pts[0, 1], "*", color="#D55E00",
            ms=14, label="Prediction start")

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(f"Kepler orbit: real vs pure-attention model\n"
                 f"(traj {idx}, first {args.cond_points} points conditioned, "
                 f"{100 - args.cond_points} predicted)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved plot -> {Path(args.out).resolve()}")

    # Report errors on the generated part
    gen_part = pred_full[args.cond_points:]
    real_part = real[args.cond_points:]
    mse = np.mean((gen_part - real_part) ** 2)
    r2 = 1.0 - np.sum((gen_part - real_part) ** 2) / np.sum(
        (real_part - real_part.mean(0)) ** 2)
    print(f"Generated-part MSE: {mse:.6f}")
    print(f"Generated-part R^2: {r2:.4f}")


if __name__ == "__main__":
    main()
