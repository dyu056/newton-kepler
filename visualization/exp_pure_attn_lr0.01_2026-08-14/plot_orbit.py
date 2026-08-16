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
    num_trajectories = int(d["num_trajectories"])

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
    trajectories = load_trajectories("data_cv", num_trajectories_needed=2 * num_trajectories)
    test_traj = trajectories[num_trajectories:]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(test_traj)) if args.traj_index is None else args.traj_index
    real = test_traj[idx]  # (100, 2)

    # Condition on first cond_points, generate the rest
    with torch.no_grad():
        cond = torch.from_numpy(real[:args.cond_points]).float().unsqueeze(0).to(device)
        gen = model.generate(cond, max_new_tokens=100 - args.cond_points)
        pred_full = gen.squeeze(0).cpu().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(real[:, 0], real[:, 1], "o-", color="#2166AC", ms=3, lw=1.2,
            label="Real (test trajectory)")
    ax.plot(pred_full[:args.cond_points, 0], pred_full[:args.cond_points, 1],
            "s", color="#888888", ms=4, label="Conditioning points")
    ax.plot(pred_full[args.cond_points:, 0], pred_full[args.cond_points:, 1],
            "o-", color="#C44E52", ms=3, lw=1.2, label="Model prediction")

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
