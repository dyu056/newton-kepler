"""Generate Spring SHM trajectories for GPT-CV next-step prediction training.

Produces pure phase-space trajectories (x, v) with no shortcut labels.
Also renders a few sample videos for visual inspection.

Output:
  data_spring/
  ├── train_trajectories.npy   — float32 (N_train, 129, 2)  columns: x, v
  ├── eval_trajectories.npy    — float32 (N_eval,  129, 2)
  ├── metadata.pt              — dict with fps, num_frames, omega_range, etc.
  └── sample_videos/           — 8 sample .mp4 videos for visual check

Usage:
  python generate_spring_data.py
  python generate_spring_data.py --num-train 20000 --num-eval 4000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow importing from spring_shortcuts/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spring_shortcuts.simulation import (
    OscillatorParameters,
    SpringRenderConfig,
    canonical_color,
    render_video,
    trajectory,
    write_video,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("data_spring"))
    p.add_argument("--num-train", type=int, default=20000)
    p.add_argument("--num-eval", type=int, default=4000)
    p.add_argument("--num-sample-videos", type=int, default=8)
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


# ── physics parameters (frozen spring_shortcuts_v4) ──
OMEGA_LOW  = 2.2
OMEGA_HIGH = 6.4
AMPLITUDE_LOW  = 0.10
AMPLITUDE_HIGH = 0.17
FPS       = 20
NUM_FRAMES = 129


def generate_trajectories(
    num: int, base_seed: int
) -> np.ndarray:
    """Generate `num` SHM trajectories in phase space.

    ω  ~ U(2.2, 6.4)
    A  ~ U(0.10, 0.17)
    φ  ~ U(0, 2π)

    Returns:
        float32 array of shape (num, 129, 2) — columns: x(t), v(t)
    """
    all_trajs: list[np.ndarray] = []
    for i in range(num):
        rng = np.random.default_rng(np.random.SeedSequence([base_seed + i, 7919]))
        omega     = float(rng.uniform(OMEGA_LOW, OMEGA_HIGH))
        amplitude = float(rng.uniform(AMPLITUDE_LOW, AMPLITUDE_HIGH))
        phase     = float(rng.uniform(0.0, 2.0 * np.pi))

        x, v = trajectory(
            OscillatorParameters(omega=omega, amplitude=amplitude, phase=phase),
            fps=FPS, num_frames=NUM_FRAMES,
        )
        all_trajs.append(np.stack([x, v], axis=1).astype(np.float32))

    return np.stack(all_trajs, axis=0)


def render_sample_videos(
    out_dir: Path,
    trajectories: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> None:
    """Render a few trajectories as .mp4 videos for visual inspection.

    Even indices → red mass, odd indices → blue mass.  Colour here is purely
    cosmetic; it is never part of the training data.
    """
    video_dir = out_dir / "sample_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    cfg = SpringRenderConfig()

    indices = rng.choice(len(trajectories), size=min(num_samples, len(trajectories)), replace=False)
    for k, idx in enumerate(indices):
        x = trajectories[idx, :, 0]  # shape (129,)
        color: str = "red" if k % 2 == 0 else "blue"
        frames = render_video(x, color, cfg)
        path = video_dir / f"sample_{k:02d}_{color}_omega_est_{_estimate_omega(x):.1f}.mp4"
        write_video(path, frames, FPS)
        print(f"  sample video: {path}")


def _estimate_omega(x: np.ndarray) -> float:
    """Rough zero-crossing frequency estimate for labelling only."""
    x = np.asarray(x, dtype=np.float64)
    sign_changes = np.diff(np.signbit(x - x.mean()))
    half_periods = np.diff(np.flatnonzero(sign_changes))
    if len(half_periods) < 2:
        return float("nan")
    T_est = 2.0 * np.median(half_periods) / FPS
    return 2.0 * np.pi / T_est if T_est > 0 else float("nan")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{out_dir} is not empty. Use --overwrite to replace."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── Train ──
    print(f"Generating {args.num_train} training trajectories …")
    train = generate_trajectories(args.num_train, base_seed=args.seed)
    print(f"  train shape: {train.shape}  x∈[{train[:,:,0].min():.4f}, {train[:,:,0].max():.4f}]")

    # ── Eval ──
    print(f"Generating {args.num_eval} eval trajectories …")
    eval_traj = generate_trajectories(args.num_eval, base_seed=args.seed + 1_000_000)
    print(f"  eval shape:  {eval_traj.shape}  x∈[{eval_traj[:,:,0].min():.4f}, {eval_traj[:,:,0].max():.4f}]")

    # ── Save arrays ──
    np.save(out_dir / "train_trajectories.npy", train)
    np.save(out_dir / "eval_trajectories.npy", eval_traj)
    print(f"Saved: {out_dir / 'train_trajectories.npy'}  ({train.nbytes/1e6:.1f} MB)")
    print(f"Saved: {out_dir / 'eval_trajectories.npy'}  ({eval_traj.nbytes/1e6:.1f} MB)")

    # ── Sample videos ──
    print(f"Rendering {args.num_sample_videos} sample videos …")
    render_sample_videos(out_dir, train, args.num_sample_videos, rng)

    # ── Metadata ──
    import torch
    meta = {
        "num_train": args.num_train,
        "num_eval": args.num_eval,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "omega_range": (OMEGA_LOW, OMEGA_HIGH),
        "amplitude_range": (AMPLITUDE_LOW, AMPLITUDE_HIGH),
        "input_dim": 2,
        "columns": ["x(t)", "v(t)"],
        "shortcut_free": True,
    }
    torch.save(meta, out_dir / "metadata.pt")

    print("\nDone.  Next step:")
    print(f"  python train.py --config configs/spring.yaml --run-name spring_test --gpu 0")


if __name__ == "__main__":
    main()
