"""Generate Spring SHM trajectories — all parameters from YAML config.

Reads the same config file as train.py, guaranteeing the generated data
exactly matches what training expects.

Usage:
  python generate_spring_data.py --config configs/spring.yaml
  python generate_spring_data.py --config configs/spring.yaml --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spring_shortcuts.simulation import (
    OscillatorParameters,
    SpringRenderConfig,
    render_video,
    trajectory,
    write_video,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="Path to config YAML (same file used by train.py)")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_config(path: Path) -> dict:
    """Load and validate the YAML config.  Returns the parsed dict."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Check required sections exist
    for section in ("generation", "data", "model", "training"):
        if section not in cfg:
            raise ValueError(f"Missing config section: {section}")

    gen = cfg["generation"]
    dat = cfg["data"]
    total = int(gen["total_trajectories"])
    per_split = int(dat["num_trajectories"])
    needed = 2 * per_split

    if total < needed:
        raise ValueError(
            f"generation.total_trajectories ({total}) must be >= "
            f"2 × data.num_trajectories ({needed}).  "
            f"train.py loads 2×{per_split}={needed} trajectories."
        )
    return cfg


def generate_trajectories(
    num: int,
    base_seed: int,
    *,
    fps: int,
    num_frames: int,
    omega_low: float,
    omega_high: float,
    amplitude_low: float,
    amplitude_high: float,
) -> np.ndarray:
    """Generate `num` SHM phase-space trajectories.

    Returns float32 array of shape (num, num_frames, 2) — columns: x(t), v(t).
    """
    rng = np.random.default_rng(np.random.SeedSequence([base_seed, 7919]))
    all_trajs: list[np.ndarray] = []

    for i in range(num):
        omega = float(rng.uniform(omega_low, omega_high))
        amp   = float(rng.uniform(amplitude_low, amplitude_high))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))

        x, v = trajectory(
            OscillatorParameters(omega=omega, amplitude=amp, phase=phase),
            fps=fps, num_frames=num_frames,
        )
        all_trajs.append(np.stack([x, v], axis=1).astype(np.float32))

    return np.stack(all_trajs, axis=0)


def render_sample_videos(
    out_dir: Path,
    trajectories: np.ndarray,
    num_samples: int,
    fps: int,
    seed: int,
) -> None:
    """Render a few trajectories as .mp4 for visual inspection."""
    video_dir = out_dir / "sample_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    cfg = SpringRenderConfig()
    rng = np.random.default_rng(seed)

    indices = rng.choice(len(trajectories), size=min(num_samples, len(trajectories)), replace=False)
    for k, idx in enumerate(indices):
        x = trajectories[idx, :, 0]
        color = "red" if k % 2 == 0 else "blue"
        frames = render_video(x, color, cfg)
        path = video_dir / f"sample_{k:02d}_{color}.mp4"
        write_video(path, frames, fps)
        print(f"  sample video: {path}")


def main() -> None:
    args = parse_args()
    cfg = _load_config(args.config)

    gen   = cfg["generation"]
    dat   = cfg["data"]
    seed  = int(cfg.get("seed", 3407))

    out_dir = Path(dat["data_dir"])
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{out_dir} is not empty. Use --overwrite to replace."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    total = int(gen["total_trajectories"])
    fps   = int(gen["fps"])
    n_frames = int(gen["num_frames"])

    print(f"Generating {total} trajectories …")
    traj = generate_trajectories(
        total, base_seed=seed,
        fps=fps, num_frames=n_frames,
        omega_low=float(gen["omega_low"]),
        omega_high=float(gen["omega_high"]),
        amplitude_low=float(gen["amplitude_low"]),
        amplitude_high=float(gen["amplitude_high"]),
    )
    print(f"  shape: {traj.shape}  x∈[{traj[:,:,0].min():.4f}, {traj[:,:,0].max():.4f}]")

    # Save a single pool; train.py handles its own 50/50 split.
    np.save(out_dir / "train_trajectories.npy", traj)
    print(f"Saved: {out_dir / 'train_trajectories.npy'}  ({traj.nbytes / 1e6:.1f} MB)")

    # Sample videos
    num_videos = int(gen.get("num_sample_videos", 8))
    if num_videos > 0:
        print(f"Rendering {num_videos} sample videos …")
        render_sample_videos(out_dir, traj, num_videos, fps, seed)

    # Metadata (for train.py auto-detection)
    import torch
    meta = {
        "num_frames": n_frames,
        "fps": fps,
        "omega_range": (float(gen["omega_low"]), float(gen["omega_high"])),
        "amplitude_range": (float(gen["amplitude_low"]), float(gen["amplitude_high"])),
        "input_dim": 2,
        "columns": ["x(t)", "v(t)"],
    }
    torch.save(meta, out_dir / "metadata.pt")

    per_split = int(dat["num_trajectories"])
    print(f"\nDone.  {total} trajectories generated for {2*per_split} needed "
          f"(2 × {per_split} per-split).")
    print(f"Next: python train.py --config {args.config} --run-name spring_test --gpu 0")


if __name__ == "__main__":
    main()
