"""Generate pure SHM phase-space trajectories — numpy only, no video.

  x(t) = A · cos(ωt + φ)
  v(t) = -Aω · sin(ωt + φ)

All parameters read from the YAML config (same file used by train.py).

Usage:
  python generate_spring_data.py --config configs/spring.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_and_validate(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    gen = cfg["generation"]
    dat = cfg["data"]
    total = int(gen["total_trajectories"])
    needed = 2 * int(dat["num_trajectories"])

    if total < needed:
        raise ValueError(
            f"generation.total_trajectories ({total}) < "
            f"2 × data.num_trajectories ({needed})"
        )
    return cfg


def generate(cfg: dict) -> np.ndarray:
    """Return float32 array of shape (total, num_frames, 2)."""
    gen   = cfg["generation"]
    seed  = int(cfg.get("seed", 3407))
    total = int(gen["total_trajectories"])
    T     = int(gen["num_frames"])
    fps   = int(gen["fps"])
    w_lo  = float(gen["omega_low"])
    w_hi  = float(gen["omega_high"])
    A_lo  = float(gen["amplitude_low"])
    A_hi  = float(gen["amplitude_high"])

    rng = np.random.default_rng(np.random.SeedSequence([seed, 7919]))
    t   = np.arange(T, dtype=np.float64) / fps              # (T,)

    # Pre-allocate
    out = np.empty((total, T, 2), dtype=np.float32)

    for i in range(total):
        omega = float(rng.uniform(w_lo, w_hi))
        amp   = float(rng.uniform(A_lo, A_hi))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))

        out[i, :, 0] = amp * np.cos(omega * t + phase)          # x(t)
        out[i, :, 1] = -amp * omega * np.sin(omega * t + phase) # v(t)

    return out


def main() -> None:
    args = parse_args()
    cfg  = load_and_validate(args.config)

    dat  = cfg["data"]
    seed = int(cfg.get("seed", 3407))

    out_dir = Path(dat["data_dir"])
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} is not empty. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = cfg["generation"]
    total   = int(gen_cfg["total_trajectories"])

    print(f"Generating {total} SHM trajectories …")
    traj = generate(cfg)
    print(f"  shape: {traj.shape}  "
          f"x∈[{traj[:,:,0].min():.4f}, {traj[:,:,0].max():.4f}]  "
          f"v∈[{traj[:,:,1].min():.4f}, {traj[:,:,1].max():.4f}]")

    np.save(out_dir / "train_trajectories.npy", traj)
    print(f"Saved: {out_dir / 'train_trajectories.npy'}  ({traj.nbytes / 1e6:.1f} MB)")

    # Metadata for train.py auto-detection
    import torch
    meta = {
        "num_frames": int(gen_cfg["num_frames"]),
        "fps":        int(gen_cfg["fps"]),
        "omega_range":    (float(gen_cfg["omega_low"]), float(gen_cfg["omega_high"])),
        "amplitude_range": (float(gen_cfg["amplitude_low"]), float(gen_cfg["amplitude_high"])),
        "input_dim": 2,
        "columns": ["x(t)", "v(t)"],
    }
    torch.save(meta, out_dir / "metadata.pt")

    per_split = int(dat["num_trajectories"])
    print(f"\nDone.  {total} trajectories for {2*per_split} needed "
          f"(2 × {per_split} per-split).")
    print(f"Next: python train.py --config {args.config} --run-name spring_test --gpu 0")


if __name__ == "__main__":
    main()
