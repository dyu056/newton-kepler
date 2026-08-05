# Spring Shortcuts — Data Generator

Deterministic synthetic video generator for the spring shortcut benchmark
(spring_shortcuts_v4).

## Physics

A mass on a spring undergoes undamped simple harmonic motion:

```
x(t) = A · cos(ωt + φ)
```

The mass is rendered as a red or blue circle connected to a fixed anchor by a
zigzag spring line.  The only "shortcut" variable in Stage 1 is **colour**:
red is correlated with a slow frequency band and blue with a fast frequency
band.

| Band | ω range (rad/s) | Canonical colour |
|------|-----------------|------------------|
| slow | [2.2, 3.0]      | red (235,48,48)  |
| fast | [5.2, 6.4]      | blue (48,96,235) |

Colour only identifies a **band**, never a single oscillator frequency —
ω is sampled uniformly and continuously within each band.

## Quick start

```bash
pip install -r requirements.txt
python build.py --config data.yaml --root data/spring_shortcuts_v4
```

This generates three splits under `data/spring_shortcuts_v4/videos/`:

| Split  | Base seeds | Videos | Conflict pairs |
|--------|-----------|--------|----------------|
| sanity | 8         | 16     | yes            |
| train  | 1024      | 2048   | no             |
| eval   | 64        | 256    | yes            |

Train contains only aligned colour-frequency pairs (red=slow, blue=fast).
Eval additionally contains conflict pairs (red=fast, blue=slow) sharing the
exact same physical trajectory.

## Output structure

```
data/spring_shortcuts_v4/
├── videos/
│   ├── train/
│   │   ├── metadata.csv        # all sample metadata
│   │   ├── audit.json           # balance / invariance audit
│   │   ├── sample_<hash>.mp4    # 129-frame 128×128 RGB video
│   │   ├── sample_<hash>.json   # per-sample metadata
│   │   └── traj_<hash>.npz      # ground-truth x(t), v(t)
│   ├── eval/
│   └── sanity/
├── metadata/
│   └── generation_config_resolved.yaml
└── build_summary.json
```

## Per-sample metadata

```json
{
  "sample_id": "sample_<blake2b>",
  "trajectory_id": "traj_<blake2b>",
  "pair_id": "pair_<blake2b>",
  "true_band": "slow",
  "color_label": "red",
  "variant": "aligned",
  "omega_true": 2.647,
  "amplitude": 0.142,
  "phase": 3.871,
  "x_star": 0.0432,
  "v_star": -0.218
}
```

Sample IDs are opaque blake2b hashes — they contain no information about
frequency band, colour, or variant.
