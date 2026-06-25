# Static-Dynamic Probe Results

Result source:

```text
results/kepler_static_dynamic/static_dynamic_num_trajectories_100_steps_20000_seed_1.npz
```

Run setup:

```text
training steps: 20000
num train trajectories: 100
num test trajectories: 100
static_window: 20
static_dim: 16
dynamic_dim: 32
n_layer_static: 1
n_layer_dynamic: 2
```

Final rollout performance:

| Metric | Value |
|---|---:|
| final train loss | 0.001830 |
| final test loss | 0.000802 |
| test rollout mean error | 0.143649 |
| test rollout R2 x | 0.884031 |
| test rollout R2 y | 0.886628 |

## Static Quantities

Linear-probe R2 for trajectory-level quantities.

| Quantity | From `z_static` R2 | From `z_dyn_last` R2 |
|---|---:|---:|
| `LRL_y` | 0.9787 | 0.9930 |
| `LRL_x` | 0.9763 | 0.9905 |
| `n_x` | 0.9445 | 0.8041 |
| `n_y` | 0.9363 | 0.8373 |
| `average_radius` | 0.8623 | 0.9656 |
| `b` | 0.8573 | 0.9558 |
| `a` | 0.8540 | 0.9713 |
| `LRL_angle` | 0.6847 | 0.6995 |
| `c` | 0.6209 | 0.8605 |
| `e` | 0.6175 | 0.8235 |
| `LRL_magnitude` | 0.6175 | 0.8235 |

## Dynamic Quantities

Linear-probe R2 for timestep-level quantities.

| Quantity | From repeated `z_static` R2 | From `z_dyn[t]` R2 |
|---|---:|---:|
| `x` | 0.3382 | 0.9955 |
| `y` | 0.2808 | 0.9949 |
| `F_direction_y` | 0.1671 | 0.9694 |
| `F_direction_x` | 0.2019 | 0.9681 |
| `r` | 0.5038 | 0.6193 |
| `r_squared` | 0.3837 | 0.5583 |
| `inv_r` | 0.5163 | 0.5479 |
| `Fy` | 0.0051 | 0.2946 |
| `F_magnitude` | 0.2161 | 0.2748 |
| `inv_r_squared` | 0.2161 | 0.2748 |
| `Fx` | 0.0055 | 0.2615 |
| `inv_r_cubed` | 0.0663 | 0.1200 |

## Summary

| Source latent | Static target mean R2 | Dynamic target mean R2 |
|---|---:|---:|
| `z_static` | 0.8136 | 0.2418 |
| `z_dyn` | 0.8841 | 0.5733 |

The model learns useful representations in both components. `z_static` captures orbital constants well, and `z_dyn[t]` captures local dynamic state very well, especially position and force direction. The decoupling is not yet clean because static orbital information remains highly decodable from `z_dyn_last`.
