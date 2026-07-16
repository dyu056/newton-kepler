"""
Standalone linear probe evaluation.
Loads a trained model (no training), extracts activations from all layers,
and computes linear probe R² scores for force and geometry targets.

Usage:
    python linear_probe.py --model {model_cv,model_mlp} --n_layer 1 [--block_size 100]

The model must already be trained (model weights loaded from results or random init).
For already-trained results, use --results_file to load from .npz instead.
"""

import argparse
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# ── Model loader ──
def build_model(model_type, n_layer, n_embd, block_size, input_dim, device):
    if model_type == 'model_cv':
        from model_cv import GPTConfigCV, GPTCV
    elif model_type == 'model_mlp':
        from model_mlp import GPTConfigCV, GPTCV
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    kwargs = dict(n_layer=n_layer, n_embd=n_embd,
                  block_size=block_size, input_dim=input_dim)
    if model_type == 'model_cv':
        kwargs['n_head'] = 1  # must divide n_embd
    config = GPTConfigCV(**kwargs)
    model = GPTCV(config).to(device)
    model.eval()
    return model


# ── Hook registration (compatible with both Attention and MLP models) ──
def setup_hooks(model):
    activation_dict = {}
    hooks = []
    n_layer = model.config.n_layer

    def extract_tensor(x):
        if isinstance(x, tuple):
            return x[0].detach()
        return x.detach()

    def make_hook(name):
        def hook(module, input, output):
            activation_dict[name] = extract_tensor(output)
        return hook

    def make_pre_hook(name):
        def hook(module, input):
            activation_dict[name] = extract_tensor(input)
        return hook

    # Global hooks
    hooks.append(model.input_embedding.register_forward_hook(make_hook('input_embed')))
    hooks.append(model.transformer.drop.register_forward_hook(make_hook('after_pos_emb')))
    hooks.append(model.transformer.ln_f.register_forward_hook(make_hook('after_ln_f')))

    # Per-block hooks
    for block_idx in range(n_layer):
        block = model.transformer.h[block_idx]
        has_attn = hasattr(block, 'attn')

        # Pre-MLP representation
        if has_attn:
            hooks.append(block.attn.register_forward_hook(make_hook(f'block_{block_idx}_attn_output')))
        else:
            hooks.append(block.ln.register_forward_hook(make_hook(f'block_{block_idx}_attn_output')))

        # Input to MLP
        hooks.append(block.mlp.register_forward_pre_hook(make_pre_hook(f'block_{block_idx}_after_attn_merge')))

        # MLP output
        hooks.append(block.mlp.register_forward_hook(make_hook(f'block_{block_idx}_mlp_output')))

        # After MLP merge (block output)
        hooks.append(block.register_forward_hook(make_hook(f'block_{block_idx}_after_mlp_merge')))

        # MLP hidden activation
        if hasattr(block.mlp, 'silu'):
            hooks.append(block.mlp.silu.register_forward_hook(make_hook(f'block_{block_idx}_mlp_hidden')))

    print(f"Registered {len(hooks)} hooks across {n_layer} blocks")
    return hooks, activation_dict


# ── Force targets ──
def compute_force_targets(positions):
    """positions: (batch, time, 2) numpy array"""
    r = np.sqrt(positions[:,:,0]**2 + positions[:,:,1]**2)
    r3 = np.maximum(r**3, 1e-10)
    Fx = -positions[:,:,0] / r3
    Fy = -positions[:,:,1] / r3
    Fm = np.sqrt(Fx**2 + Fy**2)
    return {
        'F_magnitude': Fm.flatten(),
        'Fx': Fx.flatten(),
        'Fy': Fy.flatten(),
        'F_direction_x': (Fx / Fm).flatten(),
        'F_direction_y': (Fy / Fm).flatten(),
        'r': r.flatten(),
        'inv_r': (1.0 / np.maximum(r, 1e-10)).flatten(),
        'r_squared': (r**2).flatten(),
        'inv_r_squared': (1.0 / np.maximum(r**2, 1e-10)).flatten(),
        'inv_r_cubed': (1.0 / np.maximum(r3, 1e-10)).flatten(),
        'x': positions[:,:,0].flatten(),
        'y': positions[:,:,1].flatten(),
    }


# ── Geometry targets ──
def compute_geometry_targets(orbital_params, time_steps):
    """orbital_params: list of dicts with e,a,b,c, LRL_x, LRL_y, etc."""
    e  = np.array([p['e']  for p in orbital_params])
    a  = np.array([p['a']  for p in orbital_params])
    b  = np.array([p['b']  for p in orbital_params])
    c  = np.array([p['c']  for p in orbital_params])
    Lx = np.array([p['LRL_x'] for p in orbital_params])
    Ly = np.array([p['LRL_y'] for p in orbital_params])
    nx = np.array([p['n_x'] for p in orbital_params])
    ny = np.array([p['n_y'] for p in orbital_params])
    avg_r = np.array([p['average_radius'] for p in orbital_params])
    Lm = np.sqrt(Lx**2 + Ly**2)

    # Trajectory-level values
    vals_last = {
        'e': e, 'a': a, 'b': b, 'c': c, 'average_radius': avg_r,
        'LRL_x': Lx, 'LRL_y': Ly, 'LRL_magnitude': Lm, 'n_x': nx, 'n_y': ny,
        '1/a': 1.0/a, '1/a^2': 1.0/a**2, '1/b': 1.0/b, '1/b^2': 1.0/b**2,
    }
    # Repeated for all time steps
    vals_flat = {k: np.repeat(v, time_steps) for k, v in vals_last.items()}
    return vals_flat, vals_last


# ── Linear probe ──
def run_probes(activation_dict, target_dict_flat, target_dict_last, batch_size, time_steps, target_names):
    results = {}
    for layer_name, activations in activation_dict.items():
        act_flat = activations.reshape(-1, activations.shape[-1]).cpu().numpy()
        act_last = activations[:, -1, :].cpu().numpy()
        layer_results = {}
        for name in target_names:
            if name not in target_dict_flat:
                continue
            # All positions
            probe_all = LinearRegression()
            probe_all.fit(act_flat, target_dict_flat[name])
            r2_all = r2_score(target_dict_flat[name], probe_all.predict(act_flat))
            # Last position only
            probe_last = LinearRegression()
            probe_last.fit(act_last, target_dict_last[name])
            r2_last = r2_score(target_dict_last[name], probe_last.predict(act_last))
            layer_results[name] = {'r2': r2_all, 'r2_last': r2_last}
        results[layer_name] = layer_results
    return results


# ── Main ──
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='model_mlp', choices=['model_cv','model_mlp'])
    parser.add_argument('--n_layer', type=int, default=1)
    parser.add_argument('--n_embd', type=int, default=32)
    parser.add_argument('--block_size', type=int, default=100)
    parser.add_argument('--input_dim', type=int, default=2)
    parser.add_argument('--num_trajectories', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Build model (randomly initialized — NO training)
    print(f"\nBuilding {args.model_type}, n_layer={args.n_layer}")
    model = build_model(args.model_type, args.n_layer, args.n_embd,
                        args.block_size, args.input_dim, device)

    # 2. Load test data
    print("Loading trajectories...")
    from generate_kepler_cv import generate_trajectories
    trajectories, orbital_params, _ = generate_trajectories(
        num_trajectories=args.num_trajectories, chunk_seed_offset=1000
    )
    trajectories = trajectories.astype(np.float32)
    num_points = trajectories.shape[1]

    # 3. Register hooks
    hooks, activation_dict = setup_hooks(model)
    print(f"Layers to probe: {list(activation_dict.keys()) if False else len(hooks)} hooks")

    # 4. Forward pass + collect activations
    print("Running forward pass...")
    inputs = torch.from_numpy(trajectories).to(device)
    with torch.no_grad():
        predictions, _ = model.forward(inputs, None)
    print(f"  Input: {inputs.shape}, Predictions: {predictions.shape}")

    # 5. Compute targets
    print("Computing force targets...")
    positions = trajectories  # (batch, time, 2)
    force_flat = compute_force_targets(positions)
    # Last-timestep versions
    force_last = {}
    batch_size, T = positions.shape[:2]
    for k, v in force_flat.items():
        reshaped = v.reshape(batch_size, T)
        force_last[k] = reshaped[:, -1]

    print("Computing geometry targets...")
    geo_flat, geo_last = compute_geometry_targets(orbital_params, T)

    # 6. Run probes
    force_targets = ['F_magnitude','Fx','Fy','F_direction_x','F_direction_y',
                     'r','inv_r','r_squared','inv_r_squared','inv_r_cubed','x','y']
    print("\nRunning force probes...")
    force_results = run_probes(activation_dict, force_flat, force_last, batch_size, T, force_targets)

    geo_targets = ['e','a','b','c','average_radius','LRL_x','LRL_y','LRL_magnitude',
                   'n_x','n_y','1/a','1/a^2','1/b','1/b^2']
    print("Running geometry probes...")
    geo_results = run_probes(activation_dict, geo_flat, geo_last, batch_size, T, geo_targets)

    # 7. Print summary
    key_probes = ['F_magnitude','Fx','Fy','a','b','LRL_x','LRL_y','e']
    print(f"\n{'='*90}")
    print(f"Linear Probe R² (r2_last) — {args.model_type}, n_layer={args.n_layer}")
    print(f"{'='*90}")
    header = f"{'Layer':<30}"
    for k in key_probes:
        header += f" {k:>10}"
    print(header)
    print("-"*(30 + 11*len(key_probes)))

    # Order layers naturally
    layer_order = ['input_embed', 'after_pos_emb']
    for i in range(args.n_layer):
        for suffix in ['attn_output','after_attn_merge','mlp_output','after_mlp_merge','mlp_hidden']:
            layer_order.append(f'block_{i}_{suffix}')
    layer_order.append('after_ln_f')

    for layer in layer_order:
        if layer not in activation_dict:
            continue
        line = f"{layer:<30}"
        for k in key_probes:
            src = force_results if k in force_targets else geo_results
            r2 = src.get(layer, {}).get(k, {}).get('r2_last', float('nan'))
            line += f" {r2:>10.4f}"
        print(line)

    # Cleanup
    for h in hooks:
        h.remove()
    print("\nDone.")


if __name__ == '__main__':
    main()
