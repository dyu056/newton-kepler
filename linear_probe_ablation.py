"""
A3 Common-Endpoint Ablation: Linear probe on untrained transformer.

Key differences from original:
1. Common endpoint=98 for ALL block sizes: context = [98-B+1, ..., 98], target = F at t=98
2. Trajectory-level split: 1000 unique trajectory IDs → 700 train / 300 test, reused across block sizes
3. Same model initialization: one model (block_size=99), same state_dict for all inputs
4. 5 model seeds [0,1,2,3,4]
5. All 13 layers probed, results saved as CSV

Usage:
    python linear_probe_ablation.py
    python linear_probe_ablation.py --model_seeds 0,1 --block_sizes 1,50,99
"""

from __future__ import annotations

import csv
import json
import os
import sys
import copy
import argparse
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from model_cv import GPTConfigCV, GPTCV
from kepler_cv_blocksize import (
    load_trajectories,
    setup_activation_hooks,
    compute_gravitational_force,
)


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "block_sizes": [1, 10, 50, 99],
    "common_endpoint": 98,
    "num_probe_trajectories": 1000,
    "probe_test_fraction": 0.30,
    "trajectory_sample_seed": 0,
    "probe_split_seed": 0,
    "model_seeds": [0],
    "num_total_trajectories": 2000,
    "n_layer": 2,
    "n_embd": 32,
    "n_head": 1,
    "attention_alpha": 0.0,
    "data_dir": "data_cv",
    "output_dir": "results/ablation_v2_model_et_length",
    "experiment_name": "A3_common_endpoint",
}

PROBE_TARGETS = [
    'F_magnitude', 'Fx', 'Fy',
    'F_direction_x', 'F_direction_y',
    'r', 'inv_r', 'r_squared', 'inv_r_squared', 'inv_r_cubed',
    'x', 'y',
]

DISPLAY_TARGETS = [
    'F_magnitude', 'Fx', 'Fy',
    'F_direction_x', 'F_direction_y',
    'r', 'inv_r_squared',
]

# Layer display names
LAYER_DISPLAY_NAMES = {
    'input_embed': 'input_embed',
    'after_pos_emb': 'after_pos_emb',
    'block_0_attn_output': 'b0_attn_out',
    'block_0_after_attn_merge': 'b0_post_attn',
    'block_0_mlp_hidden': 'b0_mlp_hid',
    'block_0_mlp_output': 'b0_mlp_out',
    'block_0_after_mlp_merge': 'b0_post_mlp',
    'block_1_attn_output': 'b1_attn_out',
    'block_1_after_attn_merge': 'b1_post_attn',
    'block_1_mlp_hidden': 'b1_mlp_hid',
    'block_1_mlp_output': 'b1_mlp_out',
    'block_1_after_mlp_merge': 'b1_post_mlp',
    'after_ln_f': 'after_ln_f',
}


# ============================================================
# Trajectory ID selection
# ============================================================
def select_probe_trajectory_ids(
    *,
    num_total_trajectories: int,
    num_probe_trajectories: int,
    sample_seed: int,
) -> np.ndarray:
    if num_probe_trajectories > num_total_trajectories:
        raise ValueError(
            "num_probe_trajectories cannot exceed num_total_trajectories"
        )
    rng = np.random.default_rng(sample_seed)
    return rng.choice(
        num_total_trajectories,
        size=num_probe_trajectories,
        replace=False,
    )


# ============================================================
# Common-endpoint context builder
# ============================================================
def build_common_endpoint_contexts(
    trajectories: torch.Tensor,
    trajectory_ids: np.ndarray,
    *,
    block_size: int,
    endpoint: int,
) -> torch.Tensor:
    """
    trajectories: [num_trajectories, num_points, 2]
    Returns: contexts: [num_selected_trajectories, block_size, 2]
    """
    if trajectories.ndim != 3:
        raise ValueError("trajectories must have shape [num_trajectories, num_points, 2]")

    _, num_points, coordinate_dim = trajectories.shape
    if coordinate_dim != 2:
        raise ValueError(f"expected coordinate_dim=2, got {coordinate_dim}")
    if endpoint < 0 or endpoint >= num_points:
        raise ValueError(f"endpoint={endpoint} is outside [0, {num_points - 1}]")
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")

    start = endpoint - block_size + 1
    if start < 0:
        raise ValueError(f"block_size={block_size} is too large for endpoint={endpoint}")

    trajectory_index = torch.as_tensor(trajectory_ids, dtype=torch.long, device=trajectories.device)
    selected = trajectories.index_select(dim=0, index=trajectory_index)
    contexts = selected[:, start: endpoint + 1, :]

    expected_shape = (len(trajectory_ids), block_size, coordinate_dim)
    if tuple(contexts.shape) != expected_shape:
        raise RuntimeError(f"unexpected context shape: {tuple(contexts.shape)}, expected {expected_shape}")

    return contexts


# ============================================================
# Endpoint target selector
# ============================================================
def select_endpoint_targets(
    target_tensor: torch.Tensor,
    trajectory_ids: np.ndarray,
    *,
    endpoint: int,
) -> torch.Tensor:
    """target_tensor: [num_trajectories, num_points, ...] or [num_trajectories, num_points]"""
    trajectory_index = torch.as_tensor(trajectory_ids, dtype=torch.long, device=target_tensor.device)
    selected = target_tensor.index_select(dim=0, index=trajectory_index)
    return selected[:, endpoint]


# ============================================================
# Parameter checksum
# ============================================================
def parameter_checksum(model) -> float:
    return sum(p.detach().double().sum().item() for p in model.parameters())


# ============================================================
# Model builder
# ============================================================
def build_model(*, n_layer=2, n_embd=32, block_size=99, seed=0, device=None):
    """Build an untrained model with fixed seed."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    GPTConfigCV.block_size = block_size
    GPTConfigCV.input_dim = 2
    GPTConfigCV.n_layer = n_layer
    GPTConfigCV.n_head = 1
    GPTConfigCV.n_embd = n_embd
    GPTConfigCV.attention_alpha = 0.0
    GPTConfigCV.bias = True

    model = GPTCV(GPTConfigCV).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model


# ============================================================
# Forward pass & collect activations
# ============================================================
def collect_activations(model, inputs):
    """Run forward pass and collect activations from all hooks."""
    hooks, activation_dict = setup_activation_hooks(model)
    with torch.no_grad():
        predictions, _ = model.forward(inputs, None)
        positions = inputs.cpu().numpy()
        gravitational_force = compute_gravitational_force(positions)
    return hooks, activation_dict, predictions, gravitational_force


# ============================================================
# Probe ALL layers with trajectory-level split
# ============================================================
def run_probes_all_layers(
    activation_dict,
    gravitational_force,
    *,
    train_ids_local: np.ndarray,
    test_ids_local: np.ndarray,
) -> dict:
    """
    Probe all layers using pre-defined trajectory-level train/test split.
    train_ids_local / test_ids_local: local indices [0, N_selected) into the activation batch.

    gravitational_force is computed on context positions (batch, ctx_len, 2).
    The endpoint (t=98) is the LAST position of each context.

    Returns: results[layer_name][target_name] = {train_r2, test_r2, train_mse, test_mse, ...}
    """
    results = {}
    batch_size = list(activation_dict.values())[0].shape[0]

    for layer_name, activations in activation_dict.items():
        # Last timestep only
        act_last = activations[:, -1, :].cpu().numpy()  # (batch, features)

        act_train = act_last[train_ids_local]
        act_test = act_last[test_ids_local]

        probes = {}
        # grav_force is computed on contexts (batch, block_size, 2),
        # so reshape to (batch, block_size) and take last column (= endpoint t=98)
        ctx_len = activations.shape[1]
        for target_name in PROBE_TARGETS:
            force_at_endpoint = gravitational_force[target_name].reshape(batch_size, ctx_len)[:, -1]
            force_train = force_at_endpoint[train_ids_local]
            force_test = force_at_endpoint[test_ids_local]

            probe = LinearRegression(fit_intercept=True)
            probe.fit(act_train, force_train)

            pred_train = probe.predict(act_train)
            pred_test = probe.predict(act_test)

            train_r2 = float(r2_score(force_train, pred_train))
            test_r2 = float(r2_score(force_test, pred_test))
            train_mse = float(mean_squared_error(force_train, pred_train))
            test_mse = float(mean_squared_error(force_test, pred_test))
            train_var = float(np.var(force_train))
            test_var = float(np.var(force_test))

            probes[target_name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_target_variance': train_var,
                'test_target_variance': test_var,
            }

        results[layer_name] = probes

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='A3 Common-Endpoint Ablation')
    parser.add_argument('--model_seeds', type=str, default=None,
                        help='Comma-separated model seeds (default: from CONFIG)')
    parser.add_argument('--block_sizes', type=str, default=None,
                        help='Comma-separated block sizes (default: from CONFIG)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--probe_split_seed', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    args = parser.parse_args()

    # Resolve config
    model_seeds = CONFIG['model_seeds']
    block_sizes = CONFIG['block_sizes']
    output_dir = args.output_dir or CONFIG['output_dir']
    probe_split_seed = args.probe_split_seed if args.probe_split_seed is not None else CONFIG['probe_split_seed']
    data_dir = args.data_dir or CONFIG['data_dir']

    if args.model_seeds is not None:
        model_seeds = [int(x) for x in args.model_seeds.split(',')]
    if args.block_sizes is not None:
        block_sizes = [int(x) for x in args.block_sizes.split(',')]

    common_endpoint = CONFIG['common_endpoint']
    num_probe_trajectories = CONFIG['num_probe_trajectories']
    probe_test_fraction = CONFIG['probe_test_fraction']
    trajectory_sample_seed = CONFIG['trajectory_sample_seed']

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ============================================================
    # Sanity checks
    # ============================================================
    num_points = 100
    num_total_trajectories = CONFIG['num_total_trajectories']

    assert common_endpoint + 1 < num_points, \
        f"endpoint={common_endpoint} must leave room for next-position target"
    assert max(block_sizes) <= common_endpoint + 1, \
        f"max block_size {max(block_sizes)} > endpoint+1 = {common_endpoint+1}"

    # ============================================================
    # Load trajectories
    # ============================================================
    print(f"\n{'='*80}")
    print(f"A3 Common-Endpoint Ablation")
    print(f"{'='*80}")
    print(f"Block sizes: {block_sizes}")
    print(f"Common endpoint: {common_endpoint}")
    print(f"Model seeds: {model_seeds}")
    print(f"Probe split seed: {probe_split_seed}")
    print(f"Trajectories: {num_probe_trajectories}/{num_total_trajectories}")
    print(f"Test fraction: {probe_test_fraction}")
    print(f"Output dir: {output_dir}")

    trajectories = load_trajectories(data_dir, num_trajectories_needed=num_total_trajectories)
    print(f"\nLoaded {trajectories.shape[0]} trajectories, shape={trajectories.shape}")

    # ============================================================
    # Select trajectory IDs & split (FIXED for ALL block sizes & seeds)
    # ============================================================
    selected_ids = select_probe_trajectory_ids(
        num_total_trajectories=num_total_trajectories,
        num_probe_trajectories=num_probe_trajectories,
        sample_seed=trajectory_sample_seed,
    )

    train_ids_global, test_ids_global = train_test_split(
        selected_ids,
        test_size=probe_test_fraction,
        random_state=probe_split_seed,
        shuffle=True,
    )

    # Assertions
    assert len(set(selected_ids.tolist())) == len(selected_ids), "Duplicate trajectory IDs!"
    assert set(train_ids_global).isdisjoint(set(test_ids_global)), "Train/test overlap!"
    assert len(train_ids_global) + len(test_ids_global) == len(selected_ids), \
        f"Split mismatch: {len(train_ids_global)}+{len(test_ids_global)} != {len(selected_ids)}"

    num_train = len(train_ids_global)
    num_test = len(test_ids_global)
    print(f"\nTrajectory split: {num_train} train + {num_test} test = {num_train + num_test} total")

    # Save trajectory ID files
    np.save(os.path.join(output_dir, 'selected_trajectory_ids.npy'), selected_ids)
    np.save(os.path.join(output_dir, 'probe_train_trajectory_ids.npy'), train_ids_global)
    np.save(os.path.join(output_dir, 'probe_test_trajectory_ids.npy'), test_ids_global)
    print(f"Trajectory IDs saved to {output_dir}/")

    # ============================================================
    # Prepare results CSV
    # ============================================================
    csv_path = os.path.join(output_dir, 'results.csv')
    csv_columns = [
        'experiment', 'architecture_variant', 'model_seed',
        'trajectory_sample_seed', 'probe_split_seed',
        'block_size', 'effective_context_length', 'endpoint',
        'layer', 'target', 'feature_dim',
        'train_r2', 'test_r2', 'train_mse', 'test_mse',
        'train_target_variance', 'test_target_variance',
        'parameter_checksum',
        'num_train_trajectories', 'num_test_trajectories',
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    # ============================================================
    # Run experiment for each model seed
    # ============================================================
    for seed_idx, model_seed in enumerate(model_seeds):
        print(f"\n{'='*60}")
        print(f"Model seed {model_seed} ({seed_idx+1}/{len(model_seeds)})")
        print(f"{'='*60}")

        # Build ONE model with max block_size=99
        model = build_model(
            n_layer=CONFIG['n_layer'],
            n_embd=CONFIG['n_embd'],
            block_size=99,  # max context
            seed=model_seed,
            device=device,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        checksum = parameter_checksum(model)
        print(f"  Parameter checksum: {checksum:.6f}")

        # Save initial state for potential reuse verification
        initial_state = copy.deepcopy(model.state_dict())

        for block_size in block_sizes:
            print(f"\n  --- block_size={block_size} ---")

            # Verify model state is unchanged from initial
            current_checksum = parameter_checksum(model)
            assert abs(current_checksum - checksum) < 1e-8, \
                f"Model parameters changed! checksum: {checksum:.6f} -> {current_checksum:.6f}"

            # Build common-endpoint contexts
            contexts_all = build_common_endpoint_contexts(
                torch.from_numpy(trajectories).float().to(device),
                selected_ids,
                block_size=block_size,
                endpoint=common_endpoint,
            )  # shape: (1000, block_size, 2)
            assert contexts_all.shape == (len(selected_ids), block_size, 2), \
                f"Unexpected context shape: {contexts_all.shape}"

            # Verify context alignment: last position == x, y at endpoint
            x_target_from_context = contexts_all[:, -1, 0].cpu()
            y_target_from_context = contexts_all[:, -1, 1].cpu()

            # Forward pass & collect activations
            hooks, activation_dict, predictions, grav_force = collect_activations(model, contexts_all)

            # Verify target alignment with gravitational force computation
            # grav_force is computed on contexts (shape: batch, block_size, 2)
            # The endpoint (t=98) is at the LAST position of each context
            batch_size, ctx_len = contexts_all.shape[0], contexts_all.shape[1]
            x_from_force = grav_force['x'].reshape(batch_size, ctx_len)[:, -1]
            y_from_force = grav_force['y'].reshape(batch_size, ctx_len)[:, -1]

            assert torch.allclose(x_target_from_context, torch.from_numpy(x_from_force).float(), atol=1e-5), \
                "x alignment failed: context[-1,0] != grav_force x at endpoint"
            assert torch.allclose(y_target_from_context, torch.from_numpy(y_from_force).float(), atol=1e-5), \
                "y alignment failed: context[-1,1] != grav_force y at endpoint"
            print(f"    Alignment assertions passed ✓ (ctx_len={ctx_len})")

            # Map global trajectory IDs to local indices [0, N_selected)
            # selected_ids are indices into [0, num_total_trajectories)
            # We need local indices [0, num_probe_trajectories) for train/test split
            global_to_local = {gid: lid for lid, gid in enumerate(selected_ids)}
            train_ids_local = np.array([global_to_local[gid] for gid in train_ids_global])
            test_ids_local = np.array([global_to_local[gid] for gid in test_ids_global])

            # Run probes on ALL layers
            probe_results = run_probes_all_layers(
                activation_dict,
                grav_force,
                train_ids_local=train_ids_local,
                test_ids_local=test_ids_local,
            )

            # Write to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                for layer_name in sorted(probe_results.keys()):
                    feature_dim = activation_dict[layer_name].shape[-1]
                    for target_name in PROBE_TARGETS:
                        r = probe_results[layer_name][target_name]
                        writer.writerow({
                            'experiment': CONFIG['experiment_name'],
                            'architecture_variant': 'two_full_blocks',
                            'model_seed': model_seed,
                            'trajectory_sample_seed': trajectory_sample_seed,
                            'probe_split_seed': probe_split_seed,
                            'block_size': block_size,
                            'effective_context_length': block_size,
                            'endpoint': common_endpoint,
                            'layer': layer_name,
                            'target': target_name,
                            'feature_dim': feature_dim,
                            'train_r2': r['train_r2'],
                            'test_r2': r['test_r2'],
                            'train_mse': r['train_mse'],
                            'test_mse': r['test_mse'],
                            'train_target_variance': r['train_target_variance'],
                            'test_target_variance': r['test_target_variance'],
                            'parameter_checksum': checksum,
                            'num_train_trajectories': num_train,
                            'num_test_trajectories': num_test,
                        })

            # Quick summary for after_ln_f
            layer = 'after_ln_f'
            if layer in probe_results:
                print(f"    after_ln_f summary:")
                for t in DISPLAY_TARGETS:
                    r = probe_results[layer][t]
                    print(f"      {t:<16s}: train_r2={r['train_r2']:.4f}, test_r2={r['test_r2']:.4f}")

            # Cleanup hooks
            for hook in hooks:
                hook.remove()
            activation_dict.clear()
            torch.cuda.empty_cache()

        # Cleanup model
        del model, initial_state
        torch.cuda.empty_cache()

    # ============================================================
    # Save metadata
    # ============================================================
    metadata = {
        'experiment': CONFIG['experiment_name'],
        'description': 'A3 common-endpoint ablation with trajectory-level split',
        'config': {
            'block_sizes': block_sizes,
            'common_endpoint': common_endpoint,
            'num_probe_trajectories': num_probe_trajectories,
            'probe_test_fraction': probe_test_fraction,
            'trajectory_sample_seed': trajectory_sample_seed,
            'probe_split_seed': probe_split_seed,
            'model_seeds': model_seeds,
            'num_total_trajectories': num_total_trajectories,
            'n_layer': CONFIG['n_layer'],
            'n_embd': CONFIG['n_embd'],
            'n_head': CONFIG['n_head'],
            'max_block_size': 99,
            'num_points_per_trajectory': 100,
        },
        'trajectory_files': {
            'selected': 'selected_trajectory_ids.npy',
            'train': 'probe_train_trajectory_ids.npy',
            'test': 'probe_test_trajectory_ids.npy',
        },
        'csv_file': 'results.csv',
        'csv_columns': csv_columns,
        'num_train_trajectories': num_train,
        'num_test_trajectories': num_test,
        'total_layers': sorted(LAYER_DISPLAY_NAMES.keys()),
        'probe_targets': PROBE_TARGETS,
        'timestamp': datetime.now().isoformat(),
        'python_version': sys.version,
    }

    # Try to get package versions
    try:
        metadata['torch_version'] = torch.__version__
    except Exception:
        pass
    try:
        import sklearn
        metadata['sklearn_version'] = sklearn.__version__
    except Exception:
        pass

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Done! Results saved to: {output_dir}/")
    print(f"  - {csv_path}")
    print(f"  - {output_dir}/metadata.json")
    print(f"  - {output_dir}/selected_trajectory_ids.npy")
    print(f"  - {output_dir}/probe_train_trajectory_ids.npy")
    print(f"  - {output_dir}/probe_test_trajectory_ids.npy")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
