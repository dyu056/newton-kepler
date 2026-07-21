"""Activation collection and linear probes for physical and orbital quantities."""

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from model_cv import GPTConfigCV


def setup_activation_hooks(model):
    """
    Set up hooks to collect intermediate activations from the model.
    
    For each transformer block, captures:
    1. Attention output before/after merging into residual stream
    2. MLP output before/after merging into residual stream
    3. MLP hidden activation after silu
    
    Returns:
        hooks: list of registered hooks
        activation_dict: dictionary to store activations
    """
    activation_dict = {}
    hooks = []
    n_layer = GPTConfigCV.n_layer
    
    # Helper function to extract tensor from input/output
    def extract_tensor(x):
        if isinstance(x, tuple):
            return x[0].detach()
        return x.detach()
    
    # Hook for attention output BEFORE merging into residual
    def make_attn_output_hook(block_idx):
        def hook(module, input, output):
            activation_dict[f'block_{block_idx}_attn_output'] = extract_tensor(output)
        return hook
    
    # Hook for residual stream AFTER attention merge (input to MLP)
    def make_after_attn_merge_hook(block_idx):
        def hook(module, input):
            # MLP receives the residual stream after attention merge
            activation_dict[f'block_{block_idx}_after_attn_merge'] = extract_tensor(input)
        return hook
    
    # Hook for MLP output BEFORE merging into residual
    def make_mlp_output_hook(block_idx):
        def hook(module, input, output):
            activation_dict[f'block_{block_idx}_mlp_output'] = extract_tensor(output)
        return hook
    
    # Hook for residual stream AFTER MLP merge (output of block)
    def make_after_mlp_merge_hook(block_idx):
        def hook(module, input, output):
            # Block output is the residual stream after MLP merge
            activation_dict[f'block_{block_idx}_after_mlp_merge'] = extract_tensor(output)
        return hook
    
    # Hook for MLP hidden activation after silu
    def make_mlp_hidden_hook(block_idx):
        def hook(module, input, output):
            activation_dict[f'block_{block_idx}_mlp_hidden'] = extract_tensor(output)
        return hook
    
    # Register hooks for each transformer block
    for block_idx in range(n_layer):
        block = model.transformer.h[block_idx]
        
        # 1. Attention output before merge
        hook = block.attn.register_forward_hook(make_attn_output_hook(block_idx))
        hooks.append(hook)
        
        # 2. Residual after attention merge (input to MLP)
        hook = block.mlp.register_forward_pre_hook(make_after_attn_merge_hook(block_idx))
        hooks.append(hook)
        
        # 3. MLP output before merge
        hook = block.mlp.register_forward_hook(make_mlp_output_hook(block_idx))
        hooks.append(hook)
        
        # 4. Residual after MLP merge (block output)
        hook = block.register_forward_hook(make_after_mlp_merge_hook(block_idx))
        hooks.append(hook)
        
        # 5. MLP hidden activation after silu
        hook = block.mlp.silu.register_forward_hook(make_mlp_hidden_hook(block_idx))
        hooks.append(hook)
    
    # Also register hooks for input embedding and final layer norm
    def make_input_embed_hook():
        def hook(module, input, output):
            activation_dict['input_embed'] = extract_tensor(output)
        return hook
    
    def make_after_pos_emb_hook():
        def hook(module, input, output):
            activation_dict['after_pos_emb'] = extract_tensor(output)
        return hook
    
    def make_after_ln_f_hook():
        def hook(module, input, output):
            activation_dict['after_ln_f'] = extract_tensor(output)
        return hook
    
    hook = model.input_embedding.register_forward_hook(make_input_embed_hook())
    hooks.append(hook)
    
    hook = model.transformer.drop.register_forward_hook(make_after_pos_emb_hook())
    hooks.append(hook)
    
    hook = model.transformer.ln_f.register_forward_hook(make_after_ln_f_hook())
    hooks.append(hook)
    
    print(f"Hooks registered for {n_layer} transformer blocks:")
    print(f"  For each block: attn_output, after_attn_merge, mlp_output, after_mlp_merge, mlp_hidden")
    print(f"  Additional: input_embed, after_pos_emb, after_ln_f")
    print(f"  Total hooks: {len(hooks)}")
    
    return hooks, activation_dict

def compute_gravitational_force(positions):
    """
    Compute gravitational force and related quantities from positions.
    
    Args:
        positions: array of shape (batch, time, 2) with (x, y) positions
    
    Returns:
        gravitational_force: dictionary with force components and related quantities
    """
    positions = np.asarray(positions)
    r = np.sqrt(positions[:,:,0]**2 + positions[:,:,1]**2)
    r3 = r**3
    # Avoid division by zero
    r3 = np.where(r3 < 1e-10, 1e-10, r3)
    Fx = -positions[:,:,0] / r3
    Fy = -positions[:,:,1] / r3
    F_magnitude = np.sqrt(Fx**2 + Fy**2)
    
    # Store gravitational force information
    gravitational_force = {
        'Fx': Fx.flatten(),
        'Fy': Fy.flatten(),
        'F_magnitude': F_magnitude.flatten(),
        'F_direction_x': (Fx / F_magnitude).flatten(),
        'F_direction_y': (Fy / F_magnitude).flatten(),
        # Distance-related quantities
        'r': r.flatten(),                    # r = sqrt(x^2 + y^2)
        'inv_r': (1.0 / np.where(r < 1e-10, 1e-10, r)).flatten(),  # 1/r
        'r_squared': (r**2).flatten(),        # r^2
        'inv_r_squared': (1.0 / np.where(r**2 < 1e-10, 1e-10, r**2)).flatten(),  # 1/r^2
        'inv_r_cubed': (1.0 / np.where(r**3 < 1e-10, 1e-10, r**3)).flatten(),  # 1/r^3
        # Position coordinates
        'x': positions[:,:,0].flatten(),      # x coordinate
        'y': positions[:,:,1].flatten(),      # y coordinate
    }
    
    return gravitational_force

def collect_activations(model, inputs, targets, activation_dict):
    """
    Run forward pass to collect activations.
    
    Returns:
        predictions: model predictions
        gravitational_force: computed gravitational force quantities
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        predictions, _ = model(inputs, None)
        
        # Compute gravitational force for comparison
        # Move to CPU if on GPU before converting to numpy
        positions = inputs.cpu().numpy()  # (batch, time, 2)
        gravitational_force = compute_gravitational_force(positions)
    
    print("Activations collected:")
    for name in activation_dict.keys():
        print(f"  {name}: {activation_dict[name].shape}")
    
    model.train(was_training)
    return predictions, gravitational_force

def initialize_probe_indices(train_size, eval_size, max_samples, seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed + 1000)
    train_samples = min(max_samples, train_size)
    eval_samples = min(max_samples, eval_size)
    return (
        torch.randperm(train_size, generator=generator)[:train_samples],
        torch.randperm(eval_size, generator=generator)[:eval_samples],
    )

def snapshot_activations(activation_dict):
    if not activation_dict:
        raise RuntimeError("No activations were captured")
    return {name: value.detach().cpu().clone() for name, value in activation_dict.items()}

def safe_r2_score(target, prediction):
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError(f"R² shape mismatch: {target.shape} versus {prediction.shape}")
    if target.size < 2 or np.var(target) <= 1e-12:
        return float('nan')
    return float(r2_score(target, prediction))

def reshape_token_target(value, batch_size, time_steps, name):
    value = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    if value.shape == (batch_size, time_steps):
        target = value
    elif value.shape == (batch_size, time_steps, 1):
        target = value[..., 0]
    elif value.shape == (batch_size * time_steps,):
        target = value.reshape(batch_size, time_steps)
    else:
        raise ValueError(f"{name}: unsupported target shape {value.shape}")
    if not np.isfinite(target).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return target

def run_linear_probes(train_activation_dict, train_gravitational_force,
                      eval_activation_dict, eval_gravitational_force):
    """
    Run linear probes to test if intermediate representations contain
    linear directions corresponding to gravitational force.
    
    Returns:
        probe_results: dictionary mapping layer names to probe results
    """
    probe_results = {}
    
    if set(train_activation_dict) != set(eval_activation_dict):
        raise ValueError("Train/eval activation layers do not match")
    for layer_name, activations in train_activation_dict.items():
        # Flatten activations: (batch, time, features) -> (batch*time, features)
        # Move to CPU if on GPU before converting to numpy
        train_act_flat = activations.reshape(-1, activations.shape[-1]).cpu().numpy()
        eval_activations = eval_activation_dict[layer_name]
        eval_act_flat = eval_activations.reshape(-1, eval_activations.shape[-1]).cpu().numpy()
        
        # Extract last state activations: (batch, time, features) -> (batch, features)
        batch_size, time_steps, _ = activations.shape
        train_act_last = activations[:, -1, :].cpu().numpy()
        eval_act_last = eval_activations[:, -1, :].cpu().numpy()
        
        # Probe for different gravitational force components
        probes = {}
        
        # Define probe targets
        probe_targets = [
            'F_magnitude', 'F_direction_x', 'F_direction_y', 'Fx', 'Fy',
            'r', 'inv_r', 'r_squared', 'inv_r_squared', 'inv_r_cubed', 'x', 'y'
        ]
        
        for probe_name in probe_targets:
            # Reshape gravitational force back to (batch, time) to extract last timestep
            train_force = reshape_token_target(train_gravitational_force[probe_name], batch_size, time_steps,
                                               f"{probe_name} train")
            eval_force = reshape_token_target(eval_gravitational_force[probe_name], eval_activations.shape[0],
                                              eval_activations.shape[1], f"{probe_name} eval")
            train_force_flat, eval_force_flat = train_force.reshape(-1), eval_force.reshape(-1)
            train_force_last, eval_force_last = train_force[:, -1], eval_force[:, -1]
            
            # Probe on all positions
            probe = LinearRegression()
            probe.fit(train_act_flat, train_force_flat)
            train_r2_all = safe_r2_score(train_force_flat, probe.predict(train_act_flat))
            eval_r2_all = safe_r2_score(eval_force_flat, probe.predict(eval_act_flat))
            
            # Probe on last states only
            probe_last = LinearRegression()
            probe_last.fit(train_act_last, train_force_last)
            train_r2_last = safe_r2_score(train_force_last, probe_last.predict(train_act_last))
            eval_r2_last = safe_r2_score(eval_force_last, probe_last.predict(eval_act_last))
            
            probes[probe_name] = {
                'train_r2_all': train_r2_all, 'eval_r2_all': eval_r2_all,
                'generalization_gap_all': train_r2_all - eval_r2_all,
                'train_r2_sequence_last': train_r2_last,
                'eval_r2_sequence_last': eval_r2_last,
                'generalization_gap_sequence_last': train_r2_last - eval_r2_last,
            }
        
        probe_results[layer_name] = probes
    
    # Print results
    print("\nLinear Probe Results (R² scores):")
    print("=" * 80)
    for layer_name, probes in probe_results.items():
        print(f"\n{layer_name}:")
        for probe_name, result in probes.items():
            print(f"  {probe_name:20s}: all train/eval = {result['train_r2_all']:.4f}/"
                  f"{result['eval_r2_all']:.4f}, sequence-last train/eval = "
                  f"{result['train_r2_sequence_last']:.4f}/{result['eval_r2_sequence_last']:.4f}")
    
    return probe_results

def run_geometry_probes(train_activation_dict, train_orbital_params, train_trajectory_ids,
                        eval_activation_dict, eval_orbital_params, eval_trajectory_ids):
    """Fit geometry probes on train sequences and score held-out eval sequences."""
    base_targets = ['e', 'a', 'b', 'c', 'average_radius', 'LRL_x', 'LRL_y',
                    'LRL_magnitude', 'LRL_angle', 'n_x', 'n_y']
    probe_targets = base_targets + ['1/a', '1/a^2', '1/b', '1/b^2']

    def values(params, trajectory_ids):
        if len(trajectory_ids) == 0 or np.min(trajectory_ids) < 0 or np.max(trajectory_ids) >= len(params):
            raise ValueError("Geometry trajectory IDs are outside the orbital-parameter split")
        result = {name: np.asarray([params[int(i)][name] for i in trajectory_ids]) for name in base_targets}
        result['1/a'] = 1.0 / result['a']
        result['1/a^2'] = 1.0 / result['a'] ** 2
        result['1/b'] = 1.0 / result['b']
        result['1/b^2'] = 1.0 / result['b'] ** 2
        return result

    train_values = values(train_orbital_params, train_trajectory_ids)
    eval_values = values(eval_orbital_params, eval_trajectory_ids)
    results = {}
    for layer_name, train_activations in train_activation_dict.items():
        eval_activations = eval_activation_dict[layer_name]
        train_all = train_activations.reshape(-1, train_activations.shape[-1]).cpu().numpy()
        eval_all = eval_activations.reshape(-1, eval_activations.shape[-1]).cpu().numpy()
        train_last = train_activations[:, -1, :].cpu().numpy()
        eval_last = eval_activations[:, -1, :].cpu().numpy()
        layer_results = {}
        for name in probe_targets:
            train_target_all = np.repeat(train_values[name], train_activations.shape[1])
            eval_target_all = np.repeat(eval_values[name], eval_activations.shape[1])
            probe_all = LinearRegression().fit(train_all, train_target_all)
            probe_last = LinearRegression().fit(train_last, train_values[name])
            train_r2_all = safe_r2_score(train_target_all, probe_all.predict(train_all))
            eval_r2_all = safe_r2_score(eval_target_all, probe_all.predict(eval_all))
            train_r2_last = safe_r2_score(train_values[name], probe_last.predict(train_last))
            eval_r2_last = safe_r2_score(eval_values[name], probe_last.predict(eval_last))
            layer_results[name] = {
                'train_r2_all': train_r2_all, 'eval_r2_all': eval_r2_all,
                'train_r2_sequence_last': train_r2_last,
                'eval_r2_sequence_last': eval_r2_last,
            }
        results[layer_name] = layer_results
    return results