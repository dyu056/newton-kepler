"""
Kepler orbit trajectory prediction using continuous vision transformer with variable block size.

This module generates Kepler orbit trajectories and trains a GPT-based model
to predict future positions. It includes linear probing to analyze what
physical quantities the model learns to represent.

Key differences from kepler_cv.py:
- Supports variable block_size (can be smaller than num_points_per_trajectory)
- When block_size < num_points_per_trajectory, chops trajectories into sequences
- Supports loss_mask parameter: 'all' (compute loss on all tokens) or 'last' (only last token)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import importlib.util
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from model_mlp import GPTConfigCV, GPTCV

# Configuration
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

num_points_per_trajectory = 100


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_gpu_memory_stats():
    """Get GPU memory usage statistics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}


def print_gpu_memory_stats(label=""):
    """Print GPU memory usage statistics."""
    stats = get_gpu_memory_stats()
    if torch.cuda.is_available():
        print(f"{label}GPU Memory - Allocated: {stats['allocated_gb']:.3f} GB, "
              f"Reserved: {stats['reserved_gb']:.3f} GB, "
              f"Max Allocated: {stats['max_allocated_gb']:.3f} GB")
    return stats


def clear_gpu_cache():
    """Clear GPU cache to free up memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_trajectories(data_dir='data_cv', num_trajectories_needed=None):
    """
    Load continuous trajectories from the data_cv folder.
    Supports both chunked format (new) and single file format (old, for backward compatibility).
    
    Args:
        data_dir: Directory containing the saved trajectories
        num_trajectories_needed: Number of trajectories to load (None = load all available)
    
    Returns:
        trajectories: array of shape (num_trajectories, num_points, 2)
    """
    # Check for chunked format first
    metadata_path = os.path.join(data_dir, 'metadata.pt')
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, weights_only=False)
        chunk_size = metadata.get('chunk_size', 10000)
        num_chunks = metadata.get('num_chunks', 0)
        total_available = metadata.get('num_trajectories', 0)
        
        # Determine how many trajectories to load
        if num_trajectories_needed is None:
            num_trajectories_needed = total_available
        else:
            num_trajectories_needed = min(num_trajectories_needed, total_available)
        
        # Ensure num_trajectories_needed is an integer
        num_trajectories_needed = int(num_trajectories_needed)
        chunk_size = int(chunk_size)
        num_chunks = int(num_chunks)
        
        # Only try to load chunks if we have chunks available
        if num_chunks > 0 and num_trajectories_needed > 0:
            print(f"Loading {num_trajectories_needed:,} trajectories from {num_chunks} chunks...")
            
            # Calculate which chunks we need
            num_chunks_needed = (num_trajectories_needed + chunk_size - 1) // chunk_size
            num_chunks_needed = min(num_chunks_needed, num_chunks)
            num_chunks_needed = int(num_chunks_needed)
            
            # Load chunks
            chunks = []
            trajectories_loaded = 0
            
            for chunk_idx in range(num_chunks_needed):
                chunk_filename = os.path.join(data_dir, f'trajectories_chunk_{chunk_idx:06d}.pt')
                if os.path.exists(chunk_filename):
                    chunk_data = torch.load(chunk_filename, weights_only=False)
                    if isinstance(chunk_data, torch.Tensor):
                        chunk_data = chunk_data.numpy()
                    
                    remaining_needed = num_trajectories_needed - trajectories_loaded
                    if chunk_data.shape[0] <= remaining_needed:
                        chunks.append(chunk_data)
                        trajectories_loaded += chunk_data.shape[0]
                    else:
                        # Only take what we need
                        chunks.append(chunk_data[:remaining_needed])
                        trajectories_loaded += remaining_needed
                        break
                else:
                    print(f"Warning: Chunk {chunk_idx} not found. Falling back to old format.")
                    chunks = []  # Clear chunks to trigger fallback
                    break
            
            # Concatenate chunks if we successfully loaded any
            if chunks:
                trajectories = np.concatenate(chunks, axis=0)
                # Ensure we have exactly the number needed
                if trajectories.shape[0] > num_trajectories_needed:
                    trajectories = trajectories[:num_trajectories_needed]
                print(f"Loaded {trajectories.shape[0]:,} trajectories from chunks")
                return trajectories
        
        # If chunks weren't available or loading failed, fall through to old format
        print(f"Chunked format not available or incomplete, falling back to old format...")
    
    # Fallback to old format (single file) for backward compatibility
    pt_path = os.path.join(data_dir, 'trajectories.pt')
    npy_path = os.path.join(data_dir, 'trajectories.npy')
    
    if os.path.exists(pt_path):
        trajectories = torch.load(pt_path, weights_only=False)
        if isinstance(trajectories, torch.Tensor):
            trajectories = trajectories.numpy()
        
        # Limit to num_trajectories_needed if specified
        if num_trajectories_needed is not None and trajectories.shape[0] > num_trajectories_needed:
            trajectories = trajectories[:num_trajectories_needed]
        
        print(f"Loaded {trajectories.shape[0]:,} trajectories from {pt_path}")
        return trajectories
    elif os.path.exists(npy_path):
        trajectories = np.load(npy_path)
        
        # Limit to num_trajectories_needed if specified
        if num_trajectories_needed is not None and trajectories.shape[0] > num_trajectories_needed:
            trajectories = trajectories[:num_trajectories_needed]
        
        print(f"Loaded {trajectories.shape[0]:,} trajectories from {npy_path}")
        return trajectories
    else:
        raise FileNotFoundError(
            f"Trajectories not found in {data_dir}. "
            f"Please run generate_kepler_cv.py first to generate the dataset."
        )


def chop_trajectories_into_sequences(trajectories, block_size, seed=None):
    """
    Chop trajectories into sequences of length block_size + 1 when block_size < num_points_per_trajectory.
    
    Randomly selects num_points_per_trajectory // block_size sequences per trajectory to avoid
    creating too large a dataset. For example:
    - block_size=50: randomly select 2 pieces (100/50) of length 51
    - block_size=10: randomly select 10 pieces (100/10) of length 11
    
    Args:
        trajectories: array of shape (num_trajectories, num_points, 2)
        block_size: block size for the model
        seed: random seed for reproducibility (optional)
    
    Returns:
        inputs: array of shape (num_sequences, block_size, 2)
        targets: array of shape (num_sequences, 1, 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_trajectories, num_points, _ = trajectories.shape
    seq_length = block_size + 1  # block_size input points + 1 target point
    num_sequences_per_trajectory = num_points // block_size  # e.g., 100 // 50 = 2, 100 // 10 = 10
    
    all_inputs = []
    all_targets = []
    sequence_trajectory_ids = []
    
    for trajectory_id, traj in enumerate(trajectories):
        # Calculate valid starting positions (must have at least seq_length points remaining)
        max_start = num_points - seq_length + 1
        
        # Randomly select num_sequences_per_trajectory starting positions
        if max_start <= num_sequences_per_trajectory:
            # If we can't select enough unique positions, use all available
            start_positions = np.arange(max_start)
        else:
            # Randomly sample without replacement
            start_positions = np.random.choice(max_start, size=num_sequences_per_trajectory, replace=False)
        
        # Create sequences starting at selected positions
        for i in start_positions:
            input_seq = traj[i:i+block_size]  # (block_size, 2)
            target_seq = traj[i+block_size:i+block_size+1]  # (1, 2) - just the next point
            all_inputs.append(input_seq)
            all_targets.append(target_seq)
            sequence_trajectory_ids.append(trajectory_id)
    
    inputs = np.array(all_inputs)  # (num_sequences, block_size, 2)
    targets = np.array(all_targets)  # (num_sequences, 1, 2)
    
    return inputs, targets, np.asarray(sequence_trajectory_ids, dtype=np.int64)


def setup_model(block_size, n_layer=1, n_embd=32, device=None, mlp_mult=4):
    """Setup and initialize the GPT model for continuous vision."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    GPTConfigCV.block_size = block_size
    GPTConfigCV.input_dim = 2  # 2D coordinates (x, y)
    GPTConfigCV.n_layer = n_layer
    GPTConfigCV.n_head = 1
    GPTConfigCV.n_embd = n_embd
    GPTConfigCV.attention_alpha = 0.0
    GPTConfigCV.bias = True
    GPTConfigCV.mlp_mult = mlp_mult

    model = GPTCV(GPTConfigCV)
    model = model.to(device)
    return model


def compute_loss_with_mask(predictions, targets, loss_mask='all'):
    """
    Compute MSE loss with optional masking.
    
    Args:
        predictions: tensor of shape (batch, seq_len, 2)
        targets: tensor of shape (batch, target_len, 2)
        loss_mask: 'all' to compute loss on all positions, 'last' to compute only on last position
    
    Returns:
        loss: scalar tensor
    """
    # Handle target shape: if 2D, add dimension
    if targets.dim() == 2:
        targets = targets.unsqueeze(-1)  # (b, t) -> (b, t, 1)
    
    # Handle case where targets might have different sequence length than predictions
    if targets.size(1) < predictions.size(1):
        # Only use the last predictions that match target length
        predictions = predictions[:, -targets.size(1):, :]  # Take last t_target positions
    elif targets.size(1) > predictions.size(1):
        raise ValueError(f"Target sequence length {targets.size(1)} > prediction length {predictions.size(1)}")
    
    if loss_mask == 'all':
        # Compute loss on all positions
        predictions_flat = predictions.reshape(-1)  # shape (b*t*d,)
        targets_flat = targets.reshape(-1)  # shape (b*t*d,)
        loss = F.mse_loss(predictions_flat, targets_flat)
    elif loss_mask == 'last':
        # Compute loss only on the last position
        predictions_last = predictions[:, -1, :].reshape(-1)  # shape (b*d,)
        targets_last = targets[:, -1, :].reshape(-1)  # shape (b*d,)
        loss = F.mse_loss(predictions_last, targets_last)
    else:
        raise ValueError(f"loss_mask must be 'all' or 'last', got {loss_mask}")
    
    return loss


def train_model(model, train_inputs, train_targets, test_inputs, test_targets, train_trajectories, test_trajectories,
                 n_steps=1001, lr=1e-3, weight_decay=0.0, noise_scale=0.1, prob_freq=100, batch_size=128, 
                 loss_mask='all', seed=1, train_orbital_params=None, eval_orbital_params=None,
                 train_sequence_trajectory_ids=None, eval_sequence_trajectory_ids=None):
    """
    Train the model on the trajectory data with periodic evaluation.
    
    Args:
        model: Model to train
        train_inputs: Training input trajectories
        train_targets: Training target trajectories
        test_inputs: Test input trajectories
        test_targets: Test target trajectories
        train_trajectories: Full original training trajectories for evaluation
        test_trajectories: Full original test trajectories for evaluation
        n_steps: Number of training steps
        lr: Learning rate
        weight_decay: Weight decay
        noise_scale: Scale of noise added during training
        prob_freq: Frequency of evaluation (every N steps)
        batch_size: Batch size for training (default: 128)
        loss_mask: 'all' to compute loss on all tokens, 'last' to compute only on last token
        seed: Random seed
    Returns:
        Dictionary containing:
            train_losses: list of training losses
            test_losses: list of test losses
            eval_results: list of evaluation results at each evaluation step
            eval_steps: list of step numbers where evaluation was performed
            memory_stats: dictionary with memory usage statistics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    test_losses = []
    eval_results = []
    eval_steps = []
    
    # Setup activation hooks once (will be reused for all evaluations)
    hooks, activation_dict = setup_activation_hooks(model)
    
    # Initial memory stats
    initial_memory = print_gpu_memory_stats("Initial: ")
    
    # Get training data size
    num_train_samples = train_inputs.shape[0]
    probe_train_indices, probe_eval_indices = initialize_probe_indices(
        train_inputs.shape[0], test_inputs.shape[0], 1000, seed
    )
    probe_train_trajectory_ids = train_sequence_trajectory_ids[probe_train_indices.numpy()]
    probe_eval_trajectory_ids = eval_sequence_trajectory_ids[probe_eval_indices.numpy()]
    probe_metadata = {
        'schema_version': 2, 'fit_split': 'model_train', 'eval_split': 'model_test',
        'primary_metric': 'eval_r2', 'sampling_method': 'torch.randperm',
        'sampling_with_replacement': False, 'fixed_indices_across_checkpoints': True,
        'probe_seed': seed + 1000,
        'num_train_draws': probe_train_indices.numel(),
        'num_eval_draws': probe_eval_indices.numel(),
        'num_unique_train_sequences': probe_train_indices.unique().numel(),
        'num_unique_eval_sequences': probe_eval_indices.unique().numel(),
        'num_unique_train_trajectories': int(np.unique(probe_train_trajectory_ids).size),
        'num_unique_eval_trajectories': int(np.unique(probe_eval_trajectory_ids).size),
    }
    
    for i in range(n_steps):

        if i == n_steps // 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        # Sample a random batch from training data
        # train_inputs is on CPU, so we sample indices and move to GPU
        batch_indices = torch.randint(0, num_train_samples, (batch_size,))
        batch_inputs = train_inputs[batch_indices].to(device)
        batch_targets = train_targets[batch_indices].to(device)
        
        # Training step
        inputs_noised = batch_inputs + torch.randn_like(batch_inputs) * noise_scale
        predictions, _ = model.forward(inputs_noised, None)  # Get predictions without loss
        train_loss = compute_loss_with_mask(predictions, batch_targets, loss_mask=loss_mask)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(train_loss.item())
        
        # Clear intermediate variables to save memory
        del inputs_noised, predictions, batch_inputs, batch_targets, batch_indices
        
        # Compute test loss (no gradient) - use batch
        with torch.no_grad():
            test_batch_indices = torch.randint(0, test_inputs.shape[0], (batch_size,))
            test_batch_inputs = test_inputs[test_batch_indices].to(device)
            test_batch_targets = test_targets[test_batch_indices].to(device)
            test_predictions, _ = model.forward(test_batch_inputs, None)
            test_loss = compute_loss_with_mask(test_predictions, test_batch_targets, loss_mask=loss_mask)
            del test_batch_inputs, test_batch_targets, test_batch_indices, test_predictions
            test_losses.append(test_loss.item())
        
        if i % 100 == 0:
            memory_stats = get_gpu_memory_stats()
            print(f"Step {i}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}, "
                    f"GPU Memory: {memory_stats['allocated_gb']:.3f} GB")
            # Periodically clear GPU cache to prevent memory fragmentation
            if i % 500 == 0 and i > 0:
                clear_gpu_cache()
        
        # Evaluate at specified frequency
        if i % prob_freq == 0:
            print(f"\nEvaluating at step {i+1}...")
            eval_step_results = {}
            
            # Load only a small sample for activation collection (don't reload entire dataset)
            # Use existing train_inputs/train_targets instead of reloading
            probe_train_inputs = train_inputs[probe_train_indices].to(device)
            probe_train_targets = train_targets[probe_train_indices].to(device)
            probe_eval_inputs = test_inputs[probe_eval_indices].to(device)
            probe_eval_targets = test_targets[probe_eval_indices].to(device)

            activation_dict.clear()
            train_predictions, train_force = collect_activations(
                model, probe_train_inputs, probe_train_targets, activation_dict
            )
            train_activation_dict = snapshot_activations(activation_dict)
            activation_dict.clear()
            eval_predictions, eval_force = collect_activations(
                model, probe_eval_inputs, probe_eval_targets, activation_dict
            )
            eval_activation_dict = snapshot_activations(activation_dict)

            probe_results = run_linear_probes(
                train_activation_dict, train_force, eval_activation_dict, eval_force
            )
            eval_step_results['probe_results'] = probe_results
            eval_step_results['probe_metadata'] = probe_metadata
            
            # Run geometry probes if orbital parameters are available
            if train_orbital_params is not None and eval_orbital_params is not None:
                geometry_probe_results = run_geometry_probes(
                    train_activation_dict, train_orbital_params, probe_train_trajectory_ids,
                    eval_activation_dict, eval_orbital_params, probe_eval_trajectory_ids
                )
                eval_step_results['geometry_probe_results'] = geometry_probe_results
            else:
                eval_step_results['geometry_probe_results'] = None
            
            # Clear activations from GPU after probing to free memory
            for key in list(activation_dict.keys()):
                del activation_dict[key]
            activation_dict.clear()
            del train_predictions, eval_predictions, train_force, eval_force
            del train_activation_dict, eval_activation_dict
            
            # Generate trajectory and compute error stats for train and test
            conditioning_length = 50
            block_size = model.config.block_size
            
            # Train error stats - use smaller sample and move to GPU only when needed
            # For evaluation, we need full trajectories, not chopped sequences
            train_sample_size = min(500, train_trajectories.shape[0])  # Reduced from 1000
            train_sample_indices = torch.randint(0, train_trajectories.shape[0], (train_sample_size,))
            train_sample_trajectories = train_trajectories[train_sample_indices.numpy()]  # Keep on CPU
            # Use full trajectories for evaluation (first conditioning_length points as input)
            train_sample_inputs = torch.from_numpy(train_sample_trajectories[:, :conditioning_length]).float().to(device)  # Move to GPU
            error_stats_train = generate_trajectory_and_compute_error(
                model, train_sample_inputs, train_sample_trajectories, conditioning_length, block_size=block_size
            )
            del train_sample_inputs, train_sample_indices
            
            # Test error stats - use smaller sample and move to GPU only when needed
            test_sample_size = min(500, test_trajectories.shape[0])  # Reduced from 1000
            test_sample_indices = torch.randint(0, test_trajectories.shape[0], (test_sample_size,))
            test_sample_trajectories = test_trajectories[test_sample_indices.numpy()]  # Keep on CPU
            # Use full trajectories for evaluation (first conditioning_length points as input)
            test_sample_inputs = torch.from_numpy(test_sample_trajectories[:, :conditioning_length]).float().to(device)  # Move to GPU
            error_stats_test = generate_trajectory_and_compute_error(
                model, test_sample_inputs, test_sample_trajectories, conditioning_length, block_size=block_size
            )
            del test_sample_inputs, test_sample_indices
            
            # Clear sample data
            del probe_train_inputs, probe_train_targets, probe_eval_inputs, probe_eval_targets
            clear_gpu_cache()
            eval_step_results['error_stats_train'] = error_stats_train
            eval_step_results['error_stats_test'] = error_stats_test
            eval_step_results['step'] = i + 1
            
            eval_results.append(eval_step_results)
            eval_steps.append(i + 1)
            
            # Clear GPU cache after evaluation
            clear_gpu_cache()
            
            memory_stats = print_gpu_memory_stats(f"After evaluation at step {i+1}: ")
            print(f"Evaluation at step {i+1} completed.\n")
    
    # Final memory stats
    final_memory = print_gpu_memory_stats("Final: ")
    peak_memory = get_gpu_memory_stats()

    for hook in hooks:
        hook.remove()
    activation_dict.clear()
    clear_gpu_cache()
    
    memory_stats = {
        'initial': initial_memory,
        'final': final_memory,
        'peak': peak_memory,
    }

    print(f"Final memory stats: {memory_stats}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'eval_results': eval_results,
        'eval_steps': eval_steps,
        'memory_stats': memory_stats,
    }


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
        
        # 1. Attention output before merge (or LN output for MLP-only models)
        has_attn = hasattr(block, 'attn')
        if has_attn:
            hook = block.attn.register_forward_hook(make_attn_output_hook(block_idx))
        else:
            hook = block.ln.register_forward_hook(make_attn_output_hook(block_idx))
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


def load_orbital_params(data_dir='data_cv', num_trajectories_needed=None):
    """
    Load orbital parameters from the data_cv folder.
    Supports chunked format (orbital_params_chunk_*.py files).
    
    Args:
        data_dir: Directory containing the saved orbital parameters
        num_trajectories_needed: Number of trajectories to load (None = load all available)
    
    Returns:
        orbital_params: list of dicts with keys: e, a, b, c, average_radius
    """
    # Check for chunked format
    metadata_path = os.path.join(data_dir, 'metadata.pt')
    if os.path.exists(metadata_path):
        metadata = torch.load(metadata_path, weights_only=False)
        chunk_size = metadata.get('chunk_size', 10000)
        num_chunks = metadata.get('num_chunks', 0)
        total_available = metadata.get('num_trajectories', 0)
        
        # Determine how many trajectories to load
        if num_trajectories_needed is None:
            num_trajectories_needed = total_available
        else:
            num_trajectories_needed = min(num_trajectories_needed, total_available)
        
        # Ensure num_trajectories_needed is an integer
        num_trajectories_needed = int(num_trajectories_needed)
        chunk_size = int(chunk_size)
        num_chunks = int(num_chunks)
        
        if num_chunks > 0 and num_trajectories_needed > 0:
            print(f"Loading {num_trajectories_needed:,} orbital parameters from {num_chunks} chunks...")
            
            # Calculate which chunks we need
            num_chunks_needed = (num_trajectories_needed + chunk_size - 1) // chunk_size
            num_chunks_needed = min(num_chunks_needed, num_chunks)
            num_chunks_needed = int(num_chunks_needed)
            
            # Load chunks
            all_orbital_params = []
            trajectories_loaded = 0
            
            for chunk_idx in range(num_chunks_needed):
                chunk_filename = os.path.join(data_dir, f'orbital_params_chunk_{chunk_idx:06d}.py')
                if os.path.exists(chunk_filename):
                    # Import the orbital_params from the Python file
                    spec = importlib.util.spec_from_file_location(f"orbital_params_chunk_{chunk_idx}", chunk_filename)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    chunk_data = module.orbital_params
                    
                    remaining_needed = num_trajectories_needed - trajectories_loaded
                    if len(chunk_data) <= remaining_needed:
                        all_orbital_params.extend(chunk_data)
                        trajectories_loaded += len(chunk_data)
                    else:
                        # Only take what we need
                        all_orbital_params.extend(chunk_data[:remaining_needed])
                        trajectories_loaded += remaining_needed
                        break
                else:
                    print(f"Warning: Orbital parameters chunk {chunk_idx} not found.")
                    break
            
            if all_orbital_params:
                print(f"Loaded {len(all_orbital_params):,} orbital parameters from chunks")
                return all_orbital_params
    
    # If chunks weren't available, return empty list
    print(f"Orbital parameters not found in {data_dir}")
    return []


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


def generate_trajectory_and_compute_error(model, inputs, trajectories, conditioning_length=50, block_size=None):
    """
    Generate trajectory continuation using the model for all trajectories.
    
    Args:
        model: trained model
        inputs: input trajectories (for conditioning)
        trajectories: full original trajectories (for comparison)
        conditioning_length: number of initial steps to use as conditioning
        block_size: block size of the model (needed if block_size < num_points_per_trajectory)
    
    Returns:
        error_stats: dictionary with aggregated error statistics across all trajectories
    """
    model.eval()
    
    num_trajectories = inputs.shape[0]
    num_points_per_trajectory = trajectories.shape[1]
    
    # If block_size < num_points_per_trajectory, we need to use sliding window generation
    if block_size is not None and block_size < num_points_per_trajectory:
        # Use sliding window approach: use last block_size points to predict next point
        num_steps_to_generate = num_points_per_trajectory - conditioning_length
        
        print(f"Evaluating on {num_trajectories} trajectories...")
        print(f"Conditioning length: {conditioning_length}, Generating {num_steps_to_generate} more steps per trajectory...")
        print(f"Using sliding window with block_size={block_size}")
        
        with torch.no_grad():
            # Use first conditioning_length steps as initial conditioning
            generated_seq = inputs[:, :conditioning_length].clone()  # (num_trajectories, conditioning_length, 2)
            
            # Generate one point at a time using the last block_size points
            for step in range(num_steps_to_generate):
                # Use the last block_size points as input
                current_input = generated_seq[:, -block_size:, :]  # (num_trajectories, block_size, 2)
                
                # Generate next point
                predictions, _ = model.forward(current_input, None)
                next_point = predictions[:, -1, :].unsqueeze(1)  # (num_trajectories, 1, 2)
                
                # Append to generated sequence
                generated_seq = torch.cat([generated_seq, next_point], dim=1)
            
            # Convert to numpy (move to CPU if on GPU)
            generated_np = generated_seq.cpu().numpy()  # (num_trajectories, total_length, 2)
            true_trajectories_np = trajectories  # (num_trajectories, num_points_per_trajectory, 2)
            
            # Clear GPU tensor
            del generated_seq
            
            # Extract only the generated portion (exclude conditioning_length)
            generated_only = generated_np[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)
            
            # Extract the corresponding true trajectory portion (also exclude conditioning_length)
            true_generated = true_trajectories_np[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)
            
            # Calculate errors only on the generated portion (not conditioning)
            # Align sequences in case of length mismatch
            min_len = min(generated_only.shape[1], true_generated.shape[1])
            generated_aligned = generated_only[:, :min_len, :]  # (num_trajectories, min_len, 2)
            true_aligned = true_generated[:, :min_len, :]  # (num_trajectories, min_len, 2)
            
            # Compute position errors for all trajectories
            position_errors = np.sqrt(np.sum((generated_aligned - true_aligned)**2, axis=2))  # (num_trajectories, min_len)
            all_position_errors = position_errors.flatten()  # Flatten to get all errors
            # Compute mean across trajectories before deleting
            position_errors_mean = np.mean(position_errors, axis=0)  # (min_len,)
            
            # Compute R² scores for each trajectory
            all_r2_x = []
            all_r2_y = []
            for traj_idx in range(num_trajectories):
                r2_x = r2_score(true_aligned[traj_idx, :, 0], generated_aligned[traj_idx, :, 0])
                r2_y = r2_score(true_aligned[traj_idx, :, 1], generated_aligned[traj_idx, :, 1])
                all_r2_x.append(r2_x)
                all_r2_y.append(r2_y)
            
            # Clear intermediate variables
            del generated_np, true_trajectories_np, generated_only, true_generated, generated_aligned, true_aligned, position_errors
            clear_gpu_cache()
    else:
        # Standard generation (block_size >= num_points_per_trajectory)
        num_steps_to_generate = num_points_per_trajectory - conditioning_length
        
        print(f"Evaluating on {num_trajectories} trajectories...")
        print(f"Conditioning length: {conditioning_length}, Generating {num_steps_to_generate} more steps per trajectory...")
        
        with torch.no_grad():
            # Use first N steps as conditioning for all trajectories (batch processing)
            conditioning_seq = inputs[:, :conditioning_length]  # shape: (num_trajectories, conditioning_length, 2)
            
            # Generate remaining steps for all trajectories in batch
            generated_seq = model.generate(conditioning_seq, max_new_tokens=num_steps_to_generate)
            # generated_seq shape: (num_trajectories, conditioning_length + num_steps_to_generate, 2)
            
            # Convert to numpy (move to CPU if on GPU)
            generated_np = generated_seq.cpu().numpy()  # (num_trajectories, total_length, 2)
            true_trajectories_np = trajectories  # (num_trajectories, num_points_per_trajectory, 2)
            
            # Clear GPU tensor
            del generated_seq, conditioning_seq
            
            # Extract only the generated portion (exclude conditioning_length)
            # The generated_seq includes conditioning + generated, so we take only the generated part
            generated_only = generated_np[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)
            
            # Extract the corresponding true trajectory portion (also exclude conditioning_length)
            true_generated = true_trajectories_np[:, conditioning_length:, :]  # (num_trajectories, num_steps_to_generate, 2)
            
            # Calculate errors only on the generated portion (not conditioning)
            # Align sequences in case of length mismatch
            min_len = min(generated_only.shape[1], true_generated.shape[1])
            generated_aligned = generated_only[:, :min_len, :]  # (num_trajectories, min_len, 2)
            true_aligned = true_generated[:, :min_len, :]  # (num_trajectories, min_len, 2)
            
            # Compute position errors for all trajectories
            position_errors = np.sqrt(np.sum((generated_aligned - true_aligned)**2, axis=2))  # (num_trajectories, min_len)
            all_position_errors = position_errors.flatten()  # Flatten to get all errors
            # Compute mean across trajectories before deleting
            position_errors_mean = np.mean(position_errors, axis=0)  # (min_len,)
            
            # Compute R² scores for each trajectory
            all_r2_x = []
            all_r2_y = []
            for traj_idx in range(num_trajectories):
                r2_x = r2_score(true_aligned[traj_idx, :, 0], generated_aligned[traj_idx, :, 0])
                r2_y = r2_score(true_aligned[traj_idx, :, 1], generated_aligned[traj_idx, :, 1])
                all_r2_x.append(r2_x)
                all_r2_y.append(r2_y)
            
            # Clear intermediate variables
            del generated_np, true_trajectories_np, generated_only, true_generated, generated_aligned, true_aligned, position_errors
            clear_gpu_cache()
    
    # Aggregate statistics
    # position_errors_mean was computed in the branch above
    mean_error = np.mean(all_position_errors)
    max_error = np.max(all_position_errors)
    std_error = np.std(all_position_errors)
    
    mean_r2_x = np.mean(all_r2_x)
    mean_r2_y = np.mean(all_r2_y)
    std_r2_x = np.std(all_r2_x)
    std_r2_y = np.std(all_r2_y)
    
    error_stats = {
        'position_errors': position_errors_mean,
        'mean_error': mean_error,
        'std_error': std_error,
        'max_error': max_error,
        'mean_r2_x': mean_r2_x,
        'std_r2_x': std_r2_x,
        'mean_r2_y': mean_r2_y,
        'std_r2_y': std_r2_y,
        'all_r2_x': all_r2_x,
        'all_r2_y': all_r2_y,
    }
    
    print(f"\nError Statistics (aggregated across all {num_trajectories} trajectories):")
    print(f"Mean position error: {mean_error:.6f} ± {std_error:.6f}")
    print(f"Max position error: {max_error:.6f}")
    print(f"R² scores:")
    print(f"  X coordinate: {mean_r2_x:.6f} ± {std_r2_x:.6f}")
    print(f"  Y coordinate: {mean_r2_y:.6f} ± {std_r2_y:.6f}")
    
    return error_stats


def train_one_model(block_size=100, noise_scale=0.1, lr=1e-3, n_layer=1, n_embd=16, num_trajectories=10000,
                    n_steps=1001, prob_freq=100, loss_mask='all', seed=1, batch_size=128,
                    mlp_mult=4):
    """
    Train a single model with specified hyperparameters.
    
    Args:
        block_size: Block size for the model (if < num_points_per_trajectory, trajectories will be chopped)
        noise_scale: Scale of noise added during training
        lr: Learning rate
        n_layer: Number of transformer layers
        n_embd: Embedding dimension
        num_trajectories: Number of trajectories to generate
        loss_mask: 'all' to compute loss on all tokens, 'last' to compute only on last token
    
    Returns:
        Dictionary containing model results and statistics
    """
    print(f"\n{'='*80}")
    print(f"Training model: block_size={block_size}, noise_scale={noise_scale}, lr={lr}, n_layer={n_layer}, n_embd={n_embd}, num_trajectories={num_trajectories}, loss_mask={loss_mask}")
    print(f"{'='*80}")
    
    # Ensure num_trajectories is an integer
    num_trajectories = int(num_trajectories)
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be a positive integer, got {num_trajectories}")
    
    # Load trajectories from data_cv folder
    # Only load what we need (2*num_trajectories for train+test split)
    print("Loading Kepler orbit trajectories from data_cv...")
    trajectories = load_trajectories('data_cv', num_trajectories_needed=2*num_trajectories)
    print(f"Trajectories shape: {trajectories.shape}")
    
    if trajectories.shape[0] == 0:
        raise ValueError(f"No trajectories loaded. Please check that the data files exist in data_cv/")
    
    print(f"Position range: x=[{trajectories[:,:,0].min():.3f}, {trajectories[:,:,0].max():.3f}], "
          f"y=[{trajectories[:,:,1].min():.3f}, {trajectories[:,:,1].max():.3f}]")
    
    # Split into train and test sets (50/50 split)
    num_traj = trajectories.shape[0]
    train_trajectories = trajectories[:num_traj//2]
    test_trajectories = trajectories[num_traj//2:]
    
    # Save sizes before deleting trajectories
    train_size = train_trajectories.shape[0]
    test_size = test_trajectories.shape[0]
    
    # Clear trajectories to save memory (we'll keep train_trajectories and test_trajectories for evaluation)
    # But we need to keep them for evaluation, so don't delete yet
    
    print(f"\nSplit: {train_size} training trajectories, {test_size} test trajectories")
    
    # Load orbital parameters
    print("Loading orbital parameters from data_cv...")
    orbital_params = load_orbital_params('data_cv', num_trajectories_needed=2*num_trajectories)
    if orbital_params:
        print(f"Loaded {len(orbital_params)} orbital parameters")
        # Split orbital parameters to match train/test split
        train_orbital_params = orbital_params[:num_traj//2]
        test_orbital_params = orbital_params[num_traj//2:]
    else:
        print("Warning: No orbital parameters found. Geometry probes will be skipped.")
        train_orbital_params = None
        test_orbital_params = None
    
    # Prepare inputs and targets based on block_size
    if block_size < num_points_per_trajectory:
        # Chop trajectories into sequences of length block_size + 1
        print(f"\nBlock size ({block_size}) < num_points_per_trajectory ({num_points_per_trajectory})")
        print("Chopping trajectories into sequences...")
        num_sequences_per_traj = num_points_per_trajectory // block_size
        print(f"Randomly selecting {num_sequences_per_traj} sequences per trajectory (instead of all sliding windows)")
        train_inputs_np, train_targets_np, train_sequence_trajectory_ids = chop_trajectories_into_sequences(
            train_trajectories, block_size, seed=seed
        )
        test_inputs_np, test_targets_np, eval_sequence_trajectory_ids = chop_trajectories_into_sequences(
            test_trajectories, block_size, seed=seed+1
        )
        print(f"Train sequences: {train_inputs_np.shape[0]}, Test sequences: {test_inputs_np.shape[0]}")
    else:
        # Standard approach: use full trajectories
        print(f"\nBlock size ({block_size}) >= num_points_per_trajectory ({num_points_per_trajectory})")
        print("Using full trajectories...")
        train_inputs_np = train_trajectories[:,:-1]  # shape: (train_size, num_points-1, 2)
        train_targets_np = train_trajectories[:,1:]   # shape: (train_size, num_points-1, 2)
        test_inputs_np = test_trajectories[:,:-1]  # shape: (test_size, num_points-1, 2)
        test_targets_np = test_trajectories[:,1:]   # shape: (test_size, num_points-1, 2)
        train_sequence_trajectory_ids = np.arange(train_inputs_np.shape[0], dtype=np.int64)
        eval_sequence_trajectory_ids = np.arange(test_inputs_np.shape[0], dtype=np.int64)
    
    # Convert to PyTorch tensors - keep on CPU to save GPU memory, will move batches to GPU during training
    train_inputs = torch.from_numpy(train_inputs_np).float()  # keep on CPU
    train_targets = torch.from_numpy(train_targets_np).float()   # keep on CPU
    test_inputs = torch.from_numpy(test_inputs_np).float()  # keep on CPU
    test_targets = torch.from_numpy(test_targets_np).float()   # keep on CPU
    
    print(f"\nTrain input shape: {train_inputs.shape}, Train target shape: {train_targets.shape}")
    print(f"Test input shape: {test_inputs.shape}, Test target shape: {test_targets.shape}")
    
    # Clear numpy arrays to save memory
    del train_inputs_np, train_targets_np, test_inputs_np, test_targets_np
    
    # Setup model
    print("\nSetting up model...")
    model = setup_model(block_size=block_size, n_layer=n_layer, n_embd=n_embd, device=device,
                        mlp_mult=mlp_mult)
    print_gpu_memory_stats("After model setup: ")
    
    # Initial forward pass - use smaller batch to save memory
    with torch.no_grad():
        init_batch_size = min(128, train_inputs.shape[0])
        init_train_indices = torch.randint(0, train_inputs.shape[0], (init_batch_size,))
        init_test_indices = torch.randint(0, test_inputs.shape[0], (init_batch_size,))
        
        init_train_inputs = train_inputs[init_train_indices].to(device)
        init_train_targets = train_targets[init_train_indices].to(device)
        init_test_inputs = test_inputs[init_test_indices].to(device)
        init_test_targets = test_targets[init_test_indices].to(device)
        
        train_predictions, _ = model.forward(init_train_inputs, None)
        train_loss = compute_loss_with_mask(train_predictions, init_train_targets, loss_mask=loss_mask)
        test_predictions, _ = model.forward(init_test_inputs, None)
        test_loss = compute_loss_with_mask(test_predictions, init_test_targets, loss_mask=loss_mask)
        
        print(f"Initial train loss: {train_loss.item():.6f}, Initial test loss: {test_loss.item():.6f}")
        print(f"Train predictions shape: {train_predictions.shape}")
        
        del train_predictions, test_predictions
        del init_train_inputs, init_train_targets, init_test_inputs, init_test_targets
        del init_train_indices, init_test_indices
    
    print_gpu_memory_stats("After initial forward pass: ")

    batch_size = batch_size * 100 // block_size
    
    # Training with periodic evaluation
    print("\nTraining model with periodic evaluation...")
    training_results = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, train_trajectories, test_trajectories,
        n_steps=n_steps, lr=lr, weight_decay=0.0, noise_scale=noise_scale, prob_freq=prob_freq,
        loss_mask=loss_mask, batch_size=batch_size, seed=seed,
        train_orbital_params=train_orbital_params, eval_orbital_params=test_orbital_params,
        train_sequence_trajectory_ids=train_sequence_trajectory_ids,
        eval_sequence_trajectory_ids=eval_sequence_trajectory_ids
    )
    
    train_losses = training_results['train_losses']
    test_losses = training_results['test_losses']
    eval_results = training_results['eval_results']
    eval_steps = training_results['eval_steps']
    memory_stats = training_results.get('memory_stats', {})
    
    print("\nTraining completed!")
    print_gpu_memory_stats("After training: ")
    
    # Print memory summary
    if torch.cuda.is_available() and memory_stats:
        print(f"\nMemory Summary:")
        if 'initial' in memory_stats:
            print(f"  Initial: {memory_stats['initial']['allocated_gb']:.3f} GB")
        if 'final' in memory_stats:
            print(f"  Final: {memory_stats['final']['allocated_gb']:.3f} GB")
        if 'peak' in memory_stats:
            print(f"  Peak: {memory_stats['peak']['max_allocated_gb']:.3f} GB")
        if 'initial' in memory_stats and 'final' in memory_stats:
            print(f"  Memory increase: {memory_stats['final']['allocated_gb'] - memory_stats['initial']['allocated_gb']:.3f} GB")
    
    # Get final evaluation results (last evaluation)
    final_eval = eval_results[-1] if eval_results else None
    
    # Clear large tensors (they're on CPU, but still free memory)
    del train_inputs, train_targets, test_inputs, test_targets
    del train_trajectories, test_trajectories
    clear_gpu_cache()
    
    # Return results
    results = {
        'block_size': block_size,
        'noise_scale': noise_scale,
        'lr': lr,
        'n_layer': n_layer,
        'n_embd': n_embd,
        'mlp_mult': mlp_mult,
        'num_trajectories': num_traj,
        'train_size': train_size,
        'test_size': test_size,
        'loss_mask': loss_mask,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'eval_results': eval_results,
        'eval_steps': eval_steps,
        'final_error_stats_train': final_eval['error_stats_train'] if final_eval and 'error_stats_train' in final_eval else None,
        'final_error_stats_test': final_eval['error_stats_test'] if final_eval and 'error_stats_test' in final_eval else None,
        'final_probe_results': final_eval['probe_results'] if final_eval else None,
        'final_geometry_probe_results': final_eval['geometry_probe_results'] if final_eval and 'geometry_probe_results' in final_eval else None,
        'memory_stats': memory_stats,
    }
    
    return results


def sweep_parameters(block_size_list, num_trajectories_list, noise_scale_list, loss_mask_list=['all'],
                     lr=1e-3, n_layer=1, n_embd=32, n_steps=2001, prob_freq=100, seed=1,
                     mlp_mult_list=None):
    """
    Sweep over block_size, num_trajectories, noise_scale, loss_mask, and mlp_mult parameters.
    """
    if mlp_mult_list is None:
        mlp_mult_list = [4]
    
    print(f"\n{'='*80}")
    print(f"Starting parameter sweep:")
    print(f"  block_size: {block_size_list}")
    print(f"  num_trajectories: {num_trajectories_list}")
    print(f"  noise_scale: {noise_scale_list}")
    print(f"  loss_mask: {loss_mask_list}")
    print(f"  lr: {lr} (fixed)")
    print(f"  n_layer: {n_layer} (fixed)")
    print(f"  n_embd: {n_embd} (fixed)")
    print(f"  mlp_mult: {mlp_mult_list}")
    print(f"{'='*80}\n")

    total_runs = len(block_size_list) * len(num_trajectories_list) * len(noise_scale_list) * len(loss_mask_list) * len(mlp_mult_list)
    run_count = 0

    num_traj = 10000
    
    # Pre-filter configurations to skip existing results
    configs_to_run = []
    for block_size in block_size_list:
        for noise_scale in noise_scale_list:
            for num_traj in num_trajectories_list:
                for loss_mask in loss_mask_list:
                    for mlp_mult in mlp_mult_list:
                        results_filename = f'./results/kepler_cv_blocksize/results_block_size_{block_size}_num_trajectories_{num_traj}_noise_scale_{noise_scale}_loss_mask_{loss_mask}_mlp_mult_{mlp_mult}_seed_{seed}.npz'
                        if not os.path.exists(results_filename):
                            configs_to_run.append((block_size, num_traj, noise_scale, loss_mask, mlp_mult))
                        else:
                            print(f"Skipping existing result: {results_filename}")

    print(f"\nTotal configurations: {total_runs}, Already completed: {total_runs - len(configs_to_run)}, To run: {len(configs_to_run)}")

    for block_size, num_traj, noise_scale, loss_mask, mlp_mult in configs_to_run:
        run_count += 1
        print(f"\n[{run_count}/{len(configs_to_run)}] Running: block_size={block_size}, num_trajectories={num_traj}, noise_scale={noise_scale}, loss_mask={loss_mask}, mlp_mult={mlp_mult}")
        results = train_one_model(
            block_size=block_size,
            noise_scale=noise_scale,
            lr=lr,
            n_layer=n_layer,
            n_embd=n_embd,
            num_trajectories=num_traj,
            n_steps=n_steps,
            prob_freq=prob_freq,
            loss_mask=loss_mask,
            seed=seed,
            mlp_mult=mlp_mult,
        )

        # save results to file
        results_filename = f'./results/kepler_cv_blocksize/results_block_size_{block_size}_num_trajectories_{num_traj}_noise_scale_{noise_scale}_loss_mask_{loss_mask}_mlp_mult_{mlp_mult}.npz'
        os.makedirs(os.path.dirname(results_filename), exist_ok=True)
        np.savez(results_filename, **results)
        print(f"Saved results to {results_filename}")
            
    
    print(f"\n{'='*80}")
    print(f"Parameter sweep completed! Total runs: {len(configs_to_run)}")
    print(f"{'='*80}\n")
    
    return None


def main():
    """Main execution function."""
    # Run a single model with default parameters
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_trajectories_list = [10000]
    #block_size_list = [1, 2, 5, 10, 20, 50, 100]
    #block_size_list = [60, 70, 80, 90]
    block_size_list = [100]
    noise_scale_list = [0.1]
    loss_mask_list = ['all']
    #loss_mask_list = ['all', 'last']
    #block_size_list = [100]
    #noise_scale_list = [0.1]
    #loss_mask_list = ['all']
    n_steps = 2001
    prob_freq = 10
    mlp_mult_list = [2, 4, 8, 16]
    sweep_parameters(block_size_list, num_trajectories_list, noise_scale_list, loss_mask_list,
                     n_steps=n_steps, prob_freq=prob_freq, seed=seed,
                     mlp_mult_list=mlp_mult_list)


if __name__ == "__main__":
    main()