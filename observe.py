"""Runtime and rollout diagnostics for Newton-Kepler experiments."""

import numpy as np
import torch
from sklearn.metrics import r2_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def compute_activation_rank(activation_dict, eps=1e-4, numerical_tol=1e-6):
    results = {}

    for layer_name, activation in activation_dict.items():
        x = activation.detach().float()

        # (batch, time, feature) -> (batch*time, feature)
        x = x.reshape(-1, x.shape[-1])
        x = x - x.mean(dim=0, keepdim=True)

        singular_values = torch.linalg.svdvals(x)
        energy = singular_values.square()
        total_energy = energy.sum()

        if total_energy <= eps:
            effective_rank = 0.0
            normalized_effective_rank = 0.0
            numerical_rank = 0
        else:
            probabilities = energy / total_energy
            entropy = -torch.sum(
                probabilities * torch.log(probabilities.clamp_min(eps))
            )
            effective_rank = torch.exp(entropy).item()

            max_possible_rank = min(
                x.shape[0] - 1,
                x.shape[1],
            )
            normalized_effective_rank = (
                effective_rank / max_possible_rank
                if max_possible_rank > 0
                else 0.0
            )

            threshold = numerical_tol * singular_values.max()
            numerical_rank = int(
                (singular_values > threshold).sum().item()
            )

        results[layer_name] = {
            "effective_rank": effective_rank,
            "normalized_effective_rank": normalized_effective_rank,
            "numerical_rank": numerical_rank,
        }

    return results


def set_attention_entropy_capture(model, enabled):
    """Enable or disable entropy collection on every transformer block."""
    for block in model.transformer.h:
        block.attn.capture_attention_stats = bool(enabled)
        if not enabled:
            block.attn.last_attention_stats = None


def collect_attention_entropy(model):
    """Snapshot the entropy statistics produced by the latest forward pass."""
    results = {}
    for block_index, block in enumerate(model.transformer.h):
        stats = block.attn.last_attention_stats
        if stats is None:
            raise RuntimeError(
                f"No attention statistics captured for block {block_index}. "
                "Enable capture before the forward pass."
            )
        results[f"block_{block_index}"] = dict(stats)
    return results


def collect_variance_covariance(model):
    """Snapshot per-layer variance/covariance diagnostics.

    The returned values are detached Python floats, so storing them cannot
    retain the forward computation graph. The model must have completed a
    forward pass with variance/covariance computation enabled.
    """
    results = model.variance_covariance_stats()
    if not results:
        raise RuntimeError(
            "No variance/covariance statistics were captured. Enable "
            "observe.variance_covariance or its training regularizer."
        )
    return results

def compute_weight_singular_values(model, num_singular_values=None):
    """
    Compute singular value distributions for all linear layers in the model.
    
    Args:
        model: PyTorch model
        num_singular_values: If provided, only return the first K singular values
                             (e.g., to avoid storing huge arrays). If None, return all.
    
    Returns:
        dict: layer_name -> dict with:
            - 'singular_values': list of singular values (numpy array)
            - 'max': maximum singular value
            - 'min': minimum singular value
            - 'mean': mean of singular values
            - 'std': standard deviation of singular values
            - 'effective_rank': effective rank (based on entropy)
            - 'numerical_rank': rank based on numerical threshold
    """
    results = {}
    
    for name, param in model.named_parameters():
        # Only consider weight matrices (2D or more)
        if param.dim() >= 2:
            # Get the weight as a 2D matrix by flattening leading dimensions
            weight = param.detach().float()
            # For Linear layers, shape is (out_features, in_features) or similar
            # We'll reshape to 2D: (num_rows, num_cols) where num_rows is the first dim
            if weight.dim() > 2:
                # For conv or embedding, flatten all but last dim? But here we only have Linear.
                # Keep simple: just flatten to 2D if needed.
                weight = weight.view(weight.shape[0], -1)
            # Compute SVD
            try:
                sv = torch.linalg.svdvals(weight)
            except:
                # Fallback to using CPU if GPU fails (e.g., large matrix)
                sv = torch.linalg.svdvals(weight.cpu())
                # Move back to CPU anyway
            sv = sv.cpu().numpy()
            
            # If limited, take first K
            if num_singular_values is not None and len(sv) > num_singular_values:
                sv = sv[:num_singular_values]
            
            # Compute derived statistics
            total_energy = np.sum(sv**2)
            eps = 1e-12
            if total_energy <= eps:
                effective_rank = 0.0
            else:
                probs = (sv**2) / total_energy
                # Avoid log(0)
                entropy = -np.sum(probs * np.log(probs + eps))
                effective_rank = np.exp(entropy)
            
            numerical_rank = np.sum(sv > 1e-6 * sv.max())
            
            results[name] = {
                'singular_values': sv.tolist(),
                'max': float(sv.max()),
                'min': float(sv.min()),
                'mean': float(sv.mean()),
                'std': float(sv.std()),
                'effective_rank': float(effective_rank),
                'numerical_rank': int(numerical_rank),
            }
    
    return results


def compute_activation_singular_values(activation_dict, num_singular_values=None):
    """
    Compute singular value distributions for a dictionary of activations.
    
    Args:
        activation_dict: dict of layer_name -> tensor of shape (batch, seq_len, feature)
        num_singular_values: If provided, only return the first K singular values.
    
    Returns:
        dict: layer_name -> dict with similar stats as compute_weight_singular_values
    """
    results = {}
    
    for layer_name, activation in activation_dict.items():
        # Detach, move to CPU, flatten to (N, D)
        x = activation.detach().float().cpu()
        # Reshape to (batch*seq_len, feature)
        x = x.reshape(-1, x.shape[-1])
        # Center columns
        x = x - x.mean(dim=0, keepdim=True)
        
        # Compute SVD
        try:
            sv = torch.linalg.svdvals(x)
        except:
            # Fallback
            sv = torch.linalg.svdvals(x.cpu())
        sv = sv.numpy()
        
        if num_singular_values is not None and len(sv) > num_singular_values:
            sv = sv[:num_singular_values]
        
        total_energy = np.sum(sv**2)
        eps = 1e-12
        if total_energy <= eps:
            effective_rank = 0.0
        else:
            probs = (sv**2) / total_energy
            entropy = -np.sum(probs * np.log(probs + eps))
            effective_rank = np.exp(entropy)
        
        numerical_rank = np.sum(sv > 1e-6 * sv.max())
        
        results[layer_name] = {
            'singular_values': sv.tolist(),
            'max': float(sv.max()),
            'min': float(sv.min()),
            'mean': float(sv.mean()),
            'std': float(sv.std()),
            'effective_rank': float(effective_rank),
            'numerical_rank': int(numerical_rank),
        }
    
    return results
