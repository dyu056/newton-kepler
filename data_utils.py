"""Dataset loading for Spring SHM trajectories."""

import os

import numpy as np
import torch


def load_trajectories(data_dir='data_spring', num_trajectories_needed=None):
    """Load Spring SHM trajectories from data_dir/train_trajectories.npy.

    Args:
        data_dir: Directory containing train_trajectories.npy
        num_trajectories_needed: Number of trajectories (None = all)
    Returns:
        float32 array of shape (N, num_frames, 1)
    """
    train_path = os.path.join(data_dir, "train_trajectories.npy")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"train_trajectories.npy not found in {data_dir}. "
            f"Run: python generate_spring_data.py --config configs/spring.yaml"
        )

    trajectories = np.load(train_path).astype(np.float32)

    eval_path = os.path.join(data_dir, "eval_trajectories.npy")
    if os.path.exists(eval_path):
        eval_data = np.load(eval_path).astype(np.float32)
        trajectories = np.concatenate([trajectories, eval_data], axis=0)

    if num_trajectories_needed is not None:
        trajectories = trajectories[:num_trajectories_needed]

    print(f"Loaded {trajectories.shape[0]:,} spring trajectories "
          f"from {data_dir}  (shape={trajectories.shape})")
    return trajectories

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
            input_seq = traj[i:i+block_size]
            target_seq = traj[i+block_size:i+block_size+1]
            all_inputs.append(input_seq)
            all_targets.append(target_seq)
            sequence_trajectory_ids.append(trajectory_id)
    
    inputs = np.array(all_inputs)
    targets = np.array(all_targets)

    return inputs, targets, np.asarray(sequence_trajectory_ids, dtype=np.int64)


def load_spring_metadata(data_dir: str = "data_spring") -> dict:
    """Return spring dataset metadata (fps, num_frames, omega_range)."""
    meta_path = os.path.join(data_dir, "metadata.pt")
    if not os.path.exists(meta_path):
        return {}
    return torch.load(meta_path, weights_only=False)



def load_orbital_params(data_dir="data_cv", num_trajectories_needed=None):
    """Stub: orbital parameters not available for Spring data."""
    return []
