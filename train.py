"""Single training entry point for Newton-Kepler experiments."""

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def _select_gpu_before_torch_import():
    """Apply --gpu before PyTorch or any CUDA-aware project module is imported."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=str, default=None)
    args, _ = parser.parse_known_args()

    if args.gpu is None:
        return None
    if args.gpu.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args.gpu


SELECTED_GPU = _select_gpu_before_torch_import()

# These imports must remain below _select_gpu_before_torch_import().
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from data_utils import (
    chop_trajectories_into_sequences,
    load_orbital_params,
    load_trajectories,
)
from loss import compute_loss_with_mask
from model_cv import GPTConfigCV, GPTCV
from observe import (
    clear_gpu_cache,
    collect_attention_entropy,
    collect_variance_covariance,
    compute_activation_rank,
    generate_trajectory_and_compute_error,
    get_gpu_memory_stats,
    print_gpu_memory_stats,
    set_attention_entropy_capture,
    compute_weight_singular_values,
    compute_activation_singular_values
)
from probe import (
    collect_activations,
    initialize_probe_indices,
    run_geometry_probes,
    run_linear_probes,
    setup_activation_hooks,
    snapshot_activations,
)


seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


_ACTIVE_RESULT_PATH = None
_ACTIVE_RESULT_METADATA = {}


def _atomic_save_npz(path, payload):
    """Atomically replace an NPZ file without compression overhead."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("wb") as file:
        np.savez(file, **payload)
    os.replace(temporary_path, path)


def _save_partial_training_state(
    completed_step, train_losses, total_losses, test_losses,
    variance_losses, covariance_losses, representation_mean_stds,
    weighted_variance_terms, weighted_covariance_terms, entropy_terms,
    eval_results, eval_steps,
):
    """Persist the latest recoverable metrics while training is still running."""
    if _ACTIVE_RESULT_PATH is None:
        return
    payload = dict(_ACTIVE_RESULT_METADATA)
    payload.update({
        "status": "partial",
        "completed_steps": int(completed_step),
        "train_losses": train_losses,
        "total_losses": total_losses,
        "test_losses": test_losses,
        "variance_losses": variance_losses,
        "covariance_losses": covariance_losses,
        "representation_mean_stds": representation_mean_stds,
        "weighted_variance_terms": weighted_variance_terms,
        "weighted_covariance_terms": weighted_covariance_terms,
        "entropy_terms": entropy_terms,
        "eval_results": eval_results,
        "eval_steps": eval_steps,
    })
    _atomic_save_npz(_ACTIVE_RESULT_PATH, payload)


def _result_is_complete(path):
    """Treat legacy files without a status field as complete."""
    try:
        with np.load(path, allow_pickle=True) as result:
            if "status" not in result.files:
                return True
            return str(result["status"].item()) == "complete"
    except Exception:
        return False


@dataclass(frozen=True)
class ObservablesConfig:
    probe: bool = True
    activation_rank: bool = False
    attention_entropy: bool = False
    variance_covariance: bool = False
    singular_values: bool = False
    rollout: bool = True

    @property
    def needs_periodic_eval(self):
        return any((
            self.probe,
            self.activation_rank,
            self.attention_entropy,
            self.variance_covariance,
            self.singular_values,
            self.rollout,
        ))

    @property
    def needs_activation_capture(self):
        return any((
            self.probe,
            self.activation_rank,
            self.singular_values,
        ))


def _enabled(value, default=False):
    if isinstance(value, dict):
        return bool(value.get("enabled", default))
    if value is None:
        return bool(default)
    return bool(value)


def setup_model(block_size, n_layer=2, n_head=1, n_embd=32, input_dim=1, device=None,
                varcov_enabled=False, varcov_target_std=0.1):
    """Setup and initialize the GPT model for continuous vision."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    GPTConfigCV.block_size = block_size
    GPTConfigCV.input_dim = int(input_dim)
    GPTConfigCV.n_layer = n_layer
    GPTConfigCV.n_head = n_head
    GPTConfigCV.n_embd = n_embd
    GPTConfigCV.attention_alpha = 0.0
    GPTConfigCV.bias = True
    GPTConfigCV.varcov_enabled = bool(varcov_enabled)
    GPTConfigCV.varcov_target_std = float(varcov_target_std)

    model = GPTCV(GPTConfigCV)
    model = model.to(device)
    return model


def should_run_probe(completed_step, probe_frequency, probe_schedule=None):
    """Return whether probing should run after ``completed_step`` updates.

    Schedule intervals are inclusive. When two phases share a boundary, the
    boolean result still triggers only one probe at that step.
    """
    if not probe_schedule:
        # Preserve the original i % frequency == 0 behavior, where
        # completed_step = i + 1 (steps 1, 1+frequency, ...).
        return (completed_step - 1) % probe_frequency == 0

    for phase in probe_schedule:
        start = int(phase["start"])
        end = int(phase["end"])
        every = int(phase["every"])
        if every <= 0:
            raise ValueError("probe.schedule[].every must be positive")
        if start <= completed_step <= end and (completed_step - start) % every == 0:
            return True
    return False

def train_model(model, train_inputs, train_targets, test_inputs, test_targets, train_trajectories, test_trajectories,
                 n_steps=1001, lr=1e-3, weight_decay=0.0, noise_scale=0.1, prob_freq=100, batch_size=128,
                 loss_mask='all', seed=1, train_orbital_params=None, eval_orbital_params=None,
                 optimizer_type='adamw',
                 train_sequence_trajectory_ids=None, eval_sequence_trajectory_ids=None,
                 progress_bar=None, probe_schedule=None, probe_enabled=True,
                 probe_num_train_samples=1000, probe_num_eval_samples=1000,
                 probe_verbose=False, probe_geometry_enabled=True,
                 activation_rank_enabled=False,
                 attention_entropy_enabled=False,
                 varcov_observe_enabled=False,
                 singular_values_enabled=False,
                 singular_values_num=50,
                 rollout_enabled=True,
                 varcov_reg_enabled=False,
                 varcov_reg_start=1,
                 variance_reg_weight=0.0,
                 covariance_reg_weight=0.0,
                 attention_entropy_reg_enabled=False,
                 attention_entropy_weight=0.0,
                 attention_entropy_start=1):
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
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if optimizer_type == 'gd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []
    total_losses = []
    test_losses = []
    variance_losses = []
    covariance_losses = []
    representation_mean_stds = []
    weighted_variance_terms = []
    weighted_covariance_terms = []
    entropy_terms = []  # 新增：记录注意力熵正则项
    eval_results = []
    eval_steps = []

    observables = ObservablesConfig(
        probe=probe_enabled,
        activation_rank=activation_rank_enabled,
        attention_entropy=attention_entropy_enabled,
        variance_covariance=varcov_observe_enabled,
        singular_values=singular_values_enabled,
        rollout=rollout_enabled,
    )

    if observables.needs_activation_capture:
        hooks, activation_dict = setup_activation_hooks(model, verbose=probe_verbose)
    else:
        hooks, activation_dict = [], {}

    # Initial memory stats
    initial_memory = print_gpu_memory_stats("Initial: ")

    # Get training data size
    num_train_samples = train_inputs.shape[0]
    if observables.needs_periodic_eval:
        max_probe_samples = max(
            int(probe_num_train_samples),
            int(probe_num_eval_samples),
        )
        probe_train_indices, probe_eval_indices = initialize_probe_indices(
            train_inputs.shape[0], test_inputs.shape[0], max_probe_samples, seed
        )
        probe_train_indices = probe_train_indices[:int(probe_num_train_samples)]
        probe_eval_indices = probe_eval_indices[:int(probe_num_eval_samples)]
    else:
        probe_train_indices, probe_eval_indices = None, None

    if observables.needs_activation_capture:
        probe_train_trajectory_ids = train_sequence_trajectory_ids[probe_train_indices.numpy()]
        probe_eval_trajectory_ids = eval_sequence_trajectory_ids[probe_eval_indices.numpy()]
    else:
        probe_train_trajectory_ids = None
        probe_eval_trajectory_ids = None

    latest_test_loss = float("nan")

    probe_metadata = {
        'schema_version': 2, 'fit_split': 'model_train', 'eval_split': 'model_test',
        'primary_metric': 'eval_r2', 'sampling_method': 'torch.randperm',
        'sampling_with_replacement': False, 'fixed_indices_across_checkpoints': True,
        'probe_seed': seed + 1000,
        'num_train_draws': probe_train_indices.numel() if probe_train_indices is not None else 0,
        'num_eval_draws': probe_eval_indices.numel() if probe_eval_indices is not None else 0,
        'num_unique_train_sequences': probe_train_indices.unique().numel() if probe_train_indices is not None else 0,
        'num_unique_eval_sequences': probe_eval_indices.unique().numel() if probe_eval_indices is not None else 0,
        'num_unique_train_trajectories': int(np.unique(probe_train_trajectory_ids).size) if probe_train_trajectory_ids is not None else 0,
        'num_unique_eval_trajectories': int(np.unique(probe_eval_trajectory_ids).size) if probe_eval_trajectory_ids is not None else 0,
    }

    # ── Step 0 probe (random init) ──
    if (observables.needs_periodic_eval
            and probe_schedule is not None
            and should_run_probe(0, prob_freq, probe_schedule)):
        print("\nEvaluating at step 0 (random init)...")
        eval_step_results = {}
        probe_train_inputs = train_inputs[probe_train_indices].to(device)
        probe_train_targets = train_targets[probe_train_indices].to(device)
        probe_eval_inputs = test_inputs[probe_eval_indices].to(device)
        probe_eval_targets = test_targets[probe_eval_indices].to(device)
        if observables.needs_activation_capture:
            activation_dict.clear()
            _, train_force = collect_activations(
                model, probe_train_inputs, probe_train_targets, activation_dict, verbose=probe_verbose)
            train_activation_dict = snapshot_activations(activation_dict)
            activation_dict.clear()
            _, eval_force = collect_activations(
                model, probe_eval_inputs, probe_eval_targets, activation_dict, verbose=probe_verbose)
            eval_activation_dict = snapshot_activations(activation_dict)
        if probe_enabled:
            eval_step_results['probe_results'] = run_linear_probes(
                train_activation_dict, train_force, eval_activation_dict, eval_force, verbose=probe_verbose)
        eval_step_results['probe_metadata'] = probe_metadata
        if singular_values_enabled:
            eval_step_results['weight_singular_values'] = compute_weight_singular_values(
                model, num_singular_values=singular_values_num)
            eval_step_results['activation_singular_values'] = {
                'train': compute_activation_singular_values(train_activation_dict, num_singular_values=singular_values_num),
                'eval': compute_activation_singular_values(eval_activation_dict, num_singular_values=singular_values_num)}
        eval_step_results['attention_entropy'] = None
        eval_step_results['variance_covariance'] = None
        eval_step_results['geometry_probe_results'] = None
        eval_step_results['error_stats_train'] = None
        eval_step_results['error_stats_test'] = None
        eval_step_results['step'] = 0
        eval_results.append(eval_step_results)
        eval_steps.append(0)
        del probe_train_inputs, probe_train_targets, probe_eval_inputs, probe_eval_targets
        if observables.needs_activation_capture:
            del train_force, eval_force, train_activation_dict, eval_activation_dict
        activation_dict.clear()
        print("Evaluation at step 0 completed.\n")

    for i in range(n_steps):

        if i == n_steps // 2:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        # Sample data: full-batch for GD, random batch for AdamW
        if optimizer_type == 'gd':
            batch_inputs = train_inputs.to(device)
            batch_targets = train_targets.to(device)
        else:
            batch_indices = torch.randint(0, num_train_samples, (batch_size,))
            batch_inputs = train_inputs[batch_indices].to(device)
            batch_targets = train_targets[batch_indices].to(device)

        # Training step. ``completed_step`` is the number of the optimizer
        # update being performed and is consistently one-based.
        completed_step = i + 1
        optimizer.zero_grad(set_to_none=True)
        inputs_noised = batch_inputs + torch.randn_like(batch_inputs) * noise_scale

        # 判断是否计算注意力熵
        compute_entropy = attention_entropy_reg_enabled and (completed_step >= attention_entropy_start)
        predictions, _ = model.forward(inputs_noised, None, compute_entropy=compute_entropy)
        mse_loss = compute_loss_with_mask(predictions, batch_targets, loss_mask=loss_mask)

        variance_source = None
        covariance_source = None
        if varcov_reg_enabled and completed_step >= varcov_reg_start:
            variance_source, covariance_source = model.variance_covariance_penalties()

        zero = mse_loss.new_zeros(())
        variance_term = zero
        covariance_term = zero
        entropy_term = zero
        total_loss = mse_loss

        if varcov_reg_enabled and completed_step >= varcov_reg_start:
            if variance_source is not None and variance_reg_weight > 0.0:
                variance_term = variance_reg_weight * variance_source
                total_loss = total_loss + variance_term
            if covariance_source is not None and covariance_reg_weight > 0.0:
                covariance_term = covariance_reg_weight * covariance_source
                total_loss = total_loss + covariance_term

        if compute_entropy:
            entropy_penalty = model.attention_entropy_penalty()
            if entropy_penalty is not None:
                entropy_term = attention_entropy_weight * entropy_penalty
                total_loss = total_loss + entropy_term

        total_loss.backward()
        optimizer.step()
        train_losses.append(float(mse_loss.detach().item()))
        total_losses.append(float(total_loss.detach().item()))
        variance_losses.append(
            float(variance_source.detach().item())
            if variance_source is not None else float("nan")
        )
        covariance_losses.append(
            float(covariance_source.detach().item())
            if covariance_source is not None else float("nan")
        )
        if varcov_observe_enabled:
            layer_stats = model.variance_covariance_stats()
            representation_mean_stds.append(
                float(np.mean([v["mean_std"] for v in layer_stats.values()]))
                if layer_stats else float("nan")
            )
        else:
            representation_mean_stds.append(float("nan"))
        weighted_variance_terms.append(float(variance_term.detach().item()))
        weighted_covariance_terms.append(float(covariance_term.detach().item()))
        entropy_terms.append(float(entropy_term.detach().item()))  # 记录熵正则项

        # Clear intermediate variables to save memory
        if optimizer_type == 'gd':
            del inputs_noised, predictions, batch_inputs, batch_targets
        else:
            del inputs_noised, predictions, batch_inputs, batch_targets, batch_indices

        evaluation_due = (
            observables.needs_periodic_eval
            and should_run_probe(completed_step, prob_freq, probe_schedule)
        )
        test_due = evaluation_due or i % 100 == 0
        if test_due:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                test_batch_indices = torch.randint(0, test_inputs.shape[0], (batch_size,))
                test_batch_inputs = test_inputs[test_batch_indices].to(device)
                test_batch_targets = test_targets[test_batch_indices].to(device)
                test_predictions, _ = model.forward(test_batch_inputs, None)
                test_loss = compute_loss_with_mask(
                    test_predictions, test_batch_targets, loss_mask=loss_mask
                )
                latest_test_loss = float(test_loss.item())
            model.train(was_training)
            del test_batch_inputs, test_batch_targets, test_batch_indices, test_predictions
        test_losses.append(latest_test_loss)

        if progress_bar is not None:
            progress_bar.set_postfix(
                mse=f"{mse_loss.item():.4g}",
                total=f"{total_loss.item():.4g}",
                test=f"{latest_test_loss:.4g}",
                refresh=False,
            )
            progress_bar.update(1)

        if i % 100 == 0:
            memory_stats = get_gpu_memory_stats()
            print(f"Step {completed_step}, MSE: {mse_loss.item():.6f}, "
                  f"Total Loss: {total_loss.item():.6f}, "
                  f"Test Loss: {latest_test_loss:.6f}, "
                  f"GPU Memory: {memory_stats['allocated_gb']:.3f} GB")

        # Evaluate according to the fixed frequency or piecewise schedule.
        if evaluation_due:
            print(f"\nEvaluating at step {completed_step}...")
            eval_step_results = {}

            # Load only a small sample for activation collection (don't reload entire dataset)
            # Use existing train_inputs/train_targets instead of reloading
            probe_train_inputs = train_inputs[probe_train_indices].to(device)
            probe_train_targets = train_targets[probe_train_indices].to(device)
            probe_eval_inputs = test_inputs[probe_eval_indices].to(device)
            probe_eval_targets = test_targets[probe_eval_indices].to(device)

            if attention_entropy_enabled:
                set_attention_entropy_capture(model, True)

            train_activation_dict = None
            eval_activation_dict = None
            train_force = None
            eval_force = None

            if observables.needs_activation_capture:
                activation_dict.clear()
                _, train_force = collect_activations(
                    model, probe_train_inputs, probe_train_targets, activation_dict,
                    verbose=probe_verbose
                )
                train_activation_dict = snapshot_activations(activation_dict)
            else:
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    _, _ = model(probe_train_inputs, None)
                model.train(was_training)

            train_varcov_stats = (
                collect_variance_covariance(model)
                if varcov_observe_enabled else None
            )
            train_attention_entropy = (
                collect_attention_entropy(model)
                if attention_entropy_enabled
                else None
            )

            if observables.needs_activation_capture:
                activation_dict.clear()
                _, eval_force = collect_activations(
                    model, probe_eval_inputs, probe_eval_targets, activation_dict,
                    verbose=probe_verbose
                )
                eval_activation_dict = snapshot_activations(activation_dict)
            else:
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    _, _ = model(probe_eval_inputs, None)
                model.train(was_training)

            eval_varcov_stats = (
                collect_variance_covariance(model)
                if varcov_observe_enabled else None
            )
            eval_attention_entropy = (
                collect_attention_entropy(model)
                if attention_entropy_enabled
                else None
            )

            if attention_entropy_enabled:
                set_attention_entropy_capture(model, False)
                eval_step_results['attention_entropy'] = {
                    'train': train_attention_entropy,
                    'eval': eval_attention_entropy,
                }
                train_entropy_mean = np.mean([
                    stats['normalized_mean_entropy']
                    for stats in train_attention_entropy.values()
                ])
                eval_entropy_mean = np.mean([
                    stats['normalized_mean_entropy']
                    for stats in eval_attention_entropy.values()
                ])
                print(
                    "Attention entropy normalized mean: "
                    f"train={train_entropy_mean:.6f}, eval={eval_entropy_mean:.6f}"
                )
            else:
                eval_step_results['attention_entropy'] = None

            eval_step_results['variance_covariance'] = (
                {
                    'train': train_varcov_stats,
                    'eval': eval_varcov_stats,
                }
                if varcov_observe_enabled else None
            )

            if activation_rank_enabled:
                eval_step_results['activation_rank'] = {
                    'train': compute_activation_rank(train_activation_dict),
                    'eval': compute_activation_rank(eval_activation_dict),
                }
            else:
                eval_step_results['activation_rank'] = None

            if probe_enabled:
                probe_results = run_linear_probes(
                    train_activation_dict, train_force, eval_activation_dict, eval_force,
                    verbose=probe_verbose
                )
                eval_step_results['probe_results'] = probe_results
            else:
                eval_step_results['probe_results'] = None
            eval_step_results['probe_metadata'] = probe_metadata

            # Run geometry probes if orbital parameters are available
            if (
                probe_enabled
                and probe_geometry_enabled
                and train_orbital_params is not None
                and eval_orbital_params is not None
            ):
                geometry_probe_results = run_geometry_probes(
                    train_activation_dict, train_orbital_params, probe_train_trajectory_ids,
                    eval_activation_dict, eval_orbital_params, probe_eval_trajectory_ids
                )
                eval_step_results['geometry_probe_results'] = geometry_probe_results
            else:
                eval_step_results['geometry_probe_results'] = None

            if singular_values_enabled:
                eval_step_results['weight_singular_values'] = compute_weight_singular_values(
                    model, num_singular_values=singular_values_num
                )
            else:
                eval_step_results['weight_singular_values'] = None

            # 激活奇异值（每次评估都重新计算，因为激活不同）
            if singular_values_enabled:
                eval_step_results['activation_singular_values'] = {
                    'train': compute_activation_singular_values(train_activation_dict, num_singular_values=singular_values_num),
                    'eval': compute_activation_singular_values(eval_activation_dict, num_singular_values=singular_values_num),
                }
            else:
                eval_step_results['activation_singular_values'] = None

            # Clear activations from GPU after probing to free memory
            activation_dict.clear()
            if train_force is not None:
                del train_force, eval_force
            if train_activation_dict is not None:
                del train_activation_dict, eval_activation_dict

            if rollout_enabled:
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
            else:
                error_stats_train = None
                error_stats_test = None

            # Clear sample data
            del probe_train_inputs, probe_train_targets, probe_eval_inputs, probe_eval_targets
            eval_step_results['error_stats_train'] = error_stats_train
            eval_step_results['error_stats_test'] = error_stats_test
            eval_step_results['step'] = completed_step

            eval_results.append(eval_step_results)
            eval_steps.append(completed_step)
            _save_partial_training_state(
                completed_step, train_losses, total_losses, test_losses,
                variance_losses, covariance_losses, representation_mean_stds,
                weighted_variance_terms, weighted_covariance_terms, entropy_terms,
                eval_results, eval_steps,
            )

            memory_stats = print_gpu_memory_stats(f"After evaluation at step {completed_step}: ")
            print(f"Evaluation at step {completed_step} completed.\n")

        if i % 100 == 0 and not evaluation_due:
            _save_partial_training_state(
                completed_step, train_losses, total_losses, test_losses,
                variance_losses, covariance_losses, representation_mean_stds,
                weighted_variance_terms, weighted_covariance_terms, entropy_terms,
                eval_results, eval_steps,
            )

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
        'total_losses': total_losses,
        'test_losses': test_losses,
        'variance_losses': variance_losses,
        'covariance_losses': covariance_losses,
        'representation_mean_stds': representation_mean_stds,
        'weighted_variance_terms': weighted_variance_terms,
        'weighted_covariance_terms': weighted_covariance_terms,
        'entropy_terms': entropy_terms,
        'eval_results': eval_results,
        'eval_steps': eval_steps,
        'memory_stats': memory_stats,
    }

def train_one_model(block_size=100, data_dir='data_cv', noise_scale=0.1, lr=1e-3,
                    weight_decay=0.0, n_layer=2, n_head=1, n_embd=16, input_dim=1,
                    num_trajectories=10000, n_steps=1001, prob_freq=100,
                    loss_mask='all', seed=1, batch_size=128,
                    scale_batch_by_context=True, progress_bar=None,
                    probe_schedule=None, probe_enabled=True,
                    probe_num_train_samples=1000, probe_num_eval_samples=1000,
                    probe_verbose=False, probe_geometry_enabled=True,
                    activation_rank_enabled=False,
                    attention_entropy_enabled=False,
                    varcov_observe_enabled=False,
                    singular_values_enabled=False,
                    singular_values_num=50,
                    rollout_enabled=True,
                    varcov_reg_enabled=False,
                    varcov_reg_start=1,
                    varcov_target_std=0.1,
                    variance_reg_weight=0.0,
                    covariance_reg_weight=0.0,
                    attention_entropy_reg_enabled=False,
                    attention_entropy_weight=0.0,
                    attention_entropy_start=1,
                    optimizer_type='adamw'):
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
    print(f"Training model: block_size={block_size}, noise_scale={noise_scale}, lr={lr}, weight_decay={weight_decay}, n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}, num_trajectories={num_trajectories}, loss_mask={loss_mask}")
    print(f"{'='*80}")

    # Ensure num_trajectories is an integer
    num_trajectories = int(num_trajectories)
    if num_trajectories <= 0:
        raise ValueError(f"num_trajectories must be a positive integer, got {num_trajectories}")

    # Load trajectories from data_cv folder
    # Only load what we need (2*num_trajectories for train+test split)
    print(f"Loading Kepler orbit trajectories from {data_dir}...")
    trajectories = load_trajectories(data_dir, num_trajectories_needed=2*num_trajectories)
    print(f"Trajectories shape: {trajectories.shape}")

    if trajectories.shape[0] == 0:
        raise ValueError(f"No trajectories loaded. Please check that the data files exist in data_cv/")

    print(f"Position range: x=[{trajectories[:,:,0].min():.3f}, {trajectories[:,:,0].max():.3f}], "
          f"y=[{trajectories[:,:,1].min():.3f}, {trajectories[:,:,1].max():.3f}]")

    # Auto-detect sequence length from data (Kepler=100, Spring=129, etc.)
    num_points_per_trajectory = trajectories.shape[1]

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

    if probe_enabled and probe_geometry_enabled:
        print(f"Loading orbital parameters from {data_dir}...")
        orbital_params = load_orbital_params(data_dir, num_trajectories_needed=2*num_trajectories)
        if orbital_params:
            print(f"Loaded {len(orbital_params)} orbital parameters")
            # Split orbital parameters to match train/test split
            train_orbital_params = orbital_params[:num_traj//2]
            test_orbital_params = orbital_params[num_traj//2:]
        else:
            print("Warning: No orbital parameters found. Geometry probes will be skipped.")
            train_orbital_params = None
            test_orbital_params = None
    else:
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
    model = setup_model(
        block_size=block_size, n_layer=n_layer, n_head=n_head,
        n_embd=n_embd, input_dim=input_dim, device=device,
        varcov_enabled=(varcov_observe_enabled or varcov_reg_enabled),
        varcov_target_std=varcov_target_std,
    )
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

    if scale_batch_by_context:
        batch_size = batch_size * num_points_per_trajectory // block_size
    print(f"Effective training batch size: {batch_size}")

    # Training with periodic evaluation
    print("\nTraining model with periodic evaluation...")
    training_results = train_model(
        model, train_inputs, train_targets, test_inputs, test_targets, train_trajectories, test_trajectories,
        n_steps=n_steps, lr=lr, weight_decay=weight_decay, noise_scale=noise_scale, prob_freq=prob_freq,
        loss_mask=loss_mask, batch_size=batch_size, seed=seed,
        train_orbital_params=train_orbital_params, eval_orbital_params=test_orbital_params,
        train_sequence_trajectory_ids=train_sequence_trajectory_ids,
        eval_sequence_trajectory_ids=eval_sequence_trajectory_ids,
        progress_bar=progress_bar,
        probe_schedule=probe_schedule,
        probe_enabled=probe_enabled,
        probe_num_train_samples=probe_num_train_samples,
        probe_num_eval_samples=probe_num_eval_samples,
        probe_verbose=probe_verbose,
        probe_geometry_enabled=probe_geometry_enabled,
        activation_rank_enabled=activation_rank_enabled,
        attention_entropy_enabled=attention_entropy_enabled,
        varcov_observe_enabled=varcov_observe_enabled,
        singular_values_enabled=singular_values_enabled,
        singular_values_num=singular_values_num,
        rollout_enabled=rollout_enabled,
        varcov_reg_enabled=varcov_reg_enabled,
        varcov_reg_start=varcov_reg_start,
        variance_reg_weight=variance_reg_weight,
        covariance_reg_weight=covariance_reg_weight,
        attention_entropy_reg_enabled=attention_entropy_reg_enabled,
        attention_entropy_weight=attention_entropy_weight,
        attention_entropy_start=attention_entropy_start,
        optimizer_type=optimizer_type,
    )

    train_losses = training_results['train_losses']
    total_losses = training_results['total_losses']
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
        'weight_decay': weight_decay,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'num_trajectories': num_traj,
        'train_size': train_size,
        'test_size': test_size,
        'loss_mask': loss_mask,
        'batch_size': batch_size,
        'scale_batch_by_context': scale_batch_by_context,
        'probe_schedule': probe_schedule,
        'probe_enabled': probe_enabled,
        'probe_num_train_samples': probe_num_train_samples,
        'probe_num_eval_samples': probe_num_eval_samples,
        'probe_verbose': probe_verbose,
        'probe_geometry_enabled': probe_geometry_enabled,
        'activation_rank_enabled': activation_rank_enabled,
        'attention_entropy_enabled': attention_entropy_enabled,
        'varcov_observe_enabled': varcov_observe_enabled,
        'singular_values_enabled': singular_values_enabled,
        'singular_values_num': singular_values_num,
        'rollout_enabled': rollout_enabled,
        'varcov_reg_enabled': varcov_reg_enabled,
        'varcov_reg_start': varcov_reg_start,
        'varcov_target_std': varcov_target_std,
        'variance_reg_weight': variance_reg_weight,
        'covariance_reg_weight': covariance_reg_weight,
        'attention_entropy_reg_enabled': attention_entropy_reg_enabled,
        'attention_entropy_weight': attention_entropy_weight,
        'attention_entropy_start': attention_entropy_start,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_total_loss': total_losses[-1] if total_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
        'train_losses': train_losses,
        'total_losses': total_losses,
        'test_losses': test_losses,
        'variance_losses': training_results['variance_losses'],
        'covariance_losses': training_results['covariance_losses'],
        'representation_mean_stds': training_results['representation_mean_stds'],
        'weighted_variance_terms': training_results['weighted_variance_terms'],
        'weighted_covariance_terms': training_results['weighted_covariance_terms'],
        'entropy_terms': training_results['entropy_terms'],
        'eval_results': eval_results,
        'eval_steps': eval_steps,
        'final_error_stats_train': final_eval['error_stats_train'] if final_eval and 'error_stats_train' in final_eval else None,
        'final_error_stats_test': final_eval['error_stats_test'] if final_eval and 'error_stats_test' in final_eval else None,
        'final_probe_results': final_eval['probe_results'] if final_eval else None,
        'final_geometry_probe_results': final_eval['geometry_probe_results'] if final_eval and 'geometry_probe_results' in final_eval else None,
        'memory_stats': memory_stats,
    }

    return results

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Newton-Kepler models and run linear probes."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Root directory for experiment outputs."
    )
    parser.add_argument(
        "--run-name", type=str, required=True,
        help="Experiment name used as the output subdirectory."
    )
    parser.add_argument(
        "--gpu", type=str, default=None,
        help="Physical GPU index, for example 0 or 1; use 'cpu' for CPU."
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite result files that already exist."
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The YAML config must contain a top-level mapping.")

    for section in ("data", "model", "training", "probe"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing or invalid config section: {section}")

    return config


def _config_list(config, plural_key, singular_key, default):
    """Read either a list-valued key or its backward-compatible singular key."""
    if plural_key in config:
        value = config[plural_key]
    elif singular_key in config:
        value = config[singular_key]
    else:
        value = default

    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def run_configured_experiments(config, run_dir, overwrite=False, console_stream=None):
    global _ACTIVE_RESULT_PATH, _ACTIVE_RESULT_METADATA
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    probe_config = config["probe"]
    observe_config = config.get("observe", {})
    regularization_config = config.get("regularization", {})

    experiment_seed = int(config.get("seed", 1))
    data_dir = str(data_config.get("data_dir", "data_cv"))
    num_trajectories = int(data_config.get("num_trajectories", 10000))

    # Peek at data shape to validate block_sizes before training loop
    from data_utils import load_spring_metadata
    _spring_meta = load_spring_metadata(data_dir)
    if _spring_meta:
        num_points_per_trajectory = int(_spring_meta.get("num_frames", 129))
    else:
        num_points_per_trajectory = 100  # Kepler default

    block_sizes = [
        int(value)
        for value in _config_list(
            training_config, "block_sizes", "block_size", [100]
        )
    ]
    noise_scales = [
        float(value)
        for value in _config_list(
            training_config, "noise_scales", "noise_scale", [0.1]
        )
    ]
    loss_masks = [
        str(value)
        for value in _config_list(
            training_config, "loss_masks", "loss_mask", ["all"]
        )
    ]

    n_layer = int(model_config.get("n_layer", 2))
    n_head = int(model_config.get("n_head", 1))
    n_embd = int(model_config.get("n_embd", 32))
    input_dim = int(model_config.get("input_dim", 1))
    learning_rate = float(
        training_config.get("learning_rate", training_config.get("lr", 1e-3))
    )
    weight_decay = float(training_config.get("weight_decay", 0.0))
    batch_size = int(training_config.get("batch_size", 128))
    optimizer_type = str(training_config.get("optimizer", "adamw"))
    scale_batch_by_context = bool(
        training_config.get("scale_batch_by_context", True)
    )
    n_steps = int(training_config.get("n_steps", 1001))
    progress_bar_enabled = bool(training_config.get("progress_bar", True))
    probe_enabled = _enabled(probe_config, True)
    probe_num_train_samples = int(probe_config.get("num_train_samples", 1000))
    probe_num_eval_samples = int(probe_config.get("num_eval_samples", 1000))
    probe_verbose = bool(probe_config.get("verbose", False))
    probe_geometry_enabled = bool(probe_config.get("geometry_enabled", True))
    probe_frequency = int(
        probe_config.get("frequency", probe_config.get("prob_freq", 100))
    )
    probe_schedule = probe_config.get("schedule")
    activation_rank_config = observe_config.get("activation_rank", {})
    activation_rank_enabled = _enabled(activation_rank_config, False)
    attention_entropy_config = observe_config.get("attention_entropy", {})
    attention_entropy_enabled = _enabled(attention_entropy_config, False)

    varcov_observe_config = observe_config.get("variance_covariance", {})
    varcov_observe_enabled = _enabled(varcov_observe_config, False)
    singular_values_config = observe_config.get("singular_values", {})
    singular_values_enabled = _enabled(singular_values_config, False)
    singular_values_num = (
        int(singular_values_config.get("num_singular_values", 50))
        if isinstance(singular_values_config, dict)
        else 50
    )
    evaluation_config = config.get("evaluation", {})
    if not isinstance(evaluation_config, dict):
        raise ValueError("evaluation must be a mapping")
    rollout_enabled = bool(evaluation_config.get("rollout_enabled", True))
    varcov_reg_config = regularization_config.get(
        "variance_covariance", {}
    )
    if not isinstance(varcov_reg_config, dict):
        raise ValueError(
            "regularization.variance_covariance must be a mapping"
        )
    varcov_reg_enabled = _enabled(varcov_reg_config, False)
    varcov_reg_start = int(varcov_reg_config.get("start_step", 1))
    varcov_target_std = float(varcov_reg_config.get("target_std", 0.1))
    variance_reg_weight = float(
        varcov_reg_config.get("variance_weight", 0.0)
    )
    covariance_reg_weight = float(
        varcov_reg_config.get("covariance_weight", 0.0)
    )

    # 新增：读取注意力熵正则化配置
    attention_entropy_reg_config = regularization_config.get("attention_entropy", {})
    attention_entropy_reg_enabled = _enabled(attention_entropy_reg_config, False)
    attention_entropy_weight = float(attention_entropy_reg_config.get("weight", 0.0))
    attention_entropy_start = int(attention_entropy_reg_config.get("start_step", 1))

    if num_trajectories <= 0:
        raise ValueError("data.num_trajectories must be positive")
    if n_steps <= 0:
        raise ValueError("training.n_steps must be positive")
    if batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if probe_num_train_samples <= 0 or probe_num_eval_samples <= 0:
        raise ValueError("probe sample counts must be positive")
    if singular_values_num <= 0:
        raise ValueError("observe.singular_values.num_singular_values must be positive")
    if probe_frequency <= 0:
        raise ValueError("probe.frequency must be positive")
    if probe_schedule is not None:
        if not isinstance(probe_schedule, list):
            raise ValueError("probe.schedule must be a list")
        for phase in probe_schedule:
            if not isinstance(phase, dict):
                raise ValueError("Each probe.schedule entry must be a mapping")
            missing = {"start", "end", "every"} - set(phase)
            if missing:
                raise ValueError(
                    f"probe.schedule entry is missing fields: {sorted(missing)}"
                )
            if int(phase["start"]) > int(phase["end"]):
                raise ValueError("probe.schedule start must not exceed end")
            if int(phase["every"]) <= 0:
                raise ValueError("probe.schedule every must be positive")
    if any(mask not in {"all", "last"} for mask in loss_masks):
        raise ValueError("training.loss_masks values must be 'all' or 'last'")
    if varcov_reg_start < 1:
        raise ValueError(
            "regularization.variance_covariance.start_step must be >= 1"
        )
    if varcov_target_std <= 0.0:
        raise ValueError(
            "regularization.variance_covariance.target_std must be positive"
        )
    if variance_reg_weight < 0.0 or covariance_reg_weight < 0.0:
        raise ValueError("variance/covariance weights must be non-negative")
    if attention_entropy_start < 1:
        raise ValueError("regularization.attention_entropy.start_step must be >= 1")
    if attention_entropy_weight < 0.0:
        raise ValueError("regularization.attention_entropy.weight must be non-negative")

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(block_sizes) * len(noise_scales) * len(loss_masks)
    run_index = 0
    progress_bar = (
        tqdm(
            total=total_runs * n_steps,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            mininterval=0.5,
            file=console_stream,
        )
        if progress_bar_enabled else None
    )

    try:
        for block_size in block_sizes:
            if not 1 <= block_size <= num_points_per_trajectory:
                raise ValueError(
                    f"block_size must be between 1 and "
                    f"{num_points_per_trajectory}, got {block_size}"
                )

            for noise_scale in noise_scales:
                for loss_mask in loss_masks:
                    run_index += 1
                    filename = (
                        f"block_{block_size}_noise_{noise_scale:g}_"
                        f"loss_{loss_mask}_seed_{experiment_seed}.npz"
                    )
                    result_path = results_dir / filename

                    if progress_bar is not None:
                        progress_bar.set_description(
                            f"block={block_size} noise={noise_scale:g} loss={loss_mask}",
                            refresh=True,
                        )

                    if result_path.exists() and not overwrite and _result_is_complete(result_path):
                        print(
                            f"[{run_index}/{total_runs}] "
                            f"Skipping completed result: {result_path}"
                        )
                        if progress_bar is not None:
                            progress_bar.update(n_steps)
                        continue
                    if result_path.exists() and not overwrite:
                        print(f"Restarting incomplete result: {result_path}")

                    print("\n" + "=" * 80)
                    print(f"[{run_index}/{total_runs}] Starting {filename}")
                    print("=" * 80)

                    # Reset before model initialization so every configuration uses
                    # the requested seed independently.
                    np.random.seed(experiment_seed)
                    torch.manual_seed(experiment_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(experiment_seed)

                    _ACTIVE_RESULT_PATH = result_path
                    _ACTIVE_RESULT_METADATA = {
                        "block_size": block_size,
                        "noise_scale": noise_scale,
                        "loss_mask": loss_mask,
                        "seed": experiment_seed,
                        "n_steps": n_steps,
                    }

                    try:
                        results = train_one_model(
                            block_size=block_size,
                            data_dir=data_dir,
                            noise_scale=noise_scale,
                            lr=learning_rate,
                            weight_decay=weight_decay,
                            n_layer=n_layer,
                            n_head=n_head,
                            n_embd=n_embd,
                            input_dim=input_dim,
                            num_trajectories=num_trajectories,
                            n_steps=n_steps,
                            prob_freq=probe_frequency,
                            loss_mask=loss_mask,
                            seed=experiment_seed,
                            batch_size=batch_size,
                            scale_batch_by_context=scale_batch_by_context,
                            progress_bar=progress_bar,
                            probe_schedule=probe_schedule,
                            probe_enabled=probe_enabled,
                            probe_num_train_samples=probe_num_train_samples,
                            probe_num_eval_samples=probe_num_eval_samples,
                            probe_verbose=probe_verbose,
                            probe_geometry_enabled=probe_geometry_enabled,
                            activation_rank_enabled=activation_rank_enabled,
                            attention_entropy_enabled=attention_entropy_enabled,
                            varcov_observe_enabled=varcov_observe_enabled,
                            singular_values_enabled=singular_values_enabled,
                            singular_values_num=singular_values_num,
                            rollout_enabled=rollout_enabled,
                            varcov_reg_enabled=varcov_reg_enabled,
                            varcov_reg_start=varcov_reg_start,
                            varcov_target_std=varcov_target_std,
                            variance_reg_weight=variance_reg_weight,
                            covariance_reg_weight=covariance_reg_weight,
                            attention_entropy_reg_enabled=attention_entropy_reg_enabled,
                            attention_entropy_weight=attention_entropy_weight,
                            attention_entropy_start=attention_entropy_start,
                            optimizer_type=optimizer_type,
                        )

                        final_payload = dict(results)
                        final_payload["status"] = "complete"
                        final_payload["completed_steps"] = n_steps
                        _atomic_save_npz(result_path, final_payload)
                        print(f"Saved result to: {result_path}")
                    finally:
                        _ACTIVE_RESULT_PATH = None
                        _ACTIVE_RESULT_METADATA = {}
    finally:
        if progress_bar is not None:
            progress_bar.close()


def main():
    args = parse_args()
    console_stream = sys.stdout

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    run_dir = Path(args.output_dir).expanduser() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_config_path = run_dir / "config.yaml"
    if config_path != saved_config_path.resolve():
        shutil.copy2(config_path, saved_config_path)

    print("\n" + "#" * 80)
    print(f"Run started: {datetime.now().isoformat(timespec='seconds')}")
    print(f"Config: {config_path}")
    print(f"Run name: {args.run_name}")
    print(f"Output directory: {run_dir.resolve()}")
    print(
        "Requested physical GPU: "
        f"{args.gpu if args.gpu is not None else 'environment/default'}"
    )
    print(
        "CUDA_VISIBLE_DEVICES: "
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}"
    )
    print(f"PyTorch device: {device}")
    if torch.cuda.is_available():
        print(f"Visible CUDA devices: {torch.cuda.device_count()}")
        print(f"Active CUDA device name: {torch.cuda.get_device_name(0)}")

    run_configured_experiments(
        config=config,
        run_dir=run_dir,
        overwrite=args.overwrite,
        console_stream=console_stream,
    )
    print(f"Run finished: {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()