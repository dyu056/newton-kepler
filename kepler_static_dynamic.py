"""
Train and probe a static/dynamic decoupled Kepler trajectory model.
"""

import argparse
import os

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from kepler_cv import load_trajectories, compute_gravitational_force, generate_trajectory_and_compute_error
from kepler_cv_blocksize import load_orbital_params
from model_static_dynamic import StaticDynamicConfig, StaticDynamicKeplerModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


STATIC_TARGETS = [
    "e", "a", "b", "c", "average_radius", "LRL_x", "LRL_y",
    "LRL_magnitude", "LRL_angle", "n_x", "n_y",
]

DYNAMIC_TARGETS = [
    "F_magnitude", "F_direction_x", "F_direction_y", "Fx", "Fy",
    "r", "inv_r", "r_squared", "inv_r_squared", "inv_r_cubed", "x", "y",
]


def setup_model(args):
    config = StaticDynamicConfig(
        block_size=100,
        input_dim=2,
        static_window=args.static_window,
        static_dim=args.static_dim,
        dynamic_dim=args.dynamic_dim,
        n_layer_static=args.n_layer_static,
        n_layer_dynamic=args.n_layer_dynamic,
        n_head=args.n_head,
        dropout=args.dropout,
    )
    return StaticDynamicKeplerModel(config).to(device)


def split_dataset(num_trajectories):
    trajectories = load_trajectories("data_cv", num_trajectories_needed=2 * num_trajectories)
    num_loaded = trajectories.shape[0]
    train_trajectories = trajectories[:num_loaded // 2]
    test_trajectories = trajectories[num_loaded // 2:]

    train_inputs = torch.from_numpy(train_trajectories[:, :-1]).float()
    train_targets = torch.from_numpy(train_trajectories[:, 1:]).float()
    test_inputs = torch.from_numpy(test_trajectories[:, :-1]).float()
    test_targets = torch.from_numpy(test_trajectories[:, 1:]).float()

    orbital_params = load_orbital_params("data_cv", num_trajectories_needed=2 * num_trajectories)
    train_params = orbital_params[:num_loaded // 2] if orbital_params else None
    test_params = orbital_params[num_loaded // 2:] if orbital_params else None

    return {
        "train_trajectories": train_trajectories,
        "test_trajectories": test_trajectories,
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "test_inputs": test_inputs,
        "test_targets": test_targets,
        "train_params": train_params,
        "test_params": test_params,
    }


def _fit_probe(features, targets):
    probe = LinearRegression()
    probe.fit(features, targets)
    return r2_score(targets, probe.predict(features))


def run_decoupling_probes(model, inputs, orbital_params, sample_indices=None):
    model.eval()
    inputs = inputs.to(device)
    with torch.no_grad():
        _, _, latents = model(inputs, return_latents=True)

    z_static = latents["z_static"].cpu().numpy()
    z_dyn = latents["z_dyn"].cpu().numpy()
    batch_size, time_steps, dyn_dim = z_dyn.shape
    z_dyn_flat = z_dyn.reshape(batch_size * time_steps, dyn_dim)
    z_dyn_last = z_dyn[:, -1, :]

    positions = inputs.cpu().numpy()
    dynamic_values = compute_gravitational_force(positions)

    if sample_indices is None:
        sample_indices = np.arange(batch_size)

    static_values = {}
    if orbital_params:
        for target in STATIC_TARGETS:
            static_values[target] = np.array([
                orbital_params[int(idx)].get(target, 0.0) for idx in sample_indices
            ])

    probe_results = {
        "static_from_z_static": {},
        "static_from_z_dyn_last": {},
        "dynamic_from_z_dyn": {},
        "dynamic_from_z_static_repeated": {},
    }

    for target, values in static_values.items():
        probe_results["static_from_z_static"][target] = _fit_probe(z_static, values)
        probe_results["static_from_z_dyn_last"][target] = _fit_probe(z_dyn_last, values)

    z_static_repeated = np.repeat(z_static, time_steps, axis=0)
    for target in DYNAMIC_TARGETS:
        values = dynamic_values[target]
        probe_results["dynamic_from_z_dyn"][target] = _fit_probe(z_dyn_flat, values)
        probe_results["dynamic_from_z_static_repeated"][target] = _fit_probe(z_static_repeated, values)

    return probe_results


def summarize_probe_results(probe_results):
    summary = {}
    for section, values in probe_results.items():
        if values:
            summary[f"{section}_mean_r2"] = float(np.mean(list(values.values())))
            summary[f"{section}_max_r2"] = float(np.max(list(values.values())))
    return summary


def train_model(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = split_dataset(args.num_trajectories)
    model = setup_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_inputs = data["train_inputs"]
    train_targets = data["train_targets"]
    test_inputs = data["test_inputs"]
    test_targets = data["test_targets"]

    train_losses = []
    test_losses = []
    eval_results = []
    eval_steps = []

    for step in range(args.n_steps):
        if step == args.n_steps // 2:
            for group in optimizer.param_groups:
                group["lr"] *= 0.1

        model.train()
        batch_indices = torch.randint(0, train_inputs.shape[0], (args.batch_size,))
        batch_inputs = train_inputs[batch_indices].to(device)
        batch_targets = train_targets[batch_indices].to(device)
        batch_inputs = batch_inputs + torch.randn_like(batch_inputs) * args.noise_scale

        _, loss = model(
            batch_inputs,
            batch_targets,
            static_consistency_weight=args.static_consistency_weight,
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            test_indices = torch.randint(0, test_inputs.shape[0], (args.batch_size,))
            _, test_loss = model(test_inputs[test_indices].to(device), test_targets[test_indices].to(device))
            test_losses.append(float(test_loss.item()))

        if step % args.log_freq == 0:
            print(f"Step {step}, Train Loss: {train_losses[-1]:.6f}, Test Loss: {test_losses[-1]:.6f}")

        if step % args.prob_freq == 0 or step == args.n_steps - 1:
            eval_step = step + 1
            print(f"\nEvaluating at step {eval_step}...")
            sample_size = min(args.probe_sample_size, train_inputs.shape[0])
            sample_indices = torch.randperm(train_inputs.shape[0])[:sample_size]

            probe_results = run_decoupling_probes(
                model,
                train_inputs[sample_indices],
                data["train_params"],
                sample_indices=sample_indices.numpy(),
            )
            probe_summary = summarize_probe_results(probe_results)

            train_eval_size = min(args.rollout_sample_size, train_inputs.shape[0])
            test_eval_size = min(args.rollout_sample_size, test_inputs.shape[0])
            train_eval_indices = torch.randperm(train_inputs.shape[0])[:train_eval_size]
            test_eval_indices = torch.randperm(test_inputs.shape[0])[:test_eval_size]

            error_stats_train = generate_trajectory_and_compute_error(
                model,
                train_inputs[train_eval_indices].to(device),
                data["train_trajectories"][train_eval_indices.numpy()],
                conditioning_length=args.conditioning_length,
            )
            error_stats_test = generate_trajectory_and_compute_error(
                model,
                test_inputs[test_eval_indices].to(device),
                data["test_trajectories"][test_eval_indices.numpy()],
                conditioning_length=args.conditioning_length,
            )

            eval_results.append({
                "step": eval_step,
                "probe_results": probe_results,
                "probe_summary": probe_summary,
                "error_stats_train": error_stats_train,
                "error_stats_test": error_stats_test,
            })
            eval_steps.append(eval_step)
            print(f"Probe summary: {probe_summary}")
            print(f"Evaluation at step {eval_step} completed.\n")

    final_eval = eval_results[-1] if eval_results else None
    results = {
        "config": vars(args),
        "num_train": train_inputs.shape[0],
        "num_test": test_inputs.shape[0],
        "train_losses": train_losses,
        "test_losses": test_losses,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1],
        "eval_steps": eval_steps,
        "eval_results": eval_results,
        "final_probe_summary": final_eval["probe_summary"] if final_eval else None,
        "final_error_stats_train": final_eval["error_stats_train"] if final_eval else None,
        "final_error_stats_test": final_eval["error_stats_test"] if final_eval else None,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f"static_dynamic_num_trajectories_{args.num_trajectories}_steps_{args.n_steps}_seed_{args.seed}.npz",
    )
    np.savez(output_path, **results)
    print(f"Saved results to {output_path}")
    return results, output_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Static/dynamic decoupled Kepler experiment")
    parser.add_argument("--num_trajectories", type=int, default=100)
    parser.add_argument("--n_steps", type=int, default=101)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--static_window", type=int, default=20)
    parser.add_argument("--static_dim", type=int, default=16)
    parser.add_argument("--dynamic_dim", type=int, default=32)
    parser.add_argument("--n_layer_static", type=int, default=1)
    parser.add_argument("--n_layer_dynamic", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--static_consistency_weight", type=float, default=0.0)
    parser.add_argument("--conditioning_length", type=int, default=50)
    parser.add_argument("--probe_sample_size", type=int, default=100)
    parser.add_argument("--rollout_sample_size", type=int, default=100)
    parser.add_argument("--prob_freq", type=int, default=50)
    parser.add_argument("--log_freq", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="results/kepler_static_dynamic")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
