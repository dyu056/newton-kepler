"""
Compare model_cv (with attention) vs model_mlp (no attention) training results.
Replicates the key plots from fig6_block_size_kepler_newton.ipynb for both models.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({'font.size': 14})

block_sizes = [1, 2, 5, 10, 20, 50, 100]
noise_scale = 0.1
num_traj = 10000
loss_mask = 'all'
postfix = '_last'


def load_results(results_dir, block_sizes):
    """Load results for all block sizes from a directory."""
    summary = {
        'block_sizes': [],
        'best_r2_F_magnitude': [], 'best_r2_Fx': [], 'best_r2_Fy': [],
        'best_r2_a': [], 'best_r2_b': [], 'best_r2_LRL_x': [], 'best_r2_LRL_y': [],
    }

    for bs in block_sizes:
        fname = f'results_block_size_{bs}_num_trajectories_{num_traj}_noise_scale_{noise_scale}_loss_mask_{loss_mask}.npz'
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found")
            continue

        loaded = np.load(fpath, allow_pickle=True)
        data = {key: loaded[key] for key in loaded.keys()}

        probe = data['eval_results'][-1]['probe_results']
        geo = data['eval_results'][-1]['geometry_probe_results']

        def best_r2(layer_results, key):
            best = -np.inf
            for layer_name, layer_data in layer_results.items():
                if key in layer_data:
                    r2 = layer_data[key].get('r2' + postfix, -np.inf)
                    if r2 > best:
                        best = r2
            return best

        summary['block_sizes'].append(bs)
        summary['best_r2_F_magnitude'].append(best_r2(probe, 'F_magnitude'))
        summary['best_r2_Fx'].append(best_r2(probe, 'Fx'))
        summary['best_r2_Fy'].append(best_r2(probe, 'Fy'))
        summary['best_r2_a'].append(best_r2(geo, 'a'))
        summary['best_r2_b'].append(best_r2(geo, 'b'))
        summary['best_r2_LRL_x'].append(best_r2(geo, 'LRL_x'))
        summary['best_r2_LRL_y'].append(best_r2(geo, 'LRL_y'))

    for k in summary:
        summary[k] = np.array(summary[k])
    return summary


# Load both models
print("Loading model_cv (WITH attention)...")
cv = load_results('./results/kepler_cv_blocksize', block_sizes)
print("Loading model_mlp (NO attention)...")
mlp = load_results('./results/kepler_cv_blocksize_mlp', block_sizes)

os.makedirs('./plots', exist_ok=True)

# ── Plot 1: Emergence of Gravity (Newtonian probe) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (data, title) in zip(axes, [(cv, 'With Attention (model_cv)'), (mlp, 'No Attention (model_mlp)')]):
    ax.plot(data['block_sizes'], data['best_r2_F_magnitude'], 'o-', label='$F$ magnitude')
    ax.plot(data['block_sizes'], data['best_r2_Fx'], 's--', label='$F_x$')
    ax.plot(data['block_sizes'], data['best_r2_Fy'], '^:', label='$F_y$')
    ax.set_xscale('log')
    ax.set_xlabel('Context Length')
    ax.set_ylabel('$R^2$ score')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

fig.suptitle('Emergence of Gravity: Newtonian Probes (n_layer=1)', fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig('./plots/compare_emergence_of_gravity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: compare_emergence_of_gravity.png")


# ── Plot 2: Keplerian Probes ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (data, title) in zip(axes, [(cv, 'With Attention (model_cv)'), (mlp, 'No Attention (model_mlp)')]):
    ax.plot(data['block_sizes'][1:], data['best_r2_a'][1:], 'o-', label='$a$ (semi-major)')
    ax.plot(data['block_sizes'][1:], data['best_r2_b'][1:], 's--', label='$b$ (semi-minor)')
    ax.plot(data['block_sizes'][1:], data['best_r2_LRL_x'][1:], '^:', label='$A_x$ (LRL)')
    ax.plot(data['block_sizes'][1:], data['best_r2_LRL_y'][1:], 'v-.', label='$A_y$ (LRL)')
    ax.set_xscale('log')
    ax.set_xlabel('Context Length')
    ax.set_ylabel('$R^2$ score')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim(0.75, 1.05)
    ax.axhline(y=1.0, ls='--', color='black', alpha=0.3)
    ax.grid(True, alpha=0.3)

fig.suptitle('Keplerian Model: Geometry Probes (n_layer=1)', fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig('./plots/compare_keplerian.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: compare_keplerian.png")


# ── Plot 3: Phase Transition (Kepler vs Newton scores) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (data, title) in zip(axes, [(cv, 'With Attention (model_cv)'), (mlp, 'No Attention (model_mlp)')]):
    kepler_score = (data['best_r2_a'] + data['best_r2_b'] + data['best_r2_LRL_x'] + data['best_r2_LRL_y']) / 4
    newton_score = (data['best_r2_F_magnitude'] + data['best_r2_Fx'] + data['best_r2_Fy']) / 3

    ax.plot(data['block_sizes'][1:], kepler_score[1:], 'o-', label='Keplerian score', linewidth=2)
    ax.plot(data['block_sizes'][1:], newton_score[1:], 's--', label='Newtonian score', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Context Length')
    ax.set_ylabel('$R^2$ score')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=13, loc='lower right')
    ax.set_ylim(0.8, 1.05)
    ax.axhline(y=1.0, ls='--', color='black', alpha=0.3)
    ax.grid(True, alpha=0.3)

fig.suptitle('Phase Transition: Newtonian vs Keplerian Models (n_layer=1)', fontsize=20, y=1.02)
plt.tight_layout()
plt.savefig('./plots/compare_phase_transition.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: compare_phase_transition.png")


# ── Print Summary Table ──
print("\n" + "="*100)
print("SUMMARY: R² scores comparison")
print("="*100)
print(f"{'Block':>6} | {'F_mag (cv)':>12} {'F_mag (mlp)':>13} | {'Fx (cv)':>10} {'Fx (mlp)':>11} | {'Fy (cv)':>10} {'Fy (mlp)':>11} | {'a (cv)':>8} {'a (mlp)':>9} | {'b (cv)':>8} {'b (mlp)':>9}")
print("-"*100)
for i, bs in enumerate(block_sizes):
    print(f"{bs:>6} | {cv['best_r2_F_magnitude'][i]:>12.4f} {mlp['best_r2_F_magnitude'][i]:>13.4f} | "
          f"{cv['best_r2_Fx'][i]:>10.4f} {mlp['best_r2_Fx'][i]:>11.4f} | "
          f"{cv['best_r2_Fy'][i]:>10.4f} {mlp['best_r2_Fy'][i]:>11.4f} | "
          f"{cv['best_r2_a'][i]:>8.4f} {mlp['best_r2_a'][i]:>9.4f} | "
          f"{cv['best_r2_b'][i]:>8.4f} {mlp['best_r2_b'][i]:>9.4f}")

print("\nDone! All plots saved to ./plots/")
