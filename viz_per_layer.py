"""
Per-layer R² evolution visualization for ablation experiments.
Usage: python viz_per_layer.py
Requires result .npz files in results/kepler_cv_blocksize/ and results/kepler_cv_blocksize_svb/
"""

import numpy as np, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_result(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.keys()}


def layer_order(layers):
    def key(k):
        if 'input' in k: return 0
        if 'after_pos' in k: return 1
        if 'after_ln_f' in k: return 99
        parts = k.split('_')
        blk = int(parts[1])
        suffix = '_'.join(parts[2:])
        so = {'attn_output': 0, 'after_attn_merge': 1, 'mlp_output': 2,
              'after_mlp_merge': 3, 'mlp_hidden': 4}
        return 2 + blk * 5 + so.get(suffix, 99)
    return sorted(layers, key=key)


def short_label(name):
    return (name.replace('block_', 'b').replace('_attn_output', '_attn')
            .replace('_after_attn_merge', '+attn').replace('_mlp_output', '_mlp')
            .replace('_after_mlp_merge', '+mlp').replace('_mlp_hidden', '_hid')
            .replace('input_embed', 'embed').replace('after_pos_emb', '+pos')
            .replace('after_ln_f', 'ln_f'))


def extract_layer_r2(data, layer, probe, metrics=('eval_r2_sequence_last', 'r2_last')):
    """Extract R² time series for a specific layer and probe."""
    steps, r2s = [], []
    for ev, st in zip(data['eval_results'], data['eval_steps']):
        pr = ev['probe_results']
        if layer in pr and probe in pr[layer]:
            v = pr[layer][probe]
            for m in metrics:
                if m in v:
                    steps.append(st); r2s.append(v[m]); break
    return steps, r2s


def extract_best_r2(data, probe, metrics=('eval_r2_sequence_last', 'r2_last')):
    """Extract best R² across all layers for a probe."""
    steps, r2s = [], []
    for ev, st in zip(data['eval_results'], data['eval_steps']):
        best = -np.inf
        for layer, vals in ev['probe_results'].items():
            if probe in vals:
                for m in metrics:
                    if m in vals[probe]:
                        best = max(best, vals[probe][m]); break
        steps.append(st); r2s.append(max(best, 0))
    return steps, r2s


# ────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────

EXPERIMENTS = {
    'v2': {
        'path': 'results/kepler_cv_blocksize/results_block_size_100_num_trajectories_10000_noise_scale_0.1_loss_mask_all.npz',
        'color': '#2196F3', 'marker': 'o', 'label': 'v2 (baseline)'
    },
    'svb': {
        'path': 'results/kepler_cv_blocksize_svb/results_block_size_100_num_trajectories_10000_noise_scale_0.1_loss_mask_all.npz',
        'color': '#FF5722', 'marker': 's', 'label': 'svb (ε=0.5)'
    },
}

PROBES = ['F_magnitude', 'Fx', 'Fy']
os.makedirs('plots', exist_ok=True)

# ────────────────────────────────────────────────────────────
# Load all experiments
# ────────────────────────────────────────────────────────────
data = {}
for name, cfg in EXPERIMENTS.items():
    if os.path.exists(cfg['path']):
        data[name] = load_result(cfg['path'])
        print(f'Loaded {name}: train_loss={data[name]["train_losses"][-1]:.4f} test_loss={data[name]["test_losses"][-1]:.4f}')
    else:
        print(f'WARNING: {cfg["path"]} not found, skipping {name}')
        EXPERIMENTS.pop(name)

if len(data) < 2:
    print('Need at least 2 experiments for comparison. Exiting.')
    sys.exit(1)

# Get layer list from first experiment
first_name = list(data.keys())[0]
layers = layer_order(data[first_name]['eval_results'][0]['probe_results'].keys())
labels = [short_label(l) for l in layers]
print(f'Layers ({len(layers)}): {labels}')

# ────────────────────────────────────────────────────────────
# Plot 1: Per-probe, per-layer evolution (13 subplots each)
# ────────────────────────────────────────────────────────────
for probe in PROBES:
    fig, axes = plt.subplots(4, 4, figsize=(24, 22))
    axes_flat = axes.flatten()

    for idx, layer in enumerate(layers):
        ax = axes_flat[idx]
        for name, cfg in EXPERIMENTS.items():
            steps, r2s = extract_layer_r2(data[name], layer, probe)
            if len(steps) > 0:
                ax.plot(steps, r2s, cfg['marker'] + '-', color=cfg['color'], ms=3, lw=1,
                        label=cfg['label'])
        ax.set_title(labels[idx], fontsize=10, fontweight='bold')
        ax.set_xlabel('step'); ax.set_ylabel('eval R²'); ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        if idx == 0: ax.legend(fontsize=7)

    for idx in range(len(layers), 16):
        axes_flat[idx].set_visible(False)

    fname = f'plots/per_layer_evolution_{probe}.png'
    plt.suptitle(f'{probe} — Per-Layer R² Evolution — '
                 f'{" vs ".join(cfg["label"] for cfg in EXPERIMENTS.values())}',
                 fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    print(f'Saved: {fname}')

# ────────────────────────────────────────────────────────────
# Plot 2: Best-across-layers R² evolution
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(PROBES), figsize=(6 * len(PROBES), 5))
if len(PROBES) == 1: axes = [axes]
for idx, probe in enumerate(PROBES):
    ax = axes[idx]
    for name, cfg in EXPERIMENTS.items():
        steps, r2s = extract_best_r2(data[name], probe)
        ax.plot(steps, r2s, cfg['marker'] + '-', color=cfg['color'], ms=3, lw=1.5,
                label=cfg['label'])
    ax.set_title(f'{probe} (best across layers)', fontweight='bold')
    ax.set_xlabel('step'); ax.set_ylabel('eval R²'); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1); ax.legend()
plt.tight_layout()
plt.savefig('plots/best_r2_evolution.png', dpi=150)
print('Saved: plots/best_r2_evolution.png')

# ────────────────────────────────────────────────────────────
# Plot 3: Loss comparison
# ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 5))
if len(data) == 1: axes = [axes]
for idx, (name, cfg) in enumerate(EXPERIMENTS.items()):
    ax = axes[idx]
    d = data[name]
    ax.plot(d['train_losses'], alpha=0.3, lw=0.5, color=cfg['color'])
    ax.plot(d['test_losses'], lw=1.5, color=cfg['color'])
    ax.set_yscale('log')
    ax.set_title(f'{cfg["label"]}\ntrain={d["train_losses"][-1]:.4f} test={d["test_losses"][-1]:.4f}',
                 fontweight='bold')
    ax.set_xlabel('step'); ax.set_ylabel('loss'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plots/loss_comparison.png', dpi=150)
print('Saved: plots/loss_comparison.png')

# ────────────────────────────────────────────────────────────
# Plot 4: Key layers × key probes summary grid
# ────────────────────────────────────────────────────────────
key_layers = ['input_embed', 'after_pos_emb', 'block_0_attn_output', 'block_0_mlp_hidden',
              'block_0_after_mlp_merge', 'block_1_attn_output', 'block_1_mlp_hidden',
              'block_1_after_mlp_merge', 'after_ln_f']
key_layers = [l for l in key_layers if l in layers]

fig, axes = plt.subplots(len(PROBES), len(key_layers), figsize=(4 * len(key_layers), 3.5 * len(PROBES)))
if len(PROBES) == 1: axes = [axes]

for pi, probe in enumerate(PROBES):
    for li, layer in enumerate(key_layers):
        ax = axes[pi][li] if len(PROBES) > 1 else axes[li]
        for name, cfg in EXPERIMENTS.items():
            steps, r2s = extract_layer_r2(data[name], layer, probe)
            if len(steps) > 0:
                ax.plot(steps, r2s, cfg['marker'] + '-', color=cfg['color'], ms=2, lw=0.8,
                        label=cfg['label'])
        ax.set_title(f'{probe} @ {short_label(layer)}', fontsize=9)
        ax.set_ylim(-0.1, 1.1); ax.grid(True, alpha=0.3)
        if pi == 0 and li == 0: ax.legend(fontsize=7)

plt.suptitle('Per-Layer R² Evolution — Key Layers Summary', fontsize=16, y=1.01)
plt.tight_layout()
plt.savefig('plots/key_layers_summary.png', dpi=150)
print('Saved: plots/key_layers_summary.png')

# ────────────────────────────────────────────────────────────
# Print final values table
# ────────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('FINAL EVAL STEP — Per-Layer R²')
print('=' * 70)
header = f"{'Layer':<14}"
for probe in PROBES:
    for name in EXPERIMENTS:
        header += f' {name}:{probe[:8]:>8}'
print(header)
print('-' * len(header))
for layer in layers:
    line = f'{short_label(layer):<14}'
    for probe in PROBES:
        for name in EXPERIMENTS:
            _, r2s = extract_layer_r2(data[name], layer, probe)
            line += f' {r2s[-1]:>14.4f}' if r2s else '          N/A'
    print(line)

print('\nDone! All plots saved to plots/')
