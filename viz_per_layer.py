"""
Per-layer R2 evolution visualization for ablation experiments.
Usage: python viz_per_layer.py
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
    steps, r2s = [], []
    for ev, st in zip(data['eval_results'], data['eval_steps']):
        pr = ev['probe_results']
        if layer in pr and probe in pr[layer]:
            v = pr[layer][probe]
            for m in metrics:
                if m in v:
                    steps.append(st)
                    r2s.append(v[m])
                    break
    return steps, r2s


def extract_best_r2(data, probe, metrics=('eval_r2_sequence_last', 'r2_last')):
    steps, r2s = [], []
    for ev, st in zip(data['eval_results'], data['eval_steps']):
        best = -np.inf
        for layer, vals in ev['probe_results'].items():
            if probe in vals:
                for m in metrics:
                    if m in vals[probe]:
                        best = max(best, vals[probe][m])
                        break
        steps.append(st)
        r2s.append(max(best, 0))
    return steps, r2s


# =========================================================================
# Configuration -- edit this section to compare different experiments
# =========================================================================

EXPERIMENTS = {
    'v2': {
        'path': 'results/kepler_cv_blocksize/results_block_size_100_num_trajectories_10000_noise_scale_0.1_loss_mask_all.npz',
        'color': '#2196F3', 'marker': 'o', 'label': 'v2 (baseline, no SVB)'
    },
    'svb': {
        'path': 'results/kepler_cv_blocksize_svb/results_block_size_100_num_trajectories_10000_noise_scale_0.1_loss_mask_all.npz',
        'color': '#FF5722', 'marker': 's', 'label': 'svb (epsilon=0.5)'
    },
}

PROBES = ['F_magnitude', 'Fx', 'Fy']
os.makedirs('plots', exist_ok=True)

# =========================================================================
# Load data
# =========================================================================
data = {}
for name, cfg in list(EXPERIMENTS.items()):
    if os.path.exists(cfg['path']):
        data[name] = load_result(cfg['path'])
        print(f'Loaded {name}: train_loss={data[name]["train_losses"][-1]:.4f} '
              f'test_loss={data[name]["test_losses"][-1]:.4f}')
    else:
        print(f'WARNING: {cfg["path"]} not found, skipping {name}')
        del EXPERIMENTS[name]

if len(data) < 1:
    print('No data loaded. Exiting.')
    sys.exit(1)

first_name = list(data.keys())[0]
layers = layer_order(data[first_name]['eval_results'][0]['probe_results'].keys())
labels = [short_label(l) for l in layers]
print(f'Layers ({len(layers)}): {labels}')

# =========================================================================
# Plot 1: Per-probe, per-layer evolution (13 subplots each)
# =========================================================================
for probe in PROBES:
    fig, axes = plt.subplots(4, 4, figsize=(24, 22))
    axes_flat = axes.flatten()

    for idx, layer in enumerate(layers):
        ax = axes_flat[idx]
        for name, cfg in EXPERIMENTS.items():
            steps, r2s = extract_layer_r2(data[name], layer, probe)
            if len(steps) > 0:
                ax.plot(steps, r2s, cfg['marker'] + '-', color=cfg['color'],
                        ms=5, lw=1.8, label=cfg['label'], alpha=0.85)
        ax.set_title(labels[idx], fontsize=12, fontweight='bold')
        ax.set_xlabel('step')
        ax.set_ylabel('eval R2')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

    # Common legend in upper-right corner
    handles = [plt.Line2D([0], [0], color=cfg['color'], marker=cfg['marker'],
                          ms=10, lw=2.5, label=cfg['label'])
               for cfg in EXPERIMENTS.values()]
    fig.legend(handles=handles, loc='upper right', fontsize=13, framealpha=0.95,
               bbox_to_anchor=(0.98, 0.968))

    for idx in range(len(layers), 16):
        axes_flat[idx].set_visible(False)

    fname = f'plots/per_layer_evolution_{probe}.png'
    fig.suptitle(f'{probe} -- Per-Layer R2 Evolution',
                 fontsize=20, y=1.03, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(fname, dpi=150)
    print(f'Saved: {fname}')

# =========================================================================
# Plot 2: Best-across-layers R2 evolution
# =========================================================================
fig, axes = plt.subplots(1, len(PROBES), figsize=(7 * len(PROBES), 5.5))
if len(PROBES) == 1:
    axes = [axes]
for idx, probe in enumerate(PROBES):
    ax = axes[idx]
    for name, cfg in EXPERIMENTS.items():
        steps, r2s = extract_best_r2(data[name], probe)
        ax.plot(steps, r2s, cfg['marker'] + '-', color=cfg['color'],
                ms=5, lw=2, label=cfg['label'])
    ax.set_title(f'{probe} (best across all layers)', fontweight='bold', fontsize=13)
    ax.set_xlabel('step')
    ax.set_ylabel('eval R2')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=11, loc='lower right')
fig.suptitle('Best R2 Across All Layers', fontsize=16, y=1.02, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('plots/best_r2_evolution.png', dpi=150)
print('Saved: plots/best_r2_evolution.png')

# =========================================================================
# Plot 3: Loss comparison
# =========================================================================
fig, axes = plt.subplots(1, len(data), figsize=(7 * len(data), 5.5))
if len(data) == 1:
    axes = [axes]
for idx, (name, cfg) in enumerate(EXPERIMENTS.items()):
    ax = axes[idx]
    d = data[name]
    ax.plot(d['train_losses'], alpha=0.3, lw=0.5, color=cfg['color'])
    ax.plot(d['test_losses'], lw=2, color=cfg['color'], label='test loss')
    ax.set_yscale('log')
    ax.set_title(f'{cfg["label"]}\ntrain={d["train_losses"][-1]:.4f}  '
                 f'test={d["test_losses"][-1]:.4f}', fontweight='bold', fontsize=12)
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
fig.suptitle('Training & Test Loss', fontsize=16, y=1.02, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('plots/loss_comparison.png', dpi=150)
print('Saved: plots/loss_comparison.png')

# =========================================================================
# Plot 4: Key layers summary grid
# =========================================================================
key_layers = ['input_embed', 'block_0_attn_output', 'block_0_mlp_hidden',
              'block_0_after_mlp_merge', 'block_1_attn_output', 'block_1_mlp_hidden',
              'block_1_after_mlp_merge', 'after_ln_f']
key_layers = [l for l in key_layers if l in layers]

fig, axes = plt.subplots(len(PROBES), len(key_layers),
                         figsize=(4.2 * len(key_layers), 4 * len(PROBES)))
if len(PROBES) == 1 and len(key_layers) == 1:
    axes = np.array([[axes]])
elif len(PROBES) == 1:
    axes = np.array([axes])
elif len(key_layers) == 1:
    axes = axes[:, np.newaxis]

for pi, probe in enumerate(PROBES):
    for li, layer in enumerate(key_layers):
        ax = axes[pi][li]
        for name, cfg in EXPERIMENTS.items():
            steps, r2s = extract_layer_r2(data[name], layer, probe)
            if len(steps) > 0:
                ax.plot(steps, r2s, cfg['marker'] + '-', color=cfg['color'],
                        ms=3, lw=1.2, label=cfg['label'])
        ax.set_title(f'{probe} @ {short_label(layer)}', fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        if pi == 0 and li == 0:
            ax.legend(fontsize=8)

fig.suptitle('Per-Layer R2 Evolution -- Key Layers Summary', fontsize=16, y=1.02,
             fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('plots/key_layers_summary.png', dpi=150)
print('Saved: plots/key_layers_summary.png')

# =========================================================================
# Print final values table
# =========================================================================
print('\n' + '=' * 80)
print('FINAL EVAL STEP -- Per-Layer R2 Values')
print('=' * 80)
header = f"{'Layer':<14}"
for probe in PROBES:
    for name in EXPERIMENTS:
        header += f'  {name}:{probe[:8]:>8}'
print(header)
print('-' * len(header))
for layer in layers:
    line = f'{short_label(layer):<14}'
    for probe in PROBES:
        for name in EXPERIMENTS:
            _, r2s = extract_layer_r2(data[name], layer, probe)
            line += f'  {r2s[-1]:>14.4f}' if r2s else '            N/A'
    print(line)

print('\nDone!')
