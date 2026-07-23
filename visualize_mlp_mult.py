"""
Visualize MLP width sweep results.

Generates:
  1. R² evolution plots — one figure per observable, subplots per layer,
     each showing eval_r2_sequence_last vs training step for all mlp_mult.
  2. Global max R² comparison — layers on x-axis, max eval R² on y-axis,
     grouped bars per mlp_mult.
  3. Loss curves — train + test loss for all 4 mlp_mult in a single figure.

Usage:
    python visualize_mlp_mult.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = Path('results_mlp_mult')
OUT_DIR = Path('plots_mlp_mult')
MULT_LIST = [2, 4, 8, 16]
OBS_KEYS = ['Fx', 'Fy', 'F_magnitude', 'r', 'x', 'y']

LAYER_ORDER = [
    'input_embed', 'after_pos_emb',
    'block_0_attn_output', 'block_0_after_attn_merge',
    'block_0_mlp_hidden', 'block_0_mlp_output', 'block_0_after_mlp_merge',
    'after_ln_f',
]
LAYER_LABELS = ['emb', '+pos', 'b0\na', 'b0\n+a',
                'b0\nmh', 'b0\nmo', 'b0\n+m', 'LN']

OBS_LABELS = {
    'Fx': r'$F_x$', 'Fy': r'$F_y$', 'F_magnitude': r'$|F|$',
    'r': r'$r$', 'x': r'$x$', 'y': r'$y$',
}

METRIC = 'eval_r2_sequence_last'

COLORS = {2: '#87CEEB', 4: '#5B9BD5', 8: '#2166AC', 16: '#08306B'}
MARKERS = {2: 'o', 4: 's', 8: '^', 16: 'D'}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.grid": True, "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.6, "legend.fontsize": 8,
    "grid.linestyle": "--", "grid.alpha": 0.4,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


# ── Load data ───────────────────────────────────────────────────────────
def load_data(mult):
    path = DATA_DIR / f'results_block_size_100_num_trajectories_10000_noise_scale_0.1_loss_mask_all_mlp_mult_{mult}.npz'
    d = np.load(path, allow_pickle=True)
    steps = d['eval_steps']  # (201,)
    eval_results = d['eval_results']  # (201,) of dicts
    train_losses = d['train_losses']  # (2001,)
    test_losses = d['test_losses']    # (2001,)
    return steps, eval_results, train_losses, test_losses


def extract_r2_evolution(eval_results, steps):
    """Returns {layer: {obs: [r2_at_step0, r2_at_step10, ...]}}"""
    evo = {layer: {obs: [] for obs in OBS_KEYS} for layer in LAYER_ORDER}
    for er in eval_results:
        pr = er['probe_results']
        for layer in LAYER_ORDER:
            if layer not in pr:
                for obs in OBS_KEYS:
                    evo[layer][obs].append(np.nan)
                continue
            for obs in OBS_KEYS:
                evo[layer][obs].append(pr[layer][obs][METRIC])
    return evo


def extract_max_r2(eval_results):
    """Returns {layer: {obs: max_r2}} across all steps"""
    max_r2 = {layer: {obs: -np.inf for obs in OBS_KEYS} for layer in LAYER_ORDER}
    for er in eval_results:
        pr = er['probe_results']
        for layer in LAYER_ORDER:
            if layer not in pr:
                continue
            for obs in OBS_KEYS:
                val = pr[layer][obs][METRIC]
                if val > max_r2[layer][obs]:
                    max_r2[layer][obs] = val
    return max_r2


# ── Load all ────────────────────────────────────────────────────────────
print("Loading data...")
all_data = {}
for mult in MULT_LIST:
    steps, eval_results, train_losses, test_losses = load_data(mult)
    all_data[mult] = {
        'steps': steps,
        'r2_evo': extract_r2_evolution(eval_results, steps),
        'max_r2': extract_max_r2(eval_results),
        'train_loss': train_losses,
        'test_loss': test_losses,
    }
    print(f"  mult={mult}: {len(steps)} probe checkpoints, "
          f"train loss {train_losses[-1]:.6f}, test loss {test_losses[-1]:.6f}")

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. R² EVOLUTION PLOTS  (one figure per observable)
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating R² evolution plots...")

for obs in OBS_KEYS:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for ax_idx, layer in enumerate(LAYER_ORDER):
        ax = axes[ax_idx]
        for mult in MULT_LIST:
            r2_vals = all_data[mult]['r2_evo'][layer][obs]
            s = all_data[mult]['steps']
            ax.plot(s, r2_vals, color=COLORS[mult], marker=MARKERS[mult],
                    markersize=2, markevery=len(s)//8,
                    label=f'mult={mult}', linewidth=1.4)

        ax.set_title(layer, fontsize=9)
        ax.set_xlabel('Step')
        ax.set_ylabel('R²')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'{OBS_LABELS.get(obs, obs)} — Eval R² Evolution During Training',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'evolution_{obs}.png')
    plt.close(fig)
    print(f"  evolution_{obs}.png")

# ══════════════════════════════════════════════════════════════════════════
# 2. GLOBAL MAX R² COMPARISON  (layers on x-axis, line chart)
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating global max R² comparison...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()
x = np.arange(len(LAYER_ORDER))

for ax_idx, obs in enumerate(OBS_KEYS):
    ax = axes[ax_idx]
    for mult in MULT_LIST:
        vals = [all_data[mult]['max_r2'][layer][obs] for layer in LAYER_ORDER]
        ax.plot(x, vals, color=COLORS[mult], marker=MARKERS[mult],
                markersize=9, linewidth=2.0,
                label=f'mult={mult}', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_LABELS, fontsize=8)
    ax.set_ylabel('Max eval R² (seq-last)')
    ax.set_title(OBS_LABELS.get(obs, obs), fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Global Maximum R² per Layer — MLP Width Comparison',
             fontsize=15, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT_DIR / 'max_r2_comparison.png')
plt.close(fig)
print("  max_r2_comparison.png")

# ══════════════════════════════════════════════════════════════════════════
# 3. LOSS CURVES
# ══════════════════════════════════════════════════════════════════════════
print("\nGenerating loss curves...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

for mult in MULT_LIST:
    train_loss = all_data[mult]['train_loss']
    test_loss = all_data[mult]['test_loss']
    steps = np.arange(1, len(train_loss) + 1)

    ax1.plot(steps, train_loss, color=COLORS[mult], label=f'mult={mult}',
             linewidth=1.4, alpha=0.8)
    ax2.plot(steps, test_loss, color=COLORS[mult], label=f'mult={mult}',
             linewidth=1.4, alpha=0.8)

ax1.set_title('Train Loss', fontsize=13, fontweight='bold')
ax1.set_xlabel('Step'); ax1.set_ylabel('MSE Loss')
ax1.legend(fontsize=9); ax1.set_yscale('log')
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2.set_title('Test Loss', fontsize=13, fontweight='bold')
ax2.set_xlabel('Step'); ax2.set_ylabel('MSE Loss')
ax2.legend(fontsize=9); ax2.set_yscale('log')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

fig.suptitle('Loss Curves — MLP Width Comparison', fontsize=15, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT_DIR / 'loss_curves.png')
plt.close(fig)
print("  loss_curves.png")

# ── Summary table ───────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("Final R² Summary (after_ln_f, seq-last eval, step 2001)")
print(f"{'='*80}")
header = f"{'Obs':<16}"
for m in MULT_LIST:
    header += f" {'mult='+str(m):>14}"
print(header)
print('-' * (16 + 15 * len(MULT_LIST)))
for obs in OBS_KEYS:
    line = f"{obs:<16}"
    for m in MULT_LIST:
        val = all_data[m]['r2_evo']['after_ln_f'][obs][-1]
        line += f" {val:>14.4f}"
    print(line)

print(f"\nAll plots saved to {OUT_DIR.resolve()}/")
