"""
Weight condensation analysis for 1-layer attention+MLP model.

Measures how weight singular value spectra evolve during early training:
  - effective_rank (participation ratio): how many independent directions
  - max singular value: how strongly each weight "grows"
  - utilization: effective_rank / max_possible_rank

Usage:
    python analyze_condensation.py --npz results/block_X_noise_Y_loss_Z_seed_1.npz [--max_step 100]
"""

import argparse, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


KEY_WEIGHTS = {
    'transformer.h.0.attn.c_attn.weight': 'attn QKV (384x128)',
    'transformer.h.0.attn.c_proj.weight': 'attn c_proj (128x128)',
    'transformer.h.0.mlp.c_fc.weight': 'MLP c_fc (512x128)',
    'transformer.h.0.mlp.c_proj.weight': 'MLP c_proj (128x512)',
    'input_embedding.weight': 'input embed (128x2)',
    'output_head.weight': 'output head (2x128)',
}

MAX_RANKS = {
    'input_embedding.weight': 2,
    'output_head.weight': 2,
    'transformer.h.0.attn.c_attn.weight': 128,
    'transformer.h.0.attn.c_proj.weight': 128,
    'transformer.h.0.mlp.c_fc.weight': 128,
    'transformer.h.0.mlp.c_proj.weight': 128,
}

COLORS = ['#2166AC', '#C00000', '#008B45', '#B8860B', '#68228B', '#2F4F4F']

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.grid": True, "axes.axisbelow": True,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.8, "legend.fontsize": 8,
    "grid.linestyle": "--", "grid.alpha": 0.4,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def extract_weight_evolution(npz_path, max_step=2000):
    d = np.load(npz_path, allow_pickle=True)
    er = d['eval_results']
    data = {k: {'steps': [], 'eff_rank': [], 'max_sv': [], 'mean_sv': []}
            for k in KEY_WEIGHTS}
    for e in er:
        s = e['step']
        if s > max_step:
            break
        wsv = e.get('weight_singular_values', {})
        if not wsv:
            continue
        for k in KEY_WEIGHTS:
            rk = wsv.get(k, {})
            if rk:
                data[k]['steps'].append(s)
                data[k]['eff_rank'].append(rk.get('effective_rank', np.nan))
                data[k]['max_sv'].append(rk.get('max', np.nan))
                data[k]['mean_sv'].append(rk.get('mean', np.nan))
            else:
                for key2 in data[k]:
                    if key2 != 'steps':
                        data[k][key2].append(np.nan)
    return data


def plot_condensation(data, out_dir, max_step):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Effective rank ──
    fig, ax = plt.subplots(figsize=(12, 7))
    for (k, label), c in zip(KEY_WEIGHTS.items(), COLORS):
        ax.plot(data[k]['steps'], data[k]['eff_rank'],
                color=c, marker='o', markersize=4, label=label, linewidth=1.8)
    ax.set_xlabel('Training Step'); ax.set_ylabel('Effective Rank (participation ratio)')
    ax.set_title(f'Weight Condensation: Effective Rank (first {max_step} steps)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right'); ax.set_xlim(0, max_step)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / 'weight_eff_rank.png'); plt.close(fig)

    # ── 2. Max singular value ──
    fig, ax = plt.subplots(figsize=(12, 7))
    for (k, label), c in zip(KEY_WEIGHTS.items(), COLORS):
        ax.plot(data[k]['steps'], data[k]['max_sv'],
                color=c, marker='o', markersize=4, label=label, linewidth=1.8)
    ax.set_xlabel('Training Step'); ax.set_ylabel('Max Singular Value')
    ax.set_title(f'Weight Growth: Max Singular Value (first {max_step} steps)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left'); ax.set_xlim(0, max_step)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / 'weight_max_sv.png'); plt.close(fig)

    # ── 3. Utilization (eff_rank / max_rank) ──
    fig, ax = plt.subplots(figsize=(12, 7))
    for (k, label), c in zip(KEY_WEIGHTS.items(), COLORS):
        max_r = MAX_RANKS.get(k, 128)
        util = np.array(data[k]['eff_rank']) / max_r
        ax.plot(data[k]['steps'], util,
                color=c, marker='o', markersize=4, label=label, linewidth=1.8)
    ax.set_xlabel('Training Step'); ax.set_ylabel('Utilization (eff_rank / max_rank)')
    ax.set_title(f'Weight Utilization (first {max_step} steps)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right'); ax.set_xlim(0, max_step)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / 'weight_utilization.png'); plt.close(fig)

    # ── Summary ──
    print(f"\n=== Weight condensation summary (step {max_step}) ===")
    print(f"{'Weight':<35} {'eff_rank':>10} {'max_rank':>10} {'util%':>8} {'max_sv':>10}")
    print("-" * 75)
    for k, label in KEY_WEIGHTS.items():
        er = data[k]['eff_rank'][-1]
        mx = data[k]['max_sv'][-1]
        mr = MAX_RANKS.get(k, 128)
        print(f"{label:<35} {er:>10.1f} {mr:>10} {er/mr*100:>7.1f}% {mx:>10.3f}")

    print(f"\nPlots saved to {out_dir.resolve()}/")
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, required=True,
                        help='Path to result .npz file')
    parser.add_argument('--max_step', type=int, default=100,
                        help='Only analyze first N steps (default: 100)')
    parser.add_argument('--out_dir', type=str, default='plots_condensation',
                        help='Output directory for plots')
    args = parser.parse_args()

    data = extract_weight_evolution(args.npz, args.max_step)
    plot_condensation(data, args.out_dir, args.max_step)
