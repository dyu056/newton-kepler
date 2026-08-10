"""
Visualization and RG-prediction check for the pure-attention run.

Two halves:

  PART 1  descriptive plots of the training run (loss, effective rank, attention
          entropy, per-layer ranks, force-probe R2)

  PART 2  the RG quantities the theory in
          r2_research/rg_kepler_testbed/RG_predictions_kepler_testbed.md predicts,
          measured over training rather than only at step 0:

            F        = the attention "propagator" QK^T structure
            sigma_i  = singular values of F
            rank     Conclusion 1: sigma_3/sigma_2 -> 0 (rank-2 condensation)
            cos^2 t  Conclusion 2: alignment of F with D's top eigenvector -> 1
            sigma2/1 Conclusion 3: two-mode separation, strongly d-dependent

Style follows plotting.py.  Usage:
    /opt/anaconda3/bin/python analyze_pure_attn.py
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAYERS = ['input_embed', 'after_pos_emb', 'block_0_attn_output', 'after_ln_f']
LAYER_COLORS = ["#08306B", "#2166AC", "#C44E52", "#E67E22"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.titleweight": "bold", "axes.grid": True, "axes.axisbelow": True,
    "axes.unicode_minus": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.7, "legend.fontsize": 8,
    "grid.linestyle": "--", "grid.alpha": 0.28,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="in")
    ax.grid(True, linestyle="--", alpha=0.28)


def load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    er = d['eval_results']
    out = {
        'meta': {k: d[k] for k in ('arch', 'block_size', 'lr', 'n_embd',
                                   'loss_mask', 'sigma_init') if k in d},
        'losses': np.asarray(d['train_losses'], dtype=float),
        'test_losses': (np.asarray(d['test_losses'], dtype=float)
                        if 'test_losses' in d else None),
        'steps': [], 'rank': {L: [] for L in LAYERS},
        'numrank': {L: [] for L in LAYERS},
        'ent_steps': [], 'entropy': [],
        'probe_steps': [], 'probe': {},
        'wsv_steps': [], 'wsv': {},
    }
    for e in er:
        if not isinstance(e, dict):
            continue
        s = e.get('step')

        ar = e.get('activation_rank')
        if isinstance(ar, dict) and ar.get('train'):
            out['steps'].append(s)
            for L in LAYERS:
                st = ar['train'].get(L, {})
                out['rank'][L].append(st.get('effective_rank', np.nan))
                out['numrank'][L].append(st.get('numerical_rank', np.nan))

        ae = e.get('attention_entropy')
        if isinstance(ae, dict) and isinstance(ae.get('train'), dict):
            vals = [b.get('normalized_mean_entropy') for b in ae['train'].values()
                    if isinstance(b, dict)]
            if vals:
                out['ent_steps'].append(s)
                out['entropy'].append(float(np.mean(vals)))

        pr = e.get('probe_results')
        if pr:
            out['probe_steps'].append(s)
            for layer, lp in pr.items():
                if not isinstance(lp, dict):
                    continue
                for t, m in lp.items():
                    if isinstance(m, dict) and 'eval_r2_sequence_last' in m:
                        out['probe'].setdefault((t, layer), {'steps': [], 'r2': []})
                        out['probe'][(t, layer)]['steps'].append(s)
                        out['probe'][(t, layer)]['r2'].append(
                            float(m['eval_r2_sequence_last']))

        w = e.get('weight_singular_values')
        if isinstance(w, dict):
            out['wsv_steps'].append(s)
            for name, st in w.items():
                if isinstance(st, dict):
                    out['wsv'].setdefault(name, {'steps': [], 'sv': []})
                    out['wsv'][name]['steps'].append(s)
                    out['wsv'][name]['sv'].append(
                        np.asarray(st.get('singular_values', []), dtype=float))
    return out


# ─────────────────────── PART 1: descriptive ───────────────────────

def plot_overview(r, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    s = np.asarray(r['steps'], dtype=float)
    es = np.asarray(r['ent_steps'], dtype=float)
    ent = np.asarray(r['entropy'], dtype=float)
    rank = np.asarray(r['rank']['after_ln_f'], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))

    axes[0].plot(np.arange(1, r['losses'].size + 1), r['losses'],
                 color='#08306B', label='train')
    if r['test_losses'] is not None and r['test_losses'].size:
        axes[0].plot(np.linspace(1, r['losses'].size, r['test_losses'].size),
                     r['test_losses'], color='#4C956C', alpha=.8, label='test')
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('step'); axes[0].set_ylabel('MSE')
    axes[0].set_title('Loss'); axes[0].legend()

    i = int(np.nanargmin(rank))
    axes[1].plot(s, rank, color='#C44E52', marker='o', markersize=3,
                 markevery=max(1, len(s) // 14))
    axes[1].axvline(s[i], color='#888', ls=':', lw=1.2)
    axes[1].annotate(f'min {rank[i]:.3f}\n@{int(s[i])}', (s[i], rank[i]),
                     textcoords='offset points', xytext=(12, 10), fontsize=8)
    axes[1].set_xscale('log'); axes[1].set_xlabel('step')
    axes[1].set_ylabel('effective rank')
    axes[1].set_title('Effective rank (after_ln_f)')

    j = int(np.nanargmin(ent))
    axes[2].plot(es, ent, color='#6A51A3', marker='o', markersize=3,
                 markevery=max(1, len(es) // 14))
    axes[2].axvline(es[j], color='#888', ls=':', lw=1.2)
    axes[2].annotate(f'focus->dilution\nmin {ent[j]:.4f} @{int(es[j])}',
                     (es[j], ent[j]), textcoords='offset points',
                     xytext=(10, 14), fontsize=8)
    axes[2].set_xscale('log'); axes[2].set_xlabel('step')
    axes[2].set_ylabel('normalized attention entropy')
    axes[2].set_title('Attention entropy (1.0 = uniform)')

    for ax in axes:
        _style(ax)
    fig.suptitle('Pure attention (no MLP / no residual / no LN), '
                 f"d={int(r['meta']['n_embd'])}, T={int(r['meta']['block_size'])}, "
                 f"lr={float(r['meta']['lr'])}", fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'overview.png'); plt.close(fig)

    # per-layer ranks + numerical rank
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for L, c in zip(LAYERS, LAYER_COLORS):
        axes[0].plot(s, r['rank'][L], color=c, label=L)
        axes[1].plot(s, r['numrank'][L], color=c, label=L)
    axes[0].set_ylabel('effective rank'); axes[0].set_title('Effective rank per layer')
    axes[1].set_ylabel('numerical rank'); axes[1].set_title('Numerical rank (span)')
    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel('step'); _style(ax)
    axes[0].legend(fontsize=7)
    fig.suptitle('Participation ratio moves; span does not', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'ranks_per_layer.png'); plt.close(fig)


def plot_force_probe(r, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    targets = ('Fx', 'Fy', 'F_magnitude', 'F_direction_x', 'F_direction_y')
    fig, axes = plt.subplots(1, len(targets), figsize=(4.0 * len(targets), 4.0),
                             sharey=True)
    for ax, t in zip(np.atleast_1d(axes), targets):
        for L, c in zip(LAYERS, LAYER_COLORS):
            b = r['probe'].get((t, L))
            if b:
                ax.plot(b['steps'], b['r2'], color=c, label=L)
        ax.set_xscale('log'); ax.set_xlabel('step')
        ax.set_title(t, fontsize=10); ax.set_ylim(-0.05, 1.02); _style(ax)
    np.atleast_1d(axes)[0].set_ylabel(r'probe $R^2$ (eval, last)')
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle('Force representation learning (pure attention)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'probe_r2_force.png'); plt.close(fig)


def print_summary(r):
    s = np.asarray(r['steps'], dtype=float)
    rank = np.asarray(r['rank']['after_ln_f'], dtype=float)
    es = np.asarray(r['ent_steps'], dtype=float)
    ent = np.asarray(r['entropy'], dtype=float)
    i, j = int(np.nanargmin(rank)), int(np.nanargmin(ent))

    print("\n" + "=" * 80)
    print("PART 1 — descriptive summary")
    print("=" * 80)
    m = r['meta']
    print(f"arch={m['arch']}  d={int(m['n_embd'])}  T={int(m['block_size'])}  "
          f"lr={float(m['lr'])}  loss_mask={m['loss_mask']}")
    print(f"loss        {r['losses'][0]:.5f} -> {r['losses'][-1]:.6f}")
    print(f"eff rank    {rank[0]:.3f} -> min {rank[i]:.3f} @step {int(s[i])} "
          f"-> {rank[-1]:.3f}   (dip {100*(rank[0]-rank[i])/rank[0]:.1f}%)")
    print(f"attn entropy {ent[0]:.4f} -> min {ent[j]:.4f} @step {int(es[j])} "
          f"-> {ent[-1]:.4f}   (rebound +{ent[-1]-ent[j]:.4f})")
    nr = {int(v) for v in r['numrank']['after_ln_f']}
    print(f"numerical rank: {nr}  ({'constant' if len(nr) == 1 else 'VARIES'})")
    print(f"\nrank dip bottoms at step {int(s[i])}, entropy bottoms at step "
          f"{int(es[j])} -> the two turning points are {int(es[j]) - int(s[i])} "
          f"steps apart")

    print("\nfinal force-probe R2 (best layer):")
    for t in ('Fx', 'Fy', 'F_magnitude', 'F_direction_x', 'F_direction_y'):
        best, where = -9e9, ''
        for (tt, L), b in r['probe'].items():
            if tt == t and b['r2'] and b['r2'][-1] > best:
                best, where = b['r2'][-1], L
        if where:
            print(f"  {t:16s} {best:7.3f}   ({where})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results_803_pure_attn')
    ap.add_argument('--out_dir', default='plots_803_pure_attn')
    args = ap.parse_args()

    root = Path(args.results_dir)
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        npzs = sorted(sub.glob('*.npz'))
        if not npzs:
            continue
        r = load(npzs[0])
        out = Path(args.out_dir) / sub.name
        plot_overview(r, out)
        plot_force_probe(r, out)
        print(f"  [ok] {sub.name}")
        print_summary(r)
    print(f"\nPlots -> {Path(args.out_dir).resolve()}/")


if __name__ == '__main__':
    main()
