"""
Visual analysis of the no-LayerNorm runs against the with-LayerNorm baseline.

Compares three settings on block_size=100, full-batch GD:
  _baseline_with_ln  GPTCV      lr=0.001   (from results_gd_sweep)
  noln_bs100_lr001   GPTCVNoLN  lr=0.001
  noln_bs100_lr01    GPTCVNoLN  lr=0.01

Each setting has its own folder under results_729_noln/. Plots go to
plots_729_noln/, one subfolder per setting plus a cross-setting comparison.

Style follows plotting.py (serif, inward ticks, dashed grid) and the effective-rank
extraction follows analyze_condensation.py.

Usage:
    /opt/anaconda3/bin/python analyze_noln.py
    /opt/anaconda3/bin/python analyze_noln.py --results_dir results_729_noln --out_dir plots_729_noln
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# reuse the existing probe-R2 machinery rather than reimplementing it
from plotting import (
    load_result_npz,
    extract_probe_r2_from_npz,
    plot_layer_r2,
)

# force-related linear-probe targets (probe.py compute_gravitational_force)
FORCE_TARGETS = ('Fx', 'Fy', 'F_magnitude', 'F_direction_x', 'F_direction_y')


# ── settings: folder -> (label, colour, linestyle) ──
SETTINGS = {
    '_baseline_with_ln': ('with LN, lr=1e-3', '#08306B', '-'),
    'noln_bs100_lr001':  ('no LN,   lr=1e-3', '#C44E52', '-'),
    'noln_bs100_lr01':   ('no LN,   lr=1e-2', '#E67E22', '--'),
}

LAYERS = [
    'input_embed', 'after_pos_emb', 'block_0_attn_output',
    'block_0_after_attn_merge', 'block_0_mlp_hidden',
    'block_0_mlp_output', 'block_0_after_mlp_merge', 'after_ln_f',
]

LAYER_COLORS = [
    "#08306B", "#2166AC", "#5B9BD5", "#87CEEB",
    "#8C2D2D", "#C44E52", "#E67E22", "#6A51A3",
]

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


def load_run(folder):
    """Load one setting's npz and pull out the series we plot."""
    npzs = sorted(Path(folder).glob('*.npz'))
    if not npzs:
        return None
    d = np.load(npzs[0], allow_pickle=True)
    er = d['eval_results']

    out = {
        'losses': np.asarray(d['train_losses'], dtype=float),
        'test_losses': np.asarray(d['test_losses'], dtype=float)
                       if 'test_losses' in d else None,
        'rank_steps': [], 'rank': {L: [] for L in LAYERS},
        'numrank': {L: [] for L in LAYERS},
        'sv_steps': [], 'E2': {L: [] for L in LAYERS},
        'ent_steps': [], 'entropy': [],
    }

    for e in er:
        if not isinstance(e, dict):
            continue
        step = e.get('step')

        ar = e.get('activation_rank')
        if isinstance(ar, dict) and ar.get('train'):
            out['rank_steps'].append(step)
            for L in LAYERS:
                st = ar['train'].get(L, {})
                out['rank'][L].append(st.get('effective_rank', np.nan))
                out['numrank'][L].append(st.get('numerical_rank', np.nan))

        asv = e.get('activation_singular_values')
        if isinstance(asv, dict) and asv.get('train'):
            out['sv_steps'].append(step)
            for L in LAYERS:
                sv = np.asarray(asv['train'].get(L, {}).get('singular_values', []),
                                dtype=float)
                if sv.size >= 2:
                    energy = sv ** 2
                    out['E2'][L].append(energy[:2].sum() / energy.sum())
                else:
                    out['E2'][L].append(np.nan)

        ae = e.get('attention_entropy')
        if isinstance(ae, dict) and isinstance(ae.get('train'), dict):
            vals = [b.get('normalized_mean_entropy') for b in ae['train'].values()
                    if isinstance(b, dict)]
            if vals:
                out['ent_steps'].append(step)
                out['entropy'].append(float(np.nanmean(vals)))

    return out


# ─────────────────────── per-setting plots ───────────────────────

def plot_per_setting(name, run, out_dir):
    """All layers' effective rank for one setting, linear + log x."""
    label = SETTINGS[name][0]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = np.asarray(run['rank_steps'], dtype=float)
    if steps.size == 0:
        return

    for logx, fname in [(False, 'eff_rank.png'), (True, 'eff_rank_logx.png')]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for L, c in zip(LAYERS, LAYER_COLORS):
            ax.plot(steps, run['rank'][L], color=c, marker='o', markersize=2.6,
                    markevery=max(1, len(steps) // 12), label=L)
        if logx:
            ax.set_xscale('log')
            ax.set_xlim(max(1, steps[steps > 0].min()), steps.max())
        ax.set_xlabel('Training step')
        ax.set_ylabel('Effective rank (participation ratio)')
        ax.set_title(f'Activation effective rank — {label}')
        ax.legend(fontsize=7, ncol=2, loc='best')
        _style(ax)
        fig.tight_layout()
        fig.savefig(out_dir / fname)
        plt.close(fig)

    # effective rank vs numerical rank: does any direction die?
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for L, c in zip(LAYERS, LAYER_COLORS):
        axes[0].plot(steps, run['rank'][L], color=c, label=L)
        axes[1].plot(steps, run['numrank'][L], color=c, label=L)
    for ax, ttl, yl in [(axes[0], 'Effective rank (spectral entropy)', 'eff rank'),
                        (axes[1], 'Numerical rank (1e-6 tol)', 'num rank')]:
        ax.set_xscale('log')
        ax.set_xlim(max(1, steps[steps > 0].min()), steps.max())
        ax.set_xlabel('Training step')
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        _style(ax)
    axes[1].set_yscale('log')
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f'{label}: participation ratio falls while span stays fixed',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'eff_vs_numerical_rank.png')
    plt.close(fig)

    # loss
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.arange(1, run['losses'].size + 1), run['losses'],
            color=SETTINGS[name][1], label='train')
    if run['test_losses'] is not None and run['test_losses'].size:
        ax.plot(np.linspace(1, run['losses'].size, run['test_losses'].size),
                run['test_losses'], color='#4C956C', alpha=0.75, label='test')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel('MSE loss')
    ax.set_title(f'Loss — {label}')
    ax.legend()
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / 'loss.png')
    plt.close(fig)


# ─────────────────────── cross-setting comparison ───────────────────────

def plot_comparison(runs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. after_ln_f effective rank, all settings, log x (the headline plot)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, run in runs.items():
        lbl, c, ls = SETTINGS[name]
        s = np.asarray(run['rank_steps'], dtype=float)
        ax.plot(s, run['rank']['after_ln_f'], color=c, linestyle=ls,
                marker='o', markersize=3, markevery=max(1, len(s) // 14), label=lbl)
    ax.axhline(2.0, color='#555555', linestyle=':', linewidth=1.3)
    ax.text(1.3, 2.08, 'gradient rank = 2 (algebraic floor)',
            fontsize=7.5, color='#555555')
    ax.set_xscale('log')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Effective rank of after_ln_f')
    ax.set_title('LayerNorm decides whether the rank recovers')
    ax.legend(loc='best')
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / 'compare_eff_rank_after_ln_f.png')
    plt.close(fig)

    # 2. early-window zoom (the shallow V of the with-LN run)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for name, run in runs.items():
        lbl, c, ls = SETTINGS[name]
        s = np.asarray(run['rank_steps'], dtype=float)
        m = s <= 200
        ax.plot(s[m], np.asarray(run['rank']['after_ln_f'])[m], color=c,
                linestyle=ls, marker='o', markersize=3.4, label=lbl)
    ax.set_xlabel('Training step')
    ax.set_ylabel('Effective rank of after_ln_f')
    ax.set_title('First 200 steps: shallow dip (with LN) vs monotone collapse (no LN)')
    ax.legend()
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / 'compare_early_window.png')
    plt.close(fig)

    # 3. mechanism: top-2 energy fraction E2 anti-correlates with eff rank
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3), sharex=True)
    for name, run in runs.items():
        lbl, c, ls = SETTINGS[name]
        sr = np.asarray(run['rank_steps'], dtype=float)
        ss = np.asarray(run['sv_steps'], dtype=float)
        axes[0].plot(sr, run['rank']['after_ln_f'], color=c, linestyle=ls, label=lbl)
        axes[1].plot(ss, run['E2']['after_ln_f'], color=c, linestyle=ls, label=lbl)
    axes[0].set_ylabel('Effective rank')
    axes[0].set_title('Effective rank (after_ln_f)')
    axes[1].set_ylabel(r'$E_2$ = top-2 energy fraction')
    axes[1].set_title(r'Top-2 energy fraction $E_2$')
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xlabel('Training step')
        _style(ax)
    axes[0].legend(fontsize=7.5)
    fig.suptitle('Mechanism: rank falls exactly as the rank-2 signal takes over the spectrum',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'compare_mechanism_E2.png')
    plt.close(fig)

    # 4. per-layer small multiples across settings
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.6), sharex=True)
    for ax, L in zip(axes.ravel(), LAYERS):
        for name, run in runs.items():
            lbl, c, ls = SETTINGS[name]
            s = np.asarray(run['rank_steps'], dtype=float)
            ax.plot(s, run['rank'][L], color=c, linestyle=ls, label=lbl)
        ax.set_xscale('log')
        ax.set_title(L, fontsize=9)
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel('Training step')
    for ax in axes[:, 0]:
        ax.set_ylabel('Effective rank')
    axes[0, 0].legend(fontsize=6.8)
    fig.suptitle('Effective rank per layer, all three settings', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'compare_all_layers.png')
    plt.close(fig)

    # 5. loss + attention entropy
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for name, run in runs.items():
        lbl, c, ls = SETTINGS[name]
        axes[0].plot(np.arange(1, run['losses'].size + 1), run['losses'],
                     color=c, linestyle=ls, label=lbl)
        if run['ent_steps']:
            axes[1].plot(run['ent_steps'], run['entropy'], color=c, linestyle=ls,
                         marker='o', markersize=3, label=lbl)
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_xlabel('Training step'); axes[0].set_ylabel('Train MSE')
    axes[0].set_title('Training loss')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Normalized attention entropy')
    axes[1].set_title('Attention entropy (1.0 = uniform)')
    for ax in axes:
        _style(ax)
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out_dir / 'compare_loss_entropy.png')
    plt.close(fig)


# ─────────────────────── force probe R2 ───────────────────────

def plot_force_r2_per_setting(name, npz_path, out_dir):
    """Per-layer Fx/Fy R2 grid, via plotting.plot_layer_r2 (existing code)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = load_result_npz(npz_path)
    layers = extract_probe_r2_from_npz(result, targets=('Fx', 'Fy'))
    plot_layer_r2(layers, out_dir / 'probe_r2_Fx_Fy_by_layer.png')


def collect_force_r2(npz_path, metric='eval_r2_sequence_last'):
    """target -> layer -> (steps, r2) for every force-related probe target."""
    result = load_result_npz(npz_path)
    out = {t: {} for t in FORCE_TARGETS}
    for e in (result.get('eval_results') or []):
        if not isinstance(e, dict):
            continue
        pr = e.get('probe_results')
        if not pr:
            continue
        step = e.get('step')
        for layer, lp in pr.items():
            if not isinstance(lp, dict):
                continue
            for t in FORCE_TARGETS:
                m = lp.get(t)
                if isinstance(m, dict) and metric in m:
                    b = out[t].setdefault(layer, {'steps': [], 'r2': []})
                    b['steps'].append(int(step))
                    b['r2'].append(float(m[metric]))
    return out


def plot_force_r2_targets(name, npz_path, out_dir):
    """One panel per force target, all layers overlaid, for a single setting."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = collect_force_r2(npz_path)
    label = SETTINGS[name][0]

    fig, axes = plt.subplots(1, len(FORCE_TARGETS),
                             figsize=(4.0 * len(FORCE_TARGETS), 4.0), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), FORCE_TARGETS):
        for L, c in zip(LAYERS, LAYER_COLORS):
            b = data[t].get(L)
            if not b:
                continue
            ax.plot(b['steps'], b['r2'], color=c, label=L, linewidth=1.5)
        ax.set_xscale('log')
        ax.set_xlabel('Training step')
        ax.set_title(t, fontsize=10)
        ax.set_ylim(-0.05, 1.02)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel(r'Probe $R^2$ (eval, last position)')
    np.atleast_1d(axes)[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle(f'Force representation learning — {label}', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'probe_r2_force_targets.png')
    plt.close(fig)


def plot_force_r2_comparison(runs_npz, out_dir):
    """Cross-setting: rows = force targets, cols = key layers."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {n: collect_force_r2(p) for n, p in runs_npz.items()}
    key_layers = ['after_pos_emb', 'block_0_attn_output',
                  'block_0_mlp_output', 'after_ln_f']

    fig, axes = plt.subplots(len(FORCE_TARGETS), len(key_layers),
                             figsize=(3.7 * len(key_layers), 3.1 * len(FORCE_TARGETS)),
                             sharex=True, sharey=True, squeeze=False)
    for i, t in enumerate(FORCE_TARGETS):
        for j, L in enumerate(key_layers):
            ax = axes[i][j]
            for n in runs_npz:
                lbl, c, ls = SETTINGS[n]
                b = data[n][t].get(L)
                if not b:
                    continue
                ax.plot(b['steps'], b['r2'], color=c, linestyle=ls,
                        label=lbl, linewidth=1.5)
            ax.set_xscale('log')
            ax.set_ylim(-0.05, 1.02)
            if i == 0:
                ax.set_title(L, fontsize=9)
            if j == 0:
                ax.set_ylabel(f'{t}\n' + r'$R^2$', fontsize=8.5)
            if i == len(FORCE_TARGETS) - 1:
                ax.set_xlabel('Training step')
            _style(ax)
    axes[0][0].legend(fontsize=6.8)
    fig.suptitle(r'Force probe $R^2$ (eval, last position) across settings',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'compare_probe_r2_force.png')
    plt.close(fig)

    # headline: best-over-layers R2 per target, per setting
    fig, axes = plt.subplots(1, len(FORCE_TARGETS),
                            figsize=(4.0 * len(FORCE_TARGETS), 4.0), sharey=True)
    for ax, t in zip(np.atleast_1d(axes), FORCE_TARGETS):
        for n in runs_npz:
            lbl, c, ls = SETTINGS[n]
            per_layer = data[n][t]
            if not per_layer:
                continue
            steps = sorted({s for b in per_layer.values() for s in b['steps']})
            best = []
            for s in steps:
                vals = [b['r2'][b['steps'].index(s)]
                        for b in per_layer.values() if s in b['steps']]
                best.append(max(vals) if vals else np.nan)
            ax.plot(steps, best, color=c, linestyle=ls, label=lbl, linewidth=1.7)
        ax.set_xscale('log')
        ax.set_xlabel('Training step')
        ax.set_title(t, fontsize=10)
        ax.set_ylim(-0.05, 1.02)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel(r'Best-layer probe $R^2$')
    np.atleast_1d(axes)[0].legend(fontsize=7.5)
    fig.suptitle('Best-over-layers force probe accuracy', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_dir / 'compare_probe_r2_best_layer.png')
    plt.close(fig)
    return data


def _short_layer(name):
    """Compact layer label for table output."""
    return (name.replace('block_0_', 'b0.')
                .replace('after_', 'aft_')
                .replace('_output', '_out')
                .replace('_embed', '_emb'))


def print_force_summary(data):
    print("\n" + "=" * 92)
    print("FORCE PROBE R2 (eval, last position) — final step, best layer")
    print("=" * 92)
    hdr = f"{'target':16s}" + ''.join(f"{SETTINGS[n][0][:16]:>25s}" for n in data)
    print(hdr)
    for t in FORCE_TARGETS:
        row = f"{t:16s}"
        for n in data:
            per_layer = data[n][t]
            if not per_layer:
                row += f"{'n/a':>25s}"
                continue
            best, where = -9e9, ''
            for L, b in per_layer.items():
                if b['r2'] and b['r2'][-1] > best:
                    best, where = b['r2'][-1], L
            row += f"{best:10.3f}  {_short_layer(where):<13s}"
        print(row)


def print_summary(runs):
    print("\n" + "=" * 88)
    print("SUMMARY — after_ln_f effective rank")
    print("=" * 88)
    print(f"{'setting':22s} {'start':>7} {'min':>7} {'@step':>7} {'drop%':>7} "
          f"{'end':>7} {'num_rank':>10} {'final loss':>11}")
    for name, run in runs.items():
        v = np.asarray(run['rank']['after_ln_f'], dtype=float)
        s = np.asarray(run['rank_steps'], dtype=float)
        if v.size == 0:
            continue
        i = int(np.nanargmin(v))
        nr = set(np.asarray(run['numrank']['after_ln_f']).tolist())
        print(f"{SETTINGS[name][0]:22s} {v[0]:7.2f} {v[i]:7.2f} {int(s[i]):7d} "
              f"{(v[0]-v[i])/v[0]*100:7.1f} {v[-1]:7.2f} "
              f"{('const ' + str(int(list(nr)[0]))) if len(nr)==1 else 'VARIES':>10s} "
              f"{run['losses'][-1]:11.6f}")

    print("\nPer-layer final effective rank")
    hdr = f"{'layer':26s}" + ''.join(f"{SETTINGS[n][0][:14]:>16s}" for n in runs)
    print(hdr)
    for L in LAYERS:
        row = f"{L:26s}"
        for name, run in runs.items():
            v = run['rank'][L]
            row += f"{(v[-1] if len(v) else float('nan')):16.3f}"
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results_729_noln')
    ap.add_argument('--out_dir', default='plots_729_noln')
    args = ap.parse_args()

    root = Path(args.results_dir)
    runs = {}
    runs_npz = {}
    for name in SETTINGS:
        run = load_run(root / name)
        if run is None:
            print(f"  [skip] {name}: no npz found in {root / name}")
            continue
        runs[name] = run
        npz = sorted((root / name).glob('*.npz'))[0]
        runs_npz[name] = npz
        setting_out = Path(args.out_dir) / name
        plot_per_setting(name, run, setting_out)
        plot_force_r2_per_setting(name, npz, setting_out)
        plot_force_r2_targets(name, npz, setting_out)
        print(f"  [ok]   {name}: {len(run['rank_steps'])} rank samples, "
              f"{run['losses'].size} steps")

    if len(runs) >= 2:
        cmp_out = Path(args.out_dir) / '_comparison'
        plot_comparison(runs, cmp_out)
        fdata = plot_force_r2_comparison(runs_npz, cmp_out)
        print_summary(runs)
        print_force_summary(fdata)
    else:
        print_summary(runs)
    print(f"\nPlots written to {Path(args.out_dir).resolve()}/")


if __name__ == '__main__':
    main()
