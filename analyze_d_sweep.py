"""
Analysis and visualization of the d sweep (pure-attention model, 30000 steps).

Layout: one output folder per d, plus a _comparison folder.

  plots_803_d_sweep/d32/ ... d256/     per-setting plots
  plots_803_d_sweep/_comparison/       cross-d plots + RG prediction checks

Reads:
  results_803_d_sweep/d{32,64,128,256}/*.npz          training observables
  results_803_d_sweep/rg_tracking/rg_track_d*_T50.json RG quantities

Style follows plotting.py.  Usage:
    /opt/anaconda3/bin/python analyze_d_sweep.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D_LIST = [32, 64, 128, 256]
D_COLORS = {32: "#08306B", 64: "#2166AC", 128: "#C44E52", 256: "#E67E22"}
LAYERS = ['input_embed', 'after_pos_emb', 'block_0_attn_output', 'after_ln_f']
LAYER_COLORS = ["#08306B", "#2166AC", "#C44E52", "#E67E22"]
FORCE_TARGETS = ('Fx', 'Fy', 'F_magnitude', 'F_direction_x', 'F_direction_y')

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.titleweight": "bold", "axes.grid": True, "axes.axisbelow": True,
    "axes.unicode_minus": False, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.7, "legend.fontsize": 8,
    "grid.linestyle": "--", "grid.alpha": 0.28,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="in")
    ax.grid(True, linestyle="--", alpha=0.28)


def load_train(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    er = d['eval_results']
    out = {
        'meta': {k: d[k] for k in ('arch', 'block_size', 'lr', 'n_embd',
                                   'loss_mask', 'sigma_init', 'status',
                                   'completed_steps') if k in d},
        'losses': np.asarray(d['train_losses'], dtype=float),
        'test_losses': (np.asarray(d['test_losses'], dtype=float)
                        if 'test_losses' in d else None),
        'final_test': float(d['final_test_loss']) if 'final_test_loss' in d else np.nan,
        'steps': [], 'rank': {L: [] for L in LAYERS},
        'numrank': {L: [] for L in LAYERS},
        'ent_steps': [], 'entropy': [], 'probe': {},
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
            for layer, lp in pr.items():
                if not isinstance(lp, dict):
                    continue
                for t, m in lp.items():
                    if isinstance(m, dict) and 'eval_r2_sequence_last' in m:
                        b = out['probe'].setdefault((t, layer), {'steps': [], 'r2': []})
                        b['steps'].append(s)
                        b['r2'].append(float(m['eval_r2_sequence_last']))
    return out


def load_rg(json_path):
    r = json.load(open(json_path))
    return {
        'step': np.array([x['step'] for x in r], dtype=float),
        'loss': np.array([x['loss'] for x in r]),
        'sv': np.array([x['sv'] for x in r]),
        's2_s1': np.array([x['s2_s1'] for x in r]),
        's3_s2': np.array([x['s3_s2'] for x in r]),
        'mu1': np.array([x['mu1'] for x in r]),
        'mu2': np.array([x['mu2_2'] for x in r]),
        'cos2': np.array([x['cos2'] for x in r]),
        'rkG': np.array([x['Gamma_rank'] for x in r]),
        'F_norm': np.array([x['F_norm'] for x in r]),
    }


# ─────────────────────── per-d plots ───────────────────────

def plot_one_d(d, tr, rg, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    s = np.asarray(tr['steps'], dtype=float)
    es = np.asarray(tr['ent_steps'], dtype=float)
    ent = np.asarray(tr['entropy'], dtype=float)
    rank = np.asarray(tr['rank']['after_ln_f'], dtype=float)
    c = D_COLORS[d]

    # 1. overview: loss / effective rank / entropy
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    ax[0].plot(np.arange(1, tr['losses'].size + 1), tr['losses'], color=c, label='train')
    if tr['test_losses'] is not None and tr['test_losses'].size:
        ax[0].plot(np.linspace(1, tr['losses'].size, tr['test_losses'].size),
                   tr['test_losses'], color='#4C956C', alpha=.8, label='test')
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_ylabel('MSE')
    ax[0].set_title('Loss'); ax[0].legend()

    i = int(np.nanargmin(rank))
    ax[1].plot(s, rank, color=c, marker='o', ms=3, markevery=max(1, len(s)//14))
    ax[1].axvline(s[i], color='#888', ls=':', lw=1.2)
    ax[1].annotate(f'min {rank[i]:.3f}\n@{int(s[i])}', (s[i], rank[i]),
                   textcoords='offset points', xytext=(12, 10), fontsize=8)
    ax[1].set_xscale('log'); ax[1].set_ylabel('effective rank')
    ax[1].set_title('Effective rank (after_ln_f)')

    j = int(np.nanargmin(ent))
    ax[2].plot(es, ent, color=c, marker='o', ms=3, markevery=max(1, len(es)//14))
    ax[2].axvline(es[j], color='#888', ls=':', lw=1.2)
    ax[2].annotate(f'min {ent[j]:.4f} @{int(es[j])}\nrebound {ent[-1]-ent[j]:+.4f}',
                   (es[j], ent[j]), textcoords='offset points',
                   xytext=(10, 14), fontsize=8)
    ax[2].set_xscale('log'); ax[2].set_ylabel('normalized attention entropy')
    ax[2].set_title('Attention entropy (1.0 = uniform)')
    for a in ax:
        a.set_xlabel('step'); _style(a)
    fig.suptitle(f"d={d}, T=50, 30000 steps, "
                 f"sigma_init={(0.1/(3*d))**0.25:.4f}", fontweight='bold')
    fig.tight_layout(); fig.savefig(out_dir / 'overview.png'); plt.close(fig)

    # 2. per-layer ranks
    fig, ax = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for L, lc in zip(LAYERS, LAYER_COLORS):
        ax[0].plot(s, tr['rank'][L], color=lc, label=L)
        ax[1].plot(s, tr['numrank'][L], color=lc, label=L)
    ax[0].set_ylabel('effective rank'); ax[0].set_title('Effective rank per layer')
    ax[1].set_ylabel('numerical rank'); ax[1].set_title('Numerical rank (span)')
    for a in ax:
        a.set_xscale('log'); a.set_xlabel('step'); _style(a)
    ax[0].legend(fontsize=7)
    fig.suptitle(f'd={d}: per-layer ranks', fontweight='bold')
    fig.tight_layout(); fig.savefig(out_dir / 'ranks_per_layer.png'); plt.close(fig)

    # 3. force probe
    fig, ax = plt.subplots(1, len(FORCE_TARGETS),
                           figsize=(4.0*len(FORCE_TARGETS), 4.0), sharey=True)
    for a, t in zip(np.atleast_1d(ax), FORCE_TARGETS):
        for L, lc in zip(LAYERS, LAYER_COLORS):
            b = tr['probe'].get((t, L))
            if b:
                a.plot(b['steps'], b['r2'], color=lc, label=L)
        a.set_xscale('log'); a.set_xlabel('step'); a.set_title(t, fontsize=10)
        a.set_ylim(-0.05, 1.02); _style(a)
    np.atleast_1d(ax)[0].set_ylabel(r'probe $R^2$ (eval, last)')
    np.atleast_1d(ax)[0].legend(fontsize=7)
    fig.suptitle(f'd={d}: force probe', fontweight='bold')
    fig.tight_layout(); fig.savefig(out_dir / 'probe_r2_force.png'); plt.close(fig)

    # 4. RG quantities for this d
    if rg is None:
        return
    st = rg['step'].copy(); st[st == 0] = 0.5
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.4))
    for k in range(min(5, rg['sv'].shape[1])):
        ax[0, 0].plot(st, rg['sv'][:, k], color=LAYER_COLORS[k % 4],
                      label=f'$\\sigma_{k+1}$')
    ax[0, 0].set_yscale('log'); ax[0, 0].set_title('F singular values'); ax[0, 0].legend()
    ax[0, 1].plot(st, rg['s3_s2'], color='#C44E52', marker='o', ms=3)
    ax[0, 1].axhline(0, color='#888', ls=':'); ax[0, 1].set_ylim(0, .9)
    ax[0, 1].set_title(r'C1: $\sigma_3/\sigma_2$')
    ax[0, 2].plot(st, rg['s2_s1'], color='#E67E22', marker='o', ms=3)
    ax[0, 2].set_ylim(0, 1); ax[0, 2].set_title(r'C3: $\sigma_2/\sigma_1$')
    ax[1, 0].plot(st, rg['cos2'], color='#6A51A3', marker='o', ms=3)
    ax[1, 0].set_yscale('log'); ax[1, 0].axhline(1, color='#888', ls=':')
    ax[1, 0].set_title(r'C2: $\cos^2\theta$')
    ax[1, 1].plot(st, rg['mu1'], color='#08306B', marker='o', ms=3, label=r'$\mu_1$')
    ax[1, 1].plot(st, rg['mu2'], color='#C44E52', marker='o', ms=3, label=r'$\mu_2$')
    ax[1, 1].axhline(1, color='#888', ls=':'); ax[1, 1].set_yscale('log')
    ax[1, 1].set_title(r'$\mu$ vs threshold 1'); ax[1, 1].legend()
    ax[1, 2].plot(st, rg['mu1']/np.maximum(rg['mu2'], 1e-30),
                  color='#4C956C', marker='o', ms=3)
    ax[1, 2].axhline(1, color='#888', ls=':')
    ax[1, 2].set_title(r'C4: $\mu_1/\mu_2$')
    for a in ax.ravel():
        a.set_xscale('log'); a.set_xlabel('step'); _style(a)
    fig.suptitle(f'd={d}: RG quantities', fontweight='bold')
    fig.tight_layout(); fig.savefig(out_dir / 'rg_quantities.png'); plt.close(fig)


# ─────────────────────── cross-d plots ───────────────────────

def plot_comparison(trs, rgs, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1. training observables across d
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for d, tr in trs.items():
        c = D_COLORS[d]
        ax[0].plot(np.arange(1, tr['losses'].size + 1), tr['losses'],
                   color=c, label=f'd={d}')
        s = np.asarray(tr['steps'], dtype=float)
        ax[1].plot(s, tr['rank']['after_ln_f'], color=c, label=f'd={d}')
        ax[2].plot(tr['ent_steps'], tr['entropy'], color=c, label=f'd={d}')
    ax[0].set_yscale('log'); ax[0].set_ylabel('train MSE'); ax[0].set_title('Loss')
    ax[1].set_ylabel('effective rank'); ax[1].set_title('Effective rank (after_ln_f)')
    ax[2].set_ylabel('normalized entropy'); ax[2].set_title('Attention entropy')
    for a in ax:
        a.set_xscale('log'); a.set_xlabel('step'); _style(a); a.legend(fontsize=7.5)
    fig.suptitle('Training observables vs d (pure attention, T=50, 30000 steps)',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(out_dir / 'compare_training.png'); plt.close(fig)

    # 2. RG prediction checks across d
    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.4))
    for d, rg in rgs.items():
        c = D_COLORS[d]
        st = rg['step'].copy(); st[st == 0] = 0.5
        ax[0, 0].plot(st, rg['sv'][:, 0], color=c, label=f'd={d}')
        ax[0, 1].plot(st, rg['s3_s2'], color=c, marker='o', ms=2.5, label=f'd={d}')
        ax[0, 2].plot(st, rg['s2_s1'], color=c, marker='o', ms=2.5, label=f'd={d}')
        ax[1, 0].plot(st, rg['cos2'], color=c, marker='o', ms=2.5, label=f'd={d}')
        ax[1, 1].plot(st, rg['mu2'], color=c, marker='o', ms=2.5, label=f'd={d}')
        ax[1, 2].plot(st, rg['mu1']/np.maximum(rg['mu2'], 1e-30), color=c,
                      marker='o', ms=2.5, label=f'd={d}')
    ax[0, 0].set_yscale('log'); ax[0, 0].set_title(r'$\sigma_1$ of F')
    ax[0, 1].axhline(0, color='#888', ls=':'); ax[0, 1].set_ylim(0, .9)
    ax[0, 1].set_title(r'C1: $\sigma_3/\sigma_2 \to 0$?')
    ax[0, 2].set_ylim(0, 1); ax[0, 2].set_title(r'C3: $\sigma_2/\sigma_1 \to 0$?')
    ax[1, 0].set_yscale('log'); ax[1, 0].axhline(1, color='#888', ls=':')
    ax[1, 0].set_title(r'C2: $\cos^2\theta \to 1$?')
    ax[1, 1].axhline(1, color='#555', ls='--', lw=1.3)
    ax[1, 1].set_yscale('log'); ax[1, 1].set_title(r'$\mu_2$ vs threshold 1')
    ax[1, 2].axhline(1, color='#888', ls=':')
    ax[1, 2].set_title(r'C4: $\mu_1/\mu_2$  (theory: 4.4–547)')
    for a in ax.ravel():
        a.set_xscale('log'); a.set_xlabel('step'); _style(a); a.legend(fontsize=7)
    fig.suptitle('RG predictions vs measurement, all d', fontweight='bold')
    fig.tight_layout(); fig.savefig(out_dir / 'compare_rg_predictions.png'); plt.close(fig)

    # 3. d-dependence of scalar summaries
    ds = sorted(trs)
    ent_min = [min(trs[d]['entropy']) for d in ds]
    ent_reb = [trs[d]['entropy'][-1] - min(trs[d]['entropy']) for d in ds]
    ent_arg = [trs[d]['ent_steps'][int(np.argmin(trs[d]['entropy']))] for d in ds]
    loss_end = [trs[d]['losses'][-1] for d in ds]
    s1_peak = [rgs[d]['sv'][:, 0].max() for d in ds]
    ratio = [np.mean(rgs[d]['mu1']/np.maximum(rgs[d]['mu2'], 1e-30)) for d in ds]

    fig, ax = plt.subplots(2, 3, figsize=(14.5, 8.0))
    for a, y, ttl, yl, logy in [
            (ax[0, 0], ent_min, 'entropy minimum vs d', 'min entropy', False),
            (ax[0, 1], ent_reb, 'entropy rebound vs d', 'rebound', False),
            (ax[0, 2], ent_arg, 'entropy min location vs d', 'step', True),
            (ax[1, 0], loss_end, 'final train loss vs d', 'MSE', True),
            (ax[1, 1], s1_peak, r'$\sigma_1$ peak vs d', r'peak $\sigma_1$', False),
            (ax[1, 2], ratio, r'mean $\mu_1/\mu_2$ vs d', r'$\mu_1/\mu_2$', False)]:
        a.plot(ds, y, color='#08306B', marker='o', ms=6)
        for xd, yv in zip(ds, y):
            a.annotate(f'{yv:.4g}', (xd, yv), textcoords='offset points',
                       xytext=(0, 8), fontsize=7.5, ha='center')
        a.set_xscale('log', base=2); a.set_xticks(ds)
        a.set_xticklabels([str(x) for x in ds])
        a.set_xlabel('d'); a.set_ylabel(yl); a.set_title(ttl)
        if logy:
            a.set_yscale('log')
        _style(a)
    ax[1, 2].axhline(1, color='#888', ls=':')
    fig.suptitle('d-dependence of summary quantities', fontweight='bold')
    fig.tight_layout(); fig.savefig(out_dir / 'compare_d_dependence.png'); plt.close(fig)


def print_tables(trs, rgs):
    print("\n" + "=" * 104)
    print("训练侧")
    print("=" * 104)
    print("%4s %10s %10s | %7s %7s %7s %7s | %8s %8s %7s %8s %9s" % (
        "d", "loss", "test", "er@5", "er_min", "@step", "er_end",
        "ent@5", "ent_min", "@step", "ent_end", "rebound"))
    for d in sorted(trs):
        tr = trs[d]
        s = np.asarray(tr['steps'], dtype=float)
        rk = np.asarray(tr['rank']['after_ln_f'], dtype=float)
        ent = np.asarray(tr['entropy'], dtype=float)
        es = np.asarray(tr['ent_steps'], dtype=float)
        i, j = int(np.nanargmin(rk)), int(np.nanargmin(ent))
        print("%4d %10.6f %10.6f | %7.3f %7.3f %7d %7.3f | %8.4f %8.4f %7d %8.4f %+9.4f" % (
            d, tr['losses'][-1], tr['final_test'], rk[0], rk[i], int(s[i]), rk[-1],
            ent[0], ent[j], int(es[j]), ent[-1], ent[-1]-ent[j]))

    print("\n" + "=" * 104)
    print("RG 侧")
    print("=" * 104)
    print("%4s | %8s %8s | %8s %8s | %10s %8s | %9s %7s %9s | %s" % (
        "d", "s21@0", "s21_end", "s32@0", "s32_end", "mu2_max", "all<1",
        "s1_peak", "@step", "s1_end", "rank(G)"))
    for d in sorted(rgs):
        g = rgs[d]
        i = int(np.argmax(g['sv'][:, 0]))
        print("%4d | %8.4f %8.4f | %8.4f %8.4f | %10.3e %8s | %9.1f %7d %9.1f | %s" % (
            d, g['s2_s1'][0], g['s2_s1'][-1], g['s3_s2'][0], g['s3_s2'][-1],
            g['mu2'].max(), bool((g['mu2'] < 1).all()),
            g['sv'][i, 0], int(g['step'][i]), g['sv'][-1, 0], set(g['rkG'].tolist())))

    print("\nC4 关键量 mu1/mu2（无量纲，不受归一化影响）")
    print("  理论预言: d=32 -> 177.8,  d=64 -> 111.2,  d=128 -> 546.7,  d=256 -> 4.4")
    for d in sorted(rgs):
        r = rgs[d]['mu1'] / np.maximum(rgs[d]['mu2'], 1e-30)
        print("  d=%3d  实测 step0 %.3f | min %.3f | max %.3f | mean %.3f" % (
            d, r[0], r.min(), r.max(), r.mean()))

    print("\n力探针 R2 末值（最优层）")
    hdr = "%16s" % "target" + "".join("%12s" % f"d={d}" for d in sorted(trs))
    print(hdr)
    for t in FORCE_TARGETS:
        row = "%16s" % t
        for d in sorted(trs):
            best = -9e9
            for (tt, L), b in trs[d]['probe'].items():
                if tt == t and b['r2'] and b['r2'][-1] > best:
                    best = b['r2'][-1]
            row += "%12.3f" % best
        print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results_803_d_sweep')
    ap.add_argument('--out_dir', default='plots_803_d_sweep')
    args = ap.parse_args()

    root = Path(args.results_dir)
    trs, rgs = {}, {}
    for d in D_LIST:
        npzs = sorted((root / f'd{d}').glob('*.npz'))
        jf = root / 'rg_tracking' / f'rg_track_d{d}_T50.json'
        if not npzs:
            print(f"  [skip] d={d}: no npz")
            continue
        trs[d] = load_train(npzs[0])
        rgs[d] = load_rg(jf) if jf.exists() else None
        plot_one_d(d, trs[d], rgs[d], Path(args.out_dir) / f'd{d}')
        print(f"  [ok] d={d}  ({len(trs[d]['steps'])} rank samples, "
              f"rg={'yes' if rgs[d] is not None else 'no'})")

    rgs_ok = {k: v for k, v in rgs.items() if v is not None}
    if len(trs) >= 2:
        plot_comparison(trs, rgs_ok, Path(args.out_dir) / '_comparison')
    print_tables(trs, rgs_ok)
    print(f"\nPlots -> {Path(args.out_dir).resolve()}/")


if __name__ == '__main__':
    main()
