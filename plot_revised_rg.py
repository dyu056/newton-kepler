"""
Plots for the revised-RG check (corrected D_F = F Gamma F^T / sqrt(d), SVD).

Reads results_803_d_sweep/rg_corrected/rg_track_d*_T50.json and writes
plots_803_d_sweep/_revised_rg/.

Usage: /opt/anaconda3/bin/python plot_revised_rg.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D_COLORS = {32: "#08306B", 64: "#2166AC", 128: "#C44E52", 256: "#E67E22"}
THEORY_RATIO = {32: 21.1, 64: 14.5, 128: 403.3, 256: 22.1}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.titleweight": "bold", "axes.grid": True, "axes.axisbelow": True,
    "axes.unicode_minus": False, "xtick.direction": "in", "ytick.direction": "in",
    "lines.linewidth": 1.7, "legend.fontsize": 8,
    "grid.linestyle": "--", "grid.alpha": 0.28,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _st(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="in")
    ax.grid(True, linestyle="--", alpha=0.28)


def load(p):
    r = json.load(open(p))
    g = {k: np.array([x[k] for x in r]) for k in
         ('step', 'loss', 's2_s1', 's3_s2', 'mu1_ref', 'mu2_ref',
          'mu_ratio_ref', 'sigma3_D_ref', 'm_G', 'm_F', 'Gamma_rank',
          'Gamma_ref_rank')}
    g['sv'] = np.array([x['sv'] for x in r])
    g['stepx'] = g['step'].astype(float).copy()
    g['stepx'][g['stepx'] == 0] = 0.5
    return g


def main():
    root = Path('results_803_d_sweep/rg_corrected')
    out = Path('plots_803_d_sweep/_revised_rg')
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for d in (32, 64, 128, 256):
        f = root / f'rg_track_d{d}_T50.json'
        if f.exists():
            data[d] = load(f)
    print("loaded d =", sorted(data))

    # ── Fig 1: the corrected mu ratio, the headline result ──
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for d, g in data.items():
        ax[0].plot(g['stepx'], g['mu_ratio_ref'], color=D_COLORS[d],
                   marker='o', ms=3, label=f'd={d}')
    ax[0].axhline(1, color='#555', ls='--', lw=1.3)
    ax[0].text(0.7, 1.1, r'$\mu_1=\mu_2$ (pre-correction value)',
               fontsize=7.5, color='#555')
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel('step'); ax[0].set_ylabel(r'$\mu_1/\mu_2$')
    ax[0].set_title(r'Corrected $\mu_1/\mu_2$ (rank-2 $\Gamma$, SVD)')
    ax[0].legend(fontsize=7.5)

    ds = sorted(data)
    meas = [data[d]['mu_ratio_ref'][0] for d in ds]
    theo = [THEORY_RATIO[d] for d in ds]
    w = 0.35
    xi = np.arange(len(ds))
    ax[1].bar(xi - w/2, theo, w, label='theory §0.3', color='#8C8C8C')
    ax[1].bar(xi + w/2, meas, w, label='measured (step 0)', color='#C44E52')
    for i, (t, m) in enumerate(zip(theo, meas)):
        ax[1].annotate(f'{t:.0f}', (i - w/2, t), ha='center',
                       textcoords='offset points', xytext=(0, 3), fontsize=7.5)
        ax[1].annotate(f'{m:.0f}', (i + w/2, m), ha='center',
                       textcoords='offset points', xytext=(0, 3), fontsize=7.5)
    ax[1].set_yscale('log'); ax[1].set_xticks(xi)
    ax[1].set_xticklabels([f'd={d}' for d in ds])
    ax[1].set_ylabel(r'$\mu_1/\mu_2$ at step 0')
    ax[1].set_title('Step-0 ratio: theory vs measurement')
    ax[1].legend(fontsize=7.5)
    for a in ax:
        _st(a)
    fig.suptitle(r'C2/C4: corrected $\mathcal{D}_F=F\Gamma F^\top/\sqrt{d}$ gives $\mu_1/\mu_2\gg1$',
                 fontweight='bold')
    fig.tight_layout(); fig.savefig(out / 'mu_ratio_corrected.png'); plt.close(fig)

    # ── Fig 2: coupled order parameters m_G, m_F (C5) ──
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for d, g in data.items():
        c = D_COLORS[d]
        ax[0].plot(g['stepx'], g['m_G'], color=c, marker='o', ms=3, label=f'd={d}')
        ax[1].plot(g['stepx'], g['m_F'], color=c, marker='o', ms=3, label=f'd={d}')
        ax[2].plot(g['m_G'], g['m_F'], color=c, marker='o', ms=3, label=f'd={d}')
        i = int(np.argmax(g['m_F']))
        ax[1].plot(g['stepx'][i], g['m_F'][i], marker='*', ms=13, color=c)
        ax[2].plot(g['m_G'][i], g['m_F'][i], marker='*', ms=13, color=c)
    ax[0].set_ylabel(r'$m_G=\|hh^\top\|_F/T$'); ax[0].set_title(r'$m_G$ (embedding condensation)')
    ax[1].set_ylabel(r'$m_F=\|F\|_F/T$'); ax[1].set_title(r'$m_F$ (attention strength), $\star$=peak')
    for a in ax[:2]:
        a.set_xscale('log'); a.set_xlabel('step')
    ax[2].set_xlabel(r'$m_G$'); ax[2].set_ylabel(r'$m_F$')
    ax[2].set_title(r'Trajectory in $(m_G, m_F)$ plane')
    for a in ax:
        _st(a); a.legend(fontsize=7.5)
    fig.suptitle('C5: coupled order parameters', fontweight='bold')
    fig.tight_layout(); fig.savefig(out / 'coupled_mG_mF.png'); plt.close(fig)

    # ── Fig 3: C1 and D_F rank ──
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for d, g in data.items():
        c = D_COLORS[d]
        ax[0].plot(g['stepx'], g['s3_s2'], color=c, marker='o', ms=3, label=f'd={d}')
        ax[1].plot(g['stepx'], g['s2_s1'], color=c, marker='o', ms=3, label=f'd={d}')
        ax[2].plot(g['stepx'], np.maximum(g['sigma3_D_ref'], 1e-30), color=c,
                   marker='o', ms=3, label=f'd={d}')
    ax[0].axhline(0, color='#888', ls=':'); ax[0].set_ylim(0, .9)
    ax[0].set_ylabel(r'$\sigma_3/\sigma_2$ of $F$')
    ax[0].set_title(r'C1 (revised): stays $O(1)$, not $\to 0$')
    ax[1].set_ylim(0, 1); ax[1].set_ylabel(r'$\sigma_2/\sigma_1$ of $F$')
    ax[1].set_title(r'$\sigma_2/\sigma_1$ (kinematic amplification)')
    ax[2].set_yscale('log'); ax[2].set_ylabel(r'$\sigma_3(\mathcal{D}_F)$')
    ax[2].set_title(r'$\mathcal{D}_F$ is rank 2')
    for a in ax:
        a.set_xscale('log'); a.set_xlabel('step'); _st(a); a.legend(fontsize=7.5)
    fig.suptitle(r'C1 and the rank of $\mathcal{D}_F$', fontweight='bold')
    fig.tight_layout(); fig.savefig(out / 'C1_and_DF_rank.png'); plt.close(fig)

    # ── summary numbers ──
    print("\n%4s | %9s %9s | %9s %8s | %8s %8s %8s | %11s" % (
        "d", "ratio@0", "theory", "ratio_min", "ratio_max",
        "mF_peak", "@step", "mG@peak", "g"))
    for d in ds:
        g = data[d]
        i = int(np.argmax(g['m_F']))
        gg = g['mu1_ref'][0] ** 2 / g['m_G'][i] ** 2
        print("%4d | %9.1f %9.1f | %9.1f %8.1f | %8.3f %8d %8.2f | %11.4e" % (
            d, g['mu_ratio_ref'][0], THEORY_RATIO[d], g['mu_ratio_ref'].min(),
            g['mu_ratio_ref'].max(), g['m_F'][i], int(g['step'][i]),
            g['m_G'][i], gg))
    print(f"\nPlots -> {out.resolve()}/")


if __name__ == '__main__':
    main()
