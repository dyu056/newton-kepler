"""
Reproduce the effective-rank dip under GD and decompose its cause.

Trains the same 1-layer model with full-batch GD (as in results_gd_sweep) while
recording the COMPLETE singular spectrum of each activation at every step, so
the dip can be attributed to specific singular directions.

Outputs, per step:
  eff          effective rank (participation ratio, full spectrum)
  num          numerical rank (1e-6 tol) -- tests whether directions die
  E2           fraction of total energy in the top-2 directions
  eff_tail     effective rank of the spectrum with the top-2 REMOVED
  eff_frozen   eff rank if only the top-2 evolve and the tail is held at step 0

If the dip is spectral reweighting driven by the rank-2 gradient, then across
the dip:  num is constant, E2 rises, eff_tail is ~flat, and eff_frozen tracks eff.

Run:  /opt/anaconda3/bin/python reproduce_rank_dip.py
"""

import numpy as np
import torch

from model_cv import GPTCV, GPTConfigCV
from generate_kepler_cv import generate_kepler_trajectory

torch.manual_seed(1)
np.random.seed(1)

BLOCK_SIZE, N_EMBD, NOISE, LR = 100, 128, 0.1, 1e-3
N_TRAJ, N_STEPS = 256, 120
LAYERS = ['after_pos_emb', 'block_0_mlp_output',
          'block_0_after_mlp_merge', 'after_ln_f']


def make_data(n_traj, num_points=BLOCK_SIZE + 1):
    trajs = []
    for _ in range(n_traj):
        pos, _ = generate_kepler_trajectory(
            eccentricity=np.random.uniform(0.0, 0.8),
            semi_major_axis=np.random.uniform(0.5, 2.0),
            angle=np.random.uniform(0, 2 * np.pi), num_points=num_points)
        trajs.append(pos)
    t = np.stack(trajs)
    return (torch.from_numpy(t[:, :-1]).float(),
            torch.from_numpy(t[:, 1:]).float())


def spectrum(a):
    m = a.reshape(-1, a.shape[-1]).detach().double()
    m = m - m.mean(0, keepdim=True)
    return torch.linalg.svdvals(m)


def eff_rank(sv):
    e = sv ** 2
    p = (e / e.sum()).clamp_min(1e-300)
    return float(torch.exp(-(p * p.log()).sum()))


def capture(model, x):
    """Forward pass returning the same named activations as observe.py."""
    acts = {}
    h = model.input_embedding(x)
    pos = model.transformer.wpe(torch.arange(x.shape[1], device=x.device))
    h = h + pos
    acts['after_pos_emb'] = h
    blk = model.transformer.h[0]
    h = h + blk.attn(blk.ln_1(h))
    acts['block_0_after_attn_merge'] = h
    mlp_in = blk.ln_2(h)
    hid = blk.mlp.silu(blk.mlp.c_fc(mlp_in))
    out = blk.mlp.c_proj(hid)
    acts['block_0_mlp_output'] = out
    h = h + out
    acts['block_0_after_mlp_merge'] = h
    acts['after_ln_f'] = model.transformer.ln_f(h)
    return acts


def main():
    inputs, targets = make_data(N_TRAJ)
    cfg = GPTConfigCV()
    cfg.block_size, cfg.n_layer, cfg.n_head = BLOCK_SIZE, 1, 1
    cfg.n_embd, cfg.input_dim, cfg.dropout, cfg.bias = N_EMBD, 2, 0.0, True
    model = GPTCV(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    base = {}      # step-0 spectrum per layer, for the frozen-tail control
    hist = {L: [] for L in LAYERS}

    for step in range(N_STEPS + 1):
        x = inputs + torch.randn_like(inputs) * NOISE

        if step % 5 == 0:
            with torch.no_grad():
                acts = capture(model, x)
            for L in LAYERS:
                sv = spectrum(acts[L])
                if step == 0:
                    base[L] = sv.clone()
                e = sv ** 2
                E2 = float(e[:2].sum() / e.sum())
                frozen = base[L].clone()
                frozen[:2] = sv[:2]
                hist[L].append(dict(
                    step=step, eff=eff_rank(sv),
                    num=int((sv > 1e-6 * sv[0]).sum()),
                    E2=E2, eff_tail=eff_rank(sv[2:]),
                    eff_frozen=eff_rank(frozen),
                    s1=float(sv[0]), s3=float(sv[2])))

        pred, _ = model.forward(x, None)
        loss = torch.nn.functional.mse_loss(pred.reshape(-1), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    for L in LAYERS:
        print(f"\n=== {L} ===")
        print(f"{'step':>5} {'eff':>7} {'num':>5} {'E2':>7} {'eff_tail':>9} "
              f"{'eff_frozen':>11} {'s1':>9} {'s3':>9}")
        for r in hist[L]:
            print(f"{r['step']:5d} {r['eff']:7.2f} {r['num']:5d} {r['E2']:7.4f} "
                  f"{r['eff_tail']:9.2f} {r['eff_frozen']:11.2f} "
                  f"{r['s1']:9.1f} {r['s3']:9.1f}")
        effs = [r['eff'] for r in hist[L]]
        i = int(np.argmin(effs))
        print(f"  dip: {effs[0]:.2f} -> {effs[i]:.2f} @step {hist[L][i]['step']}"
              f"  (num_rank constant: {len({r['num'] for r in hist[L]}) == 1})")


if __name__ == '__main__':
    main()
