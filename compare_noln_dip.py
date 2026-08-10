"""
Does the effective-rank dip survive without LayerNorm?

Two questions, one script:

  Q1  Are the arXiv:2605.01199 Lemma 3.1 preconditions restored?
      Test: does |grad| -> 0 as eps -> 0, with clean integer powers of eps?
      With LN the answer was no (ratios 1.17 / 6.10 / 9.91 per 10x).

  Q2  Is the effective-rank dip still there, and where?
      With LN (reproduce_rank_dip.py): after_ln_f 15.69 -> 14.39 @step 15 -> 17.54,
      num_rank pinned at 127 throughout, driven by rising top-2 energy E2.

Q2 matters because removing LN buys theoretical cleanliness but changes the dynamics
(the O(eps)->O(1) plateau). If the dip vanishes or moves, then the LN-free model is a
different phenomenon, not a cleaner window onto the same one.

Run:  /opt/anaconda3/bin/python compare_noln_dip.py
"""

import numpy as np
import torch

from model_noln import build_noln_model, rescale_init
from model_cv import GPTCV, GPTConfigCV
from generate_kepler_cv import generate_kepler_trajectory
from reproduce_rank_dip import spectrum, eff_rank

BLOCK_SIZE, N_EMBD, NOISE, LR = 100, 128, 0.1, 1e-3
N_TRAJ, N_STEPS = 256, 120
LAYERS = ['after_pos_emb', 'block_0_mlp_output',
          'block_0_after_mlp_merge', 'after_ln_f']


def make_data(n_traj=N_TRAJ, num_points=BLOCK_SIZE + 1, seed=1):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_traj):
        pos, _ = generate_kepler_trajectory(
            eccentricity=rng.uniform(0.0, 0.8),
            semi_major_axis=rng.uniform(0.5, 2.0),
            angle=rng.uniform(0, 2 * np.pi), num_points=num_points)
        out.append(pos)
    t = np.stack(out)
    return (torch.from_numpy(t[:, :-1]).float(),
            torch.from_numpy(t[:, 1:]).float())


def capture_noln(model, x):
    """Same activation names as probe.py, for an LN-free model."""
    acts = {}
    h = model.input_embedding(x)
    h = h + model.transformer.wpe(torch.arange(x.shape[1], device=x.device))
    acts['after_pos_emb'] = h
    blk = model.transformer.h[0]
    h = h + blk.attn(blk.ln_1(h))          # ln_1 is Identity
    acts['block_0_after_attn_merge'] = h
    out = blk.mlp.c_proj(blk.mlp.silu(blk.mlp.c_fc(blk.ln_2(h))))
    acts['block_0_mlp_output'] = out
    h = h + out
    acts['block_0_after_mlp_merge'] = h
    acts['after_ln_f'] = model.transformer.ln_f(h)   # Identity
    return acts


def q1_critical_point(inputs, targets):
    print("=" * 74)
    print("Q1: is the origin a critical point?  (Lemma 3.1 precondition)")
    print("=" * 74)
    for tag, noln in [("WITH LayerNorm", False), ("NO LayerNorm", True)]:
        print(f"\n{tag}")
        print(f"{'eps':>8} {'loss':>10} {'|g_Wout|':>12} {'|g_cfc|':>12} {'|g_Q|':>12}")
        prev = None
        for eps in [2e-2, 2e-3, 2e-4, 2e-5]:
            torch.manual_seed(1)
            if noln:
                m = build_noln_model(BLOCK_SIZE, 1, 1, N_EMBD)
            else:
                cfg = GPTConfigCV()
                cfg.block_size, cfg.n_layer, cfg.n_head = BLOCK_SIZE, 1, 1
                cfg.n_embd, cfg.input_dim = N_EMBD, 2
                cfg.dropout, cfg.bias = 0.0, True
                m = GPTCV(cfg)
            rescale_init(m, eps)
            x = inputs + torch.randn_like(inputs) * NOISE
            pred, _ = m.forward(x, None)
            loss = torch.nn.functional.mse_loss(pred.reshape(-1), targets.reshape(-1))
            gw, gc, ga = torch.autograd.grad(
                loss, [m.output_head.weight, m.transformer.h[0].mlp.c_fc.weight,
                       m.transformer.h[0].attn.c_attn.weight])
            gq = ga.split(N_EMBD, 0)[0]
            print(f"{eps:8.0e} {loss.item():10.5f} {gw.norm():12.4e} "
                  f"{gc.norm():12.4e} {gq.norm():12.4e}")
            if prev:
                r = [prev[i] / v for i, v in enumerate(
                    [gw.norm(), gc.norm(), gq.norm()])]
                print(f"{'':8s} {'ratio/10x':>10s} {r[0]:12.2f} {r[1]:12.2f} {r[2]:12.2e}")
            prev = (gw.norm(), gc.norm(), gq.norm())
    print("\nratio ~10 => grad ~ eps^1 (critical point);  ~1 => O(1) (not one)")


def q2_dip(inputs, targets, noln):
    torch.manual_seed(1)
    if noln:
        model = build_noln_model(BLOCK_SIZE, 1, 1, N_EMBD)
        cap = capture_noln
    else:
        cfg = GPTConfigCV()
        cfg.block_size, cfg.n_layer, cfg.n_head = BLOCK_SIZE, 1, 1
        cfg.n_embd, cfg.input_dim, cfg.dropout, cfg.bias = N_EMBD, 2, 0.0, True
        model = GPTCV(cfg)
        from reproduce_rank_dip import capture as cap
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    hist = {L: [] for L in LAYERS}
    losses = []
    for step in range(N_STEPS + 1):
        x = inputs + torch.randn_like(inputs) * NOISE
        if step % 5 == 0:
            with torch.no_grad():
                acts = cap(model, x)
            for L in LAYERS:
                sv = spectrum(acts[L])
                e = sv ** 2
                hist[L].append(dict(step=step, eff=eff_rank(sv),
                                    num=int((sv > 1e-6 * sv[0]).sum()),
                                    E2=float(e[:2].sum() / e.sum())))
        pred, _ = model.forward(x, None)
        loss = torch.nn.functional.mse_loss(pred.reshape(-1), targets.reshape(-1))
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    return hist, losses


def main():
    inputs, targets = make_data()
    q1_critical_point(inputs, targets)

    print("\n" + "=" * 74)
    print("Q2: does the effective-rank dip survive?")
    print("=" * 74)
    res = {}
    for tag, noln in [("WITH LayerNorm", False), ("NO LayerNorm", True)]:
        hist, losses = q2_dip(inputs, targets, noln)
        res[tag] = (hist, losses)
        print(f"\n{tag}   loss {losses[0]:.5f} -> {losses[-1]:.5f}")
        print(f"  {'layer':26s} {'eff@0':>7} {'min':>7} {'@step':>6} "
              f"{'drop%':>7} {'end':>7} {'E2@0':>7} {'E2@min':>7} {'num':>6}")
        for L in LAYERS:
            v = [r['eff'] for r in hist[L]]
            i = int(np.argmin(v))
            nrs = {r['num'] for r in hist[L]}
            print(f"  {L:26s} {v[0]:7.2f} {v[i]:7.2f} {hist[L][i]['step']:6d} "
                  f"{(v[0]-v[i])/v[0]*100:7.2f} {v[-1]:7.2f} "
                  f"{hist[L][0]['E2']:7.4f} {hist[L][i]['E2']:7.4f} "
                  f"{'const' if len(nrs)==1 else 'VARIES':>6s}")

    print("\n" + "-" * 74)
    print("verdict")
    for tag in res:
        hist, losses = res[tag]
        v = [r['eff'] for r in hist['after_ln_f']]
        i = int(np.argmin(v))
        dip = (v[0] - v[i]) / v[0] * 100
        print(f"  {tag:16s} after_ln_f dip {dip:5.2f}% @step {hist['after_ln_f'][i]['step']:3d}"
              f"   loss drop {losses[0]-losses[-1]:.5f}")


if __name__ == '__main__':
    main()
