"""
Numerical verification of the effective-rank-decline theory.

Reproduces the block_size=100 / GD setup of results_gd_sweep and tests, at the
actual initialization, each claim the theory makes:

  C1  input_embedding gradient has rank <= 2 (exactly, not approximately)
  C2  output_head gradient has rank <= 2
  C3  all interior weight gradients have rank <= 2 + (attention rank inflation)
  C4  dL/dW_Q and dL/dW_K are suppressed relative to dL/dW_out by O(eps^2)
  C5  effective rank of activations is NOT bounded by the gradient rank
      (numerical rank stays full; only the spectrum is reweighted)

Run:  /opt/anaconda3/bin/python verify_rank_theory.py
"""

import numpy as np
import torch

from model_cv import GPTCV, GPTConfigCV
from generate_kepler_cv import generate_kepler_trajectory

torch.manual_seed(1)
np.random.seed(1)

BLOCK_SIZE = 100
N_EMBD = 128
NOISE = 0.1
N_TRAJ = 64


def make_data(n_traj=N_TRAJ, num_points=BLOCK_SIZE + 1):
    trajs = []
    for _ in range(n_traj):
        e = np.random.uniform(0.0, 0.8)
        a = np.random.uniform(0.5, 2.0)
        ang = np.random.uniform(0, 2 * np.pi)
        pos, _ = generate_kepler_trajectory(
            eccentricity=e, semi_major_axis=a, angle=ang, num_points=num_points)
        trajs.append(pos)
    trajs = np.stack(trajs)
    return (torch.from_numpy(trajs[:, :-1]).float(),
            torch.from_numpy(trajs[:, 1:]).float())


def build_model():
    cfg = GPTConfigCV()
    cfg.block_size, cfg.n_layer, cfg.n_head = BLOCK_SIZE, 1, 1
    cfg.n_embd, cfg.input_dim, cfg.dropout, cfg.bias = N_EMBD, 2, 0.0, True
    return GPTCV(cfg)


def rank_report(name, g, tol=1e-8):
    sv = torch.linalg.svdvals(g.reshape(g.shape[0], -1).double())
    nz = int((sv > tol * max(sv[0].item(), 1e-30)).sum())
    p = (sv ** 2 / (sv ** 2).sum()).clamp_min(1e-300)
    eff = float(torch.exp(-(p * p.log()).sum()))
    print(f"  {name:38s} shape={tuple(g.shape)!s:12s} "
          f"num_rank={nz:4d} eff_rank={eff:6.2f} |g|={g.norm():.3e}")
    return nz, eff, sv


def main():
    inputs, targets = make_data()
    model = build_model()
    model.train()

    x = inputs + torch.randn_like(inputs) * NOISE
    pred, _ = model.forward(x, None)
    loss = torch.nn.functional.mse_loss(pred.reshape(-1), targets.reshape(-1))
    grads = torch.autograd.grad(loss, [p for _, p in model.named_parameters()],
                                allow_unused=True)
    gd = {n: g for (n, _), g in zip(model.named_parameters(), grads)}

    print(f"\nloss at init = {loss.item():.6f}   |pred| = {pred.norm():.3e} "
          f" |target| = {targets.norm():.3e}")

    print("\n=== C1-C3: rank of weight gradients at initialization ===")
    order = ['input_embedding.weight', 'output_head.weight',
             'transformer.wpe.weight',
             'transformer.h.0.attn.c_attn.weight',
             'transformer.h.0.attn.c_proj.weight',
             'transformer.h.0.mlp.c_fc.weight',
             'transformer.h.0.mlp.c_proj.weight']
    for n in order:
        if gd.get(n) is not None:
            rank_report(n, gd[n])

    print("\n=== C4: attention Q/K gradient suppression ===")
    g_attn = gd['transformer.h.0.attn.c_attn.weight']
    gq, gk, gv = g_attn.split(N_EMBD, dim=0)
    gout = gd['output_head.weight']
    print(f"  |dL/dW_Q|   = {gq.norm():.4e}")
    print(f"  |dL/dW_K|   = {gk.norm():.4e}")
    print(f"  |dL/dW_V|   = {gv.norm():.4e}")
    print(f"  |dL/dW_out| = {gout.norm():.4e}")
    print(f"  ratio |g_Q| / |g_V|   = {(gq.norm()/gv.norm()):.4e}")
    print(f"  ratio |g_Q| / |g_out| = {(gq.norm()/gout.norm()):.4e}")
    eps = 0.02
    print(f"  eps^2 = {eps**2:.4e}   (predicted scale of the Q/K suppression)")

    print("\n=== C5: activation spectra — is any direction absent? ===")
    acts = {}
    h = model.input_embedding(x)
    acts['input_embed'] = h
    pos = model.transformer.wpe(torch.arange(BLOCK_SIZE))
    acts['after_pos_emb'] = h + pos
    for name, a in acts.items():
        a2 = a.reshape(-1, a.shape[-1]).detach().double()
        a2 = a2 - a2.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(a2)
        p = (sv ** 2 / (sv ** 2).sum()).clamp_min(1e-300)
        eff = float(torch.exp(-(p * p.log()).sum()))
        nz = int((sv > 1e-6 * sv[0]).sum())
        print(f"  {name:20s} eff_rank={eff:6.2f}  num_rank={nz:4d}  "
              f"top5={[round(v,2) for v in sv[:5].tolist()]}")

    print("\n=== C5b: the dip is spectral reweighting, not rank loss ===")
    print("  Synthetic test: freeze the tail, grow only the top-2 singular values.")
    sv = torch.linalg.svdvals(
        (acts['after_pos_emb'].reshape(-1, N_EMBD).detach().double()
         - acts['after_pos_emb'].reshape(-1, N_EMBD).detach().double().mean(0)))

    def eff_of(s):
        p = (s ** 2 / (s ** 2).sum()).clamp_min(1e-300)
        return float(torch.exp(-(p * p.log()).sum()))

    for scale in [1.0, 1.05, 1.10, 1.20, 1.40]:
        s = sv.clone()
        s[:2] *= scale
        print(f"    top-2 scaled by {scale:4.2f} -> eff_rank {eff_of(s):6.2f} "
              f"(num_rank unchanged = {int((s > 1e-6*s[0]).sum())})")


if __name__ == '__main__':
    main()
