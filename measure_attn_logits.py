"""
Measure the actual pre-softmax attention logits, and test whether the
softmax Taylor expansion used in the theory is justified.

The theory (effective_rank_theory.md §4.5) claims that at initialization
S = QK^T/sqrt(d) ~ O(eps^2) ~ 0, so softmax(S) is near-uniform and its Jacobian
diag(a) - a a^T kills the first-order gradient, freezing W_Q/W_K.

That argument needs an expansion of softmax about S = 0:

    softmax(S)_ij = 1/m * (1 + (S_ij - S_bar_i) + O(|S - S_bar|^2))

which is only valid when the *spread* of logits within a row is << 1. This script
measures that spread instead of assuming it, at several training steps, and reports:

  max|S|        largest logit magnitude
  row spread    max_j S_ij - min_j S_ij  (what actually controls the expansion)
  std(S)        per-row logit dispersion
  rel_err_1st   ||softmax(S) - uniform*(1 + dS)|| / ||softmax(S)||   (1st-order Taylor error)
  rel_err_2nd   same with the 2nd-order term included
  max dev       max_ij |a_ij - 1/m_i|  (departure from exact uniformity)

Run:  /opt/anaconda3/bin/python measure_attn_logits.py
      /opt/anaconda3/bin/python measure_attn_logits.py --no_ln
"""

import argparse
import math

import numpy as np
import torch

from model_cv import GPTCV, GPTConfigCV
from model_noln import GPTCVNoLN, GPTConfigCVNoLN
from data_utils import load_trajectories

BLOCK_SIZE, N_EMBD, NOISE = 100, 128, 0.1
CHECK_STEPS = [0, 1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def pre_softmax_logits(model, x):
    """Return the causally-masked pre-softmax scores S, exactly as model_cv computes them."""
    blk = model.transformer.h[0]
    h = model.input_embedding(x)
    h = h + model.transformer.wpe(torch.arange(x.shape[1], device=x.device))
    z = blk.ln_1(h)                      # Identity for the no-LN model
    B, T, C = z.size()
    attn = blk.attn
    q, k, _ = attn.c_attn(z).split(attn.n_embd, dim=2)
    nh = attn.n_head
    k = k.view(B, T, nh, C // nh).transpose(1, 2)
    q = q.view(B, T, nh, C // nh).transpose(1, 2)
    S = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    mask = attn.bias[:, :, :T, :T] == 0
    return S, mask


def taylor_diagnostics(S, mask):
    """Quantify logit spread and the error of expanding softmax about S=0."""
    S = S.double()
    Sm = S.masked_fill(mask, float('-inf'))
    A = torch.softmax(Sm, dim=-1)                     # exact attention

    valid = (~mask).to(S.dtype)                       # 1 where a key is visible
    m = valid.sum(-1, keepdim=True)                   # visible count per row
    Sz = S * valid                                    # zero out masked logits
    Sbar = (Sz.sum(-1, keepdim=True) / m)             # row mean over visible
    dS = (Sz - Sbar * valid)                          # centred logits

    # spread within each row, over visible entries only
    big = S.masked_fill(mask, float('-inf')).amax(-1)
    small = S.masked_fill(mask, float('inf')).amin(-1)
    spread = (big - small)

    # Taylor expansions of softmax about S = 0 (uniform point)
    unif = valid / m
    first = unif * (1.0 + dS)
    second = unif * (1.0 + dS + 0.5 * (dS ** 2)) - unif * unif * 0.5 * (dS ** 2).sum(-1, keepdim=True)

    def rel(approx):
        num = ((approx - A) * valid).pow(2).sum(-1).sqrt()
        den = (A * valid).pow(2).sum(-1).sqrt().clamp_min(1e-30)
        return float((num / den).mean())

    dev = float(((A - unif).abs() * valid).max())
    # `valid` broadcasts from (1,1,T,T); expand before counting or the
    # normalisation is short by a factor of B*n_head.
    n_valid = valid.expand_as(S).sum()
    return {
        'max_abs_S': float(S.masked_fill(mask, 0).abs().max()),
        'mean_abs_S': float(S.masked_fill(mask, 0).abs().sum() / n_valid),
        'max_spread': float(spread.max()),
        'mean_spread': float(spread.mean()),
        'std_S': float((dS.pow(2).sum() / n_valid).sqrt()),
        'rel_err_1st': rel(first),
        'rel_err_2nd': rel(second),
        'max_dev_from_uniform': dev,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no_ln', action='store_true', help='use the LayerNorm-free model')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n_traj', type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)
    tr = load_trajectories('data_cv', args.n_traj)
    x_all = torch.from_numpy(tr[:, :-1]).float()
    y_all = torch.from_numpy(tr[:, 1:]).float()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_all, y_all = x_all.to(dev), y_all.to(dev)

    if args.no_ln:
        cfg = GPTConfigCVNoLN()
    else:
        cfg = GPTConfigCV()
    cfg.block_size, cfg.n_layer, cfg.n_head = BLOCK_SIZE, 1, 1
    cfg.n_embd, cfg.input_dim, cfg.dropout, cfg.bias = N_EMBD, 2, 0.0, True
    model = (GPTCVNoLN(cfg) if args.no_ln else GPTCV(cfg)).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    tag = 'NO LayerNorm' if args.no_ln else 'WITH LayerNorm'
    print(f"\n{tag}, block_size={BLOCK_SIZE}, lr={args.lr}, "
          f"1/sqrt(d)={1/math.sqrt(N_EMBD):.4f}")
    print("=" * 108)
    print(f"{'step':>5} {'loss':>9} | {'max|S|':>9} {'mean|S|':>9} {'max spread':>11} "
          f"{'mean spread':>12} {'std(S)':>9} | {'1st-ord err':>11} {'2nd-ord err':>11} {'max dev':>9}")
    print("-" * 108)

    nsteps = max(CHECK_STEPS)
    for step in range(nsteps + 1):
        xn = x_all + torch.randn_like(x_all) * NOISE
        if step in CHECK_STEPS:
            with torch.no_grad():
                S, mask = pre_softmax_logits(model, xn[:256])
                d = taylor_diagnostics(S, mask)
                p, _ = model(xn[:256], None)
                loss = torch.nn.functional.mse_loss(
                    p.reshape(-1), y_all[:256].reshape(-1)).item()
            print(f"{step:5d} {loss:9.5f} | {d['max_abs_S']:9.3e} {d['mean_abs_S']:9.3e} "
                  f"{d['max_spread']:11.3e} {d['mean_spread']:12.3e} {d['std_S']:9.3e} | "
                  f"{d['rel_err_1st']:11.3e} {d['rel_err_2nd']:11.3e} "
                  f"{d['max_dev_from_uniform']:9.3e}")
        p, _ = model(xn, None)
        loss = torch.nn.functional.mse_loss(p.reshape(-1), y_all.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    print("\nTaylor expansion about S=0 is valid while the row SPREAD << 1.")
    print("Rule of thumb: spread < 0.1 => 1st order good to ~1%; spread > 1 => expansion invalid.")


if __name__ == '__main__':
    main()
