"""
Track the RG order parameters during training of the pure-attention model.

The saved .npz only holds singular VALUES of individual weight matrices, not the
matrices themselves, so F / Gamma / G / D cannot be reconstructed from it. This
script retrains the same configuration and computes the RG quantities inline.

CORRECTED 2026-08-03 per the revised RG_predictions_kepler_testbed.md §0.2:
  * D_F = F @ Gamma @ F^T / sqrt(d)   (was: sqrt(d) * F @ Gamma @ F)
  * mu_alpha = singular values of D_F via SVD, NOT sqrt(|eigvals|) of a symmetrised D
  * Gamma is now computed BOTH ways and both are reported:
      Gamma_auto  = autograd dL/dS with the single-target loss  -> rank 1
      Gamma_ref   = the analytic form at uniform attention used by
                    compute_rg_parameters.py:233-254, where every one of the T
                    positions carries an error against the target -> rank 2
    The mu_1/mu_2 ratio depends strongly on which is used, so both are tracked.
  * also tracks m_G and m_F (§3 of the revised doc) for the coupled description

Definitions follow r2_research/rg_kepler_testbed/compute_rg_parameters.py:

  F_ij   = the pre-softmax score matrix S = QK^T/sqrt(d)   (the "propagator")
  Gamma  = dL/dS at the current parameters
  G      = W_Q W_K^T  contracted through the embedding (coupling matrix)
  D      = sqrt(d) * G * Gamma * G,  eigenvalues mu^2

RG predictions being checked (RG_predictions_kepler_testbed.md):

  C1  rank-2 condensation:   sigma_3/sigma_2 of F -> 0
  C2  alignment:             cos^2(theta) = (v1^T F v1)^2 / ||F||^4 -> 1
                             and log(1 - cos^2) linear early, slope -2 mu_1
  C3  two-mode separation:   sigma_2/sigma_1 -> 0 when mu_2 < 1 (d=128),
                             -> ~1 when mu_2 large (d=256)
  C4  d controls crossover sharpness

Run:  /opt/anaconda3/bin/python track_rg_quantities.py --d 128 --steps 20000
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from model_8_1_att import build_pure_attn_model, calibrated_sigma
from data_utils import load_trajectories, chop_trajectories_into_sequences


def make_data(T, n_traj, seed=1):
    tr = load_trajectories('data_cv', n_traj)
    xi, yi, _ = chop_trajectories_into_sequences(tr, T, seed=seed)
    return torch.from_numpy(xi).float(), torch.from_numpy(yi).float()


def rg_quantities(model, x, y, d, T, v1_ref=None):
    """Compute F's spectrum, Gamma, D's eigenvalues and the alignment cos^2(theta).

    Uses one batch; Gamma needs a backward pass so this is not under no_grad.
    """
    attn = model.transformer.h[0].attn
    h = model.input_embedding(x) + model.transformer.wpe(
        torch.arange(x.shape[1], device=x.device))
    B, Tx, _ = h.shape
    q, k, v = attn.c_attn(h).split(d, dim=2)
    q = q.view(B, Tx, 1, d).transpose(1, 2)
    k = k.view(B, Tx, 1, d).transpose(1, 2)
    v = v.view(B, Tx, 1, d).transpose(1, 2)

    S = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    S.retain_grad()
    mask = attn.bias[:, :, :Tx, :Tx] == 0
    A = torch.softmax(S.masked_fill(mask, float('-inf')), dim=-1)
    o = attn.c_proj((A @ v).transpose(1, 2).reshape(B, Tx, d))
    pred = model.output_head(o)

    # single-target loss (loss_mask='last'), matching the RG setup
    loss = torch.nn.functional.mse_loss(pred[:, -1, :].reshape(-1),
                                       y[:, -1, :].reshape(-1))
    model.zero_grad(set_to_none=True)
    if S.grad is not None:
        S.grad = None
    loss.backward()

    with torch.no_grad():
        F = S.detach()[0, 0].double()                 # (T,T) propagator
        Fm = F.masked_fill(mask[0, 0], 0.0)
        sv = torch.linalg.svdvals(Fm)

        Gam = S.grad[0, 0].double()                   # (T,T)
        Gam_sv = torch.linalg.svdvals(Gam)

        # G: coupling through W_Q W_K^T in embedding space
        Wqkv = attn.c_attn.weight.detach().double()
        Wq, Wk = Wqkv[:d], Wqkv[d:2 * d]
        G = (Wq.T @ Wk) / math.sqrt(d)                # (d,d)
        G_sv = torch.linalg.svdvals(G)

        # revised doc §0.2: D_F = F Gamma F^T / sqrt(d), singular values via SVD
        D_auto = (Fm @ Gam @ Fm.T) / math.sqrt(d)
        U_a, sd_a, Vh_a = torch.linalg.svd(D_auto)

        # reference-style Gamma: analytic at uniform attention, all T positions
        # carry error -> rank 2 (compute_rg_parameters.py:233-254)
        Wout = model.output_head.weight.detach().double()
        tgt_last = y[0, -1].detach().double()
        Delta = pred[0].detach().double() - tgt_last.unsqueeze(0)
        Vm = v[0, 0].detach().double()
        Gam_ref = (1.0 / Tx) * ((Delta @ Wout) @ (Vm - Vm.mean(0, keepdim=True)).T)
        D_ref = (Fm @ Gam_ref @ Fm.T) / math.sqrt(d)
        sd_r = torch.linalg.svdvals(D_ref)
        g_ref_sv = torch.linalg.svdvals(Gam_ref)

        # coupled order parameters (revised doc §3)
        hh = h[0].detach().double()
        m_G = float((hh @ hh.T).norm() / Tx)
        m_F = float(Fm.norm() / Tx)

        mu2 = sd_a[:4].tolist()
        v1 = Vh_a[0, :]

        # alignment order parameter (Conclusion 2)
        ref = v1 if v1_ref is None else v1_ref
        num = (ref @ Fm @ ref) ** 2
        den = (Fm.pow(2).sum()) ** 2
        cos2 = float(num / den.clamp_min(1e-300))

        return {
            'loss': float(loss.detach()),
            'sv': sv[:6].tolist(),
            's2_s1': float(sv[1] / sv[0].clamp_min(1e-300)),
            's3_s2': float(sv[2] / sv[1].clamp_min(1e-300)),
            'F_norm': float(Fm.norm()),
            # --- autograd Gamma (single-target loss) -> rank 1 ---
            'Gamma_rank': int((Gam_sv > 1e-10 * Gam_sv[0]).sum()),
            'Gamma_top': float(Gam_sv[0]),
            'G_top': float(G_sv[0]),
            'mu_auto': sd_a[:4].tolist(),
            'mu1': float(sd_a[0]),
            'mu2_2': float(sd_a[1]),
            'mu_ratio_auto': float(sd_a[0] / sd_a[1].clamp_min(1e-300)),
            # --- reference analytic Gamma at uniform attention -> rank 2 ---
            'Gamma_ref_rank': int((g_ref_sv > 1e-10 * g_ref_sv[0]).sum()),
            'mu_ref': sd_r[:4].tolist(),
            'mu1_ref': float(sd_r[0]),
            'mu2_ref': float(sd_r[1]),
            'mu_ratio_ref': float(sd_r[0] / sd_r[1].clamp_min(1e-300)),
            'sigma3_D_ref': float(sd_r[2]),
            # --- coupled order parameters (revised doc §3) ---
            'm_G': m_G,
            'm_F': m_F,
            'cos2': cos2,
            'v1': v1,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--T', type=int, default=50)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--noise', type=float, default=0.1)
    ap.add_argument('--n_traj', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--out', default='results_803_pure_attn/rg_tracking')
    args = ap.parse_args()

    torch.manual_seed(1); np.random.seed(1)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_all, y_all = make_data(args.T, args.n_traj)
    x_all, y_all = x_all.to(dev), y_all.to(dev)

    model = build_pure_attn_model(block_size=args.T, n_embd=args.d, device=dev)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    marks = sorted(set([0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000,
                        1250, 1500, 2000, 3000, 4000, 5000, 6500, 8000, 10000,
                        13000, 16000, 20000]))
    marks = [m for m in marks if m <= args.steps]

    xb, yb = x_all[:args.batch], y_all[:args.batch]
    rec, v1_0 = [], None
    print(f"\nd={args.d} T={args.T} lr={args.lr} sigma_init={calibrated_sigma(args.d):.4f}")
    print(f"{'step':>6} {'loss':>9} {'s1(F)':>9} {'s2/s1':>7} {'s3/s2':>7} "
          f"{'rkG':>4} {'rkGref':>7} {'mu1_ref':>10} {'mu2_ref':>10} "
          f"{'ratio_ref':>10} {'m_G':>9} {'m_F':>8}")
    for step in range(args.steps + 1):
        if step in marks:
            xn = xb + torch.randn_like(xb) * args.noise
            q = rg_quantities(model, xn, yb, args.d, args.T, v1_ref=v1_0)
            if v1_0 is None:
                v1_0 = q['v1']
            rec.append({k: v for k, v in q.items() if k != 'v1'} | {'step': step})
            print(f"{step:6d} {q['loss']:9.5f} {q['sv'][0]:9.3f} {q['s2_s1']:7.4f} "
                  f"{q['s3_s2']:7.4f} {q['Gamma_rank']:4d} {q['Gamma_ref_rank']:7d} "
                  f"{q['mu1_ref']:10.3e} {q['mu2_ref']:10.3e} "
                  f"{q['mu_ratio_ref']:10.2f} {q['m_G']:9.2f} {q['m_F']:8.3f}")

        xn = x_all + torch.randn_like(x_all) * args.noise
        pred, _ = model(xn, None)
        loss = torch.nn.functional.mse_loss(pred[:, -1, :].reshape(-1),
                                           y_all[:, -1, :].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    outp = Path(args.out); outp.mkdir(parents=True, exist_ok=True)
    f = outp / f'rg_track_d{args.d}_T{args.T}.json'
    json.dump(rec, open(f, 'w'), indent=1)
    print(f"\nsaved -> {f}")


if __name__ == '__main__':
    main()
