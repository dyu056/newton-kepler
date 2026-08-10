"""
Test the theory's block_size prediction with step-0 resolution.

Theory (effective_rank_theory.md §4.2): the dip depth is set by how much
position-tail energy the rank-2 signal has to outrun. Longer T means a longer,
flatter tail and therefore a deeper relative dip.

The recorded runs cannot test this at small T, because activation_rank is absent
at step 0 and the dip minimum already sits at the first eval step (step 5). Here
we record from step 0 with every-step resolution over the first 40 steps.

Run:  /opt/anaconda3/bin/python sweep_dip_vs_blocksize.py
"""

import numpy as np
import torch

from model_cv import GPTCV, GPTConfigCV
from generate_kepler_cv import generate_kepler_trajectory
from reproduce_rank_dip import capture, spectrum, eff_rank

NOISE, LR, N_STEPS = 0.1, 1e-3, 40
T_LIST = [2, 5, 10, 20, 50, 100]
LAYER = 'after_ln_f'
N_POINTS = 101


def trajectories(n_traj, num_points=N_POINTS, seed=1):
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_traj):
        pos, _ = generate_kepler_trajectory(
            eccentricity=rng.uniform(0.0, 0.8),
            semi_major_axis=rng.uniform(0.5, 2.0),
            angle=rng.uniform(0, 2 * np.pi), num_points=num_points)
        out.append(pos)
    return np.stack(out)


def chop(trajs, T):
    """Mirror data_utils.chop_trajectories_into_sequences (contiguous windows)."""
    xs, ys = [], []
    n_points = trajs.shape[1]
    for traj in trajs:
        for s in range(0, n_points - T, T):
            xs.append(traj[s:s + T])
            ys.append(traj[s + 1:s + T + 1])
    return (torch.from_numpy(np.stack(xs)).float(),
            torch.from_numpy(np.stack(ys)).float())


def run_one(T, n_traj=200):
    torch.manual_seed(1)
    trajs = trajectories(n_traj)
    x_all, y_all = chop(trajs, T)
    # keep the position-sample count comparable across T
    cap = max(1, 20000 // T)
    x_all, y_all = x_all[:cap], y_all[:cap]

    cfg = GPTConfigCV()
    cfg.block_size, cfg.n_layer, cfg.n_head = T, 1, 1
    cfg.n_embd, cfg.input_dim, cfg.dropout, cfg.bias = 128, 2, 0.0, True
    model = GPTCV(cfg)
    opt = torch.optim.SGD(model.parameters(), lr=LR)

    rec = []
    for step in range(N_STEPS + 1):
        xn = x_all + torch.randn_like(x_all) * NOISE
        with torch.no_grad():
            sv = spectrum(capture(model, xn)[LAYER])
        e = sv ** 2
        rec.append(dict(step=step, eff=eff_rank(sv),
                        num=int((sv > 1e-6 * sv[0]).sum()),
                        E2=float(e[:2].sum() / e.sum())))
        pred, _ = model.forward(xn, None)
        loss = torch.nn.functional.mse_loss(pred.reshape(-1), y_all.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return rec


def main():
    print(f"layer = {LAYER}, full-batch GD, lr={LR}, recording from step 0\n")
    print(f"{'T':>4} {'eff@0':>7} {'min':>7} {'@step':>6} {'drop%':>7} "
          f"{'E2@0':>7} {'E2@min':>7} {'num_rank':>10}")
    summary = []
    for T in T_LIST:
        rec = run_one(T)
        effs = [r['eff'] for r in rec]
        i = int(np.argmin(effs))
        drop = (effs[0] - effs[i]) / effs[0] * 100
        nrs = {r['num'] for r in rec}
        print(f"{T:4d} {effs[0]:7.2f} {effs[i]:7.2f} {rec[i]['step']:6d} "
              f"{drop:7.2f} {rec[0]['E2']:7.4f} {rec[i]['E2']:7.4f} "
              f"{str(sorted(nrs)):>10s}")
        summary.append((T, effs[0], effs[i], rec[i]['step'], drop))

    print("\nprediction: drop% increases monotonically with T")
    drops = [s[4] for s in summary]
    print("  observed drop%:", [f"{d:.2f}" for d in drops])
    print("  monotonic non-decreasing:",
          all(drops[i] <= drops[i + 1] + 1e-9 for i in range(len(drops) - 1)))


if __name__ == '__main__':
    main()
