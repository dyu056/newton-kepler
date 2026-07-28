"""
Capture attention weight matrices during training and visualize as grayscale.

Grabs the softmax attention matrix (B, H, T, T) at steps 0,10,...,100,
averages over batch and heads, and saves grayscale heatmaps.

Usage:
    python capture_attention.py --gpu 1
"""

import argparse, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from model_cv import GPTConfigCV, GPTCV
from data_utils import load_trajectories, chop_trajectories_into_sequences
from loss import compute_loss_with_mask


def capture_attention_evolution(config, out_dir='plots_attn_evo', gpu=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() and gpu != 'cpu' else 'cpu')
    if gpu is not None and gpu != 'cpu':
        torch.cuda.set_device(int(gpu))

    seed = int(config.get('seed', 1))
    np.random.seed(seed); torch.manual_seed(seed)

    # ── Data ──
    dc = config['data']
    n_traj = int(dc['num_trajectories'])
    block_size = int(config['training']['block_sizes'][0])
    n_steps = int(config['training']['n_steps'])
    lr = float(config['training']['learning_rate'])
    noise_scale = float(config['training']['noise_scale'])
    batch_size = int(config['training']['batch_size'])
    loss_mask = config['training']['loss_mask']

    trajectories = load_trajectories(dc['data_dir'], num_trajectories_needed=2 * n_traj)
    train_traj = trajectories[:n_traj]; test_traj = trajectories[n_traj:]

    if block_size < 100:
        train_inp, train_tgt, _ = chop_trajectories_into_sequences(train_traj, block_size, seed=seed)
        test_inp, test_tgt, _ = chop_trajectories_into_sequences(test_traj, block_size, seed=seed+1)
    else:
        train_inp = torch.from_numpy(train_traj[:, :-1, :]).float()
        train_tgt = torch.from_numpy(train_traj[:, 1:, :]).float()
        test_inp = torch.from_numpy(test_traj[:, :-1, :]).float()
        test_tgt = torch.from_numpy(test_traj[:, 1:, :]).float()

    train_inp = torch.from_numpy(train_inp).float() if not torch.is_tensor(train_inp) else train_inp.float()
    train_tgt = torch.from_numpy(train_tgt).float() if not torch.is_tensor(train_tgt) else train_tgt.float()
    num_train = train_inp.shape[0]

    # ── Model ──
    mc = config['model']
    GPTConfigCV.block_size = block_size; GPTConfigCV.input_dim = 2
    GPTConfigCV.n_layer = int(mc['n_layer']); GPTConfigCV.n_head = int(mc['n_head'])
    GPTConfigCV.n_embd = int(mc['n_embd']); GPTConfigCV.attention_alpha = 0.0
    GPTConfigCV.bias = True; GPTConfigCV.varcov_enabled = False; GPTConfigCV.varcov_target_std = 0.1
    model = GPTCV(GPTConfigCV).to(device)
    model.train()

    # ── Hook: capture attention weights ──
    attn_cache = {}  # layer_name -> attention matrix (averaged over batch & heads)

    def make_hook(layer_name):
        def hook(module, input, output):
            # The attention matrix is computed inside CausalSelfAttention.forward()
            # We need to access it. We'll use a different approach: intercept the softmax output.
            pass
        return hook

    # Register hooks — capture from block's attention module
    for idx, block in enumerate(model.transformer.h):
        layer_name = f'block_{idx}'
        block.attn.capture_attention_stats = True
        # Override _record_attention_stats to also save the matrix
        orig_record = block.attn._record_attention_stats

        def make_capture_fn(lname):
            def capture(probabilities):
                avg = probabilities.detach().float().mean(dim=(0, 1)).cpu().numpy()
                attn_cache[lname] = avg
            return capture

        block.attn._record_attention_stats = make_capture_fn(layer_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    # ── Step 0: capture ──
    attn_snapshots = {}  # step -> {layer: matrix}

    def capture_step0():
        model.eval()
        with torch.no_grad():
            idx = torch.randint(0, num_train, (batch_size,))
            bx = train_inp[idx].to(device)
            model(bx, None)
        model.train()
        snap = {k: v.copy() for k, v in attn_cache.items()}
        attn_snapshots[0] = snap
        print(f'Step 0 captured: {list(snap.keys())} layers, shapes={[v.shape for v in snap.values()]}')

    capture_step0()

    # ── Training loop with periodic capture ──
    capture_steps = list(range(100, n_steps + 1, 100))
    next_capture_idx = 0

    for i in range(n_steps):
        completed_step = i + 1

        if completed_step == n_steps // 2:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1

        idx = torch.randint(0, num_train, (batch_size,))
        bx = train_inp[idx].to(device)
        by = train_tgt[idx].to(device)
        inputs_noised = bx + torch.randn_like(bx) * noise_scale

        preds, _ = model.forward(inputs_noised, None)
        loss = compute_loss_with_mask(preds, by, loss_mask=loss_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Capture after training step (for step 10,20,...,100)
        if (next_capture_idx < len(capture_steps)
                and completed_step == capture_steps[next_capture_idx]):
            model.eval()
            with torch.no_grad():
                idx = torch.randint(0, num_train, (batch_size,))
                bx = train_inp[idx].to(device)
                model(bx, None)
            model.train()
            snap = {k: v.copy() for k, v in attn_cache.items()}
            attn_snapshots[completed_step] = snap
            print(f'Step {completed_step} captured')
            next_capture_idx += 1

    # ── Visualization ──
    n_layers = len(model.transformer.h)
    n_steps_captured = len(attn_snapshots)
    steps_list = sorted(attn_snapshots.keys())

    for layer_idx in range(n_layers):
        layer_name = f'block_{layer_idx}'
        n_cols = 7
        n_rows = (n_steps_captured + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(3 * n_cols, 3 * n_rows))
        axes = axes.flatten()

        for ax_idx, step in enumerate(steps_list):
            mat = attn_snapshots[step].get(layer_name)
            if mat is None:
                continue
            ax = axes[ax_idx]
            im = ax.imshow(mat, cmap='Greys', aspect='auto', vmin=0, vmax=mat.max())
            ax.set_title(f'Step {step}', fontsize=8)
            ax.set_xlabel('Key pos'); ax.set_ylabel('Query pos')
            ax.tick_params(labelsize=6)

        for j in range(n_steps_captured, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f'{layer_name} — Attention Weight Evolution (black=high)',
                     fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(out_dir / f'attn_evo_{layer_name}.png', dpi=150)
        plt.close(fig)
        print(f'Saved: {out_dir / f"attn_evo_{layer_name}.png"}')

    # Also save as NPZ for later use
    np.savez_compressed(out_dir / 'attn_snapshots.npz',
                        steps=steps_list,
                        snapshots=attn_snapshots)
    print(f'Saved: {out_dir / "attn_snapshots.npz"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/basic.yaml',
                        help='Path to YAML config')
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='plots_attn_evo')
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    capture_attention_evolution(config, out_dir=args.out_dir, gpu=args.gpu)
