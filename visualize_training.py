"""
可视化训练结果：ablation_v2 vs ablation_svb
生成 Loss、R² 演化、逐层对比等全套图像。
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ── 加载数据 ──
v2 = np.load('results/kepler_cv_blocksize/results_block_size_100_num_trajectories_10000_noise_scale_0.1_loss_mask_all.npz',
             allow_pickle=True)
svb = np.load('results/kepler_cv_blocksize_svb/results_block_size_100_num_trajectories_10000_noise_scale_0.1_loss_mask_all.npz',
              allow_pickle=True)
v2d = {k: v2[k] for k in v2.keys()}
svbd = {k: svb[k] for k in svb.keys()}

os.makedirs('plots', exist_ok=True)

# ── 配色 ──
C_V2 = '#2196F3'
C_SVB = '#FF5722'

# ============================================================
# Figure 1: Loss 曲线
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (d, name, c) in zip(axes, [(v2d, 'ablation_v2 (no SVB)', C_V2),
                                     (svbd, 'ablation_svb (SVB ε=0.5)', C_SVB)]):
    ax.plot(d['train_losses'], alpha=0.3, linewidth=0.5, color=c)
    ax.plot(d['test_losses'], linewidth=1.5, color=c, label='test')
    ax.set_yscale('log')
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('step'); ax.set_ylabel('loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.95, f'train={d["train_losses"][-1]:.4f}\ntest={d["test_losses"][-1]:.4f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Training & Test Loss', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('plots/01_loss.png', dpi=150, bbox_inches='tight')
print('Saved: 01_loss.png')

# ============================================================
# Figure 2: 力场 R² 演化（每个探针一条曲线）
# ============================================================
def extract_r2_evolution(data, probe_names, metric='eval_r2_sequence_last'):
    """提取所有 eval step 上每个探针的最佳 R²（跨层取 max）"""
    result = {k: [] for k in probe_names}
    steps = []
    for eval_data, step in zip(data['eval_results'], data['eval_steps']):
        steps.append(step)
        for pn in probe_names:
            best = -np.inf
            for layer, vals in eval_data['probe_results'].items():
                if pn in vals and metric in vals[pn]:
                    v = vals[pn][metric]
                    if v > best: best = v
            result[pn].append(max(best, 0))  # clamp to >= 0
    return steps, result

force_names = ['F_magnitude', 'Fx', 'Fy',
               'F_direction_x', 'F_direction_y',
               'r', 'inv_r', 'r_squared', 'inv_r_squared', 'inv_r_cubed', 'x', 'y']

s_v2, r_v2 = extract_r2_evolution(v2d, force_names)
s_svb, r_svb = extract_r2_evolution(svbd, force_names)

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
for idx, (pn, ax) in enumerate(zip(force_names, axes.flatten())):
    ax.plot(s_v2, r_v2[pn], 'o-', color=C_V2, markersize=3, linewidth=1, label='v2')
    ax.plot(s_svb, r_svb[pn], 's-', color=C_SVB, markersize=3, linewidth=1, label='svb')
    ax.set_title(pn, fontweight='bold')
    ax.set_xlabel('step'); ax.set_ylabel('R²'); ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=7)

plt.suptitle('Force Probe R² Evolution (eval_r2_sequence_last, best across layers)', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('plots/02_force_r2_evolution.png', dpi=150, bbox_inches='tight')
print('Saved: 02_force_r2_evolution.png')

# ============================================================
# Figure 3: 最终 eval 步的逐层 R² 对比
# ============================================================
last_v2 = v2d['eval_results'][-1]['probe_results']
last_svb = svbd['eval_results'][-1]['probe_results']

# 排序层
def sort_layers(d):
    keys = list(d.keys())
    order = {'input_embed': 0, 'after_pos_emb': 1, 'after_ln_f': 99}
    result = []
    for k in keys:
        if k in order:
            result.append((order[k], k))
        elif k.startswith('block_'):
            parts = k.split('_')
            blk = int(parts[1])
            suffix = '_'.join(parts[2:])
            s_order = {'attn_output': 0, 'after_attn_merge': 1, 'mlp_output': 2, 'after_mlp_merge': 3, 'mlp_hidden': 4}
            result.append((2 + blk*5 + s_order.get(suffix, 99), k))
    result.sort()
    return [k for _, k in result]

layers = sort_layers(last_v2)
labels = [l.replace('block_','b').replace('_attn_output','_attn').replace('_after_attn_merge','+attn')
           .replace('_mlp_output','_mlp').replace('_after_mlp_merge','+mlp').replace('_mlp_hidden','_hid')
           .replace('input_embed','embed').replace('after_pos_emb','+pos').replace('after_ln_f','ln_f')
          for l in layers]

key_force = ['F_magnitude', 'Fx', 'Fy', 'r', 'x', 'y']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, (pn, ax) in enumerate(zip(key_force, axes.flatten())):
    r2_v2_layer = [last_v2[l][pn].get('eval_r2_sequence_last', last_v2[l][pn].get('r2_last', 0)) if l in last_v2 and pn in last_v2[l] else 0 for l in layers]
    r2_svb_layer = [last_svb[l][pn].get('eval_r2_sequence_last', last_svb[l][pn].get('r2_last', 0)) if l in last_svb and pn in last_svb[l] else 0 for l in layers]

    x = range(len(layers))
    ax.plot(x, r2_v2_layer, 'o-', color=C_V2, markersize=6, linewidth=2, label='v2')
    ax.plot(x, r2_svb_layer, 's-', color=C_SVB, markersize=6, linewidth=2, label='svb')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(-0.1, 1.1); ax.set_title(pn, fontweight='bold')
    ax.set_ylabel('R²'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

plt.suptitle('Per-Layer Force Probe R² — Final Eval Step', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('plots/03_per_layer_force.png', dpi=150, bbox_inches='tight')
print('Saved: 03_per_layer_force.png')

# ============================================================
# Figure 4: 预测误差 evolution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (d, name, c) in zip(axes, [(v2d, 'v2', C_V2), (svbd, 'svb', C_SVB)]):
    errors = []
    steps = []
    for eval_data, step in zip(d['eval_results'], d['eval_steps']):
        err = eval_data.get('error_stats_test', {})
        if 'position_errors' in err:
            errors.append(np.mean(err['position_errors']))
            steps.append(step)
    ax.plot(steps, errors, 'o-', color=c, markersize=4, linewidth=1.5)
    ax.set_title(f'{name} — Mean Position Error', fontweight='bold')
    ax.set_xlabel('step'); ax.set_ylabel('error'); ax.grid(True, alpha=0.3)

plt.suptitle('Prediction Error Evolution', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('plots/04_error_evolution.png', dpi=150, bbox_inches='tight')
print('Saved: 04_error_evolution.png')

# ============================================================
# Figure 5: R² 差值热力图 (svb - v2)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 6))

all_force = ['F_magnitude','Fx','Fy','F_direction_x','F_direction_y',
             'r','inv_r','r_squared','inv_r_squared','inv_r_cubed','x','y']
diff_matrix = np.zeros((len(layers), len(all_force)))
for i, layer in enumerate(layers):
    for j, pn in enumerate(all_force):
        v2_val = last_v2[layer][pn].get('eval_r2_sequence_last', 0) if layer in last_v2 and pn in last_v2[layer] else 0
        svb_val = last_svb[layer][pn].get('eval_r2_sequence_last', 0) if layer in last_svb and pn in last_svb[layer] else 0
        diff_matrix[i, j] = svb_val - v2_val

im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
ax.set_xticks(range(len(all_force))); ax.set_xticklabels(all_force, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(layers))); ax.set_yticklabels(labels, fontsize=8)
ax.set_title('SVB − V2: R² Difference Heatmap (blue = SVB better, red = V2 better)', fontweight='bold')
plt.colorbar(im, ax=ax, label='ΔR²')
plt.tight_layout()
plt.savefig('plots/05_diff_heatmap.png', dpi=150, bbox_inches='tight')
print('Saved: 05_diff_heatmap.png')

print('\nDone! All plots in plots/')
