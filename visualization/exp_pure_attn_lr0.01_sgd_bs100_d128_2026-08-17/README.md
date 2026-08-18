# exp_pure_attn_lr0.01_sgd_bs100_d128_2026-08-17

## 实验目的

纯 attention 模型（无 MLP/残差/LN）在开普勒轨道预测上的自回归 rollout 能力检验。
训练用逐位置监督（loss_mask: all，切段 target 错位完整序列）+ SGD 全批次。

## 参数

- 架构：GPTPureAttn，n_layer=1, n_head=1, n_embd=128
- block_size=100（attention 矩阵 100×100）, lr=0.01, SGD 全批次（无 weight decay）
- n_steps=25000, loss_mask=all, seed=1, noise_scale=0.1
- attention 熵/矩阵每 20 步采样；probe R² 关闭；SVD 走 CPU（GPU 0 显存损坏规避）

## 关键发现

- 最终 train loss 0.0891 / test loss 0.0803
- rollout（前 20 点条件 → 生成 80 点）：MSE 0.143, R² -0.165
- 对比：同配置早期运行 R² -0.495 → 完整 25000 步训练后 -0.165（训练时长仍关键）
- attention entropy 最终 ~0.59（中度聚焦）

## 文件

| 文件 | 说明 |
|------|------|
| `orbit_r2.png` | 真实轨道 vs 模型自回归预测 |
| `training_curves.png` | loss + attention entropy 曲线 |
| `attention_evolution.gif` | attention 矩阵演化（每 20 步一帧） |
| `plot_orbit.py` / `plot_training_curves.py` / `make_attn_gif.py` | 生成脚本 |
| `config.yaml` / `training.log` | 配置与日志 |
| `.npz` / `.pt` | 结果数据与模型权重 |
