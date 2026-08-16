# exp_pure_attn_lr0.01_2026-08-14

## 实验目的

验证纯 attention 模型（无 MLP/残差/LayerNorm）在开普勒轨道预测任务上
是否真正学到了轨道结构：通过自回归生成和线性 probe 检验模型表征。

## 参数

- 架构：`GPTPureAttn`（model_att.py），n_layer=1, n_head=1, n_embd=128
- block_size=50, lr=0.01, batch=128, n_steps=20000, loss_mask=last
- seed=1, noise_scale=0.1, 优化器 AdamW（weight_decay=0.0，无中途 LR 衰减）

## 关键发现

- 最终 train MSE ≈ 0.021（step 1401 时曾达 0.018，后期因无 LR 衰减小幅回升）
- attention entropy：初始 ~0.997 → 训练中聚焦到 ~0.83
- attention 矩阵演化：见 `attention_evolution.gif`（每 10 步一帧，共 2000 帧）
- 模型权重保存在服务器对应 `.pt` 文件，用 `plot_orbit.py` 可画真实 vs 预测轨道

## 文件

| 文件 | 说明 |
|------|------|
| `plot_orbit.py` | 真实轨道 vs 模型自回归预测对比图 |
| `make_attn_gif.py` | attention 矩阵灰度 GIF 生成脚本 |
| `config.yaml` | 本次实验配置 |
| `training.log` | 训练日志 |
| `block_50_noise_0.1_loss_last_seed_1.npz` | 训练结果（指标、attention 矩阵、entropy） |
| `attention_evolution.gif` | attention 矩阵演化动画 |
