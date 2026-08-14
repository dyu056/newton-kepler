# Newton-Kepler: Pure Attention 模型实验

## 模型是什么

`model_att.py` 里是**纯 attention 模型**（`GPTPureAttn`），对比标准 GPT 结构，删掉了三样东西：

| 组件 | 标准 GPT | 本模型 |
|------|---------|--------|
| Attention（单头 causal） | ✓ | ✓ |
| MLP（SiLU 前馈层） | ✓ | ✗ 删除 |
| 残差连接（`x = x + f(x)`） | ✓ | ✗ 删除 |
| LayerNorm | ✓ | ✗ 删除 |

**数据流**：`(x, y) → input_embedding (2→128) → +位置编码 → attention → output_head (128→2) → (x', y')`

这样设计是为了让模型与 RG 理论推导（`r2_research/rg_kepler_testbed/`）的假设一致：残差会让有效映射变成 `G + I`，MLP 的 SiLU Jacobian 逐样本变化，LN 使原点不再是梯度流的临界点——都会破坏理论的推导前提。

**初始化**：`sigma_init = (0.1/(3d))^{1/4}`（d=128 时 ≈ 0.127），使初始注意力分数 std ≈ 0.1，保证 softmax 有非平凡梯度且 Taylor 展开有效。

## 任务

开普勒轨道轨迹预测：给定前 T-1 个 (x, y) 坐标，预测下一个坐标。损失为 MSE。

## 训练

```bash
python train.py --config configs/basic.yaml --run-name <name> --gpu 0
```

超参数在 `configs/basic.yaml`（按规范：本地改参数，再上传服务器，不在服务器上改）。

## 观测量

| 观测量 | 频率（yaml 控制） | 说明 |
|--------|------------------|------|
| attention entropy | `observe.attention_entropy.every: 10` | softmax 权重每行熵 / log(可见 key 数)，1.0=均匀，→0=死盯一个位置 |
| attention 矩阵 | `observe.attention_matrix.every: 10` | T×T 矩阵（batch/head 平均），用于灰度 GIF |
| effective rank | probe schedule | 激活矩阵奇异值分布的 participation ratio |
| linear probe R² | probe schedule | 各层表征对力/位置的可线性解码度 |
| loss / test loss | 每 100 步 | MSE |

## 可视化

按《机器学习实验规范》，可视化统一放 `visualization/exp_<参数>_<日期>/`：

```bash
python visualization/exp_pure_attn_lr0.001_2026-08-14/make_attn_gif.py \
    --npz <results.npz> --out attention_evolution.gif
```

## 文件结构

```
train.py              # 训练入口
model_att.py          # 纯 attention 模型（本实验核心）
probe.py              # activation hooks + attention entropy 采集 + linear probes
observe.py            # rank / 奇异值等观测量
data_utils.py         # 数据加载
configs/basic.yaml    # 实验配置
visualization/        # 可视化（按规范每实验一个子路径）
```
