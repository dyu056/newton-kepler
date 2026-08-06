# Spring SHM — 用 GPT 学习简谐运动

用一个单层 causal attention Transformer（GPTCV）学习弹簧振子的简谐运动轨迹预测。模型只看到位置序列 `x(t)`，做 next-step prediction——给定前 N 个时刻的位置，预测下一时刻的位置。速度不由数据显式提供，模型必须从位置差分中隐式推断。

核心问题：**一个极简 Transformer 在只看到纯轨迹数据的情况下，能否学到真实的物理动力学？**

---

## 目录

1. [数据](#数据)
2. [模型设计](#模型设计)
3. [文件结构](#文件结构)
4. [快速开始](#快速开始)
5. [配置文件详解](#配置文件详解)
6. [训练管线](#训练管线)
7. [依赖](#依赖)

---

## 数据

### 物理方程

无阻尼简谐运动，只用解析解，不做任何数值积分：

```
x(t) = A · cos(ωt + φ)           —— 位置
```

模型的输入只有 `x(t)` 序列。速度 `v(t) = -Aω·sin(ωt+φ)` 不进入训练数据——模型必须像 Kepler 原始设定一样，从位置差分 `Δx/Δt` 中隐式学习速度概念。

### 自由参数

| 参数 | 符号 | 范围 | 分布 | 含义 |
|------|------|------|------|------|
| 角频率 | ω | [2.2, 6.4] rad/s | 均匀 | 决定振荡快慢 |
| 振幅 | A | [0.10, 0.17] | 均匀 | 决定振荡大小 |
| 初相位 | φ | [0, 2π) | 均匀 | 决定初始位置 |

三个参数**独立随机采样**，每个三元组 `(ω, A, φ)` 唯一确定一条轨迹。

### 时间离散化

```
总帧数: 129
帧率:   20 fps
时长:   6.45 秒
时间步: Δt = 1/20 = 0.05 秒

t_i = i / 20,   i = 0, 1, 2, ..., 128
```

ω 的范围 [2.2, 6.4] 意味着周期范围约 [0.98, 2.86] 秒，每条轨迹包含 2~6 个完整振荡周期。

### 数据格式

每条轨迹保存为 1D 位置序列：

```
shape: (129, 1)
  列 0 = x(t)   — 位置，范围约 [-0.17, 0.17]（归一化坐标）
```

ω 的范围 [2.2, 6.4] 意味着周期范围约 [0.98, 2.86] 秒，每条轨迹包含 2~6 个完整振荡周期。

### 数据产出

```
data_spring/
├── train_trajectories.npy   ← float32, shape = (N, 129, 1)
└── metadata.pt              ← {num_frames, fps, omega_range, input_dim=1, ...}
```

- 所有轨迹存在一个 `.npy` 文件中
- 训练时 `train.py` 自动做 50/50 分割（前半 train，后半 test）
- **无 shortcut 标签**——数据里只有轨迹坐标，没有颜色、频段、类别标记

### 数据生成器

`generate_spring_data.py`，约 110 行，仅依赖 `numpy`。核心逻辑：

```python
t = np.arange(129, dtype=np.float64) / 20.0           # 时间轴
for i in range(N):
    omega = rng.uniform(2.2, 6.4)
    amp   = rng.uniform(0.10, 0.17)
    phase = rng.uniform(0.0, 2 * np.pi)
    data[i, :, 0] = amp * np.cos(omega * t + phase)       # x — 仅此一列
```

---

## 模型设计

### GPTCV：连续变量 GPT

定义在 `model_cv.py`，是一个 decoder-only Transformer，关键区别在于输入/输出是连续值而非离散 token。

```
输入张量: (batch_size, seq_len, 1)     ← 位置序列 x(t)
                    │
    ┌───────────────┴───────────────┐
    │  Linear(1 → n_embd)           │  ← 可学习的连续值嵌入
    │  + Position Embedding         │  ← 标准 learned positional encoding
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │  Transformer Block × n_layer  │
    │  ┌─────────────────────────┐  │
    │  │ LayerNorm               │  │
    │  │ CausalSelfAttention     │  │  ← 因果掩码：token_i 只能看到 token_{0..i}
    │  │   · Q, K, V 投影        │  │
    │  │   · Flash Attention     │  │
    │  │   · optional: α 缩放    │  │     attn = attn * (log pos)^α
    │  ├─────────────────────────┤  │
    │  │ LayerNorm               │  │
    │  │ MLP                     │  │
    │  │   · Linear(embd → 4×embd)│  │
    │  │   · SiLU 激活            │  │
    │  │   · Linear(4×embd → embd)│  │
    │  │   · optional: var/cov   │  │     正则化惩罚
    │  └─────────────────────────┘  │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │  LayerNorm (final)            │
    │  Linear(n_embd → 1)           │  ← 投影回位置空间
    └───────────────┬───────────────┘
                    │
输出: (batch_size, seq_len, 1)     ← 预测的下一时刻 x̂

损失: MSE(prediction, target)
```

### 关键设计选择

| 设计 | 选择 | 原因 |
|------|------|------|
| 输入维度 | 1 `(x)` | 仅位置，模型从差分 Δx/Δt 隐式推断速度 |
| 嵌入方式 | `Linear(1→embd)` | 连续值不需要 token embedding table |
| 位置编码 | learned | GPT-2 风格，模型可以学习 SHM 的时间结构 |
| 注意力掩码 | causal | 强制 next-step prediction，不偷看未来 |
| 激活函数 | SiLU | 连续回归任务中比 GELU 更平滑 |
| 输出层 | `Linear(embd→1)` | 直接回归，不量化 |
| 默认配置 | `n_layer=1, n_head=1, n_embd=128, input_dim=1` | 最小可学习单元 |

### 为什么是 1 层 1 头

`n_layer=1, n_head=1` 是最简 Transformer。单层 causal attention 在数学上等价于：

```
预测(x_{t+1}) = Σ_{i=0}^{t} w_i · x_i
```

其中权重 `w_i` 由 query-key 相似度 + softmax 决定。模型只能用**一个加权平均**来预测下一步——它无法叠加非线性层来构造高阶近似。这个极简设计让模型的归纳偏置完全暴露：

- 如果学到的 `w_i` 集中在最近几步 → 模型靠局部有限差分外推（隐式学会了速度）
- 如果学到的 `w_i` 覆盖整个历史 → 模型靠全局模式匹配（记住了完整周期）

---

## 文件结构

```
newton-kepler/
│
├── generate_spring_data.py   ← 数据生成（117行，纯numpy）
├── train.py                  ← 训练入口（支持多block_size扫参）
│
├── model_cv.py               ← GPTCV 模型定义
├── data_utils.py             ← 数据加载（Kepler/Spring双后端自动检测）
├── loss.py                   ← MSE loss + var/cov 正则化
├── observe.py                ← 运行时诊断（秩、熵、奇异值、rollout）
├── probe.py                  ← 线性探针（从中间表示解码物理量）
│
├── configs/
│   └── spring.yaml           ← ★ 唯一参数文件
│
├── requirements.txt          ← torch, numpy, scipy, matplotlib, sklearn, pyyaml, tqdm
└── README.md
```

**各文件职责**：

| 文件 | 被谁调用 | 职责 |
|------|---------|------|
| `generate_spring_data.py` | 用户直接运行 | 读 YAML → 生成 `.npy` 轨迹 |
| `train.py` | 用户直接运行 | 读 YAML → 加载数据 → 训练 → 保存结果 |
| `model_cv.py` | `train.py` | `GPTCV` 类定义，前向传播 |
| `data_utils.py` | `train.py` | `load_trajectories()` 自动检测数据格式 |
| `loss.py` | `train.py`, `model_cv.py` | MSE + var/cov 惩罚 |
| `observe.py` | `train.py` | 训练过程中采集秩、熵、rollout 误差 |
| `probe.py` | `train.py` | 冻结模型 → 从中间层训练线性探针 |

---

## 快速开始

### 1. 安装

```bash
cd newton-kepler
pip install -r requirements.txt
```

### 2. 生成数据

```bash
python generate_spring_data.py --config configs/spring.yaml
```

### 3. 训练

```bash
python train.py --config configs/spring.yaml --run-name spring_test --gpu 0
```

用 CPU 训练的话把 `--gpu 0` 换成 `--gpu cpu`。

### 4. 查看结果

```bash
ls outputs/spring_test/results/
# block_2_noise_0.1_loss_all_seed_3407.npz
# block_5_noise_0.1_loss_all_seed_3407.npz
# ...
```

每个 `.npz` 包含：`train_losses`, `test_losses`, `eval_results`（探针R²、激活秩、注意力熵、rollout误差等）。

### 调试模式

改 `configs/spring.yaml` 把数字缩小，几分钟跑完：

```yaml
generation:
  total_trajectories: 200

data:
  num_trajectories: 100

training:
  n_steps: 200
  block_sizes: [20, 50]
```

然后重新生成数据（加 `--overwrite` 覆盖旧数据）：

```bash
python generate_spring_data.py --config configs/spring.yaml --overwrite
python train.py --config configs/spring.yaml --run-name debug_test --gpu 0
```

---

## 配置文件详解

`configs/spring.yaml` 是**唯一参数来源**——数据生成和训练共享同一份配置。

### generation — 数据生成

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `total_trajectories` | 20000 | 总共生成多少条轨迹 |
| `num_sample_videos` | 0 | 保留字段（当前版本不生成视频） |
| `fps` | 20 | 帧率 |
| `num_frames` | 129 | 每条轨迹的时间步数 |
| `omega_low / omega_high` | 2.2 / 6.4 | 角频率范围 (rad/s) |
| `amplitude_low / amplitude_high` | 0.10 / 0.17 | 振幅范围（归一化坐标） |

### data — 训练时数据加载

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `data_dir` | data_spring | 数据目录 |
| `num_trajectories` | 10000 | 每个 split 的轨迹数（train/test各10000） |

**一致性规则**：`total_trajectories >= 2 × num_trajectories`。不满足则在生成阶段直接报错。

### model — 模型架构

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `n_layer` | 1 | Transformer 层数 |
| `n_head` | 1 | 每层注意力头数 |
| `n_embd` | 128 | 隐藏维度 |
| `input_dim` | 1 | 输入维度（1=仅位置，和 Kepler 的 2D 坐标对应） |

### training — 训练超参

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `block_sizes` | [2,5,20,50,100] | 待扫的上下文长度列表 |
| `noise_scale` | 0.1 | 输入噪声标准差 |
| `learning_rate` | 0.001 | 学习率 |
| `weight_decay` | 0.0 | 权重衰减 |
| `batch_size` | 128 | 批大小（GD模式下为全批量） |
| `n_steps` | 2000 | 每轮训练步数 |
| `loss_mask` | all | `all`=所有token，`last`=仅最后一步 |
| `optimizer` | gd | `gd`=全批量SGD，`adamw`=AdamW |

### probe — 线性探针

训练过程中，定期冻结模型权重，从中间层激活训练线性回归器来预测位置/速度。R² 分数衡量模型表示中编码了多少物理信息。

### observe — 可观测指标

- `activation_rank`：激活矩阵的有效秩
- `attention_entropy`：注意力分布的归一化熵
- `variance_covariance`：MLP 隐藏层的方差/协方差统计
- `singular_values`：权重矩阵和激活矩阵的奇异值

这些指标被记录但不参与训练损失。

---

## 训练管线

`train.py` 加载 YAML 后，对 `block_sizes` 中的每个值独立训练一轮：

```
for block_size in [2, 5, 20, 50, 100]:

    1. 加载数据: load_trajectories(data_dir, need=2×num_trajectories)
       ├── 检测 data_spring/train_trajectories.npy → Spring 格式
       └── 50/50 切分: train / test

    2. 如果 block_size < num_frames:
       └── chop_trajectories_into_sequences()
           将每条 129 帧轨迹切成多个 block_size 长的片段

    3. 初始化模型: GPTCV(block_size, n_layer=1, n_head=1, n_embd=128, input_dim=1)

    4. 训练循环 (n_steps 步):
       ├── 前向: 输入 (B, block_size, 1) → 预测下一个 x
       ├── 损失: MSE(预测, 真实)
       ├── 反向传播 + 优化器步进
       └── 定期评估:
            ├── test loss
            ├── 线性探针 (R² of position decoding)
            ├── 激活秩、注意力熵
            ├── 权重/激活奇异值
            └── rollout 误差 (自回归生成64步，对比真实轨迹)

    5. 保存结果 → outputs/<run_name>/results/block_{N}_noise_0.1_loss_all_seed_3407.npz
```

### block_size 实验的意义

`block_size` 是模型能看到的历史长度：

| block_size | 物理含义 | 预期行为 |
|-----------|---------|---------|
| 2 | 只看到上一步 | 必须从两步差分 Δx/Δt 外推——纯局部动力学 |
| 5 | 0.25 秒历史 | 大约 1/4 个周期 |
| 20 | 1 秒历史 | 约半个到 1 个周期 |
| 50 | 2.5 秒历史 | 1~2 个完整周期 |
| 100 | 5 秒历史 | 几乎整条轨迹 |

Kepler 论文的核心发现是：**短上下文时模型学牛顿力学（局部外推），长上下文时学开普勒力学（全局模式匹配）**。Spring SHM 为这个问题提供了一个更干净的测试平台——因为真实动力学就是线性的，任何偏离都来自模型的归纳偏置。

---

## 依赖

```
torch>=2.0
numpy
scipy
matplotlib
scikit-learn
pyyaml
tqdm
```

全部是标准科学计算栈，无需 GPU 专用库之外的任何东西。
