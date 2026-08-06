# Spring SHM — 用 GPT 学习简谐运动

用一个单层 causal attention Transformer（GPTCV）学习弹簧振子的简谐运动（SHM）轨迹预测。模型在相空间 `(x, v)` 中做 next-step prediction，等价于学习一个 2D 动力学系统的离散时间演化映射。

---

## 模型设计

### GPTCV：连续变量 GPT

模型定义在 `model_cv.py`，是一个标准的 decoder-only Transformer，专门适配连续值输入/输出：

```
输入: (batch, seq_len, 2)     ← 2D 相空间坐标 (x, v)
  │
  ├── Linear(input_dim=2 → n_embd=128)    ← 可学习的连续值嵌入
  ├── + Position Embedding (learned)
  │
  ├── Block × n_layer                       ← Transformer 层
  │     ├── LayerNorm
  │     ├── CausalSelfAttention             ← 因果掩码，只看过去
  │     │     └── optional: α 位置缩放 (log N)^α
  │     ├── LayerNorm
  │     └── MLP (4× expansion, SiLU)
  │           └── optional: var/cov 正则化
  │
  ├── LayerNorm (final)
  └── Linear(n_embd → 2)                   ← 输出下一时刻的 (x, v)

损失: MSE(prediction, target)     ← 对所有 token 或仅最后一个
```

**关键设计选择**：

| 设计 | 选择 | 原因 |
|------|------|------|
| 输入维度 | 2（x 和 v） | 相空间 (x,v) 完整描述 SHM 状态；与 Kepler (x,y) 数学同构 |
| 位置编码 | learned embedding | 标准 GPT-2 风格，而非 RoPE |
| 注意力掩码 | causal（只看过去） | 强制模型做 next-step 预测，不能偷看未来 |
| 激活函数 | SiLU | 比 GELU 在连续回归任务上更平滑 |
| 输出 | 直接回归连续值 | 无需离散化/分桶，保留完整精度 |

### 为什么相空间 (x, v) 能工作

简谐运动 `x(t) = A·cos(ωt + φ)` 在相空间中的轨迹是一个**椭圆**：

```
v(t) = -Aω·sin(ωt + φ)

(x/A)² + (v/(Aω))² = 1    ← 椭圆方程
```

这和 Kepler 问题的椭圆轨道在数学上完全一致——Kepler 模型学的是平面椭圆，Spring 模型学的也是平面椭圆。因此 `input_dim=2` 的 GPTCV 架构可以零改动地复用。

### 1 层 attention 的用意

`n_layer=1, n_head=1` 是最简配置。单层 causal attention 等价于对过去轨迹做加权平均来预测下一步——模型无法叠加多层非线性变换，只能学到一个"最优加权窗口"。这个极简设计让模型的归纳偏置（inductive bias）暴露得最清楚：它到底学会了真正的 SHM 动力学，还是只是记住了训练数据的统计模式？

---

## 数据

### 物理模型

无阻尼简谐运动，解析解（不做数值积分）：

```
x(t) = A · cos(ωt + φ)
v(t) = -Aω · sin(ωt + φ)
```

### 采样空间

| 参数 | 范围 | 分布 |
|------|------|------|
| 角频率 ω | [2.2, 6.4] rad/s | 均匀 |
| 振幅 A | [0.10, 0.17] | 均匀 |
| 初相位 φ | [0, 2π) | 均匀 |

每个 `(ω, A, φ)` 三元组唯一确定一条 129 帧（20 fps，6.45 秒）的轨迹。

### 数据格式

```
train_trajectories.npy  →  float32 数组, shape = (N, 129, 2)
                           列 0 = x(t)  — 水平位置
                           列 1 = v(t)  — 瞬时速度
```

- 所有轨迹拼接为一个 `.npy` 文件
- 训练时 `train.py` 自动做 50/50 随机分割
- **不含任何 shortcut 标签**（无颜色、无频段标记）——模型只能从轨迹本身学动力学

### 样本视频

数据生成时同步渲染 `.mp4` 样本视频（`data_spring/sample_videos/`），包含弹簧、质量块、锚点的完整可视化。颜色（红/蓝）仅用于区分不同样本，不进入训练数据。

---

## 快速开始

### 安装

```bash
pip install -r requirements.txt
pip install -r spring_shortcuts/requirements.txt
```

### 第一步：生成数据

```bash
python generate_spring_data.py --config configs/spring.yaml
```

所有参数（轨迹数量、频率范围、振幅范围、帧率等）均由 `configs/spring.yaml` 中的 `generation` 段控制。生成的数据存到 `data_spring/`。

### 第二步：训练

```bash
python train.py --config configs/spring.yaml --run-name spring_test --gpu 0
```

训练会遍历 `block_sizes` 中配置的每个上下文长度，每个跑 `n_steps` 步。结果（loss 曲线、探针诊断、rollout 误差等）存到 `outputs/spring_test/results/`。

### 调试模式

```yaml
# 在 configs/spring.yaml 中改小数字：
generation:
  total_trajectories: 200     # 只用 200 条

data:
  num_trajectories: 100       # train=100, test=100

training:
  n_steps: 200                # 只跑 200 步
  block_sizes: [20, 50]       # 只扫两个 block size
```

然后重新生成数据 + 训练，几分钟就能跑完一轮。

---

## 配置文件

`configs/spring.yaml` 是整个项目的**唯一参数来源**。数据生成和训练都读同一个文件，改一个数字两边自动同步。

```yaml
generation:          # 数据生成参数
  total_trajectories: 20000
  omega_low: 2.2
  omega_high: 6.4
  ...

data:                # 训练时的数据加载参数
  data_dir: data_spring
  num_trajectories: 10000   # per-split，训练加载 2× 这个数

model:               # 模型架构
  n_layer: 1
  n_head: 1
  n_embd: 128

training:            # 训练超参
  block_sizes: [2, 5, 20, 50, 100]
  learning_rate: 0.001
  ...

probe:               # 线性探针配置
observe:             # 可观测指标（rank、熵、奇异值等）
regularization:      # 正则化项
evaluation:          # rollout 评估
```

**一致性保证**：数据生成时自动校验 `generation.total_trajectories >= 2 × data.num_trajectories`，不满足直接报错。

---

## 文件结构

```
newton-kepler/
├── train.py                     ← 训练入口
├── model_cv.py                  ← GPTCV 模型定义（不改）
├── data_utils.py                ← 数据加载（Kepler + Spring 双后端）
├── loss.py                      ← MSE loss + var/cov 正则化
├── observe.py                   ← 运行时诊断（rank、熵、rollout）
├── probe.py                     ← 线性探针
├── generate_spring_data.py      ← Spring 数据生成
├── configs/
│   └── spring.yaml              ← ★ 唯一参数文件
├── spring_shortcuts/            ← SHM 物理引擎 + 视频渲染
│   ├── simulation.py            ← 核心：轨迹生成、渲染、检测
│   ├── build.py                 ← 独立视频数据集生成（可选）
│   ├── data.yaml                ← 视频生成默认配置
│   ├── requirements.txt
│   └── README.md
├── requirements.txt
└── README.md
```

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
opencv-python       ← 仅视频质量检测需要
imageio             ← 仅视频渲染需要
imageio-ffmpeg      ← 仅视频编码需要
Pillow              ← 仅视频渲染需要
```
