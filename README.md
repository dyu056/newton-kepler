# Newton–Kepler

This is the github repo for the paper ["From Kepler to Newton: Inductive Biases Guide Learned World Models in Transformers"](https://arxiv.org/pdf/2602.06923).

Summary: [Vafa et al.](https://arxiv.org/abs/2507.06952) showed that a transformer can predict planetary motion (e.g., Earth orbiting the Sun) remarkably well, even though its internal representations do not explicitly encode gravitational forces. Does this mean transformers are not valid world models? **No.** In fact, our new paper shows that sometimes the transformer is learning a *Keplerian world model*, rather than a *Newtonian world model*. Even more intriguingly, when we replace full-context attention with *local attention* (i.e., reduce the context length), the transformer begins to behave like a **Newtonian world model**. As the context length is varied, we observe a sharp and fascinating *phase transition* between these two types of world models.

For more discussion about the implications for world models, read this [blog](https://kindxiaoming.github.io/blog/2026/kepler-newton/).

<img width="1060" height="719" alt="intro-plot" src="https://github.com/user-attachments/assets/b3223bb1-3dc2-4744-9278-8acc0eb6da0c" />

## Overview

Experiments and analysis for probing inductive biases of transformers on physics-inspired tasks: **1D sine waves** and **2D Kepler orbits** (Newtonian gravity).

This folder contains:

- **Training scripts** for classification and regression transformers on discretized trajectories.
- **Analysis notebooks** that reproduce figures and analyze training dynamics, spatial representations, scaling, and block-size (context length) effects.
- **Dataset generation** for sine waves and Kepler orbits.

Notebooks expect either pretrained checkpoints (Fig 2) or results from the training scripts; each notebook describes its dependencies in the first cells.

## Structure

| Path | Description |
|------|-------------|
| **fig2_vafa_spatial_map/** | Spatial map analysis of the pretrained transformer by Vafa et al. Loads a pretrained checkpoint and analyzes representations. **Requires checkpoint** (see below). |
| **fig3a_1d_sine_wave_embedding_evolution.ipynb** | 1D sine wave: embedding evolution over training. |
| **fig3b_1d_sine_wave_scaling_law.ipynb** | 1D sine wave: scaling law analysis. Uses results from `sine.py`. |
| **fig4_kepler_noisy_context_learning.ipynb** | Kepler: noisy context learning. |
| **fig5_regression_classification_comparison.ipynb** | Regression vs classification (Kepler). Uses results from `kepler.py` and `kepler_cv.py`. |
| **fig6_block_size_kepler_newton.ipynb** | Block size (context length) experiments for Kepler/Newton. Uses results from `kepler_cv_blocksize.py`. |
| **generate_dataset.py** | Generate and save large sine wave trajectory dataset. |
| **generate_kepler_cv.py** | Generate Kepler orbit trajectories (continuous variables) for training. |
| **sine.py** | Train transformers on 1D sine wave data. Saves to `./results/sine/`. |
| **kepler.py** | Train classification transformers on Kepler data. Saves to `./results/kepler/`. |
| **kepler_cv.py** | Train regression transformers on Kepler (continuous) data. Saves to `./results/kepler_cv/`. |
| **kepler_cv_blocksize.py** | Train with varying block size (context length). Saves to `./results/kepler_cv_blocksize/`. |
| **model.py**, **model_cv.py** | Model definitions (shared by training scripts). |

## Checkpoint (Fig 2)

The **fig2_vafa_spatial_map** analysis loads a pretrained checkpoint. If `./ckpt` or `./ckpt.pt` is missing, the notebook will raise an error and ask you to download it.

1. Download the pretrained checkpoint (e.g. `ckpt.pt`) from [inductivebiasprobe repo](https://github.com/keyonvafa/inductive-bias-probes) or their [Google Drive link](https://drive.google.com/drive/folders/1qfdwQE2LpeZ1shGyrbX5y4xwze-DLbCF).
2. Place the file in **fig2_vafa_spatial_map/** as `./ckpt.pt`.
3. Re-run the notebook.

Run the notebook from **fig2_vafa_spatial_map/** so that `./ckpt.pt` is in the current working directory.

## Quick start

**Small toy models you can quickly train on a CPU**
* `fig3a_*.ipynb` contains a 1D sine-wave example.
* `fig4_*.ipynb` contains a 2D Kepler example. 

## Variance-Covariance Regularization (`ablation_env.py`)

基于 `ablation_v2.py` 的严格探针流水线，额外加入中间表示的**方差-协方差正则化**，防止表示坍缩和特征冗余。

### 动机

神经网络中间层容易出现两种退化：
1. **维度坍缩**：某些特征维度的方差趋于 0，成为"死维度"
2. **特征冗余**：不同维度高度相关，32 维表示实际只有少数有效维度

### 数学形式

在 MSE loss 上追加两项惩罚：

$$L_{\text{total}} = L_{\text{MSE}} + \lambda_v \cdot L_{\text{var}} + \lambda_c \cdot L_{\text{cov}}$$

#### 方差惩罚 $L_{\text{var}}$

对每个 MLP 投影层的输出 $z \in \mathbb{R}^{B \times T \times d}$，计算每维标准差 $\sigma_j$：

$$L_{\text{var}} = \frac{1}{d}\sum_{j=1}^{d} \max(0,\; 1.0 - \sigma_j)$$

鼓励每维标准差至少为 1.0，防止该维度退化。

#### 协方差惩罚 $L_{\text{cov}}$

对协方差矩阵 $C = \frac{1}{n-1} z^T z$，惩罚非对角元素的平方：

$$L_{\text{cov}} = \frac{1}{d(d-1)}\sum_{i \neq j} C_{ij}^2$$

鼓励不同特征维度彼此独立，降低冗余。

### 实现

通过 `RegHooks` 类向每个 Transformer block 的 MLP 输出层（`c_proj`）注册 forward hook，每次前向传播自动截获中间表示并计算惩罚项。训练循环中与 MSE 组装为 `total_loss`，单次 `backward()` 统一反向传播。

### 参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `variance_reg_weight` | 0.0 | $\lambda_v$，方差正则强度 |
| `covariance_reg_weight` | 0.0 | $\lambda_c$，协方差正则强度 |

设两个 $\lambda = 0$ 时等价于 `ablation_v2.py`。

### 实验结果（block_size=100, n_layer=2, λ=0.05）

| Metric | v2（无正则） | env（λ=0.05） |
|--------|------------|---------------|
| test_loss | **0.00165** | 0.00221 |
| \|F\| eval R² | 0.919 | **0.944** |
| Fx eval R² | 0.875 | **0.921** |
| Fy eval R² | 0.645 | **0.820** |

方差-协方差正则化显著改善了力场表示质量（Fy +0.17），代价是预测精度轻微下降。

---

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. The list includes: PyTorch (>=2.0), NumPy, SciPy, Matplotlib, scikit-learn, PyYAML, and optionally tqdm for progress bars. Figure notebooks may use extra plotting options; install any missing packages as prompted.
