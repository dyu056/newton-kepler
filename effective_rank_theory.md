# 训练初期有效秩下降的严格理论

> 对象：`newton-kepler` 的 `model_cv.py` / `train.py`，1 层 Attention+MLP，连续 (x,y) 轨道预测。
> 主要数据：`results_gd_sweep/block_100_noise_0.1_loss_all_seed_1.npz`（block_size=100，全批 GD）。
> 数值验证：`verify_rank_theory.py`、`reproduce_rank_dip.py`（可直接运行复现本文每一条断言）。

---

## 0. 与前一版文档的分歧

`r2_research/attention_mlp_theory_rigorous.md` 的推导链条是：

$$\frac{\partial\mathcal{L}}{\partial\hat{\mathbf{y}}} = \frac{2}{BT}(\hat{\mathbf{y}} - \mathbf{y})
\;\xrightarrow{\;\hat{\mathbf{y}}\approx 0\;}\; O(1) \;\Longrightarrow\; \text{秩坍缩}$$

需要区分两个问题。

**偏导里出现 $\mathbf{y}$ 本身没有错。** $\mathcal{L}(\hat{\mathbf{y}};\mathbf{y})$ 是 $\hat{\mathbf{y}}$ 与 $\mathbf{y}$ 的二元函数，对第一个变量求偏导，第二个变量当然会留下来。任何监督学习的梯度都必须含标签，否则无法携带训练信号。

**真正的错误在下一步。** 它把 $\hat{\mathbf{y}}\approx 0$ 代入，得到"梯度量级 $O(1)$"，再由此论证秩坍缩。这是无效的：

1. **量级与秩无关。** 把数据整体乘 $10^{-3}$，所有梯度量级变成 $O(10^{-3})$，但每个梯度矩阵的秩一个也不会变。用量级论证秩，是范畴错误。
2. **该量级判断依赖数据的数值范围**（文档自己也承认"如果坐标被缩放到 [0,0.01]，结论就不同"）。一个依赖坐标单位的论证，不可能解释一个与单位无关的现象。
3. **它解释错了对象。** 该文档的核心证据是"所有权重有效秩在 step 5 暴跌然后 2000 步不变"。这个"现象"不存在——见 §1。

本文重写为：秩的结论**只**来自 Jacobian 的因子分解结构，与梯度量级、数据单位、初始化尺度全部无关。

---

## 1. 先修正事实：被解释的现象是什么

### 1.1 "权重有效秩暴跌后冻结"是记录假象

`results_hyt/*/results/*.npz` 中，`eval_results[i]['weight_singular_values']` 在**所有** eval step 上是同一个 Python 对象：

```
results_hyt/observe   steps 5->1000  identical=True  maxdiff=0.000e+00
   same dict object across evals? True | same inner list? True
```

同期 loss 从 0.597 降到 0.0057。谱逐位不变而 loss 降两个数量级，只能是旧版 `train.py` 存了引用而非快照。`results_gd_sweep/*` 不受影响（`identical=False`），当前 `train.py:620-625` 每次 eval 重算，代码侧已修好。

### 1.2 权重谱其实几乎不动

用未受污染的 bs=100 run，权重奇异值 2000 步的真实变化：

| 权重 | step 0 (top-1) | step 2000 (top-1) | 相对变化 |
|------|----------------|-------------------|----------|
| `mlp.c_fc` | 0.67365 | 0.67379 | +0.02% |
| `attn.c_attn` | 0.60160 | 0.60167 | +0.01% |
| `mlp.c_proj` | 0.47764 | 0.47871 | +0.22% |
| `output_head` | 0.24578 | 0.31768 | **+29%** |
| `input_embedding` | 0.23683 | 0.21627 | −8.7% |

只有直接接触 2 维信号的 `output_head` / `input_embedding` 明显移动，内部大矩阵基本静止。所以"权重凝聚"不是本任务的现象。

### 1.3 真实现象：激活有效秩的浅 V 型

bs=100，全批 GD，未截断谱（`activation_rank`，注意 `activation_singular_values` 被 `num_singular_values: 5` 截断，不能用于秩）：

| 层 | step 5 | 谷底 | 峰值 | step 2000 | 降幅 |
|----|--------|------|------|-----------|------|
| `input_embed` | 1.98 | — | — | 1.98 | 0% |
| `after_pos_emb` | 10.88 | 10.78 @20 | 11.71 @185 | 10.79 | 0.9% |
| `block_0_attn_output` | 2.73 | — | 3.08 @500 | 3.05 | 0% |
| `block_0_after_attn_merge` | 16.40 | 16.37 @10 | 18.75 @135 | 16.39 | 0.2% |
| `block_0_mlp_output` | 14.93 | **13.78 @20** | 17.29 @185 | 15.76 | **7.7%** |
| `block_0_after_mlp_merge` | 12.77 | **12.22 @15** | 19.71 @2000 | 19.71 | **4.3%** |
| `after_ln_f` | 15.38 | **14.75 @15** | 18.92 @2000 | 18.92 | **4.1%** |

需要解释的是：**残差流与 MLP 输出的有效秩在前 15–20 步下降约 4–8%，随后回升并超过初值。** 幅度不大，位置固定，只出现在 MLP 之后的层。

### 1.4 关键约束：没有任何方向消失

同一批数据的数值秩（阈值 $10^{-6}\sigma_{\max}$）在全部 59 个 eval step 上取值集合为单元素：

```
after_pos_emb            {100}
block_0_after_attn_merge {127}
block_0_mlp_hidden       {512}
block_0_mlp_output       {128}
after_ln_f               {127}
```

**这一条否决了所有"秩坍缩 / 方向被压缩掉"的解释。** 任何正确的理论必须解释一个参与率下降、而张成空间维数严格不变的过程。

---

## 2. 符号与设定

严格对应代码。

| 符号 | 含义 | 代码位置 |
|------|------|----------|
| $d=128$ | 残差流宽度 | `n_embd`（config `basic.yaml`） |
| $d_{\text{ff}}=512$ | MLP 隐层宽度 $4d$ | `MLP.__init__`，`model_cv.py:151` |
| $p=2$ | 输入/输出维数 | `input_dim`，坐标 $(x,y)$ |
| $T$ | 序列长度 | `block_size`（主用 100） |
| $B$ | batch 中序列数 | 全批 GD 时为全训练集 |
| $N=BT$ | 位置样本总数 | — |
| $\mathbf{W}_E\in\mathbb{R}^{d\times p}$ | 输入嵌入 | `input_embedding`，`model_cv.py:229` |
| $\mathbf{W}_U\in\mathbb{R}^{p\times d}$ | 输出头 | `output_head`，`model_cv.py:231` |
| $\mathbf{P}\in\mathbb{R}^{T\times d}$ | 位置嵌入 | `transformer.wpe`，`model_cv.py:220` |
| $\varepsilon=0.02$ | 初始化标准差 | `_init_weights`，`model_cv.py:255` |
| $\mathbf{H}\in\mathbb{R}^{N\times d}$ | 某层激活（行为样本） | `probe.py` hook |

前向（`model_cv.py:283-352`，单层）：

$$\mathbf{Z}_0 = \mathbf{X}\mathbf{W}_E^\top + \mathbf{1}\mathbf{P}, \qquad
\mathbf{Z}_1 = \mathbf{Z}_0 + \mathrm{Attn}(\mathrm{LN}_1(\mathbf{Z}_0)),$$
$$\mathbf{Z}_2 = \mathbf{Z}_1 + \mathrm{MLP}(\mathrm{LN}_2(\mathbf{Z}_1)), \qquad
\hat{\mathbf{Y}} = \mathrm{LN}_f(\mathbf{Z}_2)\,\mathbf{W}_U^\top.$$

损失（`loss.py:31`，`loss_mask='all'`）：$\mathcal{L}=\frac{1}{Np}\sum_{n,j}(\hat{Y}_{nj}-Y_{nj})^2$。

**有效秩定义**（`observe.py:217-230`，注意先中心化再 SVD）：对奇异值 $\sigma_1\ge\cdots\ge\sigma_R$，令 $q_i=\sigma_i^2/\sum_k\sigma_k^2$，

$$\operatorname{erank} = \exp\Big(-\sum_i q_i\log q_i\Big).$$

这是**谱熵的指数**，即参与率。两条性质本文反复用到：

- **P1（尺度不变）** 对任意 $c\ne 0$，$\operatorname{erank}(c\mathbf{H})=\operatorname{erank}(\mathbf{H})$。故有效秩的任何变化都**不可能**由整体幅度变化解释，只能由谱**形状**变化解释。
- **P2（对能量集中单调递减）** 若把能量从小奇异方向搬到大奇异方向（majorization 意义下 $q$ 变得更集中），$\operatorname{erank}$ 严格下降。$\operatorname{erank}$ 与数值秩无任何函数关系：谱 $(1,\delta,\dots,\delta)$ 的数值秩是 $R$，而 $\delta\to0$ 时有效秩 $\to 1$。

P1 立刻说明：§0 那种基于梯度量级的论证，对本文要解释的量原则上无解释力。

---

## 3. 核心引理：梯度的秩由输出瓶颈锁定

这一节是全文唯一的秩来源，纯代数，不涉及任何量级、初始化或数据分布假设。

### 3.1 反传信号穿过 $\mathbf{W}_U$ 后落入 $\le p$ 维

记 $\mathbf{G}=\partial\mathcal{L}/\partial\hat{\mathbf{Y}}\in\mathbb{R}^{N\times p}$。对 MSE，$\mathbf{G}=\frac{2}{Np}(\hat{\mathbf{Y}}-\mathbf{Y})$——**这里出现 $\mathbf{Y}$ 是正确且必要的**，它就是训练信号的载体。下面的论证完全不需要知道 $\mathbf{G}$ 的数值，只需要它的**形状** $N\times p$。

设 $\mathbf{H}_f=\mathrm{LN}_f(\mathbf{Z}_2)$，则 $\hat{\mathbf{Y}}=\mathbf{H}_f\mathbf{W}_U^\top$，于是

$$\frac{\partial\mathcal{L}}{\partial\mathbf{H}_f} = \mathbf{G}\,\mathbf{W}_U \in\mathbb{R}^{N\times d}.$$

**引理 1.** $\operatorname{rank}\!\big(\partial\mathcal{L}/\partial\mathbf{H}_f\big)\le p=2$，且其行空间 $\subseteq\operatorname{row}(\mathbf{W}_U)$，与 $\mathbf{G}$ 的取值无关。

*证明.* $\operatorname{rank}(\mathbf{G}\mathbf{W}_U)\le\min\{\operatorname{rank}\mathbf{G},\operatorname{rank}\mathbf{W}_U\}\le\min\{p,p\}=p$。每一行 $(\mathbf{G}\mathbf{W}_U)_n=\mathbf{G}_n\mathbf{W}_U$ 是 $\mathbf{W}_U$ 的 $p$ 个行向量的线性组合。∎

关键在于 $\mathbf{W}_U\in\mathbb{R}^{2\times128}$ 的行空间只有 2 维：**无论上游网络多宽、多深，回传到残差流的误差信号在每个位置都被钉在同一个 2 维子空间里。** 这是架构的硬约束，不是近似。

### 3.2 逐元素算子与残差加法不提升秩

反传继续向下，穿过 $\mathrm{LN}_f$、残差分叉、$\mathrm{MLP}$、$\mathrm{Attn}$。需要一个逐层的秩守恒陈述。

**引理 2.** 设 $\boldsymbol{\Delta}\in\mathbb{R}^{N\times d}$ 满足 $\operatorname{rank}\boldsymbol{\Delta}\le r$。则

1. （线性映射）对任意 $\mathbf{M}$，$\operatorname{rank}(\boldsymbol{\Delta}\mathbf{M})\le r$；
2. （残差相加）若 $\boldsymbol{\Delta}',\boldsymbol{\Delta}''$ 秩分别 $\le r',r''$，则 $\operatorname{rank}(\boldsymbol{\Delta}'+\boldsymbol{\Delta}'')\le r'+r''$；
3. （逐元素乘 / 行内混合）逐元素乘一个一般矩阵 $\mathbf{S}$（如 $\mathrm{SiLU}'$）满足 $\operatorname{rank}(\mathbf{S}\odot\boldsymbol{\Delta})\le r\cdot\operatorname{rank}(\mathbf{S})$，且当 $\mathbf{S}$ 逐行为常数时秩 $\le r$。

*证明.* (1) 标准。(2) 秩的次可加性。(3) Hadamard 积的秩满足 $\operatorname{rank}(\mathbf{S}\odot\boldsymbol{\Delta})\le\operatorname{rank}(\mathbf{S})\operatorname{rank}(\boldsymbol{\Delta})$；行内常数时 $\mathbf{S}\odot\boldsymbol{\Delta}=\operatorname{diag}(\mathbf{s})\boldsymbol{\Delta}$，退化为 (1)。∎

(3) 是唯一的秩放大来源：$\mathrm{SiLU}'$ 与 $\mathrm{LN}$ 的 Jacobian 都逐样本变化，严格上界会随层数相乘。所以理论只能给出"$\ge 2$、且被非线性有限放大"——精确值必须实测。这正是 §4 要做的。

### 3.3 权重梯度是外积和

对任意线性层 $\mathbf{U}=\mathbf{V}\mathbf{W}^\top$（$\mathbf{V}$ 为输入激活，$\boldsymbol{\Delta}=\partial\mathcal{L}/\partial\mathbf{U}$）：

$$\frac{\partial\mathcal{L}}{\partial\mathbf{W}} = \boldsymbol{\Delta}^\top\mathbf{V}
= \sum_{n=1}^{N}\boldsymbol{\delta}_n\mathbf{v}_n^\top .$$

**引理 3.** $\operatorname{rank}(\partial\mathcal{L}/\partial\mathbf{W})\le\min\{\operatorname{rank}\boldsymbol{\Delta},\ \operatorname{rank}\mathbf{V}\}$。

于是 $N$ 个秩-1 外积求和后，秩不由 $N$ 决定，而由两个因子的秩决定。结合引理 1–2：**所有权重梯度的秩被 $p=2$ 与非线性放大因子共同压住，与 $B$、$T$、$N$ 全部无关。**

### 3.4 数值验证

`verify_rank_theory.py` 在真实模型的真实初始化上直接对 $\mathcal{L}$ 求 autograd 梯度并测秩：

```
input_embedding.weight        shape=(128, 2)   num_rank=  2  eff_rank= 2.00
output_head.weight            shape=(2, 128)   num_rank=  2  eff_rank= 1.97
transformer.h.0.attn.c_attn   shape=(384,128)  num_rank=128  eff_rank= 2.00
transformer.h.0.attn.c_proj   shape=(128,128)  num_rank=124  eff_rank= 1.97
transformer.h.0.mlp.c_fc      shape=(512,128)  num_rank=128  eff_rank= 2.01
transformer.h.0.mlp.c_proj    shape=(128,512)  num_rank=128  eff_rank= 2.00
```

**每一个内部权重梯度的有效秩都是 2.00±0.03。** 数值秩满（128）说明非线性确实把信号洒到了所有方向，但能量集中度精确对应 2 个有效方向——引理 1 的瓶颈以有效秩的形式精确显现。这比"$\le 4\text{–}5$"的旧估计更强，且不含任何拟合参数。

---

## 4. 有效秩下降的机制：两时标谱重加权

现在把引理 1–3 接到观测量上。

### 4.1 初始谱的两段结构

$t=0$ 时残差流 $\mathbf{Z}_0=\mathbf{X}\mathbf{W}_E^\top+\mathbf{1}\mathbf{P}$ 是两项之和：

- **信号段**：$\mathbf{X}\mathbf{W}_E^\top$ 的秩恰为 2（$\mathbf{X}$ 的列空间 2 维），承载全部输入信息；
- **位置段**：$\mathbf{1}\mathbf{P}$ 贡献 $\min(T,d)$ 个方向，元素 $\sim\mathcal{N}(0,\varepsilon^2)$，彼此近正交、量级相当。

实测（`verify_rank_theory.py`，bs=100 初始化）：

```
input_embed     eff_rank= 1.98  num_rank=  2   top5=[20.13, 17.67, 0, 0, 0]
after_pos_emb   eff_rank=10.60  num_rank=101   top5=[20.19, 17.73, 3.24, 3.16, 3.08]
```

谱形状是 **2 大 + 一条长而平的尾巴**：$\sigma_{1,2}\approx 20,18$；$\sigma_{3..101}\approx 3.2$ 缓降。有效秩 10.6 就是这个两段谱的参与率——它既不是 2（信号维数），也不是 101（数值秩），而是两段能量比的函数。

### 4.2 信号段与尾段的演化速率不同

由 §3：训练信号在**每一步**都是秩 2 的（有效秩意义上）。梯度下降对残差流的影响因此分成两类：

- 信号段 $\sigma_1,\sigma_2$ 沿 $\operatorname{row}(\mathbf{W}_U)$ 方向被系统性驱动，逐步累积；
- 位置段 $\sigma_3,\dots$ 只受二阶效应（$\mathbf{W}_E,\mathbf{P}$ 的间接更新）扰动，无一阶驱动。

设 $E_2=(\sigma_1^2+\sigma_2^2)/\sum_k\sigma_k^2$。信号段增长而尾段近静止 $\Rightarrow$ $E_2$ 上升 $\Rightarrow$ 由 P2，$\operatorname{erank}$ **下降**。这正是观测到的浅 V 型左支。

bs=100 记录数据（`after_ln_f`）：

| step | erank | $\sigma_1$ | $\sigma_3$ | $\sigma_1/\sigma_3$ |
|------|-------|-----------|-----------|--------------------|
| 5 | 15.38 | 4361.6 | 2103.7 | 2.073 |
| 15 | **14.75** | 4380.7 | 2045.1 | **2.142** |
| 20 | 14.77 | 4370.0 | 2041.8 | 2.140 |
| 45 | 15.77 | 4239.4 | 2102.2 | 1.952 |
| 105 | 17.84 | 3805.8 | 2113.5 | 1.756 |

**谷底（step 15）与 $\sigma_1/\sigma_3$ 的峰值（2.142）逐点对齐**，两者相关性在三层上一致。有效秩不是被"压缩"，是被前两维**挤**下去的。

### 4.3 决定性检验：冻结尾巴

若机制正确，则"只让 top-2 按实测演化、尾段冻结在初值"应当复现整条下降曲线。`reproduce_rank_dip.py` 实跑全批 GD 并记录完整 128 维谱：

`after_ln_f`：

| step | erank | num_rank | $E_2$ | erank(去掉top-2) | erank(冻结尾巴) |
|------|-------|----------|-------|-----------------|----------------|
| 0 | 15.69 | 127 | 0.5515 | 42.95 | 15.69 |
| 10 | 14.57 | 127 | 0.5763 | 43.60 | 15.13 |
| 15 | **14.39** | 127 | **0.5804** | 43.74 | **15.04** |
| 20 | 14.40 | 127 | 0.5803 | 43.78 | 15.05 |
| 40 | 15.16 | 127 | 0.5639 | 43.32 | 15.51 |
| 120 | 17.54 | 127 | 0.5114 | 41.29 | 18.94 |

四条同时成立：

1. **`num_rank` 恒为 127**，全程零方向死亡——排除坍缩类解释；
2. **$E_2$ 与 erank 严格反相**，谷底即 $E_2$ 峰值；
3. **尾段自身的有效秩几乎不动**（42.95→43.78，且是**上升**的）——尾巴没有被压缩，反而略微变平；
4. **冻结尾巴的控制量复现了 V 型**（15.69→15.04→18.94），证明 top-2 的相对增长是充分原因。

`block_0_after_mlp_merge` 同样：13.06→11.91@15→16.13，`num_rank` 恒 128，$E_2$ 0.6046→0.6340→0.5375。

合成实验进一步给出定量灵敏度（只放大 top-2，尾巴不动）：

```
top-2 ×1.00 -> erank 10.60      top-2 ×1.20 -> erank  7.68
top-2 ×1.05 -> erank  9.69      top-2 ×1.40 -> erank  6.01
top-2 ×1.10 -> erank  8.92      (num_rank 全程 = 101)
```

top-2 只需增长 5% 就能让有效秩掉 0.9。实测 4–8% 的降幅对应的信号增长量级完全在合理范围。

### 4.4 为什么只有 MLP 之后的层明显

回看 §1.3：`after_pos_emb` 只降 0.9%，`attn_output` 不降，而 `mlp_output` / `after_mlp_merge` / `after_ln_f` 降 4–8%。

原因是 §3 的瓶颈信号**注入位置**不同。$\partial\mathcal{L}/\partial\mathbf{H}_f$ 的 2 维结构直接作用在 $\mathrm{LN}_f$ 前的残差流上；`c_proj` 把 MLP 的输出写回残差流时，写入的增量本身有效秩为 2（§3.4 实测 2.00）。因此 MLP 输出端是 2 维信号进入表示的**第一现场**，$E_2$ 变化最大。而 `after_pos_emb` 位于所有写入之前，只能通过 $\mathbf{W}_E$ 的缓慢更新间接受影响——它的 $\sigma_{1,2}$ 在 2000 步里只从 175.8/150.5 走到 175.5/153.7，故几乎不降。

### 4.5 回升：注意力的迟滞

V 型右支需要新方向获得能量。机制是注意力的**初始抑制随后解除**。

$t=0$ 时 $\mathbf{Q},\mathbf{K}$ 均为 $O(\varepsilon)$，故 $\mathbf{S}=\mathbf{Q}\mathbf{K}^\top/\sqrt{d}=O(\varepsilon^2)$，softmax 输出接近均匀。softmax 的 Jacobian 为 $\operatorname{diag}(\mathbf{a})-\mathbf{a}\mathbf{a}^\top$，在均匀点 $\mathbf{a}=\frac{1}{m}\mathbf{1}$ 处作用于任意上游信号时，行内均值被减去，一阶项相消，$\partial\mathcal{L}/\partial\mathbf{S}=O(\varepsilon^2)$。

实测（`verify_rank_theory.py`）：

```
|dL/dW_Q|   = 1.3400e-01      |dL/dW_V|   = 2.9608e+00
|dL/dW_K|   = 1.3186e-01      |dL/dW_out| = 8.1901e+00
ratio |g_Q|/|g_V| = 4.53e-02
```

$\mathbf{W}_Q,\mathbf{W}_K$ 的梯度比同一矩阵内的 $\mathbf{W}_V$ 小 22 倍，比输出头小 61 倍。故前若干步注意力实质冻结，$\mathrm{Attn}$ 只是一个近似均匀的平均器，**不产生新方向**（`attn_output` 有效秩 2.73，全程近乎不变）。

随着 $\mathbf{W}_E,\mathbf{W}_U$ 把残差流幅度推大，$\mathbf{S}$ 脱离 $O(\varepsilon^2)$，注意力开始分化并向残差流写入位置相关的新方向，尾段获得一阶驱动，$E_2$ 转而下降，有效秩回升并最终超过初值（`after_ln_f` 15.38→18.92）。记录数据与此一致：`attn_output` 有效秩在 step 500 才涨到 3.08，`after_pos_emb` 的峰值出现在 step 185，都晚于 step 15–20 的谷底。

注意力熵在整个 bs=100 run 里保持 `normalized_mean_entropy` = 1.000，说明 $T=100$ 时注意力始终接近均匀——这解释了为何 bs=100 的回升主要来自 $\mathrm{MLP}$ 与 $\mathbf{W}_E$ 而非注意力分化，也解释了 §1.3 中 `after_mlp_merge` 单调升到 19.71 而 `after_pos_emb` 回落的差异。

---

## 5. 可检验预测与检验结果

理论若正确，$T$ 越大 $\Rightarrow$ 位置尾段越长越平 $\Rightarrow$ 初始 $E_2$ 越小 $\Rightarrow$ 信号段要"挤"过的尾巴越多 $\Rightarrow$ 相对降幅越大。

记录数据无法检验小 $T$：`activation_rank` 在 step 0 缺失（`train.py:352-357` 的 step-0 分支只记 `weight/activation_singular_values`），而 $T\le20$ 的谷底恰好落在首个 eval step（step 5），说明**下降发生在 step 5 之前**、被采样分辨率截断了。`sweep_dip_vs_blocksize.py` 从 step 0 起逐步记录：

| $T$ | erank@0 | 谷底 | @step | 降幅 | $E_2$@0 | $E_2$@谷底 | num_rank |
|-----|---------|------|-------|------|---------|-----------|----------|
| 2 | 3.48 | 3.43 | 9 | 1.60% | 0.7902 | 0.8056 | 96–98 |
| 5 | 4.97 | 4.89 | 8 | 1.68% | 0.7235 | 0.7347 | 127 |
| 10 | 7.18 | 6.88 | 10 | 4.19% | 0.6480 | 0.6677 | 127 |
| 20 | 8.16 | 7.89 | 10 | 3.30% | 0.6744 | 0.6850 | 127 |
| 50 | 12.33 | 11.35 | 17 | 7.88% | 0.5949 | 0.6192 | 127 |
| 100 | 15.57 | 14.26 | 19 | 8.42% | 0.5544 | 0.5836 | 127 |

**成立的部分：**

- **下降在所有 $T$ 上都存在**（1.6%–8.4%）。§1.3 里 $T=2,5,20$ 的"无下降"是采样假象，不是物理差异——这本身是理论提出并被证实的一条预测。
- $E_2$ 在**每一个** $T$ 上都从初值上升到谷底值，无一例外；`num_rank` 在每个 $T$ 上恒定。核心机制（§4.3 的四条）普适。
- 初始 $E_2$ 随 $T$ 单调下降（0.790→0.554），与"尾段随 $T$ 变长变平"的预测一致。
- 谷底位置随 $T$ 后移（step 8–10 → 17–19），符合"尾段能量越多、信号段需要更久才能挤过"的图像。

**不成立的部分：** 降幅并非严格单调——$T=20$（3.30%）低于 $T=10$（4.19%）。$T=20$ 的初始 $E_2$（0.6744）也高于 $T=10$（0.6480），与单调趋势相悖。位置嵌入是 $\mathcal{N}(0,\varepsilon^2)$ 随机矩阵，$T\le20$ 时其谱的样本波动足以掩盖 $T$ 的系统效应（$T=2$ 的 `num_rank` 在 96–98 间抖动即为佐证）。因此正确的表述是**降幅随 $T$ 总体增大但非严格单调**，单调性只在 $T\ge50$ 稳定。要判定 $T\le20$ 的次序需多 seed 平均，目前每个 $T$ 只有单 seed，尚未做。

---

## 6. 与 `Focus and Dilution`（arXiv:2605.01199）的关系

该文（Chen, Lin, Xu, Luo, ICML 2026）在一层 Transformer + Markov 数据上给出四阶段：rank-1 凝聚 → 注意力聚焦 → 质量重分配稀释 → 不对称提升退化。可移植与不可移植的部分需要分清。

**可移植（结构性，与 token 无关）：**

| 该文机制 | 本任务对应 | 状态 |
|---------|-----------|------|
| $\partial\mathcal{L}/\partial\Phi\vert_{\theta=0}=0$，注意力初始冻结 | $\vert g_Q\vert/\vert g_V\vert=4.5\times10^{-2}$ | ✅ 实测确认（§4.5） |
| $\partial\mathcal{L}/\partial M\vert_{\theta=0}$ 低秩且非零 | 全部内部权重梯度 erank=2.00 | ✅ 实测确认（§3.4） |
| 低秩梯度驱动表示低秩化 | $E_2$ 上升驱动 erank 下降 | ✅ 实测确认（§4.3） |
| 注意力后激活 → 新方向出现 | 谷底后回升 | ✅ 定性一致（§4.5） |

其 Thm 3.2 的实质是**代数的**：凝聚源于梯度矩阵的低秩性，而低秩性源于架构瓶颈（该文是 $M=W_1W_0$ 的 rank-1 结构，本任务是 $p=2$ 的输入输出瓶颈）。这一层不依赖离散 token，可以移植，且本文 §3 把它做成了比该文更强的形式（有效秩恒等于 2，而非仅给上界）。

**不可移植：**

- **rank-1 凝聚方向由稳态分布 $\pi$ 显式给定**（$W_0/\Vert W_0\Vert\to\frac{\pi}{\Vert\pi\Vert}\alpha_1^\top$）。本任务无 token 词表、无 $\pi$；凝聚方向由数据协方差 $\mathbb{E}[\mathbf{y}\mathbf{x}^\top]$ 决定，无闭式。
- **"高频 token 偏置"与 focus 阶段**（$A_i\to e_1^\top$）。需要离散频率不对称。bs=100 全程 `normalized_mean_entropy`=1.000，注意力从未聚焦到任何单一位置——focus 阶段在本任务**不存在**。
- **循环性**。该文的 focus–dilution 是多轮的，由低频 token 间的 $O(\delta)$ 不对称逐级解锁。本任务的有效秩曲线是单谷单调回升（`after_mlp_merge` 12.77→12.22→19.71），2000 步内无第二个谷。

**结论：** 该文的**凝聚**部分（Stage I，源自 Chen & Luo 2025）适用且已被本文强化；**focus–dilution 循环**（Stage II–IV）不适用于连续坐标任务。把本任务的有效秩下降解释成 focus–dilution 的第一个循环是过度类比——它是纯粹的 rank-2 谱重加权，缺少该理论赖以成立的离散频率结构。`r2_research/focus_dilution_analysis.md` §2.1 把 mlp_hidden 的秩下降归给"rank-1 凝聚先于注意力增长"，方向对，但那份文档引用的具体数值（"Step 0 有效秩 5.1 → Step 55 降至 3.4"）来自被 `num_singular_values: 5` 截断的谱，5 是截断上限而非测量值，不能作为证据。

---

## 7. 结论

1. **被解释的现象需要先修正。** "权重有效秩 step 5 暴跌后冻结"是 `results_hyt/*` 的对象别名 bug（§1.1）；真实权重谱 2000 步内变化在小数第四位（§1.2）。真实现象是**激活**有效秩在前 15–20 步下降 4–8% 后回升（§1.3）。

2. **秩的结论只需代数，不需量级。** $\partial\mathcal{L}/\partial\mathbf{H}_f=\mathbf{G}\mathbf{W}_U$ 的行空间被 $\mathbf{W}_U$ 的 2 维行空间锁死（引理 1），经引理 2–3 传播到所有权重梯度。实测每个内部权重梯度有效秩 = 2.00（§3.4）。此论证与数据单位、初始化尺度、$B$、$T$、$N$ 全部无关，因此不重复旧文档"由 $\hat{\mathbf{y}}\approx0$ 推出 $O(1)$ 再推出秩"的范畴错误。

3. **下降的机制是谱重加权，不是坍缩。** 秩-2 的学习信号驱动 $\sigma_1,\sigma_2$ 增长，而位置嵌入贡献的长尾无一阶驱动、近乎静止；$E_2$ 上升，参与率按 P2 下降。四条独立诊断同时成立：数值秩恒定、$E_2$ 与 erank 反相、尾段自身 erank 不降反升、冻结尾巴可复现 V 型（§4.3）。

4. **回升源于注意力迟滞的解除。** 均匀 softmax 处一阶梯度相消使 $\vert g_Q\vert/\vert g_V\vert=4.5\times10^{-2}$，注意力前期不产生新方向；幅度长大后尾段获得驱动，$E_2$ 转降，有效秩回升并超过初值（§4.5）。

5. **一条预测被证实、一条被部分证伪。** 下降在所有 $T$ 上都存在（记录数据的"无下降"是 step-0 缺失导致的采样假象）；但降幅随 $T$ 增大**非严格单调**（$T=20$ 反常），$T\le20$ 时随机位置嵌入的谱波动掩盖了系统效应，判定次序需多 seed（§5）。

### 复现

```bash
/opt/anaconda3/bin/python verify_rank_theory.py       # §3.4, §4.1, §4.5 的实测
/opt/anaconda3/bin/python reproduce_rank_dip.py       # §4.3 四条诊断
/opt/anaconda3/bin/python sweep_dip_vs_blocksize.py   # §5 的 T 扫描
```

### 尚未解决

- $\sigma_1,\sigma_2$ 的增长率没有闭式。要得到谷底位置 $t^\ast(T)$ 的解析式，需在 $\mathrm{LN}$ 存在下解耦残差流动力学。
- 引理 2(3) 只给出乘法型上界，无法预言 $\mathrm{SiLU}'$ 与 $\mathrm{LN}$ 究竟注入多少方向。实测有效秩恒为 2 说明放大在能量意义上可忽略，但缺少证明。
- 多 seed 下 $T\le20$ 的降幅次序未定（§5）。
