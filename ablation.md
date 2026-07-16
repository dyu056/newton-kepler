# Kepler 未训练模型：当前两组关键消融实验 Prompt
> 目标：解释未训练 Transformer 在不同 `block_size` 下，为什么在完全没有训练的情况下，对 \(F,F_x,F_y\) 已经得到较高且差异明显的线性 probe \(R^2\)。
> 当前阶段只做两组最关键实验：
> 1. **A3：完全对齐 trajectory、endpoint、split 与初始化，只改变 context length。**
> 2. **随机特征来源分解：区分“线性 probe 本身的拟合能力”与“随机初始化网络提供的非线性特征”。**
> 暂时不要做 entropy loss、attention distance、weight decay 或任何模型训练。

# 0. 当前现象

当前未训练模型结果中：

- `x/y` 的线性 probe \(R^2\) 接近 1；
- `r`、force direction 的 \(R^2\) 接近 1；
- `F_magnitude` 和 `inv_r_squared` 可达到约 \(0.5\sim0.87\)；
- 不同 `block_size` 之间差异明显；
- 某些设置下 \(F_x\) 和 \(F_y\) 差异很大；
- 当前 backbone 未训练，但 linear probe 会在 probe-train split 上拟合。

我们需要区分三个来源：

\[\boxed{\text{probe 有限样本拟合}+\text{输入本身的几何信息}+\text{随机网络生成的非线性特征}}\]

---

# 1. 最高优先级约束

## 1.1 本阶段禁止

- 不训练 backbone；
- 不修改数据生成；
- 不使用物理辅助标签训练 backbone；
- 不修改 probe target；
- 不改成滑动窗口；
- 不加入 attention regularization；
- 不加入 weight decay sweep；
- 不使用测试集选超参数；
- 不覆盖已有结果。

## 1.2 必须固定

- 同一批 trajectory IDs；
- 同一组 probe train/test trajectory split；
- 同一个 endpoint；
- 同一个 model initialization；
- 同一个 probe 实现；
- 同一个采样数量；
- 同一个 target alignment；
- 同一组 model seeds 与 split seeds。

---

# 2. 先做代码审查

修改前先定位并报告：

```text
未训练模型消融 entrypoint:
sequence/window 构造函数:
trajectory 数据 shape:
block_size 的真实语义:
last hidden 对应的时间点:
F/Fx/Fy label 对应的时间点:
probe 实现:
probe 是否带 intercept:
probe split 当前按 sequence 还是 trajectory:
model 初始化位置:
absolute positional embedding:
final LayerNorm:
结果保存路径:
```

必须确认以下时间对齐：

设输入最后一个位置是 \(r_t\)，则：

- hidden state 是否对应 \(t\)；
- force probe label 是否为 \(F_t\)；
- next-position target 是否为 \(r_{t+1}\)。

本 prompt 默认使用：

\[
h_t \longrightarrow F_t,
\]

但若现有代码不是这样，必须先报告，不能静默修改。

---

# 3. 实验一：A3 严格 common-endpoint 消融

## 3.1 实验目的

排除以下混杂：

- 不同 block size 使用不同 trajectory；
- 同一 trajectory 同时进入 probe train/test；
- 不同 block size 的 target 时间点分布不同；
- 不同 block size 使用不同随机模型初始化；
- 不同 block size 抽到不同样本。

最终要求：

\[\boxed{\text{每个样本的 trajectory 和 endpoint 完全相同，只有 context length 不同}}\]

---

## 3.2 样本定义

数据共有：

```python
num_trajectories = 2000
num_points = 100
```

位置索引为：

```text
0, 1, ..., 99
```

若 hidden 的最后输入 token 为 \(t\)，并需要 next-position target \(t+1\)，则最大合法 endpoint 为：

```python
common_endpoint = 98
```

此时 probe target 为：

```text
F at t = 98
```

next-position target 为：

```text
position at t = 99
```

使用实际 context lengths：

```python
block_sizes = [1, 10, 50, 99]
```

不要把实际长度 99 标成 100。

对每个 trajectory \(i\) 和 block size \(B\)，构造：

\[
X_{i,B}
=
\left(
r_{i,\,98-B+1},
\ldots,
r_{i,\,98}
\right).
\]

所有 \(B\) 对应同一个：

\[
F_{i,98},\quad F_{x,i,98},\quad F_{y,i,98}.
\]

---

## 3.3 trajectory-level split

只抽取唯一 trajectory，不抽 sequence。

推荐第一轮：

```python
num_probe_trajectories = 1000
probe_test_fraction = 0.30
trajectory_sample_seed = 0
probe_split_seed = 0
```

先抽取 1000 个唯一 trajectory ID：

```python
import numpy as np
from sklearn.model_selection import train_test_split


def select_probe_trajectory_ids(
    *,
    num_total_trajectories: int,
    num_probe_trajectories: int,
    sample_seed: int,
) -> np.ndarray:
    if num_probe_trajectories > num_total_trajectories:
        raise ValueError(
            "num_probe_trajectories cannot exceed "
            "num_total_trajectories"
        )

    rng = np.random.default_rng(sample_seed)

    return rng.choice(
        num_total_trajectories,
        size=num_probe_trajectories,
        replace=False,
    )


selected_ids = select_probe_trajectory_ids(
    num_total_trajectories=2000,
    num_probe_trajectories=1000,
    sample_seed=trajectory_sample_seed,
)

train_ids, test_ids = train_test_split(
    selected_ids,
    test_size=probe_test_fraction,
    random_state=probe_split_seed,
    shuffle=True,
)
```

这组 `train_ids/test_ids` 必须被所有 block size 复用。

保存：

```text
selected_trajectory_ids.npy
probe_train_trajectory_ids.npy
probe_test_trajectory_ids.npy
```

---

## 3.4 common-endpoint context 构造代码

```python
from __future__ import annotations

import numpy as np
import torch


def build_common_endpoint_contexts(
    trajectories: torch.Tensor,
    trajectory_ids: np.ndarray,
    *,
    block_size: int,
    endpoint: int,
) -> torch.Tensor:
    """
    trajectories:
        [num_trajectories, num_points, 2]

    Returns:
        contexts:
        [num_selected_trajectories, block_size, 2]
    """
    if trajectories.ndim != 3:
        raise ValueError(
            "trajectories must have shape "
            "[num_trajectories, num_points, coordinate_dim]"
        )

    _, num_points, coordinate_dim = trajectories.shape

    if coordinate_dim != 2:
        raise ValueError(
            f"expected coordinate_dim=2, got {coordinate_dim}"
        )

    if endpoint < 0 or endpoint >= num_points:
        raise ValueError(
            f"endpoint={endpoint} is outside [0, {num_points - 1}]"
        )

    if block_size < 1:
        raise ValueError(
            f"block_size must be positive, got {block_size}"
        )

    start = endpoint - block_size + 1

    if start < 0:
        raise ValueError(
            f"block_size={block_size} is too large for "
            f"endpoint={endpoint}"
        )

    trajectory_index = torch.as_tensor(
        trajectory_ids,
        dtype=torch.long,
        device=trajectories.device,
    )

    selected = trajectories.index_select(
        dim=0,
        index=trajectory_index,
    )

    contexts = selected[:, start : endpoint + 1, :]

    expected_shape = (
        len(trajectory_ids),
        block_size,
        coordinate_dim,
    )

    if tuple(contexts.shape) != expected_shape:
        raise RuntimeError(
            f"unexpected context shape: {tuple(contexts.shape)}, "
            f"expected {expected_shape}"
        )

    return contexts
```

---

## 3.5 target 构造

必须直接按同一个 endpoint 取 label，而不是从 sequence iterator 推断。

```python
def select_endpoint_targets(
    target_tensor: torch.Tensor,
    trajectory_ids: np.ndarray,
    *,
    endpoint: int,
) -> torch.Tensor:
    """
    target_tensor:
        [num_trajectories, num_points, target_dim]
        or [num_trajectories, num_points]
    """
    trajectory_index = torch.as_tensor(
        trajectory_ids,
        dtype=torch.long,
        device=target_tensor.device,
    )

    selected = target_tensor.index_select(
        dim=0,
        index=trajectory_index,
    )

    return selected[:, endpoint]
```

必须对以下量执行一致性断言：

```python
assert torch.allclose(
    x_target,
    contexts[:, -1, 0],
)

assert torch.allclose(
    y_target,
    contexts[:, -1, 1],
)
```

若断言失败，说明 target 和 hidden 的时间对齐错误，立即停止实验。

---

## 3.6 同一个随机模型初始化

推荐做法：

1. 创建一个支持最大 context 99 的模型；
2. 保存初始化 `state_dict`；
3. 每个 block size 加载同一份初始化；
4. 或直接复用同一个 frozen model 对不同长度输入前向。

```python
import copy
import torch


def build_reference_untrained_model(
    model_config,
    *,
    model_seed: int,
):
    torch.manual_seed(model_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    model = build_model(model_config)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    initial_state = copy.deepcopy(model.state_dict())

    return model, initial_state
```

若模型可直接接受变长输入，优先复用同一个实例：

```python
with torch.no_grad():
    activations_b1 = collect(model, contexts_b1)
    activations_b10 = collect(model, contexts_b10)
```

如果必须重建模型，则：

```python
model.load_state_dict(initial_state, strict=True)
```

必须打印并保存参数 checksum，确认不同 block size 完全一致：

```python
def parameter_checksum(model) -> float:
    return sum(
        parameter.detach().double().sum().item()
        for parameter in model.parameters()
    )
```

---

## 3.7 Probe 协议

保持当前主 probe 不变。

若当前为 OLS：

```python
from sklearn.linear_model import LinearRegression


probe = LinearRegression(
    fit_intercept=True,
)

probe.fit(
    hidden_train,
    target_train,
)
```

必须：

- train 使用 `train_ids`；
- test 使用 `test_ids`；
- 所有 block size 使用同一 split；
- 不重新随机切 sequence；
- 不把同一 trajectory 放进两边。

记录：

```text
train R²
test R²
train MSE
test MSE
train target variance
test target variance
```

不要再把负的 `train R² - test R²` 称为 underfitting。

---

## 3.8 A3 必须输出

对每个 model seed、block size、layer、target 保存：

```text
model_seed
trajectory_sample_seed
probe_split_seed
block_size
effective_context_length
endpoint
layer
target
train_r2
test_r2
train_mse
test_mse
train_target_variance
test_target_variance
parameter_checksum
```

第一轮 model seeds：

```python
model_seeds = [0, 1, 2, 3, 4]
```

probe split seeds 第一轮固定为：

```python
probe_split_seed = 0
```

若结果仍然波动大，再增加 split seeds。

输出汇总：

\[
\operatorname{mean}\pm\operatorname{std}
\]

并画：

```text
random-backbone R² vs block_size
Fx R² and Fy R² vs block_size
Fx-Fy R² difference vs model_seed
```

---

# 4. 实验二：随机特征来源分解

## 4.1 关键判断

“改成单层 attention 和 MLP”是必要的，但它不能单独回答：

> 高 \(R^2\) 是 linear probe 自己训练出来的，还是随机初始化网络提供的？

因为无论 backbone 是一层还是两层，linear probe 都会训练。

真正需要三个层次的 control：

### Control 1：与输入无关的随机 Gaussian features

若 linear probe 仅凭有限样本和高维特征就能得到高 test \(R^2\)，那么独立 Gaussian features 也会高。

预期：

\[
R^2_{\mathrm{Gaussian,test}}\approx 0
\]

或小于 0。

### Control 2：raw \((x,y)\) 和 random linear embedding

随机线性 embedding 不会增加线性可表达能力：

\[
h=W(x,y).
\]

在 \(W\) 满秩、probe 带 intercept 时：

\[
\operatorname{span}(h)
=
\operatorname{span}(x,y).
\]

因此应有：

\[
R^2_{\mathrm{random\ linear}}
\approx
R^2_{\mathrm{raw}\ xy}.
\]

### Control 3：随机非线性模块

只有 LayerNorm、GELU MLP、context-dependent attention 等非线性变换，才可能让 \(r,1/r^2,F_x,F_y\) 更容易被线性读取。

---

## 4.2 最小但充分的 architecture ladder

使用实验一的同一组：

- trajectory IDs；
- endpoint；
- split；
- model seeds；
- block sizes；
- targets。

依次比较：

| ID | 表示 | 回答的问题 |
|---|---|---|
| C0 | independent Gaussian features | probe 本身能否在无信息特征上制造高 test \(R^2\) |
| C1 | raw last-token \((x_t,y_t)\) | 输入本身的线性可解码基线 |
| C2 | random linear `input_embed` | 随机线性初始化是否改变线性可解码性 |
| C3 | `input_embed + LayerNorm` | LayerNorm 非线性贡献 |
| C4 | one-block attention-only | 随机 attention/context mixing 贡献 |
| C5 | one-block MLP-only | 随机 GELU features 贡献 |
| C6 | one full block | attention 与 MLP 的联合贡献 |
| C7 | original two full blocks | depth 的增量贡献 |

另外，所有 Transformer 变体都同时保存：

```text
before_final_ln
after_final_ln
```

用于判断最终 LayerNorm 是否是高 \(R^2\) 的主要来源。

---

# 5. Gaussian feature control

为每个 hidden dimension 构造与输入和 target 独立的特征：

```python
import numpy as np


def make_independent_gaussian_features(
    *,
    num_samples: int,
    feature_dim: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    return rng.standard_normal(
        size=(num_samples, feature_dim),
    ).astype(np.float32)
```

建议至少测试：

```python
feature_dims = [2, 32, 128]
```

解释：

- 2：与 raw xy 维数相同；
- 32：与 residual hidden 相同；
- 128：与 MLP hidden 相同。

Gaussian feature 必须按同一 trajectory split 进入 probe。

若 Gaussian test \(R^2\) 明显为正，优先检查：

- 数据泄漏；
- target 与 feature 构造是否共享 seed/索引；
- train/test 切分；
- 是否错误地在全数据上 fit scaler；
- 是否报告了 train R² 而非 test R²。

---

# 6. Raw xy 与 random linear embedding

## 6.1 Raw xy

```python
raw_last_xy = contexts[:, -1, :]
```

## 6.2 Random linear embedding

直接使用模型的 `input_embed` hook。

也可增加一个明确的独立实现作为 sanity check：

```python
import torch
from torch import nn


class FrozenRandomLinearEmbedding(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        seed: int,
    ) -> None:
        super().__init__()

        generator = torch.Generator()
        generator.manual_seed(seed)

        weight = torch.randn(
            input_dim,
            hidden_dim,
            generator=generator,
        ) / (input_dim ** 0.5)

        self.register_buffer(
            "weight",
            weight,
            persistent=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return x @ self.weight
```

必须验证：

\[
R^2(\text{raw xy})
\approx
R^2(\text{random linear embedding})
\]

若差异很大，检查：

- probe 是否带 intercept；
- feature 是否中心化；
- embedding 是否还包含 LN、activation 或 position-dependent 操作；
- target 是否对齐；
- 矩阵秩是否为 2。

---

# 7. 单层 attention / MLP 消融的正确实现

## 7.1 不要分别随机初始化三个不同模型

否则 attention-only、MLP-only、full-block 的差异会混入初始化差异。

应创建同一个 full random block，然后使用固定 gate 控制分支：

\[
x_1=x+g_A\,\mathrm{Attn}(\mathrm{LN}_1(x)),
\]

\[
x_2=x_1+g_M\,\mathrm{MLP}(\mathrm{LN}_2(x_1)),
\]

其中：

```text
attention-only: g_A=1, g_M=0
MLP-only:       g_A=0, g_M=1
full block:     g_A=1, g_M=1
identity:       g_A=0, g_M=0
```

gate 不是可训练参数。

---

## 7.2 参考代码

必须适配当前 pre-LN/post-LN 实现。以下假设当前为 pre-LN：

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BranchGates:
    attention: float = 1.0
    mlp: float = 1.0


class AblatableTransformerBlock(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        mlp: nn.Module,
        ln_1: nn.Module,
        ln_2: nn.Module,
    ) -> None:
        super().__init__()

        self.attention = attention
        self.mlp = mlp
        self.ln_1 = ln_1
        self.ln_2 = ln_2

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        gates: BranchGates,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        activations: dict[str, torch.Tensor] = {}

        attention_input = self.ln_1(hidden)
        activations["attention_input"] = attention_input

        attention_output = self.attention(attention_input)
        activations["attention_output"] = attention_output

        after_attention = (
            hidden
            + gates.attention * attention_output
        )
        activations["after_attention"] = after_attention

        mlp_input = self.ln_2(after_attention)
        activations["mlp_input"] = mlp_input

        mlp_output = self.mlp(mlp_input)
        activations["mlp_output"] = mlp_output

        after_mlp = (
            after_attention
            + gates.mlp * mlp_output
        )
        activations["after_mlp"] = after_mlp

        return after_mlp, activations
```

运行：

```python
gate_settings = {
    "identity": BranchGates(
        attention=0.0,
        mlp=0.0,
    ),
    "attention_only": BranchGates(
        attention=1.0,
        mlp=0.0,
    ),
    "mlp_only": BranchGates(
        attention=0.0,
        mlp=1.0,
    ),
    "full_block": BranchGates(
        attention=1.0,
        mlp=1.0,
    ),
}
```

所有设置复用同一组参数。

---

## 7.3 单层与双层

推荐不要独立随机初始化 `n_layer=1` 和 `n_layer=2`。

创建原始两层随机模型：

```text
block_0
block_1
final_ln
```

然后比较：

### One-block representation

```python
one_block_hidden = block_0_output
one_block_after_final_ln = final_ln(
    one_block_hidden
)
```

### Two-block representation

```python
two_block_hidden = block_1_output
two_block_after_final_ln = final_ln(
    two_block_hidden
)
```

这样：

- one-block 与 two-block 共享 input embedding；
- one-block 使用双层模型中的第一层；
- depth 增量只来自第二层；
- 不混入重新初始化差异。

必须保存：

```text
block_0_before_final_ln
block_0_after_final_ln
block_1_before_final_ln
block_1_after_final_ln
```

---

# 8. LayerNorm 单独控制

由于 LayerNorm 是非线性的，不能把它当作普通线性缩放忽略。

至少比较：

```text
input_embed
LN(input_embed)
block output before final LN
block output after final LN
```

参考：

```python
with torch.no_grad():
    embedded = model.input_projection(contexts)
    embedded_last = embedded[:, -1, :]

    embedded_ln_last = model.final_ln(
        embedded
    )[:, -1, :]
```

若模型的 final LN 只允许最后 block 输出，创建同参数副本或直接调用同一 module；不要重新初始化一个 LN。

---

# 9. Shuffled-label control

为了确认 probe pipeline 不会制造虚假泛化，增加：

```python
def shuffle_train_labels(
    targets,
    *,
    seed: int,
):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(targets))
    return targets[permutation]
```

严格区分两种 control：

### Train-label shuffle

只打乱 probe-train labels，test labels 保持真实：

```text
fit: hidden_train -> shuffled target_train
evaluate: hidden_test -> true target_test
```

预期 test \(R^2\le 0\)。

### Full-label shuffle

整体打乱后再按固定 split 切分，只作为额外检查。

主 control 使用 train-label shuffle。

---

# 10. 运行矩阵

第一轮只运行：

```python
block_sizes = [1, 10, 50, 99]
model_seeds = [0, 1, 2, 3, 4]
probe_split_seed = 0
common_endpoint = 98
```

architecture variants：

```text
gaussian_d2
gaussian_d32
gaussian_d128
raw_xy
input_embed
input_embed_after_ln
attention_only_before_final_ln
attention_only_after_final_ln
mlp_only_before_final_ln
mlp_only_after_final_ln
one_full_block_before_final_ln
one_full_block_after_final_ln
two_full_blocks_before_final_ln
two_full_blocks_after_final_ln
```

targets：

```text
x
y
r
inv_r_squared
F_magnitude
Fx
Fy
F_direction_x
F_direction_y
```

其中：

```text
F_magnitude == inv_r_squared
```

若 \(GM=1\)，汇总时只把它们视为一个独立 target。

---

# 11. 结果解释规则

## 11.1 probe 本身的有限样本效应

如果：

\[
R^2_{\mathrm{Gaussian,test}}\approx0,
\]

则高 \(R^2\) 不是 linear probe 单纯凭高维随机特征“拟合出来”的。

如果 Gaussian test \(R^2\) 很高，则优先判定 pipeline 有问题。

## 11.2 输入本身的贡献

如果：

\[
R^2_{\mathrm{raw}\ xy}
\approx
R^2_{\mathrm{input\ embed}},
\]

说明 random linear embedding 没有增加信息，这是理论预期。

## 11.3 LayerNorm 的贡献

如果：

\[
R^2_{\mathrm{embed+LN}}
\gg
R^2_{\mathrm{embed}},
\]

说明高 \(R^2\) 很大部分来自随机初始化网络中的 normalization-induced nonlinear features。

## 11.4 MLP 的贡献

如果：

\[
R^2_{\mathrm{MLP-only}}
\gg
R^2_{\mathrm{embed+LN}},
\]

说明 GELU random features 是主要来源。

## 11.5 Attention 的贡献

比较：

\[
R^2_{\mathrm{attention-only}}(B)
\]

随 block size 是否变化。

若 attention-only 已产生明显 block-size dependence，则不同 context 的随机 mixing 是重要来源。

特别注意 \(B=1\)：

- causal attention 只有一个 key；
- softmax 权重恒为 1；
- 不存在跨 token mixing；
- 但 pre-attention LayerNorm 仍是非线性的。

因此，`attention-only, B=1` 不是纯线性 control。

## 11.6 深度贡献

如果：

\[
R^2_{\mathrm{two-block}}
-
R^2_{\mathrm{one-block}}
\]

显著大于随机 seed 方差，说明第二层增加了有效随机非线性基函数。

---

# 12. 必须输出的图表

## 图 1：A3 common endpoint

横轴：

```text
block_size
```

纵轴：

```text
test R²
```

曲线：

```text
F
Fx
Fy
```

误差条：

```text
mean ± std over model seeds
```

## 图 2：Architecture ladder

横轴：

```text
Gaussian
raw xy
linear embed
embed + LN
attention only
MLP only
one full block
two full blocks
```

纵轴：

```text
test R²
```

对每个 block size 单独画图，或使用 facet。

## 图 3：Fx/Fy 对称性

画：

\[
R^2(F_x)-R^2(F_y)
\]

随 model seed 与 architecture variant 的变化。

## 图 4：训练 probe 的 sanity control

比较：

```text
true labels
shuffled train labels
independent Gaussian features
```

---

# 13. 结果文件格式

保存 CSV：

```text
results/random_probe_key_ablations/results.csv
```

列至少包括：

```text
experiment
architecture_variant
model_seed
trajectory_sample_seed
probe_split_seed
block_size
endpoint
layer
target
feature_dim
train_r2
test_r2
train_mse
test_mse
train_target_variance
test_target_variance
parameter_checksum
num_train_trajectories
num_test_trajectories
```

保存 metadata：

```text
results/random_probe_key_ablations/metadata.json
```

包含：

- git commit；
- config；
- trajectory IDs 文件路径；
- target alignment；
- effective context lengths；
- model parameter count；
- Python/PyTorch/sklearn 版本。

---

# 14. 必须实现的断言

```python
assert len(set(selected_ids.tolist())) == len(selected_ids)

assert set(train_ids).isdisjoint(set(test_ids))

assert len(train_ids) + len(test_ids) == len(selected_ids)

assert common_endpoint + 1 < num_points

assert max(block_sizes) <= common_endpoint + 1
```

对每个 block size：

```python
assert contexts.shape == (
    len(selected_ids),
    block_size,
    2,
)

assert torch.allclose(
    contexts[:, -1, 0],
    x_target,
)

assert torch.allclose(
    contexts[:, -1, 1],
    y_target,
)
```

对不同 block size：

```python
assert np.array_equal(
    selected_ids_for_b1,
    selected_ids_for_b99,
)

assert np.array_equal(
    train_ids_for_b1,
    train_ids_for_b99,
)

assert np.array_equal(
    test_ids_for_b1,
    test_ids_for_b99,
)

assert endpoint_for_b1 == endpoint_for_b99
```

对初始化：

```python
assert checksum_b1 == checksum_b10
assert checksum_b10 == checksum_b50
assert checksum_b50 == checksum_b99
```

---

# 15. Agent 执行顺序

严格按顺序：

1. 只读审查；
2. 报告当前 target alignment；
3. 实现 common-endpoint dataset builder；
4. 实现 trajectory-level fixed split；
5. 添加对齐断言；
6. 固定并复用同一个随机初始化；
7. 重跑 A3；
8. 保存 A3 结果；
9. 实现 Gaussian feature control；
10. 实现 raw xy 和 random linear control；
11. 实现 branch gates；
12. 运行 attention-only、MLP-only、one-block、two-block；
13. 加入 before/after final LN；
14. 加入 shuffled-label control；
15. 汇总 5 个 model seeds；
16. 输出图和结论。

不要在完成这两组实验前实现任何训练 trick。

---

# 16. 最终需要回答的问题

Agent 最终报告必须明确回答：

1. 在严格 common endpoint 和 trajectory split 下，随机模型的 block-size 差异是否仍存在？
2. 该差异是否大于 model-seed 方差？
3. Gaussian independent features 的 test \(R^2\) 是否接近 0？
4. raw xy 与 random linear embedding 是否一致？
5. LayerNorm 是否显著提高 \(r\)、direction、force 的可解码性？
6. MLP-only 是否已经产生高 \(F\) \(R^2\)？
7. attention-only 是否产生明显 block-size dependence？
8. 一层到两层的增量是否稳定？
9. \(F_x/F_y\) 不对称是否在多 seed 后消失？
10. 训练前的高 \(R^2\) 主要来自：
    - probe finite-sample artifact；
    - input geometry；
    - LayerNorm；
    - random MLP features；
    - random attention；
    - model depth；
    - absolute positional embedding；
    中的哪一项？
