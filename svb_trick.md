# SVB: Singular Value Bounding

## 问题

神经网络权重矩阵 $W \in \mathbb{R}^{m \times n}$ 在训练中可能出现奇异值分布极端化——少数方向过度放大、多数方向退化——导致表示能力浪费。

## 数学定义

对 $W$ 做奇异值分解：

$$W = U \cdot \text{diag}(\sigma_1, \dots, \sigma_k) \cdot V^T, \quad k = \min(m, n)$$

将每个奇异值 $\sigma_i$ 裁剪到区间 $[\frac{1}{1+\varepsilon},\; 1+\varepsilon]$：

$$\hat{\sigma}_i = \text{clamp}\left(\sigma_i,\; \frac{1}{1+\varepsilon},\; 1+\varepsilon\right)$$

重组矩阵并写回：

$$W \leftarrow U \cdot \text{diag}(\hat{\sigma}_1, \dots, \hat{\sigma}_k) \cdot V^T$$

$\varepsilon=0.2$ 时区间为 $[0.833,\; 1.2]$。

## 代码

```python
import torch
import torch.nn as nn

def svb_apply(model: nn.Module, epsilon: float = 0.2) -> None:
    if epsilon <= 0.0:
        return
    lower = 1.0 / (1.0 + epsilon)
    upper = 1.0 + epsilon
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.data
        if w.shape[0] >= w.shape[1]:
            U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        else:
            U, S, Vh = torch.linalg.svd(w.T, full_matrices=False)
            U, S, Vh = Vh.T, S, U.T
        S = S.clamp(lower, upper)
        module.weight.data.copy_((U * S.unsqueeze(0)) @ Vh)
```

## 训练循环中调用

```python
# optimizer.step() 之后
if epsilon > 0 and (step + 1) % freq == 0:
    svb_apply(model, epsilon)
```

推荐 `freq=100`，`epsilon=0.2`。

## 为什么有效

SVB 是**后处理**而非梯度惩罚——不修改 loss、不干扰 SGD 动力学。训练自由进行，每 N 步做一次"整理"：剪掉过大和过小的奇异值，防止权重矩阵的维度坍塌或过度敏感。

## 在我们实验中的效果

| | Baseline | +SVB (ε=0.2) |
|---|:---:|:---:|
| Fx R² | 0.652 | **0.678** |
| Fy R² | 0.585 | **0.613** |
| Val Loss | 0.00220 | **0.00207** |

n_embd=32, n_layer=2, 7 组正则化实验中唯一有效的 trick。
