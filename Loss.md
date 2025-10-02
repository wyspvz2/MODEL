# PyTorch 常见损失函数详解

本教程介绍 PyTorch 中常用的损失函数，包括均方误差损失、交叉熵损失和 KL 散度损失，包含数学公式、原理解释和示例代码。

---

## 1. MSELoss（均方误差损失）

数学公式：

$$
\text{MSELoss} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

解释：

- 用于回归任务  
- 衡量预测值 $\hat{y}$ 与真实值 $y$ 的平方差  
- 对异常值较敏感

```python
import torch
import torch.nn as nn

mse_loss = nn.MSELoss()
y_pred = torch.tensor([0.5, 0.8, 1.2])
y_true = torch.tensor([0.0, 1.0, 1.0])
loss = mse_loss(y_pred, y_true)
print("MSELoss:", loss.item())
```

## 2. CrossEntropyLoss（交叉熵损失）

**数学公式（多分类）**：

$$
\text{CrossEntropyLoss} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log \hat{y}_{i,c}
$$

**解释**：

- 用于分类任务  
- $y_{i,c}$ 是真实类别的 one-hot 编码，$hat{y}_{i,c}$ 是预测概率  
- PyTorch 的 `CrossEntropyLoss` 内部包含 `Softmax`，不需要手动计算概率

**示例代码**：

```python
import torch
import torch.nn as nn

cross_entropy = nn.CrossEntropyLoss()
y_pred = torch.tensor([[2.0, 1.0, 0.1]])  # logits
y_true = torch.tensor([0])                # 类别索引
loss = cross_entropy(y_pred, y_true)
print("CrossEntropyLoss:", loss.item())
```
## 3. KLDivLoss（KL 散度损失）

**数学公式**：

$$
\text{KLDivLoss}(P \parallel Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$

**解释**：

- 用于衡量两个概率分布 $P$ 和 $Q$ 的差异  
- 常用于知识蒸馏或概率分布拟合  
- PyTorch 要求输入为 **log 概率**（`log_target=False` 时输入为 log 概率）

**示例代码**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

kl_div = nn.KLDivLoss(reduction='batchmean')
p = F.log_softmax(torch.tensor([[0.2, 0.5, 0.3]]), dim=1)
q = torch.tensor([[0.1, 0.6, 0.3]])
loss = kl_div(p, q)
print("KLDivLoss:", loss.item())
