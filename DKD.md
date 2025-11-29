# Decoupled Knowledge Distillation (DKD)

## 1. 方法概述
<img width="570" height="789" alt="image" src="https://github.com/user-attachments/assets/3987b96a-cc21-4fdd-b9d6-95838d53f161" />

Decoupled Knowledge Distillation（DKD）是一种改进的知识蒸馏方法，它将 **教师-学生间的知识传递**拆分为 **Target-Class Knowledge Distillation (TCKD)** 和 **Non-Target-Class Knowledge Distillation (NCKD)** 两部分：

- **TCKD**：关注样本的目标类别预测概率  
- **NCKD**：关注样本的非目标类别预测概率分布  

通过对两部分分别加权，学生可以更准确地学习教师在目标类和非目标类上的知识。

---

## 2. 数学公式

### 2.1 softmax 输出

对于教师和学生的 logits:

$$
f^t_i, f^s_i \in \mathbb{R}^C, \quad i=1,\dots,B
$$

先计算 softmax 预测概率：

$$
p^t_i = \text{softmax}(f^t_i), \quad p^s_i = \text{softmax}(f^s_i)
$$

### 2.2 Target-class KD（TCKD）

只关注样本真实类别 $y_i$：

$$
\mathcal{L}_{\text{TCKD}} = \frac{1}{B} \sum_{i=1}^{B} p^t_{i,y_i} \cdot \Big( \log p^t_{i,y_i} - \log p^s_{i,y_i} \Big)
$$

### 2.3 Non-target-class KD（NCKD）

只关注非目标类别：

$$
\begin{aligned}
\hat{p}^t_i &= \frac{p^t_i \odot (1 - \mathbf{1}_{y_i})}{\sum_{j \neq y_i} p^t_{i,j}} \\
\hat{p}^s_i &= \frac{p^s_i \odot (1 - \mathbf{1}_{y_i})}{\sum_{j \neq y_i} p^s_{i,j}} \\
\mathcal{L}_{\text{NCKD}} &= \frac{1}{B} \sum_{i=1}^B \sum_{j \neq y_i} \hat{p}^t_{i,j} \cdot \big( \log \hat{p}^t_{i,j} - \log \hat{p}^s_{i,j} \big)
\end{aligned}
$$

### 2.4 总损失

$$
\mathcal{L}_{\text{DKD}} = \alpha \, \mathcal{L}_{\text{TCKD}} + \beta \, \mathcal{L}_{\text{NCKD}}
$$

其中 $\alpha, \beta$ 为权重超参数。

---

## 3. 张量示例

假设 batch size $B=2$，类别数 $C=3$，标签：

$$
y = [0, 2]
$$

教师 logits：

$$
f^t = \begin{bmatrix} [2.0, 1.0, 0.5] \\ [0.2, 0.5, 1.5] \end{bmatrix}
$$

学生 logits：

$$
f^s = \begin{bmatrix} [1.5, 1.2, 0.3] \\ [0.1, 0.7, 1.2] \end{bmatrix}
$$

1. **Softmax**（示例计算第一个样本）：

$$
p^t_0 = \text{softmax}([2.0, 1.0, 0.5]) \approx [0.57, 0.26, 0.17]
$$  
$$
p^s_0 = \text{softmax}([1.5, 1.2, 0.3]) \approx [0.48, 0.35, 0.17]
$$

2. **TCKD**（第一样本，目标类别 0）：

$$
\text{TCKD}_0 = 0.57 \cdot (\log 0.57 - \log 0.48) \approx 0.10
$$

3. **NCKD**（第一样本，非目标类别 1 和 2）：

$$
\hat{p}^t_0 = [0, 0.26/(0.26+0.17), 0.17/(0.26+0.17)] = [0, 0.60, 0.40]
$$  
$$
\hat{p}^s_0 = [0, 0.35/(0.35+0.17), 0.17/(0.35+0.17)] = [0, 0.67, 0.33]
$$  
$$
\text{NCKD}_0 = 0.6 (\log 0.6 - \log 0.67) + 0.4 (\log 0.4 - \log 0.33) \approx 0.03
$$

4. 总损失（假设 $\alpha=1, \beta=8$）：

$$
\mathcal{L}_{\text{DKD},0} = 1 \cdot 0.10 + 8 \cdot 0.03 = 0.34
$$

---

## 4. PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DKDLoss(nn.Module):
    """Decoupled Knowledge Distillation (TCKD + NCKD)"""
    def __init__(self, alpha=1.0, beta=8.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8

    def forward(self, fS, fT, labels):
        B, C = fS.shape
        device = fS.device

        # softmax probabilities
        pS = F.softmax(fS, dim=1)
        pT = F.softmax(fT, dim=1)

        idx = torch.arange(B, device=device)

        # ----------------- TCKD -----------------
        tckd = (pT[idx, labels] * (torch.log(pT[idx, labels]+1e-8) - torch.log(pS[idx, labels]+1e-8))).mean()

        # ----------------- NCKD -----------------
        mask = torch.ones_like(pS, dtype=torch.bool)
        mask[idx, labels] = False

        expS = fS.exp()
        expT = fT.exp()

        denomS = expS.masked_fill(~mask, 0.0).sum(dim=1, keepdim=True) + 1e-8
        denomT = expT.masked_fill(~mask, 0.0).sum(dim=1, keepdim=True) + 1e-8

        pS_hat = expS.masked_fill(~mask, 0.0) / denomS
        pT_hat = expT.masked_fill(~mask, 0.0) / denomT

        nckd = (pT_hat * (torch.log(pT_hat+1e-8) - torch.log(pS_hat+1e-8))).sum(dim=1).mean()

        # ----------------- total -----------------
        loss = self.alpha * tckd + self.beta * nckd
        return loss

# ===========================
# 示例
# ===========================
B, C = 2, 3
labels = torch.tensor([0, 2])
fT = torch.tensor([[2.0, 1.0, 0.5],
                   [0.2, 0.5, 1.5]])
fS = torch.tensor([[1.5, 1.2, 0.3],
                   [0.1, 0.7, 1.2]])

loss_fn = DKDLoss(alpha=1.0, beta=8.0)
loss = loss_fn(fS, fT, labels)
print("DKD loss:", loss.item())
