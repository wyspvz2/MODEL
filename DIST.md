# DIST: Distillation via Inter-class and Intra-class Correlation
![Uploading image.png…]()

## 1. 方法概述

DIST（Distillation via Inter-class and Intra-class Correlation）是一种关系蒸馏方法，通过匹配学生网络和教师网络在类别预测概率上的相关性来传递知识。

它不直接替代分类损失，而是在交叉熵基础上增加一项关系损失，让学生学习教师对 **类间（inter-class）和类内（intra-class）** 的分布关系。

### 1.1 Inter-class Relation（类间关系）

- 对每个样本，教师和学生预测的概率向量之间的相关性。
- 数学公式（块级公式）：

$$
L_{inter} = \frac{1}{B} \sum_{i=1}^{B} \Big(1 - corr(y_i^s, y_i^t)\Big)
$$

其中 $y_i^s, y_i^t \in \mathbb{R}^C$ 是学生和教师的 softmax 预测，B 是 batch size，corr(y_i, y_i) 为 Pearson 相关系数。

### 1.2 Intra-class Relation（类内关系）

- 对每个类别，统计 batch 内样本在该类别上的预测概率趋势，然后匹配教师和学生的相关性。
- 数学公式：

$$
L_{intra} = \frac{1}{C} \sum_{j=1}^{C} \Big(1 - corr(y_j^s, y_j^t)\Big)
$$

其中 $y_j^s, y_j^t \in \mathbb{R}^B$ 表示学生/教师在 batch 中样本对类别 $j$ 的预测概率。

### 1.3 总损失

$$
\mathcal{L}_{DIST} = \beta \cdot \tau^2 L_{inter} + \gamma \cdot \tau^2 L_{intra}
$$

---

## 2. 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DIST(nn.Module):
    def __init__(self, beta=2.0, gamma=2.0, tau=6.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.eps = 1e-8

    def pearson_corr(self, x, y):
        xm = x.mean(dim=1, keepdim=True)
        ym = y.mean(dim=1, keepdim=True)
        x_c = x - xm
        y_c = y - ym
        num = (x_c * y_c).sum(dim=1)
        denom = torch.sqrt((x_c**2).sum(dim=1) + self.eps) * torch.sqrt((y_c**2).sum(dim=1) + self.eps)
        return num / (denom + self.eps)

    def forward(self, logits_s, logits_t):
        B, C = logits_s.shape
        y_s = F.softmax(logits_s / self.tau, dim=1)
        y_t = F.softmax(logits_t / self.tau, dim=1)

        # Inter-class correlation
        corr_inter = self.pearson_corr(y_s, y_t)
        L_inter = (1.0 - corr_inter).mean()

        # Intra-class correlation
        y_s_T = y_s.transpose(0,1)
        y_t_T = y_t.transpose(0,1)
        corr_intra = self.pearson_corr(y_s_T, y_t_T)
        L_intra = (1.0 - corr_intra).mean()

        # Temperature scaling
        scale = self.tau ** 2
        L_inter *= scale
        L_intra *= scale

        loss = self.beta * L_inter + self.gamma * L_intra
        return loss

# 示例
if __name__ == "__main__":
    B, C = 8, 10
    logits_t = torch.randn(B, C)
    logits_s = torch.randn(B, C)

    dist_loss_fn = DIST(beta=2.0, gamma=2.0, tau=6.0)
    loss = dist_loss_fn(logits_s, logits_t)
    print("DIST loss:", loss.item())
