# DIST: Distillation via Inter-class and Intra-class Correlation

## 1. 方法概述

DIST（Distillation via Inter-class and Intra-class Correlation）是一种 **关系蒸馏方法**，旨在通过匹配学生网络和教师网络在 **类别预测概率上的相关性** 来传递知识。  
它不直接替代分类损失，而是在标准交叉熵基础上增加一项关系损失，使学生学习教师对 **类间（inter-class）和类内（intra-class）** 的分布关系。

### 1.1 Inter-class Relation（类间关系）

- 对每个样本，教师和学生预测的概率向量之间的相关性。  
- 数学公式：

$$
L_\text{inter} = \frac{1}{B} \sum_{i=1}^{B} \Big( 1 - \text{corr}(y_i^s, y_i^t) \Big)
$$

其中 $y_i^s, y_i^t \in \mathbb{R}^C$ 是学生和教师的 softmax 预测，$B$ 是 batch size，$\text{corr}(\cdot,\cdot)$ 为 Pearson 相关系数。

### 1.2 Intra-class Relation（类内关系）

- 对每个类别，统计 batch 内样本在该类别上的预测概率趋势，然后匹配教师和学生的相关性。
- 数学公式：

$$
L_\text{intra} = \frac{1}{C} \sum_{j=1}^{C} \Big( 1 - \text{corr}(y_j^s, y_j^t) \Big)
$$

其中 $y_j^s, y_j^t \in \mathbb{R}^B$ 表示学生/教师在 batch 中样本对类别 $j$ 的预测概率。

### 1.3 总损失

- 引入权重 $\beta, \gamma$ 以及 softmax 温度 $\tau$：

$$
\mathcal{L}_\text{DIST} = \beta \cdot \tau^2 L_\text{inter} + \gamma \cdot \tau^2 L_\text{intra}
$$

---

## 2. 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DIST(nn.Module):
    """
    蒸馏关系损失模块：基于 Inter-class 和 Intra-class 相关性匹配。
    """
    def __init__(self, beta=2.0, gamma=2.0, tau=6.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.eps = 1e-8  # 防止除零

    def pearson_corr(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Pearson correlation between rows of x and rows of y.
        x, y: shape (N, D)
        Returns: shape (N,) each row's correlation coefficient.
        """
        xm = x.mean(dim=1, keepdim=True)
        ym = y.mean(dim=1, keepdim=True)
        x_c = x - xm
        y_c = y - ym
        num = (x_c * y_c).sum(dim=1)
        denom = torch.sqrt((x_c**2).sum(dim=1) + self.eps) * torch.sqrt((y_c**2).sum(dim=1) + self.eps)
        corr = num / (denom + self.eps)
        return corr

    def forward(self, logits_s: torch.Tensor, logits_t: torch.Tensor) -> torch.Tensor:
        """
        计算蒸馏关系损失。
        logits_s, logits_t: [B, C] 预测概率向量
        """
        B, C = logits_s.shape

        # 1. softmax with temperature
        y_s = F.softmax(logits_s / self.tau, dim=1)
        y_t = F.softmax(logits_t / self.tau, dim=1)

        # 2. Inter-class correlation
        corr_inter = self.pearson_corr(y_s, y_t)  # [B,]
        L_inter = (1.0 - corr_inter).mean()

        # 3. Intra-class correlation
        y_s_T = y_s.transpose(0, 1)  # [C, B]
        y_t_T = y_t.transpose(0, 1)  # [C, B]
        corr_intra = self.pearson_corr(y_s_T, y_t_T)  # [C,]
        L_intra = (1.0 - corr_intra).mean()

        # 4. Temperature scaling
        scale = self.tau**2
        L_inter = scale * L_inter
        L_intra = scale * L_intra

        # 5. Total loss
        loss = self.beta * L_inter + self.gamma * L_intra
        return loss

# ==============================
# 示例
# ==============================
if __name__ == "__main__":
    B, C = 8, 10  # batch size = 8, 类别数 = 10
    logits_t = torch.randn(B, C)  # 教师预测
    logits_s = torch.randn(B, C)  # 学生预测

    dist_loss_fn = DIST(beta=2.0, gamma=2.0, tau=6.0)
    loss = dist_loss_fn(logits_s, logits_t)
    print("DIST loss:", loss.item())
