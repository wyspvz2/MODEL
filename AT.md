# AT: Attention Transfer

## 1. 方法概述

AT（Attention Transfer）是一种 **中间特征蒸馏方法**，提出者为 Zagoruyko & Komodakis（2017）。  
它的核心思想是将教师网络中间层的 **空间注意力图（Attention Map）** 传递给学生网络，让学生学习教师关注的空间区域。

注意力图通常从卷积特征生成，例如对每个通道求平方后求和并归一化：

$$
A(F) = \frac{\sum_{c=1}^{C} F_c^2}{\left\|\sum_{c=1}^{C} F_c^2 \right\|_2}
$$

- $F \in \mathbb{R}^{C \times H \times W}$ 是某层卷积特征图  
- $A(F) \in \mathbb{R}^{H \times W}$ 是注意力图  
- $\|\cdot\|_2$ 表示 L2 范数  

最终 AT 损失为 **学生与教师注意力图的 L2 距离**：

$$
\mathcal{L}_{\text{AT}} = \frac{1}{B} \sum_{i=1}^{B} \left\| A(F_i^s) - A(F_i^t) \right\|_2^2
$$

其中：
- $B$ 是 batch size  
- $F_i^s$ 和 $F_i^t$ 分别是学生和教师的第 $i$ 个样本的中间特征  

---

## 2. 代码实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ATLoss(nn.Module):
    """
    Attention Transfer Loss
    匹配学生和教师的中间特征注意力图
    """
    def __init__(self, eps=1e-8):
        super(ATLoss, self).__init__()
        self.eps = eps

    def attention_map(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Compute attention map from feature map.
        feature: [B, C, H, W]
        returns: [B, H, W]
        """
        # 平方求和
        att = feature.pow(2).sum(dim=1)
        # L2 归一化
        att = att / (att.norm(p=2, dim=(1,2), keepdim=True) + self.eps)
        return att

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
        """
        feat_s, feat_t: [B, C, H, W]
        """
        A_s = self.attention_map(feat_s)
        A_t = self.attention_map(feat_t)
        # L2 距离
        loss = F.mse_loss(A_s, A_t)
        return loss

# ==============================
# 示例
# ==============================
if __name__ == "__main__":
    B, C, H, W = 4, 16, 8, 8
    feat_t = torch.randn(B, C, H, W)
    feat_s = torch.randn(B, C, H, W)

    at_loss_fn = ATLoss()
    loss = at_loss_fn(feat_s, feat_t)
    print("AT loss:", loss.item())
