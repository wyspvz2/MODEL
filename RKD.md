# Relational Knowledge Distillation (RKD)

## 1. 方法概述

Relational Knowledge Distillation (RKD) 是一种**关系蒸馏方法**，与传统知识蒸馏不同，它不直接匹配教师和学生的输出概率，而是通过匹配 **特征间的关系** 来传递知识。  

RKD 主要关注两种关系：

1. **距离关系（Distance-based RKD, RKD-D）**  
   - 学生与教师在特征空间中的样本间距离保持一致  
   - 强调样本间的相对位置关系  
2. **角度关系（Angle-based RKD, RKD-A）**  
   - 学生与教师在特征空间中样本向量构成的角度保持一致  
   - 强调特征分布的几何结构

---

## 2. 数学公式

设教师和学生中间特征为：

$$
X^t = [x_1^t, x_2^t, ..., x_B^t] \in \mathbb{R}^{B \times D}, \quad 
X^s = [x_1^s, x_2^s, ..., x_B^s] \in \mathbb{R}^{B \times D}
$$

### 2.1 距离关系（RKD-D）

对每对样本 $(i,j)$，计算特征距离：

$$
d_{ij}^t = \| x_i^t - x_j^t \|_2, \quad
d_{ij}^s = \| x_i^s - x_j^s \|_2
$$

归一化距离（平均距离）：

$$
\tilde{d}_{ij}^t = \frac{d_{ij}^t}{\mu^t}, \quad 
\tilde{d}_{ij}^s = \frac{d_{ij}^s}{\mu^s}, \quad 
\mu^t = \frac{2}{B(B-1)} \sum_{i<j} d_{ij}^t
$$

RKD-D 损失为 **Smooth L1**（Huber）：

$$
\mathcal{L}_\text{RKD-D} = \frac{2}{B(B-1)} \sum_{i<j} \text{SmoothL1}(\tilde{d}_{ij}^s - \tilde{d}_{ij}^t)
$$

### 2.2 角度关系（RKD-A）

对于三元组 $(i,j,k)$，计算教师和学生特征构成的角度：

$$
\theta_{ijk}^t = \cos^{-1} \frac{(x_i^t - x_j^t) \cdot (x_k^t - x_j^t)}{\|x_i^t - x_j^t\|_2 \, \|x_k^t - x_j^t\|_2}
$$

对应学生角度 $\theta_{ijk}^s$，RKD-A 损失：

$$
\mathcal{L}_\text{RKD-A} = \frac{1}{\text{#triplets}} \sum_{i,j,k} \text{SmoothL1}(\theta_{ijk}^s - \theta_{ijk}^t)
$$

### 2.3 总损失

一般将两部分加权：

$$
\mathcal{L}_\text{RKD} = \lambda_d \mathcal{L}_\text{RKD-D} + \lambda_a \mathcal{L}_\text{RKD-A}
$$

---

## 3. 张量示例

假设 batch size $B=3$，特征维度 $D=2$：

教师特征：

$$
X^t = \begin{bmatrix} [1.0, 0.0] \\ [0.0, 1.0] \\ [1.0, 1.0] \end{bmatrix}, 
\quad
X^s = \begin{bmatrix} [0.9, 0.1] \\ [0.1, 0.9] \\ [1.1, 1.0] \end{bmatrix}
$$

1. RKD-D 距离对 $(0,1)$：

$$
d_{01}^t = \sqrt{(1-0)^2 + (0-1)^2} = \sqrt{2} \approx 1.414
$$
$$
d_{01}^s = \sqrt{(0.9-0.1)^2 + (0.1-0.9)^2} = \sqrt{1.28} \approx 1.131
$$

平均距离 $\mu^t = \frac{d_{01}^t + d_{02}^t + d_{12}^t}{3} \approx 1.414$，归一化：

$$
\tilde{d}_{01}^t \approx 1.414 / 1.414 = 1.0, \quad 
\tilde{d}_{01}^s \approx 1.131 / 1.0 = 1.131
$$

RKD-D 对应损失：

$$
\text{SmoothL1}(1.131 - 1.0) \approx 0.131
$$

2. RKD-A 角度计算示例：

教师角度 $(0,1,2)$：

$$
\vec{01} = [-1,1], \quad \vec{21} = [1,0] \implies
\theta_{012}^t = \cos^{-1} \frac{(-1*1 + 1*0)}{\sqrt{2}\cdot1} = \cos^{-1}(-0.707) \approx 135^\circ
$$

学生角度类似计算，然后 SmoothL1 差值。

---

## 4. PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations, permutations

class RKDLoss(nn.Module):
    """Relational Knowledge Distillation (Distance + Angle)"""
    def __init__(self, lambda_d=25.0, lambda_a=50.0):
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_a = lambda_a
        self.smooth_l1 = nn.SmoothL1Loss()

    def pairwise_distance(self, x):
        """x: [B, D]"""
        B = x.size(0)
        dist = torch.zeros(B, B, device=x.device)
        for i, j in combinations(range(B), 2):
            dist_ij = (x[i] - x[j]).norm()
            dist[i,j] = dist[j,i] = dist_ij
        return dist

    def forward(self, fS, fT):
        B, D = fS.shape

        # -------- RKD-D --------
        dT = self.pairwise_distance(fT)
        dS = self.pairwise_distance(fS)
        meanT = dT[dT>0].mean()
        meanS = dS[dS>0].mean()
        dT_norm = dT / (meanT + 1e-8)
        dS_norm = dS / (meanS + 1e-8)
        mask = torch.ones_like(dT_norm, dtype=torch.bool)
        mask.fill_diagonal_(0)
        loss_d = self.smooth_l1(dS_norm[mask], dT_norm[mask])

        # -------- RKD-A --------
        loss_a = 0.0
        triplets = list(permutations(range(B), 3))
        if len(triplets) > 0:
            angles_T, angles_S = [], []
            for i,j,k in triplets:
                vT1 = fT[i] - fT[j]
                vT2 = fT[k] - fT[j]
                vS1 = fS[i] - fS[j]
                vS2 = fS[k] - fS[j]
                cosT = F.cosine_similarity(vT1.unsqueeze(0), vT2.unsqueeze(0))
                cosS = F.cosine_similarity(vS1.unsqueeze(0), vS2.unsqueeze(0))
                angles_T.append(cosT)
                angles_S.append(cosS)
            angles_T = torch.cat(angles_T)
            angles_S = torch.cat(angles_S)
            loss_a = self.smooth_l1(angles_S, angles_T)

        loss = self.lambda_d * loss_d + self.lambda_a * loss_a
        return loss

# =========================
# 示例
# =========================
fT = torch.tensor([[1.0,0.0],[0.0,1.0],[1.0,1.0]])
fS = torch.tensor([[0.9,0.1],[0.1,0.9],[1.1,1.0]])

rkd_loss_fn = RKDLoss(lambda_d=25.0, lambda_a=50.0)
loss = rkd_loss_fn(fS, fT)
print("RKD loss:", loss.item())
