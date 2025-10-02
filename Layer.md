# PyTorch 常用层讲解与示例

本教程介绍 PyTorch 中一些核心神经网络层：Linear、Conv2d、LSTM，  
包括它们的数学公式、文字解释和对应实现代码示例。

---

## 1. Linear 层（全连接层）

### 数学公式

输入向量 $x \in \mathbb{R}^{\text{in\_features}}$，输出向量 $y \in \mathbb{R}^{\text{out\_features}}$:

$$
y = x W^T + b, \quad 
W \in \mathbb{R}^{\text{out\_features} \times \text{in\_features}}, \quad 
b \in \mathbb{R}^{\text{out\_features}}
$$

解释：

- 每个输出节点是输入节点的加权和加上偏置  
- 常用于全连接网络、MLP 等

### 示例代码

```python
import torch
import torch.nn as nn

class LinearExample(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        # y = x W^T + b
        return self.linear(x)

# 示例
x = torch.randn(4, 10)  # batch_size=4, in_features=10
model_linear = LinearExample(10, 5)
y = model_linear(x)
print("Linear 输出形状:", y.shape)  # torch.Size([4, 5])
