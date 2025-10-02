# PyTorch 常用层讲解与示例

本教程介绍 PyTorch 中一些核心神经网络层：Linear、Conv2d、LSTM，包括它们的数学公式和对应实现代码。

---

## 1. Linear 层 (全连接层)

**数学公式**：

输入向量 $x \in \mathbb{R}^{d_{in}}$，输出向量 $y \in \mathbb{R}^{d_{out}}$：

$$
y = x W^\top + b, \quad W \in \mathbb{R}^{d_{out} \times d_{in}}, \quad b \in \mathbb{R}^{d_{out}}
$$

**解释**：

- 每个输出节点是输入节点的加权和加上偏置  
- 常用于全连接网络、MLP 等

**示例代码**：

```python
import torch
import torch.nn as nn

class LinearExample(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.linear(x)

# 示例
x = torch.randn(4, 10)
model_linear = LinearExample(10, 5)
y = model_linear(x)
print("Linear 输出形状:", y.shape)

```
## 2. Conv2d 层 (二维卷积层)

**数学公式**：

输入张量 $X \in \mathbb{R}^{C_{in} \times H \times W}$，卷积核 $K \in \mathbb{R}^{C_{out} \times C_{in} \times k_h \times k_w}$：

$$
Y_{o,i,j} = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} K_{o,c,m,n} \cdot X_{c,i+m,j+n} + b_o
$$

**解释**：

- 对每个输出通道，卷积核在输入各通道上进行加权求和，并加偏置  
- 常用于图像特征提取、卷积神经网络

**示例代码**：

```python
import torch
import torch.nn as nn

class Conv2dExample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv(x)

# 示例
x_img = torch.randn(2, 3, 32, 32)  # batch_size=2, 3通道, 32x32图像
model_conv = Conv2dExample(3, 6, 5)  # 输出6个通道, kernel 5x5
y_img = model_conv(x_img)
print("Conv2d 输出形状:", y_img.shape)

