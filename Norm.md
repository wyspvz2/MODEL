# PyTorch 归一化与正则化详解

本教程介绍 PyTorch 中常用的归一化和正则化方法，包括 BatchNorm、LayerNorm、Dropout，包含数学公式、原理解释和示例代码。

---

# BatchNorm（批归一化）

数学公式：

对于输入 $x \in \mathbb{R}^{N \times C \times H \times W}$（N=batch size, C=通道数）：

$$
\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \quad
y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c
$$

其中：

- $\mu_c = \frac{1}{NHW} \sum_{n,h,w} x_{n,c,h,w}$  
- $\sigma_c^2 = \frac{1}{NHW} \sum_{n,h,w} (x_{n,c,h,w} - \mu_c)^2$  
- $\gamma_c, \beta_c$ 是可学习参数，<img width="33" height="32" alt="image" src="https://github.com/user-attachments/assets/fe35df7d-8149-4366-bfa8-becb8b0b460b" />
 是防止除零的小常数

解释：

- 通过标准化每个批次的特征，缓解内部协变量偏移（Internal Covariate Shift）  
- 可加速训练并提高收敛性  

示例代码：

```python
import torch
import torch.nn as nn

x = torch.randn(4, 3, 8, 8)  # batch_size=4, channels=3, H=W=8
bn = nn.BatchNorm2d(num_features=3)
y = bn(x)
print("BatchNorm 输出形状:", y.shape)
```

# LayerNorm（层归一化）

**数学公式**：

对于输入 $x \in \mathbb{R}^{N \times D}$ 或 $x \in \mathbb{R}^{N \times C \times H \times W}$，在特征维度上归一化：

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad
y_i = \gamma \hat{x}_i + \beta
$$

其中：

- $\mu, \sigma^2$ 是在特征维度上计算的均值和方差  
- $\gamma, \beta$ 是可学习的缩放和平移参数  
- $\epsilon$ 是防止除零的小常数

**解释**：

- 对每个样本独立归一化，不依赖批次大小  
- 常用于 Transformer 或 RNN 结构中  

**示例代码**：

```python
import torch
import torch.nn as nn

x = torch.randn(2, 5)  # batch_size=2, features=5
ln = nn.LayerNorm(normalized_shape=5)
y = ln(x)
print("LayerNorm 输出:\n", y)
```
