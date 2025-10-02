# PyTorch 归一化与正则化详解

本教程介绍 PyTorch 中常用的归一化和正则化方法，包括 BatchNorm、LayerNorm、Dropout，包含数学公式、原理解释和示例代码。

---

**BatchNorm（批归一化）**

数学公式：

对于输入 $x \in \mathbb{R}^{N \times C \times H \times W}$（N=batch size, C=通道数）：

$$
\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}, \quad
y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c
$$

其中：

- $\mu_c = \frac{1}{NHW} \sum_{n,h,w} x_{n,c,h,w}$  
- $\sigma_c^2 = \frac{1}{NHW} \sum_{n,h,w} (x_{n,c,h,w} - \mu_c)^2$  
- $\gamma_c, \beta_c$ 是可学习参数，$\epsilon$ 是防止除零的小常数

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
