# PyTorch 常用层讲解与示例
# 本教程介绍 PyTorch 中一些核心神经网络层：Linear、Conv2d、LSTM，
# 包括它们的数学公式和对应实现代码。

# =====================================================
# 1. Linear 层 (全连接层)
# =====================================================
# 数学公式：
# 输入向量 x ∈ ℝ^(in_features)，输出向量 y ∈ ℝ^(out_features):
# y = x W^T + b,  W ∈ ℝ^(out_features × in_features), b ∈ ℝ^(out_features)
# 解释：
# - 每个输出节点是输入节点的加权和加上偏置
# - 常用于全连接网络、MLP 等

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

# =====================================================
# 2. Conv2d 层 (二维卷积层)
# =====================================================
# 数学公式：
# 输入张量 X ∈ ℝ^(C_in × H × W)，卷积核 K ∈ ℝ^(C_out × C_in × k_h × k_w):
# Y_{o, i, j} = Σ_{c=0}^{C_in-1} Σ_{m=0}^{k_h-1} Σ_{n=0}^{k_w-1} K_{o,c,m,n} * X_{c, i+m, j+n} + b_o
# 解释：
# - 对每个输出通道，卷积核在输入各通道上进行加权求和，并加偏置
# - 常用于图像特征提取、卷积神经网络

class Conv2dExample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # Y = Conv2d(X)
        return self.conv(x)

# 示例
x_img = torch.randn(2, 3, 32, 32)  # batch_size=2, 3通道, 32x32图像
model_conv = Conv2dExample(3, 6, 5)  # 输出6个通道, kernel 5x5
y_img = model_conv(x_img)
print("Conv2d 输出形状:", y_img.shape)  # torch.Size([2, 6, 28, 28])

# =====================================================
# 3. LSTM 层 (长短期记忆网络)
# =====================================================
# 数学公式：
# 输入 x_t, 前一隐藏状态 h_{t-1}, 前一细胞状态 c_{t-1}：
# i_t = σ(W_i x_t + U_i h_{t-1} + b_i)  # 输入门
# f_t = σ(W_f x_t + U_f h_{t-1} + b_f)  # 遗忘门
# o_t = σ(W_o x_t + U_o h_{t-1} + b_o)  # 输出门
# g_t = tanh(W_g x_t + U_g h_{t-1} + b_g) # 候选细胞状态
# c_t = f_t * c_{t-1} + i_t * g_t       # 当前细胞状态
# h_t = o_t * tanh(c_t)                  # 当前隐藏状态
# 解释：
# - LSTM 用门控机制控制信息流，解决长期依赖问题
# - 常用于序列数据，如文本、时间序列

class LSTMExample(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # 输出 h_t, c_t
        out, (h_n, c_n) = self.lstm(x)
        return out, (h_n, c_n)

# 示例
x_seq = torch.randn(3, 5, 10)  # batch_size=3, seq_len=5, input_size=10
model_lstm = LSTMExample(10, 7)
y_seq, (h_n, c_n) = model_lstm(x_seq)
print("LSTM 输出形状:", y_seq.shape)  # torch.Size([3, 5, 7])
print("LSTM 最终隐藏状态形状:", h_n.shape)  # torch.Size([1, 3, 7])
