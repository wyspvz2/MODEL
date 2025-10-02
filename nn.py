import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义网络结构
        self.fc1 = nn.Linear(28*28, 128)  # 全连接层1
        self.relu = nn.ReLU()             # 激活函数
        self.fc2 = nn.Linear(128, 10)    # 全连接层2
        self.softmax = nn.Softmax(dim=1) # 输出概率归一化

    def forward(self, x):
        # 前向计算流程
        x = x.view(-1, 28*28)  # 展平输入为 (batch_size, 784)
        x = self.fc1(x)        # 第一个全连接层
        x = self.relu(x)       # ReLU 激活
        x = self.fc2(x)        # 第二个全连接层
        x = self.softmax(x)    # Softmax 转为概率分布
        return x

# 创建模型实例
model = SimpleNN()
