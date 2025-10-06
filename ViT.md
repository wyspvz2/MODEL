#  Transformer Encoder 的数学过程和代码示例

---

##  一、总体目标

Transformer Encoder 的核心目标是：

> 将输入序列（如句子或图像 patch 向量）转化为全局上下文相关的高层语义表示。

换句话说：  
每个位置（单词 / patch）最终都能“理解”整个序列的语义关系。

---

##  二、输入形式

输入通常是嵌入向量序列：

$$X = [x_1, x_2, \dots, x_N] \in \mathbb{R}^{N \times D}$$

其中：
- $N$：序列长度（NLP 中是词数，ViT 中是 patch 数）
- $D$：embedding 维度（如 512、768）

如果是 ViT，还需要：
1. 添加一个 `[CLS]` token：
   $$X' = [x_{cls}; x_1; x_2; \dots; x_N]$$
2. 加上位置编码：
   $$Z_0 = X' + E_{pos}$$

 **目的：**
- `[CLS]` 用于汇聚全局特征。
- 位置编码 $E_{pos}$ 保留序列顺序信息（因为注意力机制本身是无序的）。

---

##  三、Encoder 的层结构

每一层（Layer）包含两个主要部分：

1. **多头自注意力（Multi-Head Self-Attention）**
2. **前馈网络（Feed-Forward Network, FFN）**

并且都包含：
- 残差连接（Residual Connection）
- 层归一化（LayerNorm）

---

##  四、Self-Attention 的数学过程与目的

### Step 1️: 线性映射（构造 Q, K, V）

对输入 $Z_{l-1}$ 做三次线性变换：

$$Q = Z_{l-1} W_Q, \quad K = Z_{l-1} W_K, \quad V = Z_{l-1} W_V$$

其中：
- $W_Q, W_K, W_V \in \mathbb{R}^{D \times D_h}$
- $D_h = D / h$ 是单个注意力头的维度
- $h$：注意力头的数量

 **目的：**
- $Q$（Query）表示“我想关注什么”
- $K$（Key）表示“我能提供什么”
- $V$（Value）表示“我包含的信息内容”

---

### Step 2️: 相似度计算（Query 与 Key）

计算注意力得分矩阵：

$$S = \frac{Q K^T}{\sqrt{D_h}}$$

然后进行 softmax 归一化：

$$A = \text{softmax}(S)$$

 **目的：**
- 衡量每个 token 与其它 token 的相关性；
- $\sqrt{D_h}$ 防止内积值过大，稳定梯度；
- softmax 使得权重在 [0,1] 之间且可解释为“注意力分布”。

---

### Step 3️: 加权求和（根据注意力聚合信息）

$$Z' = A V$$

 **目的：**
每个 token 得到整个序列的信息加权汇总。  
即每个位置“看到了”其它所有位置的内容。

---

### Step 4️: 多头机制（Multi-Head）

$$\text{MHA}(Z) = [Z'_1; Z'_2; \dots; Z'_h] W_O$$

 **目的：**
不同的注意力头在关注不同的语义关系（局部、全局、颜色、形状、上下文等）。

---

```python

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # QKV 线性层
        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, D = x.shape  # batch, seq_len, embed_dim
        
        # 生成 Q, K, V
        Q = self.W_Q(x)  # [B, N, D]
        K = self.W_K(x)
        V = self.W_V(x)
        
        # 分头
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, D_h]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, h, N, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # 加权求和
        out = torch.matmul(attn_weights, V)  # [B, h, N, D_h]
        
        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        out = self.W_O(out)
        return out

```

##  五、残差连接与归一化

每一层都会加上输入的残差，并做 LayerNorm：

$$Z'_l = \text{LayerNorm}(Z_{l-1} + \text{MHA}(Z_{l-1}))$$

 **目的：**
- 残差连接保证信息流通、防止梯度消失；
- LayerNorm 稳定训练，使分布平衡。

---

##  六、前馈网络（Feed-Forward Network, FFN）

对每个位置独立地做非线性变换：

$$\text{FFN}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2$$

$$Z_l = \text{LayerNorm}(Z'_l + \text{FFN}(Z'_l))$$

 **目的：**
- 对每个 token 进行更复杂的特征映射；
- 增加非线性表达能力；
- 第二次残差保证特征不丢失。

```python
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```
---

##  七、数学流程总结

完整的第 $l$ 层 Encoder 公式：

$$
\begin{aligned}
Q &= Z_{l-1} W_Q, \quad K = Z_{l-1} W_K, \quad V = Z_{l-1} W_V \\
A &= \text{softmax}\left(\frac{QK^T}{\sqrt{D_h}}\right) \\
Z'_l &= \text{LayerNorm}(Z_{l-1} + A V W_O) \\
Z_l &= \text{LayerNorm}(Z'_l + \text{FFN}(Z'_l))
\end{aligned}
$$

---

##  八、最终输出与任务目标

经过 $L$ 层编码后：

$$Z_L = \text{Encoder}(Z_0)$$

- 若是 **NLP**：  
  输出序列中每个位置的特征可用于翻译、文本生成等。
- 若是 **ViT**：  
  取 `[CLS]` 位置的向量：
  $$y = \text{Linear}(Z_L^{[CLS]})$$  
  作为分类结果。

---

##  九、Encoder 的目的总结表

| 模块 | 数学形式 | 目的 |
|------|-----------|------|
| Patch Embedding / Token Embedding | $X \to Z_0$ | 将输入转为向量 |
| Positional Encoding | $Z_0 + E_{pos}$ | 保留顺序信息 |
| Q, K, V 线性变换 | $Q=ZW_Q, K=ZW_K, V=ZW_V$ | 表征关注关系 |
| 注意力权重 | $A = \text{softmax}(QK^T / \sqrt{D_h})$ | 计算相关性 |
| 加权求和 | $AV$ | 汇聚上下文信息 |
| 多头拼接 | $[AV_1;\dots;AV_h]W_O$ | 多视角建模 |
| 残差 + Norm | $Z' = \text{LN}(Z+AVW_O)$ | 稳定梯度 |
| FFN + 残差 | $Z = \text{LN}(Z'+\text{FFN}(Z'))$ | 非线性表达 |

---


