import torch
import torch.nn as nn
import math

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

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.mha = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Multi-Head Attention + Residual + LayerNorm
        x = self.norm1(x + self.mha(x))
        # Feed-Forward + Residual + LayerNorm
        x = self.norm2(x + self.ffn(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_hidden_dim, seq_len):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_hidden_dim)
            for _ in range(num_layers)
        ])
        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
    def forward(self, x):
        # x: [B, N, D]
        x = x + self.pos_embed[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, num_layers=12, num_heads=12, ff_hidden_dim=3072, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_hidden_dim, seq_len=(img_size // patch_size)**2 + 1)
        self.mlp_head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # [B, N, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, D]
        x = self.encoder(x)
        cls_output = x[:, 0]  # [CLS] token
        out = self.mlp_head(cls_output)
        return out

