import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.vision_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim))
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, visual_feats, text_feats, text_mask=None):
        # 输入维度:
        # visual_feats: [B, N, D]
        # text_feats: [B, L, D]
        # text_mask: [B, L]
        
        q = self.vision_proj(visual_feats)
        k = self.text_proj(text_feats)
        v = text_feats
        
        attn_output, attn_weights = self.attention(
            query=q,
            key=k,
            value=v,
            key_padding_mask=text_mask
        )
        
        # 残差连接+LayerNorm
        output = self.norm(visual_feats + self.dropout(attn_output))
        return output, attn_weight)
