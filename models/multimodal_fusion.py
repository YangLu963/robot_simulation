import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionNetwork(nn.Module):
    def __init__(self, 
                 visual_feat_dim=768, 
                 text_feat_dim=768,
                 hidden_dim=512,
                 num_heads=8):
        super().__init__()
        
        # 视觉特征适配层
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 文本特征适配层
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 模态门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 输出层
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 输出6DoF抓取位姿
        )

    def forward(self, visual_feats, text_feats, text_mask=None):
        """
        输入:
            visual_feats: [B, N, D_v] 视觉特征
            text_feats: [B, L, D_t] 文本特征  
            text_mask: [B, L] 文本padding掩码
        输出:
            pose: [B, 6] (3D位置+旋转)
            attention_weights: 注意力权重
            fusion_gate: 融合门控值
        """
        # 特征投影
        v_proj = self.visual_proj(visual_feats)
        t_proj = self.text_proj(text_feats)
        
        # 跨模态注意力
        attn_output, _ = self.cross_attn(
            query=v_proj,
            key=t_proj,
            value=t_proj,
            key_padding_mask=text_mask
        )
        
        # 特征池化
        visual_global = torch.mean(attn_output, dim=1)
        text_global = torch.mean(t_proj, dim=1)
        
        # 门控融合
        gate = self.gate(torch.cat([visual_global, text_global], dim=-1))
        fused_feature = gate * visual_global + (1 - gate) * text_global
        
        # 输出预测
        pose = self.output_head(fused_feature)
        return {
            'pose': pose,
            'attention_weights': attn_output,
            'fusion_gate': gate
        }

class MultitaskHead(nn.Module):
    """用于同时预测动作类型和目标位姿"""
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.action_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        return {
            'action_logits': self.action_head(x),
            'target_pose': self.regression_head(x)
        }
