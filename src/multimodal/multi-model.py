# === src/multimodal/multimodal_model.py ===
import torch
import torch.nn as nn
from .vision_encoder import VisionEncoder
from .text_encoder import TextEncoder
from .fusion_network import MultimodalFusionNetwork

class MultimodalModel(nn.Module):
    def __init__(self, vision_dim=256, text_dim=256, hidden_dim=512, state_dim=32, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.vision_encoder = VisionEncoder(output_dim=vision_dim)
        self.text_encoder = TextEncoder(output_dim=text_dim)
        
        # 修正：MultimodalFusionNetwork 需要分别处理视觉和文本特征
        self.fusion = MultimodalFusionNetwork(
            visual_feat_dim=vision_dim,
            text_feat_dim=text_dim,
            hidden_dim=hidden_dim
        )
        
        # 输出层需要调整维度以匹配融合网络的输出
        self.fc_out = nn.Linear(hidden_dim, 128)  # 输出128维特征
        
    def forward(self, image, text, state):
        # 修正：正确提取视觉和文本特征
        vision_output = self.vision_encoder(image)
        text_output = self.text_encoder(text)
        
        # 提取全局特征
        vision_feat = vision_output['global_feature']  # [B, vision_dim]
        text_feat = text_output['text_embedding']      # [B, text_dim]
        text_mask = text_output.get('attention_mask')  # [B, seq_len]
        
        # 修正：使用正确的融合网络接口
        fused_output = self.fusion(
            visual_feats=vision_feat.unsqueeze(1),  # [B, 1, vision_dim]
            text_feats=text_output['word_embeddings'],  # [B, L, text_dim]
            text_mask=text_mask
        )
        
        # 拼接机器人状态
        fused_with_state = torch.cat([fused_output['pose'], state], dim=-1)
        return self.fc_out(fused_with_state)

    def encode_obs(self, obs, step):
        """
        将环境的原始观测 obs 和任务步骤 step 编码为 RL 可用特征
        """
        # 1. 取图像 + 机器人低维状态
        image = obs["image"].to(self.device)
        state = obs.get("state", torch.zeros(1, 32).to(self.device))
        
        # 2. 从 step 生成指令文本
        if isinstance(step, dict):
            text_input = f"{step.get('action', '')} {step.get('target', {}).get('object_id', '')}"
        else:
            text_input = f"{step.action} {step.target.object_id if step.target else ''}"
        
        # 3. 修正：正确调用文本编码器
        with torch.no_grad():
            text_output = self.text_encoder(text_input)
            vision_output = self.vision_encoder(image)
            
            # 提取特征
            vision_feat = vision_output['global_feature']
            text_feat = text_output['text_embedding']
            text_mask = text_output.get('attention_mask')
            
            # 使用融合网络
            fused_output = self.fusion(
                visual_feats=vision_feat.unsqueeze(0).unsqueeze(1),  # [1, 1, vision_dim]
                text_feats=text_output['word_embeddings'],  # [1, L, text_dim]
                text_mask=text_mask
            )
            
            # 拼接状态特征
            features = torch.cat([fused_output['pose'], state], dim=-1)
            features = self.fc_out(features)
            
        return features.unsqueeze(0)  # 增加 batch 维度
