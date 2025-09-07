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
        
        # 融合视觉、文本、机器人状态
        fusion_input_dim = vision_dim + text_dim + state_dim
        self.fusion = MultimodalFusionNetwork(fusion_input_dim, hidden_dim)
        
        self.fc_out = nn.Linear(hidden_dim, 128)  # 输出128维特征
        
    def forward(self, image, text, state):
        vision_feat = self.vision_encoder(image)
        text_feat = self.text_encoder(text)
        fused_input = torch.cat([vision_feat, text_feat, state], dim=-1)
        fused_feat = self.fusion(fused_input)
        return self.fc_out(fused_feat)

    def encode_obs(self, obs, step):
        """
        将环境的原始观测 obs 和任务步骤 step 编码为 RL 可用特征
        """
        # 1. 取图像 + 机器人低维状态
        image = obs["image"].to(self.device)
        state = obs.get("state", torch.zeros(1, 32)).to(self.device)

        # 2. 从 step 生成指令文本
        if isinstance(step, dict):
            text_input = f"{step.get('action', '')} {step.get('target', {}).get('object_id', '')}"
        else:
            # 如果是 Pydantic 对象
            text_input = f"{step.action} {step.target.object_id if step.target else ''}"

        # 3. 转换为张量
        text_feat = text_input  # TextEncoder 里会处理字符串
        
        with torch.no_grad():
            features = self.forward(image, text_feat, state)
        return features.unsqueeze(0)  # 增加 batch 维度
