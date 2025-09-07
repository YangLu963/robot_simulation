import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel
from torchvision import transforms

class VisionEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224').to(device)
        self.device = device
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, rgb_image):
        # 输入: PIL.Image 或 [H,W,C] numpy数组
        inputs = self.feature_extractor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 标准化处理
        pixel_values = inputs['pixel_values']
        pixel_values = self.normalize(pixel_values.squeeze(0)).unsqueeze(0)
        
        outputs = self.model(pixel_values=pixel_values)
        return {
            'global_feature': outputs.last_hidden_state[:, 0],
            'patch_features': outputs.last_hidden_state[:, 1:],
            'attention_weights': outputs.attentions
        }