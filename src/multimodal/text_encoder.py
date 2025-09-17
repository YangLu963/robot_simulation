import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class TextEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.model = BertModel.from_pretrained('bert-base-chinese').to(device)
        self.device = device

    def forward(self, text):
        if isinstance(text, str):
            text = [text]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self.device)
        
        outputs = self.model(**inputs)
        return {
            'text_embedding': outputs.last_hidden_state[:, 0],
            'word_embeddings': outputs.last_hidden_state,
            'attention_mask': inputs['attention_mask']
        }
