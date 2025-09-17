#!/bin/bash
# scripts/download_models.sh
# Download required pretrained models

echo "Downloading pretrained models..."

MODEL_DIR="../models/pretrained"
mkdir -p $MODEL_DIR

# Download vision encoder (ViT) - 使用huggingface的方式
echo "Downloading ViT-B/16..."
python3 -c "
from transformers import ViTFeatureExtractor, ViTModel
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')
feature_extractor.save_pretrained('$MODEL_DIR/vit-base-patch16-224')
model.save_pretrained('$MODEL_DIR/vit-base-patch16-224')
"

# Download text encoder (BERT)
echo "Downloading BERT-base-chinese..."
python3 -c "
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model.save_pretrained('$MODEL_DIR/bert-base-chinese')
tokenizer.save_pretrained('$MODEL_DIR/bert-base-chinese')
"

echo "All models downloaded to $MODEL_DIR"
