#!/bin/bash
# scripts/download_models.sh
# Download required pretrained models

echo "Downloading pretrained models..."

MODEL_DIR="../models/pretrained"
mkdir -p $MODEL_DIR

# Download vision encoder (ViT)
echo "Downloading ViT-B/16..."
wget -P $MODEL_DIR https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-16.pt

# Download text encoder (BERT)
echo "Downloading BERT-base-chinese..."
python -c "
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model.save_pretrained('$MODEL_DIR/bert-base-chinese')
tokenizer.save_pretrained('$MODEL_DIR/bert-base-chinese')
"

echo "All models downloaded to $MODEL_DIR"