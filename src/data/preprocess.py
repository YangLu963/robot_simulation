# src/data/preprocess.py
from pathlib import Path
import json
from typing import List, Dict, Any
import random

from .robot_instruction import RobotInstruction  # Your existing model

def prepare_instruction_data(raw_data_dir: Path, output_dir: Path, train_ratio: float = 0.8):
    """Prepare train/val/test splits from raw instruction data."""
    
    raw_file = raw_data_dir / "instructions.json"
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_file}")
    
    with open(raw_file, 'r', encoding='utf-8') as f:
        raw_instructions = json.load(f)
    
    # Validate and filter instructions
    valid_instructions = []
    for item in raw_instructions:
        try:
            RobotInstruction(**item)  # Validation
            valid_instructions.append(item)
        except Exception as e:
            print(f"Invalid instruction skipped: {e}")
            continue
    
    # Shuffle and split
    random.shuffle(valid_instructions)
    n_total = len(valid_instructions)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * 0.1)
    
    splits = {
        'train': valid_instructions[:n_train],
        'val': valid_instructions[n_train:n_train + n_val],
        'test': valid_instructions[n_train + n_val:]
    }
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, data in splits.items():
        output_file = output_dir / f"{split_name}_instructions.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Data preparation complete: {n_total} instructions")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")