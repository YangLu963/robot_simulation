# src/data/data_loader.py
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader

# Import your existing instruction model
from ..robot_instruction import RobotInstruction

class RobotInstructionDataset(Dataset):
    """Dataset for loading and processing structured robot instructions."""

    def __init__(self, data_dir: Path, split: str = 'train', max_samples: int = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.data_file = self.data_dir / f"{split}_instructions.json"
        self.instructions = self._load_instructions()
        
        if max_samples:
            self.instructions = self.instructions[:max_samples]

    def _load_instructions(self) -> List[Dict[str, Any]]:
        """Load and validate instructions using Pydantic model."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Instruction file not found: {self.data_file}")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Validate each instruction using your existing Pydantic model
        validated_instructions = []
        for item in raw_data:
            try:
                # This validates against your RobotInstruction schema
                instruction = RobotInstruction(**item)
                validated_instructions.append(instruction.dict())
            except Exception as e:
                print(f"Invalid instruction skipped: {e}")
                continue
                
        return validated_instructions

    def __len__(self):
        return len(self.instructions)

    def __getitem__(self, idx):
        instruction = self.instructions[idx]
        
        # Convert to format suitable for multimodal model
        return {
            'instruction': instruction,
            'intent': instruction['intent'],
            'objects': instruction['objects'],
            'steps': instruction['steps']
        }

def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 2):
    """Create train/val/test data loaders."""
    data_dir = Path(data_dir)
    
    datasets = {
        'train': RobotInstructionDataset(data_dir, 'train'),
        'val': RobotInstructionDataset(data_dir, 'val'),
        'test': RobotInstructionDataset(data_dir, 'test')
    }
    
    loaders = {
        split: DataLoader(dataset, batch_size=batch_size, 
                         shuffle=(split == 'train'),
                         num_workers=num_workers,
                         collate_fn=collate_instructions)
        for split, dataset in datasets.items()
    }
    
    return loaders

def collate_instructions(batch):
    """Custom collate function for instruction batches."""
    # Implement based on your multimodal model's input requirements
    return batch