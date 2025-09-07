# src/visualization/plot_utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def plot_training_history(history_path: Path, save_path: Path = None):
    """Plot training history from saved logs."""
    if not history_path.exists():
        print("Training history not found")
        return
    
    # Read and parse training logs
    # This would need to match your actual logging format
    epochs = []
    rewards = []
    
    with open(history_path, 'r') as f:
        for line in f:
            if 'Reward:' in line:
                parts = line.split('Reward:')
                if len(parts) > 1:
                    reward = float(parts[1].strip())
                    rewards.append(reward)
                    epochs.append(len(epochs) + 1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rewards, 'b-', label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Performance')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_instruction_stats(data_dir: Path):
    """Visualize statistics about robot instructions."""
    from src.data.data_loader import RobotInstructionDataset
    
    dataset = RobotInstructionDataset(data_dir, 'train')
    
    # Count intent types
    intent_counts = {}
    for item in dataset:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(intent_counts.keys(), intent_counts.values())
    plt.title('Instruction Intent Distribution')
    plt.xlabel('Intent')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()