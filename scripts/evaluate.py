# scripts/evaluate.py
import torch
import json
from pathlib import Path
import yaml

# Import your existing modules
from src.multimodal.multimodal_model import MultimodalModel
from src.reinforcement_learning.sac import SACAgent
from src.data.data_loader import create_data_loaders

def evaluate_policy(model_checkpoint: str, config_path: str = 'config/fusion_config.yaml'):
    """Evaluate the trained policy on test instructions."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model (aligns with your multimodal_model.py)
    model = MultimodalModel(
        vision_dim=config['fusion']['visual_feat_dim'],
        text_dim=config['fusion']['text_feat_dim'],
        hidden_dim=config['fusion']['hidden_dim']
    ).to(device)
    
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()
    
    # Create data loader
    data_loaders = create_data_loaders('data/processed')
    test_loader = data_loaders['test']
    
    success_count = 0
    total_count = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            # TODO: Implement actual evaluation logic
            # This would involve running the policy in environment
            total_count += len(batch)
            
    success_rate = success_count / total_count if total_count > 0 else 0
    print(f"Evaluation completed: Success rate = {success_rate:.3f}")
    
    # Save results
    results = {
        'model': model_checkpoint,
        'success_rate': success_rate,
        'total_episodes': total_count
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return success_rate

if __name__ == '__main__':
    evaluate_policy('../models/multimodal_fusion.pt')