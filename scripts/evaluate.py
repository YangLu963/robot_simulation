# scripts/evaluate.py
import torch
import json
from pathlib import Path
import yaml
import sys
import os

# 修复导入路径：添加src目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.multimodal.multimodal_model import MultimodalModel
from src.reinforcement_learning.sac import SACAgent
from src.data.data_loader import create_data_loaders
from src.reinforcement_learning.environment import PyBulletRobotEnv

def evaluate_policy(model_checkpoint: str, config_path: str = 'config/fusion_config.yaml'):
    """Evaluate the trained policy on test instructions."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化环境
    env = PyBulletRobotEnv(render=False)
    
    # 初始化多模态编码器
    multimodal_encoder = MultimodalModel(
        vision_dim=config['fusion']['visual_feat_dim'],
        text_dim=config['fusion']['text_feat_dim'],
        hidden_dim=config['fusion']['hidden_dim']
    ).to(device)
    
    # 加载训练好的策略
    obs_dim = 128  # 多模态特征维度
    action_dim = env.action_space.shape[0]
    policy = SACAgent(obs_dim, action_dim, device=device)
    
    # 加载模型权重
    policy.actor.load_state_dict(torch.load(model_checkpoint, map_location=device))
    policy.actor.eval()
    
    # 创建数据加载器
    data_loaders = create_data_loaders('data/processed')
    test_loader = data_loaders['test']
    
    success_count = 0
    total_count = 0
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for batch in test_loader:
            for instruction in batch:
                try:
                    # 在实际环境中执行任务
                    obs = env.reset()
                    done = False
                    success = False
                    
                    while not done:
                        # 使用多模态编码器处理观测和指令
                        obs_features = multimodal_encoder.encode_obs(
                            obs={'image': obs, 'state': obs},  # 需要根据实际观测调整
                            step={'action': 'execute', 'target': {'object_id': 0}}  # 简化步骤
                        )
                        
                        # 策略选择动作
                        action = policy.choose_action(obs_features)
                        
                        # 执行动作
                        next_obs, reward, done, info = env.step(action)
                        obs = next_obs
                        
                        # 简单的成功判断：如果奖励达到阈值则认为成功
                        if reward > 0.9:  # 需要根据实际任务调整
                            success = True
                            break
                    
                    if success:
                        success_count += 1
                    total_count += 1
                    
                except Exception as e:
                    print(f"Evaluation failed for instruction: {e}")
                    continue
            
    success_rate = success_count / total_count if total_count > 0 else 0
    print(f"Evaluation completed: Success rate = {success_rate:.3f} ({success_count}/{total_count})")
    
    # Save results
    results = {
        'model': model_checkpoint,
        'success_rate': success_rate,
        'success_count': success_count,
        'total_episodes': total_count
    }
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return success_rate

if __name__ == '__main__':
    evaluate_policy('../models/multimodal_fusion.pt')
