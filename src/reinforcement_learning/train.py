# === src/reinforcement_learning/train.py ===
import torch
import numpy as np
import random
from collections import deque
from pathlib import Path
import sys

# 修复导入路径
sys.path.append(str(Path(__file__).parent.parent))
from .sac_agent import SACAgent
from .environment import PyBulletRobotEnv
from src.multimodal.multimodal_model import MultimodalModel
from src.data.data_loader import RobotInstructionDataset

def train():
    # 初始化环境
    env = PyBulletRobotEnv(render=False)
    
    # 初始化多模态编码器
    multimodal_encoder = MultimodalModel(device="cuda")
    
    # 加载任务指令数据
    data_dir = Path("data/processed")
    dataset = RobotInstructionDataset(data_dir, 'train')
    
    # SAC智能体（使用多模态特征维度）
    obs_dim = 128  # 与multimodal_model.fc_out输出维度一致
    action_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, action_dim, device="cuda")

    replay_buffer = deque(maxlen=100000)
    episodes = 50
    batch_size = 64

    for ep in range(episodes):
        # 随机选择一个任务指令
        task_idx = random.randint(0, len(dataset) - 1)
        task_instruction = dataset[task_idx]
        
        # 重置环境
        raw_obs = env.reset()
        episode_reward = 0
        done = False
        step_idx = 0

        while not done:
            # 使用多模态编码器编码观测和任务指令
            with torch.no_grad():
                obs_features = multimodal_encoder.encode_obs(
                    obs={"image": raw_obs, "state": raw_obs},  # 需要根据实际观测结构调整
                    step=task_instruction['steps'][min(step_idx, len(task_instruction['steps'])-1)]
                )
            
            # SAC选择动作
            action = agent.choose_action(obs_features).flatten()

            # 执行动作
            next_raw_obs, reward, done, info = env.step(action)
            
            # 编码下一个观测
            with torch.no_grad():
                next_obs_features = multimodal_encoder.encode_obs(
                    obs={"image": next_raw_obs, "state": next_raw_obs},
                    step=task_instruction['steps'][min(step_idx, len(task_instruction['steps'])-1)]
                )
            
            # 存储多模态特征经验
            replay_buffer.append((
                obs_features.cpu().numpy(), 
                action, 
                reward, 
                next_obs_features.cpu().numpy(), 
                done
            ))

            raw_obs = next_raw_obs
            episode_reward += reward
            step_idx += 1

            # 更新SAC
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states = torch.tensor(np.array(states), dtype=torch.float32).to(agent.device)
                actions = torch.tensor(np.array(actions), dtype=torch.float32).to(agent.device)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(agent.device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(agent.device)
                dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(agent.device)

                agent.update(states, actions, rewards, next_states, dones)

        print(f"Episode {ep+1}, Reward: {episode_reward:.2f}, Task: {task_instruction['intent']}")

    # 保存模型
    torch.save(multimodal_encoder.state_dict(), "models/multimodal_encoder.pth")
    torch.save(agent.actor.state_dict(), "models/sac_actor.pth")

if __name__ == "__main__":
    train()
