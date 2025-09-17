# === src/reinforcement_learning/train.py ===
import torch
import numpy as np
from .sac_agent import SACAgent  # 修复：相对导入
from .environment import PyBulletRobotEnv  # 修复：相对导入
from collections import deque
import random
import sys
from pathlib import Path

# 添加路径以便导入多模态模块
sys.path.append(str(Path(__file__).parent.parent))
from src.multimodal.multimodal_model import MultimodalModel

def train():
    env = PyBulletRobotEnv(render=False)
    
    # 初始化多模态编码器（修复问题2）
    multimodal_encoder = MultimodalModel(device="cuda")
    
    # 使用多模态特征维度而不是原始观测维度（修复问题1）
    obs_dim = 128  # multimodal_model.fc_out的输出维度
    action_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, action_dim, device="cuda")

    replay_buffer = deque(maxlen=100000)
    episodes = 50
    batch_size = 64

    for ep in range(episodes):
        # 获取原始观测（包含图像和状态）
        raw_obs = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # 使用多模态编码器编码观测（修复问题1、2）
            with torch.no_grad():
                # 假设环境返回的obs是字典：{'image': image_data, 'state': joint_states}
                obs_features = multimodal_encoder.encode_obs(raw_obs, step=None)  # 需要根据实际情况提供step
                
            action = agent.choose_action(obs_features).cpu().numpy().flatten()

            next_raw_obs, reward, done, info = env.step(action)
            
            # 编码下一个观测
            with torch.no_grad():
                next_obs_features = multimodal_encoder.encode_obs(next_raw_obs, step=None)
            
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

            # 更新 SAC
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32).to(agent.device)
                actions = torch.tensor(np.array(actions), dtype=torch.float32).to(agent.device)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(agent.device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(agent.device)
                dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(agent.device)

                agent.update(states, actions, rewards, next_states, dones)

        print(f"Episode {ep+1}, Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    train()
