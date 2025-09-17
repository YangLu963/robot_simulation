# === src/reinforcement_learning/train.py ===
import torch
import numpy as np
from .sac_agent import SACAgent  # 修复：相对导入
from .environment import PyBulletRobotEnv  # 修复：相对导入
from collections import deque
import random

def train():
    env = PyBulletRobotEnv(render=False)
    obs_dim = env.observation_space.shape[0]  # 保持原有关节状态维度
    action_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, action_dim)

    replay_buffer = deque(maxlen=100000)
    episodes = 50
    batch_size = 64

    for ep in range(episodes):
        obs = env.reset()  # 关节状态观测
        episode_reward = 0
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent.device)
            action = agent.choose_action(obs_tensor).cpu().numpy().flatten()

            next_obs, reward, done, info = env.step(action)
            replay_buffer.append((obs, action, reward, next_obs, done))

            obs = next_obs
            episode_reward += reward

            # 更新 SAC
            if len(replay_buffer) > batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

                states = torch.tensor(states, dtype=torch.float32).to(agent.device)
                actions = torch.tensor(actions, dtype=torch.float32).to(agent.device)
                rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(agent.device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(agent.device)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(agent.device)

                agent.update(states, actions, rewards, next_states, dones)

        print(f"Episode {ep+1}, Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    train()
