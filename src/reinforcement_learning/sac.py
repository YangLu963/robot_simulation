# sac.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# 策略网络 (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        # 你的观测是图像，所以需要一个视觉编码器。
        # 这里的 obs_dim 其实是图像编码器输出的特征维度
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出动作的均值和标准差
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_features):
        x = F.relu(self.fc1(obs_features))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # 限制标准差的范围以保证训练稳定
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, obs_features):
        mean, std = self.forward(obs_features)
        normal_dist = Normal(mean, std)
        # 从正态分布中采样动作
        action = normal_dist.rsample()
        # 计算动作的对数概率，这是SAC算法的核心
        log_prob = normal_dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
d
# 价值网络 (Critic)
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        # Q1 网络
        self.fc1_q1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)

        # Q2 网络
        self.fc1_q2 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs_features, action):
        # 将观测和动作拼接在一起作为输入
        x_q1 = torch.cat([obs_features, action], dim=-1)
        x_q1 = F.relu(self.fc1_q1(x_q1))
        x_q1 = F.relu(self.fc2_q1(x_q1))
        q1 = self.fc3_q1(x_q1)

        x_q2 = torch.cat([obs_features, action], dim=-1)
        x_q2 = F.relu(self.fc1_q2(x_q2))
        x_q2 = F.relu(self.fc2_q2(x_q2))
        q2 = self.fc3_q2(x_q2)
        
        return q1, q2

# SAC智能体
class SACAgent:
    def __init__(self, obs_dim, action_dim, device="cuda", hidden_dim=256,
                 gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, batch_size=256):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        # 初始化 Actor 网络和两个 Q 网络
        self.actor = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q_network1 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q_network2 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        
        # 目标 Q 网络，用于稳定训练
        self.target_q_network1 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_q_network2 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_q_network1.load_state_dict(self.q_network1.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer1 = optim.Adam(self.q_network1.parameters(), lr=lr)
        self.q_optimizer2 = optim.Adam(self.q_network2.parameters(), lr=lr)

    def choose_action(self, obs_features):
        """根据观测选择动作，用于与环境交互"""
        with torch.no_grad():
            mean, std = self.actor(obs_features)
            normal_dist = Normal(mean, std)
            action = normal_dist.sample()
            # Tanh 激活，将动作限制在 [-1, 1] 之间
            return torch.tanh(action).cpu().numpy()

    def update(self, states, actions, rewards, next_states, dones):
        """SAC 完整的更新逻辑"""
        # 1. 更新 Q 网络
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_states)
            next_q1, next_q2 = self.target_q_network1(next_states, next_action), self.target_q_network2(next_states, next_action)
            min_next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_prob
            target_q = rewards + (1 - dones) * self.gamma * min_next_q
            
        q1, q2 = self.q_network1(states, actions), self.q_network2(states, actions)
        q_loss1 = F.mse_loss(q1, target_q)
        q_loss2 = F.mse_loss(q2, target_q)
        
        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        self.q_optimizer1.step()

        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        self.q_optimizer2.step()
        
        # 2. 更新策略网络 (Actor)
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.q_network1(states, new_actions), self.q_network2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 3. 软更新目标网络
        for param, target_param in zip(self.q_network1.parameters(), self.target_q_network1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q_network2.parameters(), self.target_q_network2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# 策略网络 (Actor)
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        normal = Normal(mean, std)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        return torch.tanh(action), log_prob


# 价值网络 (Critic)
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# SAC智能体
class SACAgent:
    def __init__(self, obs_dim, action_dim, config, device="cpu"):
        self.device = device
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.alpha = config.get("alpha", 0.2)
        self.batch_size = config.get("batch_size", 64)
        lr = config.get("lr", 3e-4)

        self.actor = PolicyNetwork(obs_dim, action_dim).to(device)
        self.q1 = QNetwork(obs_dim, action_dim).to(device)
        self.q2 = QNetwork(obs_dim, action_dim).to(device)
        self.target_q1 = QNetwork(obs_dim, action_dim).to(device)
        self.target_q2 = QNetwork(obs_dim, action_dim).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        self.memory = []
        self.max_memory = config.get("replay_size", 100000)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[i] for i in batch])

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_q1(next_states, next_actions)
            target_q2 = self.target_q2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_value = rewards + (1 - dones) * self.gamma * target_q

        q1_value = self.q1(states, actions)
        q2_value = self.q2(states, actions)
        q1_loss = F.mse_loss(q1_value, target_value)
        q2_loss = F.mse_loss(q2_value, target_value)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        actor_loss = (self.alpha * log_probs - torch.min(q1_new, q2_new)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        for target, source in zip(self.target_q1.parameters(), self.q1.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
        for target, source in zip(self.target_q2.parameters(), self.q2.parameters()):
            target.data.copy_(self.tau * source.data + (1 - self.tau) * target.data)
