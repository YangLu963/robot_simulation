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
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs_features):
        x = F.relu(self.fc1(obs_features))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, obs_features):
        mean, std = self.forward(obs_features)
        normal_dist = Normal(mean, std)
        action = normal_dist.rsample()
        log_prob = normal_dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
# 修复：删除多余的字符 'd'

# 价值网络 (Critic)
class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1_q1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)

        self.fc1_q2 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs_features, action):
        x_q1 = torch.cat([obs_features, action], dim=-1)
        x_q1 = F.relu(self.fc1_q1(x_q1))
        x_q1 = F.relu(self.fc2_q1(x_q1))
        q1 = self.fc3_q1(x_q1)

        x_q2 = torch.cat([obs_features, action], dim=-1)
        x_q2 = F.relu(self.fc1_q2(x_q2))
        x_q2 = F.relu(self.fc2_q2(x_q2))
        q2 = self.fc3_q2(x_q2)
        return q1, q2

# SAC智能体（只保留一个版本）
class SACAgent:
    def __init__(self, obs_dim, action_dim, device="cuda", hidden_dim=256,
                 gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, batch_size=256):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size

        self.actor = PolicyNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q_network1 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.q_network2 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        
        self.target_q_network1 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_q_network2 = QNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.target_q_network1.load_state_dict(self.q_network1.state_dict())
        self.target_q_network2.load_state_dict(self.q_network2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optimizer1 = optim.Adam(self.q_network1.parameters(), lr=lr)
        self.q_optimizer2 = optim.Adam(self.q_network2.parameters(), l
