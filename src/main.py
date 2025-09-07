# === src/main.py ===
import torch
import os
from multimodal.multimodal_model import MultimodalModel
from reinforcement_learning.environment import RobotEnv
from reinforcement_learning.sac import SACAgent

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = "../outputs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "logs.txt")
    log_file = open(log_path, "a")

    model = MultimodalModel().to(device)
    env = RobotEnv()

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, action_dim, device=device)

    num_episodes = 5
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = agent.choose_action(state_tensor)[0]
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        log_line = f"[Main] Episode {episode+1}/{num_episodes} | Reward: {total_reward:.2f}"
        print(log_line)
        log_file.write(log_line + "\n")
        log_file.flush()

    log_file.close()
