# hrl_controller.py
import random
import numpy as np
import torch
from collections import deque
from typing import Dict, Any
import logging
import yaml

from .llm_planner import LLMTaskPlanner
from .multimodal.multimodal_model import MultimodalModel
from .reinforcement_learning.sac import SACAgent  # 修复：相对导入
from .reinforcement_learning.environment import RobotEnv  # 修复：相对导入
from .robot_instruction import RobotInstruction

class HierarchicalRLController:
    def __init__(self, rl_config_path: str = "configs/rl_config.yaml", device: str = 'cuda'):
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.config = self._load_config(rl_config_path)
        self.planner = LLMTaskPlanner()
        self.memory = deque(maxlen=self.config.get('replay_buffer_size', 10000))
        self.train_counter = 0

        # 初始化多模态编码器
        self.multimodal_encoder = MultimodalModel(
            vision_dim=self.config['obs_space']['vision_dim'],
            text_dim=self.config['obs_space']['text_dim'],
            hidden_dim=self.config['obs_space']['hidden_dim']
        ).to(self.device)

        # 初始化 SAC 智能体
        obs_dim = 128
        self.policy = SACAgent(
            obs_dim=obs_dim,
            action_dim=self.config['action_space']['dim'],
            device=self.device,
            **self.config.get('policy_params', {})
        )

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f).get('rl', {})

    def execute(self, instruction: str, env: RobotEnv) -> None:
        """
        输入自然语言指令 -> 调用 LLM 分解 -> 执行分解后的任务
        """
        try:
            # 1. 使用 LLM + TaskTemplate 生成结构化任务
            plan: RobotInstruction = self.planner.plan_task(instruction)

            self.logger.info(f"Task plan received: {plan.model_dump_json(indent=2)}")  # 修复：正确的方法名

            # 2. 按步骤执行
            for step in plan.steps:
                self._execute_step(step, env)

        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            raise

    def _execute_step(self, step: RobotInstruction.Step, env: RobotEnv) -> None:
        """
        执行单个子任务步骤
        """
        # 修复：确保观测格式与多模态编码器匹配
        raw_obs = env.reset()
        obs = {
            'image': raw_obs,  # 需要根据实际环境调整
            'state': raw_obs   # 需要根据实际环境调整
        }

        for _ in range(self.config.get('max_step_per_task', 200)):
            # 1. 将原始观测和 step 指令输入多模态编码器
            obs_features = self.multimodal_encoder.encode_obs(obs, step.model_dump())  # 修复：正确的方法名

            # 2. SAC 智能体选择动作
            action = self.policy.choose_action(obs_features)

            # 3. 执行动作
            next_raw_obs, reward, done, info = env.step(action)
            next_obs = {
                'image': next_raw_obs,
                'state': next_raw_obs
            }

            # 4. 存储经验
            next_obs_features = self.multimodal_encoder.encode_obs(next_obs, step.model_dump())
            self.memory.append((obs_features, action, reward, next_obs_features, done))

            # 5. 更新策略
            self._update_policy()

            obs = next_obs
            if done:
                break

    def _update_policy(self) -> None:
        """
        使用经验回放更新 SAC 策略
        """
        if len(self.memory) < self.policy.batch_size:
            return

        batch = random.sample(self.memory, self.policy.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states, dim=0).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.cat(next_states, dim=0).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        self.policy.update(states, actions, rewards, next_states, dones)
