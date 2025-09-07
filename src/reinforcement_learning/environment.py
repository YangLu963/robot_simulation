# === src/reinforcement_learning/environment.py ===
import gym
import numpy as np
import pybullet as p
import pybullet_data
import os
from PIL import Image

class PyBulletRobotEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render
        if self.render_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # 载入平面和机器人
        self.plane_id = p.loadURDF("plane.urdf")
        robot_path = os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/model.urdf")
        self.robot_id = p.loadURDF(robot_path, useFixedBase=True)

        # 初始化数据保存目录
        self.img_dir = "data/images"
        os.makedirs(self.img_dir, exist_ok=True)
        self.step_idx = 0  # 用于图像命名

        # 获取关节数量
        self.num_joints = p.getNumJoints(self.robot_id)

        # Gym 空间定义
        self.observation_space = gym.spaces.Box(
            low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        self.max_steps = 200
        self.step_count = 0

    def render_image(self):
        """捕获并保存当前视角的RGB图像"""
        width, height = 224, 224
        _, _, rgbImg, _, _ = p.getCameraImage(width=width, height=height)
        img = Image.fromarray(rgbImg)
        path = os.path.join(self.img_dir, f"{self.step_idx}.png")
        img.save(path)
        return path

    def reset(self):
        self.step_count = 0
        self.step_idx = 0
        for j in range(self.num_joints):
            p.resetJointState(self.robot_id, j, 0.0)
        return self._get_obs()

    def step(self, action):
        self.step_count += 1
        
        # 执行动作
        for j in range(self.num_joints):
            current_pos = p.getJointState(self.robot_id, j)[0]
            target_pos = current_pos + float(action[j]) * 0.05
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_pos,
                force=200
            )
        p.stepSimulation()

        # 保存当前图像
        image_path = self.render_image()
        self.step_idx += 1

        # 返回观测和奖励
        obs = self._get_obs()
        reward = -np.linalg.norm(obs)
        done = self.step_count >= self.max_steps
        info = {"image_path": image_path}  # 可选：将图像路径传递给外部

        return obs, reward, done, info

    def _get_obs(self):
        return np.array([p.getJointState(self.robot_id, j)[0] for j in range(self.num_joints)], dtype=np.float32)

    def render(self, mode="human"):
        if not self.render_mode:
            print("Environment created in DIRECT mode. Restart with render=True to see GUI.")