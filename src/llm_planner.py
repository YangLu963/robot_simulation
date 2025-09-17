# === src/controllers/llm_controller.py ===
import logging
from typing import Dict, Any
from .hrl_controller import HierarchicalRLController
from ..llm.llm_api import ChatGPTAPI   # 修复：相对导入
from ..multimodal.text_encoder import TextEncoder  # 修复：相对导入
from ..task_templates.task_template import TaskTemplate  # 修复：相对导入
from ..reinforcement_learning.environment import RobotEnv  # 修复：相对导入
from ..robot_instruction import RobotInstruction  # 修复：相对导入

class LLMController:
    def __init__(self, rl_config_path: str = "configs/rl_config.yaml", device: str = 'cuda'):
        self.logger = logging.getLogger(__name__)
        self.hrl_controller = HierarchicalRLController(rl_config_path, device)
        self.chatgpt = ChatGPTAPI()
        self.text_encoder = TextEncoder(output_dim=256).to(device)
        self.task_template = TaskTemplate()
        self.device = device

    def plan_task(self, instruction: str) -> RobotInstruction:
        """
        使用 ChatGPT + TaskTemplate 将自然语言任务分解为结构化计划
        """
        try:
            prompt = self.task_template.format(instruction)
            response = self.chatgpt.generate(prompt)
            self.logger.info(f"Task decomposed: {response}")
            return RobotInstruction.from_llm_response(response)
        except Exception as e:
            self.logger.error(f"Task planning failed: {str(e)}")
            raise

    def execute_instruction(self, instruction: str, env: RobotEnv) -> None:
        """
        高层入口：接收自然语言指令，调用 LLM 分解，然后交给 HRLController 执行
        """
        plan = self.plan_task(instruction)

        # 修复：使用正确的步骤访问方式
        for step in plan.steps:  # plan.steps 而不是 plan['steps']
            # 修复：使用hrl_controller现有的execute方法
            # 需要确保hrl_controller能够处理步骤和文本特征
            self.hrl_controller.execute(f"Execute step: {step.action}", env)

    def interactive_session(self, env: RobotEnv):
        """
        人机交互模式：持续接收用户输入并执行
        """
        print("进入交互模式（输入 exit 退出）")
        while True:
            user_input = input("请输入任务指令: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            self.execute_instruction(user_input, env)
