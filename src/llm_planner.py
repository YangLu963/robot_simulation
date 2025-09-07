# === src/controllers/llm_controller.py ===
import logging
from typing import Dict, Any
from .hrl_controller import HierarchicalRLController
from llm.llm_api import ChatGPTAPI   # 你封装的 GPT 接口
from multimodal.text_encoder import TextEncoder
from task_templates.task_template import TaskTemplate
from reinforcement_learning.environment import RobotEnv

from robot_instruction import RobotInstruction

class LLMController:

    def __init__(self, rl_config_path: str = "configs/rl_config.yaml", device: str = 'cuda'):
        self.logger = logging.getLogger(__name__)
        self.hrl_controller = HierarchicalRLController(rl_config_path, device)
        self.chatgpt = ChatGPTAPI()
        self.text_encoder = TextEncoder(output_dim=256).to(device)
        self.task_template = TaskTemplate()
        self.device = device

    from robot_instruction import RobotInstruction

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

        for step in plan['steps']:
            # 将每个 step 的语言特征转为向量，供多模态融合
            text_feature = self.text_encoder.encode(step['goal']).to(self.device)
            self.hrl_controller.execute_step_with_text(env, step, text_feature)

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
