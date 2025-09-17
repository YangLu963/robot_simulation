# 🤖 MultiModal HRL for Robotic Task Planning

A robotic task planning system based on **Multimodal Fusion** and **Hierarchical Reinforcement Learning (HRL)**. It parses natural language instructions into structured task plans and executes them through reinforcement learning in a simulation environment.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Table of Contents
- [✨ Core Features](#-core-features)
- [📁 Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Configuration](#️-configuration)
- [🧩 Core Modules](#-core-modules)
- [📊 Data Format](#-data-format)
- [🧪 Testing & Evaluation](#-testing--evaluation)
- [📝 Citation](#-citation)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🆘 Support](#-support)

## ✨ Core Features

- **Natural Language Understanding**: Leverages Large Language Models (GPT-4) to parse user instructions into structured `RobotInstruction`.
- **Multimodal Perception**: Fuses visual (ViT) and textual (BERT) features to generate environmental state representations.
- **Hierarchical Control**: High-level (LLM) for task planning, low-level (SAC) for action execution.
- **Simulation Environment**: Integrated with PyBullet physics engine for realistic robot simulation.
- **Strong Typing Validation**: Uses Pydantic models to ensure structured and valid task instructions.

## 📁 Project Structure
project_root/
├── config/
│ ├── llm_config.yaml # LLM configuration (API keys, model parameters)
│ ├── rl_config.yaml # Reinforcement Learning config (SAC hyperparameters, environment spaces)
│ └── fusion_config.yaml # Multimodal fusion network configuration
├── src/
│ ├── controllers/ # Control layer
│ │ ├── hrl_controller.py # Hierarchical RL Controller
│ │ └── llm_planner.py # LLM Task Planner
│ ├── multimodal/ # Multimodal fusion module
│ │ ├── multimodal_model.py # Multimodal model (main entry)
│ │ ├── vision_encoder.py # Vision Encoder (ViT)
│ │ ├── text_encoder.py # Text Encoder (BERT)
│ │ └── fusion_network.py # Cross-modal attention fusion network
│ ├── reinforcement_learning/ # Reinforcement Learning module
│ │ ├── sac.py # Soft Actor-Critic algorithm
│ │ ├── environment.py # PyBullet simulation environment
│ │ └── train.py # Training script
│ ├── data/ # Data utilities
│ │ ├── robot_instruction.py# Task instruction Pydantic model (Core)
│ │ ├── data_loader.py # Data loader
│ │ └── preprocess.py # Data preprocessing utilities
│ └── visualization/ # Visualization tools
│ └── plot_utils.py # Plotting utility functions
├── scripts/
│ ├── download_models.sh # Download pre-trained models
│ ├── evaluate.py # Model evaluation script
│ └── deploy.py # Model deployment script (to ONNX)
├── models/
│ └── llm_planner/
│ └── task_templates.py # LLM task templates
└── multimodal_fusion.py
├── requirements.txt # Python dependencies
├── LICENSE # MIT License
└── README.md # Project description
└── .DS_Store


text

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/YangLu963/work.git
cd work

# Create a conda environment (recommended)
conda create -n robot-hrl python=3.8
conda activate robot-hrl

# Install core dependencies
pip install torch torchvision torchaudio
pip install transformers einops pydantic pybullet gymnasium
pip install openai python-dotenv

# Alternatively, use requirements.txt
pip install -r requirements.txt
2. Download Pre-trained Models
bash
# Run the download script
bash scripts/download_models.sh
3. Configure API Keys
bash
# Copy the environment variables template
cp .env.example .env
# Edit the .env file and add your OpenAI API key
OPENAI_API_KEY=sk-your-api-key-here
4. Run a Demo
bash
# Test the LLM planner (Parse natural language instruction)
python -c "
from src.controllers.llm_planner import LLMController
controller = LLMController()
plan = controller.plan_task('Put the red block into the blue box')
print(plan.json(indent=2))
"

# Run the main training program
python src/main.py
⚙️ Configuration
LLM Configuration (config/llm_config.yaml)
yaml
llm:
  api_key: ""  # Your OpenAI API Key
  model: "gpt-4-1106-preview"
  temperature: 0.7
  max_tokens: 1024
  task_templates_path: "models/llm_planner/task_templates.json"
Reinforcement Learning Configuration (config/rl_config.yaml)
yaml
rl:
  policy:
    type: "sac"
    gamma: 0.99
    tau: 0.005
    lr: 3e-4
  obs_space:
    rgb_shape: [224, 224, 3]
    joint_dim: 7
  action_space:
    dim: 7
    low: [-1.0, -1.0, -1.0, -3.14, -3.14, -3.14, -0.5]
    high: [1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 0.5]
🧩 Core Modules
Task Instruction Structure
python
# Strongly-typed instruction model based on Pydantic
class RobotInstruction(BaseModel):
    intent: Literal["transfer", "arrange", "clean", "fetch"]
    objects: List[Object]
    steps: List[Step]
    safety_constraints: List[SafetyConstraint] = []
Multimodal Fusion Network
python
# Fuses visual and textual features
class MultimodalFusionNetwork(nn.Module):
    def __init__(self, visual_feat_dim=768, text_feat_dim=768, hidden_dim=512):
        super().__init__()
        self.visual_proj = nn.Linear(visual_feat_dim, hidden_dim)
        self.text_proj = nn.Linear(text_feat_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8)
📊 Data Format
Task instruction data is in JSON format, compliant with the Pydantic model definition:

json
{
  "intent": "transfer",
  "objects": [
    {"id": 1, "name": "cup", "attributes": {"color": "red"}}
  ],
  "steps": [
    {
      "action": "pick",
      "target": {"object_id": 1},
      "preconditions": [{"type": "object_visible", "object_id": 1}]
    }
  ]
}
🧪 Testing & Evaluation
bash
# Run the evaluation script
python scripts/evaluate.py --model_path models/multimodal_fusion.pt

# Generate training curves
python -c "
from src.visualization.plot_utils import plot_training_history
plot_training_history('outputs/training_log.txt')
"
📝 Citation
If you use this project in your research, please cite:

bibtex
@software{yang2025multimodalhrl,
  title = {MultiModal HRL for Robotic Task Planning},
  author = {Yang Lu},
  year = {2025},
  url = {https://github.com/YangLu963/work}
}
🤝 Contributing
Contributions are welcome! Please feel free to submit Issues and Pull Requests. Ensure you:

Follow the existing code style.

Add appropriate test cases.

Update relevant documentation.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🆘 Support
If you encounter any problems:

📧 Email: luyang96377@gmail.com

🐛 Issues: GitHub Issues

💬 Discussion: Feel free to open a discussion topic
