import os
import json

subset_names = ['training', 'evaluation']

def load_arc_data(arc_data_path):
    required_files = []
    for subset in subset_names:
        required_files.append(os.path.join(arc_data_path, f'arc-agi_{subset}_challenges.json'))
        required_files.append(os.path.join(arc_data_path, f'arc-agi_{subset}_solutions.json'))

    # 检查文件是否都存在
    if not all(map(os.path.isfile, required_files)):
        raise FileNotFoundError(f"Some ARC data files are missing in {arc_data_path}")

    # 读取文件内容
    data = {}
    for subset in subset_names:
        with open(os.path.join(arc_data_path, f'arc-agi_{subset}_challenges.json'), 'r') as f:
            challenges = json.load(f)
        with open(os.path.join(arc_data_path, f'arc-agi_{subset}_solutions.json'), 'r') as f:
            solutions = json.load(f)
        data[subset] = {'challenges': challenges, 'solutions': solutions}

    print(f"Loaded ARC data from {arc_data_path}")
    return data

# 调用示例
arc_data_path = '../data/arc'  # 你的本地数据目录
arc_data = load_arc_data(arc_data_path)
