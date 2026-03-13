# scripts/utils_io.py

# 导入 json，用于读写 JSON 文件
import json

# 导入 Path，用于更稳地处理路径
from pathlib import Path


def ensure_dir(path: str) -> None:
    """
    确保目录存在；如果不存在就自动创建。
    
    参数:
    - path: 目录路径字符串
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> dict:
    """
    读取 JSON 文件并返回 Python 字典。
    
    参数:
    - path: JSON 文件路径
    
    返回:
    - dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    """
    将 Python 字典写入 JSON 文件。
    
    参数:
    - data: 要保存的数据
    - path: 输出 JSON 文件路径
    """
    # 先确保父目录存在
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入 JSON，ensure_ascii=False 让中文正常显示
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)