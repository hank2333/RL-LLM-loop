# scripts/summarize_results.py

# =========================
# 1. 标准库导入
# =========================
import argparse
from pathlib import Path
import sys


# =========================
# 2. 加入项目根目录到 sys.path
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入我们自己写的 JSON 工具
from scripts.utils_io import load_json, save_json


def parse_args():
    """
    解析命令行参数。
    
    返回:
    - argparse.Namespace，包含 input 和 output 路径
    """
    parser = argparse.ArgumentParser(description="Summarize raw RL metrics into summary.json")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw_metrics.json"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to summary.json"
    )

    return parser.parse_args()


def compute_dominant_action(action_distribution: dict):
    """
    从动作分布中找出占比最高的动作和对应比例。
    
    参数:
    - action_distribution: 例如
      {
        "remove": 0.53,
        "decoy": 0.0,
        "monitor": 0.47
      }
    
    返回:
    - dominant_action: str
    - dominant_action_ratio: float
    """
    # 如果动作分布为空，给默认值
    if not action_distribution:
        return "unknown", 0.0

    # max(..., key=...) 会根据 value 最大的项返回对应的 key
    dominant_action = max(action_distribution, key=action_distribution.get)

    # 取出这个动作的占比
    dominant_action_ratio = float(action_distribution[dominant_action])

    return dominant_action, dominant_action_ratio


def judge_stability(return_std: float) -> str:
    """
    根据回报标准差粗略判断训练/评估稳定性。
    
    规则:
    - < 1.0  → stable
    - < 3.0  → slightly_unstable
    - >= 3.0 → unstable
    """
    if return_std < 1.0:
        return "stable"
    elif return_std < 3.0:
        return "slightly_unstable"
    else:
        return "unstable"


def judge_performance(avg_return: float) -> str:
    """
    根据平均回报粗略判断表现水平。
    
    规则:
    - >= 8   → good
    - >= 0   → medium
    - < 0    → poor
    """
    if avg_return >= 8.0:
        return "good"
    elif avg_return >= 0.0:
        return "medium"
    else:
        return "poor"


def judge_action_collapse(dominant_action_ratio: float) -> bool:
    """
    判断是否存在明显动作塌缩。
    
    Day 2 的简单规则:
    - 如果最大动作占比 >= 0.85，则认为可能塌缩
    """
    return dominant_action_ratio >= 0.85


def build_notes(
    stability_flag: str,
    performance_level: str,
    success_rate: float,
    action_distribution: dict,
    dominant_action: str,
    action_collapse_flag: bool
):
    """
    生成简短规则化说明。
    
    返回:
    - list[str]
    """
    notes = []

    # 关于稳定性
    if stability_flag == "stable":
        notes.append("Policy performance is stable under current evaluation setting.")
    elif stability_flag == "slightly_unstable":
        notes.append("Policy performance shows mild variance across evaluation episodes.")
    else:
        notes.append("Policy performance is unstable and may require conservative tuning.")

    # 关于表现水平
    if performance_level == "good":
        notes.append("Current average return is strong under the current configuration.")
    elif performance_level == "medium":
        notes.append("Current average return is acceptable but still has room for improvement.")
    else:
        notes.append("Current average return is weak under the current configuration.")

    # 关于成功率
    if success_rate >= 0.99:
        notes.append("Success rate is very high in the current evaluation.")
    elif success_rate <= 0.2:
        notes.append("Success rate is low and indicates poor task completion.")
    
    # 关于动作塌缩
    if action_collapse_flag:
        notes.append(f"Action usage is highly concentrated on '{dominant_action}', suggesting possible action collapse.")

    # 关于未被使用的动作
    for action_name, ratio in action_distribution.items():
        if ratio == 0.0:
            notes.append(f"Action '{action_name}' is unused in the evaluated policy.")

    return notes


def build_summary(raw_metrics: dict, input_path: str) -> dict:
    """
    根据 raw_metrics 构建 summary 字典。
    
    参数:
    - raw_metrics: 从 raw_metrics.json 读取的原始数据
    - input_path: 原始文件路径
    
    返回:
    - summary dict
    """
    # 取出最核心的评估指标
    eval_metrics = raw_metrics["eval_metrics"]

    avg_return = float(eval_metrics["avg_return"])
    return_std = float(eval_metrics["return_std"])
    avg_episode_length = float(eval_metrics["avg_episode_length"])
    success_rate = float(eval_metrics["success_rate"])
    action_distribution = eval_metrics["action_distribution"]

    # 计算主导动作及比例
    dominant_action, dominant_action_ratio = compute_dominant_action(action_distribution)

    # 规则判断
    stability_flag = judge_stability(return_std)
    performance_level = judge_performance(avg_return)
    action_collapse_flag = judge_action_collapse(dominant_action_ratio)

    # 生成诊断说明
    notes = build_notes(
        stability_flag=stability_flag,
        performance_level=performance_level,
        success_rate=success_rate,
        action_distribution=action_distribution,
        dominant_action=dominant_action,
        action_collapse_flag=action_collapse_flag
    )

    # 组装最终摘要
    summary = {
        "experiment_name": raw_metrics["experiment_name"],
        "round_id": raw_metrics["round_id"],
        "seed": raw_metrics["seed"],
        "algo": raw_metrics["algo"],
        "train_params": raw_metrics["train_params"],
        "metrics": {
            "avg_return": avg_return,
            "return_std": return_std,
            "avg_episode_length": avg_episode_length,
            "success_rate": success_rate,
            "action_distribution": action_distribution,
            "dominant_action": dominant_action,
            "dominant_action_ratio": dominant_action_ratio,
            "stability_flag": stability_flag
        },
        "diagnostics": {
            "action_collapse_flag": action_collapse_flag,
            "performance_level": performance_level,
            "notes": notes
        },
        "source_files": {
            "raw_metrics": input_path
        }
    }

    return summary


def main():
    """
    主入口函数：
    - 读取 raw_metrics.json
    - 构建 summary
    - 保存 summary.json
    """
    # 解析命令行参数
    args = parse_args()

    # 读取原始结果
    raw_metrics = load_json(args.input)

    # 构建摘要
    summary = build_summary(raw_metrics, args.input)

    # 保存摘要文件
    save_json(summary, args.output)

    # 控制台打印摘要信息，方便调试
    print("\n===== Summary Generated =====")
    print(f"Input raw metrics: {args.input}")
    print(f"Output summary: {args.output}")
    print(f"Avg return: {summary['metrics']['avg_return']:.4f}")
    print(f"Return std: {summary['metrics']['return_std']:.4f}")
    print(f"Dominant action: {summary['metrics']['dominant_action']}")
    print(f"Dominant action ratio: {summary['metrics']['dominant_action_ratio']:.4f}")
    print(f"Stability flag: {summary['metrics']['stability_flag']}")
    print(f"Action collapse flag: {summary['diagnostics']['action_collapse_flag']}")


if __name__ == "__main__":
    main()