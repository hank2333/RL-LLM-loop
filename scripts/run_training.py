# scripts/run_training.py

# =========================
# 1. 标准库导入
# =========================

# argparse 用于解析命令行参数
import argparse

# time 用于记录训练耗时
import time

# datetime 用于保存开始和结束时间
from datetime import datetime

# Path 用于更稳地处理路径
from pathlib import Path

# sys 用于修改 Python 模块搜索路径
import sys


# =========================
# 2. 第三方库导入
# =========================

# numpy 用于数值统计
import numpy as np

# stable-baselines3 提供 PPO 算法
from stable_baselines3 import PPO


# =========================
# 3. 把项目根目录加入 sys.path
# =========================
# 这样脚本运行时可以找到 envs/ 和 scripts/ 下的模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入我们自己写的工具函数
from scripts.utils_io import load_json, save_json, ensure_dir

# 导入 toy 环境
from envs.toy_env import ToyRedBlueEnv


def parse_args():
    """
    解析命令行参数。
    
    返回:
    - argparse.Namespace
    """
    parser = argparse.ArgumentParser(description="Run one RL training round.")
    
    # --config 参数用于指定训练配置文件路径
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to train_config.json"
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """
    设置 numpy 随机种子。
    
    说明:
    stable-baselines3 的 PPO 在创建模型时也会接收 seed，
    环境 reset 时也会传 seed，所以这里主要做额外一致性控制。
    """
    np.random.seed(seed)


def build_env(config: dict) -> ToyRedBlueEnv:
    """
    根据配置构造环境实例。
    
    参数:
    - config: 训练配置字典
    
    返回:
    - ToyRedBlueEnv 实例
    """
    # 从 config 中读取 max_steps；如果没有就默认 20
    max_steps = config["env"].get("max_steps", 20)

    # 创建环境实例
    env = ToyRedBlueEnv(max_steps=max_steps)

    return env


def build_model(config: dict, env) -> PPO:
    """
    根据配置构造 PPO 模型。
    
    参数:
    - config: 训练配置字典
    - env: 环境实例
    
    返回:
    - PPO 模型
    """
    train_params = config["train_params"]

    # 创建 PPO 模型
    # policy="MlpPolicy" 表示使用多层感知机策略网络
    # verbose=1 表示打印一些训练信息，方便调试
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=train_params["learning_rate"],
        n_steps=train_params["n_steps"],
        ent_coef=train_params["ent_coef"],
        verbose=1,
        seed=config["seed"]
    )

    return model


def train_model(model: PPO, config: dict) -> PPO:
    """
    执行训练。
    
    参数:
    - model: PPO 模型
    - config: 训练配置
    
    返回:
    - 训练后的模型
    """
    total_timesteps = config["train_params"]["total_timesteps"]

    # 调用 stable-baselines3 的 learn 方法进行训练
    model.learn(total_timesteps=total_timesteps)

    return model


def evaluate_model(model: PPO, config: dict) -> dict:
    """
    使用训练后的模型做评估，并统计评估指标。
    
    参数:
    - model: 训练后的 PPO 模型
    - config: 配置字典
    
    返回:
    - 一个 metrics 字典，至少包含:
        avg_return
        return_std
        avg_episode_length
        action_counts
        action_distribution
        success_rate
    """
    # 单独创建一个评估环境，避免和训练环境混在一起
    eval_env = build_env(config)

    # 读取评估 episode 数
    n_eval_episodes = config["eval_params"]["n_eval_episodes"]

    # 读取随机种子
    base_seed = config["seed"]

    # 记录每个 episode 的总回报
    episode_returns = []

    # 记录每个 episode 的长度
    episode_lengths = []

    # 记录成功次数
    success_count = 0

    # 全部评估 episode 的动作总计数
    total_action_counts = {
        "remove": 0,
        "decoy": 0,
        "monitor": 0
    }

    # 逐个 episode 评估
    for episode_idx in range(n_eval_episodes):
        # 每个 episode 用不同 seed，但可复现
        obs, info = eval_env.reset(seed=base_seed + episode_idx)

        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            # 用确定性策略进行评估
            action, _ = model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = eval_env.step(int(action))

            # 累积奖励
            episode_return += reward

            # 累积步数
            episode_length += 1

            # episode 是否结束
            done = terminated or truncated

        # 记录这一轮 episode 的结果
        episode_returns.append(float(episode_return))
        episode_lengths.append(int(episode_length))

        # 读取当前 episode 的动作计数并累加到总计数中
        ep_action_counts = info["action_counts"]
        total_action_counts["remove"] += ep_action_counts["remove"]
        total_action_counts["decoy"] += ep_action_counts["decoy"]
        total_action_counts["monitor"] += ep_action_counts["monitor"]

        # 这里定义一个简单成功判定：
        # 如果 episode 不是因为风险过高被自然终止，而是因为走完 max_steps 被截断，
        # 说明系统撑到了最后，可以视作成功
        if truncated and not terminated:
            success_count += 1

    # 计算平均回报
    avg_return = float(np.mean(episode_returns))

    # 计算回报标准差
    return_std = float(np.std(episode_returns))

    # 计算平均 episode 长度
    avg_episode_length = float(np.mean(episode_lengths))

    # 成功率
    success_rate = float(success_count / n_eval_episodes)

    # 动作总数
    total_actions = sum(total_action_counts.values())

    # 避免分母为 0
    if total_actions > 0:
        action_distribution = {
            "remove": total_action_counts["remove"] / total_actions,
            "decoy": total_action_counts["decoy"] / total_actions,
            "monitor": total_action_counts["monitor"] / total_actions
        }
    else:
        action_distribution = {
            "remove": 0.0,
            "decoy": 0.0,
            "monitor": 0.0
        }

    # 汇总成 metrics 字典
    metrics = {
        "avg_return": avg_return,
        "return_std": return_std,
        "avg_episode_length": avg_episode_length,
        "success_rate": success_rate,
        "action_counts": total_action_counts,
        "action_distribution": action_distribution
    }

    return metrics


def build_raw_metrics(config: dict, eval_metrics: dict, start_dt: datetime, end_dt: datetime) -> dict:
    """
    构造最终要落盘的 raw_metrics.json 内容。
    
    参数:
    - config: 训练配置
    - eval_metrics: 评估指标
    - start_dt: 训练开始时间
    - end_dt: 训练结束时间
    
    返回:
    - 原始结果字典
    """
    duration_sec = (end_dt - start_dt).total_seconds()

    raw_metrics = {
        "experiment_name": config["experiment_name"],
        "round_id": config["round_id"],
        "seed": config["seed"],
        "algo": config["algo"],
        "train_params": config["train_params"],
        "eval_params": config["eval_params"],
        "eval_metrics": eval_metrics,
        "runtime": {
            "train_start_time": start_dt.isoformat(),
            "train_end_time": end_dt.isoformat(),
            "duration_sec": duration_sec
        },
        "status": "success"
    }

    return raw_metrics


def main():
    """
    主入口函数。
    """
    # 解析命令行参数
    args = parse_args()

    # 读取训练配置
    config = load_json(args.config)

    # 设置随机种子
    set_seed(config["seed"])

    # 获取输出目录
    results_dir = config["output"]["results_dir"]

    # 确保输出目录存在
    ensure_dir(results_dir)

    # 记录训练开始时间
    start_dt = datetime.now()

    # 也记录一个 perf timer，后面你要更精确统计可用
    _ = time.perf_counter()

    # 创建训练环境
    train_env = build_env(config)

    # 创建 PPO 模型
    model = build_model(config, train_env)

    # 执行训练
    model = train_model(model, config)

    # 训练结束时间
    end_dt = datetime.now()

    # 执行评估
    eval_metrics = evaluate_model(model, config)

    # 构造输出结果
    raw_metrics = build_raw_metrics(
        config=config,
        eval_metrics=eval_metrics,
        start_dt=start_dt,
        end_dt=end_dt
    )

    # 输出文件路径
    raw_metrics_path = str(Path(results_dir) / "raw_metrics.json")

    # 保存结果 JSON
    save_json(raw_metrics, raw_metrics_path)

    # 控制台打印一个简短结果，方便你确认
    print("\n===== Training Finished =====")
    print(f"Saved raw metrics to: {raw_metrics_path}")
    print(f"Average return: {eval_metrics['avg_return']:.4f}")
    print(f"Return std: {eval_metrics['return_std']:.4f}")
    print(f"Average episode length: {eval_metrics['avg_episode_length']:.4f}")
    print(f"Success rate: {eval_metrics['success_rate']:.4f}")
    print(f"Action distribution: {eval_metrics['action_distribution']}")


# 只有直接运行当前脚本时，才执行 main()
if __name__ == "__main__":
    main()