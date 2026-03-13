# envs/toy_env.py

# 导入 gymnasium，这是较新的 Gym 接口实现
import gymnasium as gym

# spaces 用来定义动作空间和观测空间
from gymnasium import spaces

# numpy 用于数值计算和随机数生成
import numpy as np


class ToyRedBlueEnv(gym.Env):
    """
    一个最小可运行的 toy 红蓝对抗风格环境。
    
    环境设计思路：
    - 系统有一个风险值 risk_level，范围大致在 [0, 10]
    - 智能体每一步选择一个动作：
        0 = remove   （强力清除，风险下降更多，但成本更高）
        1 = decoy    （部署诱饵，风险下降适中，成本适中）
        2 = monitor  （监控观察，成本低，但风险可能自然上升）
    - 每一步都会返回 reward（奖励）
    - episode 在达到最大步数或风险过高时结束
    """

    # 告诉 gym 这个环境支持什么渲染模式
    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 20):
        """
        环境初始化函数。
        
        参数:
        - max_steps: 每个 episode 的最大步数
        """
        super().__init__()

        # 保存最大步数
        self.max_steps = max_steps

        # 动作空间：3 个离散动作
        # 0=remove, 1=decoy, 2=monitor
        self.action_space = spaces.Discrete(3)

        # 观测空间：
        # 我们这里用一个 2 维向量表示状态：
        # [risk_level, step_fraction]
        #
        # risk_level: 当前系统风险，范围 [0, 10]
        # step_fraction: 当前步数比例，范围 [0, 1]
        #
        # dtype 用 float32，和 RL 框架更兼容
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # 当前风险值，reset 时会初始化
        self.risk_level = None

        # 当前 episode 已走的步数
        self.current_step = None

        # 用于统计当前 episode 的动作次数
        self.action_counts = None

        # gymnasium 推荐使用 np_random 而不是全局随机数
        self.np_random = None

    def _get_obs(self) -> np.ndarray:
        """
        构造当前观测向量。
        
        返回:
        - 一个 shape=(2,) 的 numpy 数组
        """
        # step_fraction 表示已经走了多少比例的步数
        step_fraction = self.current_step / self.max_steps

        # 返回 float32 类型观测
        return np.array([self.risk_level, step_fraction], dtype=np.float32)

    def _get_info(self) -> dict:
        """
        构造 info 字典。
        
        info 不直接参与训练奖励，但可以用于调试和日志统计。
        """
        return {
            "risk_level": float(self.risk_level),
            "current_step": int(self.current_step),
            "action_counts": self.action_counts.copy()
        }

    def reset(self, seed=None, options=None):
        """
        重置环境，开始一个新 episode。
        
        gymnasium 的 reset 返回:
        - observation
        - info
        """
        # 让父类处理 seed，这样 self.np_random 会被正确初始化
        super().reset(seed=seed)

        # episode 开始时，初始风险设为一个中等值
        # 这里给一个随机初值，让环境稍微有点变化
        self.risk_level = float(self.np_random.uniform(4.0, 6.0))

        # 当前步数归零
        self.current_step = 0

        # 动作计数器清零
        self.action_counts = {
            "remove": 0,
            "decoy": 0,
            "monitor": 0
        }

        # 获取初始观测
        observation = self._get_obs()

        # 获取调试信息
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        """
        执行一步环境转移。
        
        参数:
        - action: 智能体选择的动作
        
        返回:
        - observation: 下一个观测
        - reward: 奖励
        - terminated: 是否自然终止
        - truncated: 是否因时间上限截断
        - info: 额外信息
        """
        # 进入一步
        self.current_step += 1

        # 自然风险增长项：
        # 每一步系统风险会有一点随机波动
        natural_risk_change = float(self.np_random.normal(loc=0.4, scale=0.3))

        # 根据动作决定风险变化和动作成本
        if action == 0:
            # remove：风险下降最多，但成本较高
            self.risk_level += natural_risk_change - 1.2
            action_cost = 0.30
            self.action_counts["remove"] += 1

        elif action == 1:
            # decoy：风险下降适中，成本中等
            self.risk_level += natural_risk_change - 0.7
            action_cost = 0.15
            self.action_counts["decoy"] += 1

        elif action == 2:
            # monitor：不主动干预，风险可能上升，成本最低
            self.risk_level += natural_risk_change - 0.1
            action_cost = 0.05
            self.action_counts["monitor"] += 1

        else:
            # 理论上不会进这里，因为动作空间已经约束了动作范围
            raise ValueError(f"Invalid action: {action}")

        # 为了避免观测超出范围太离谱，这里对风险做裁剪
        self.risk_level = float(np.clip(self.risk_level, 0.0, 10.0))

        # 设计奖励：
        # 奖励越高表示状态越好
        # 我们希望风险越低越好，同时动作成本不要太高
        #
        # 基础思路：
        # - 高风险时给负奖励
        # - 低风险时损失较小
        # - 再扣掉动作成本
        #
        # 这里 reward 是一个比较平滑、简单的设计，适合 toy 环境
        reward = 1.0 - (self.risk_level / 5.0) - action_cost

        # terminated 表示“自然结束”
        # 例如风险达到极高，系统被攻破
        terminated = self.risk_level >= 8.5

        # truncated 表示“时间到了被截断”
        truncated = self.current_step >= self.max_steps

        # 获取下一个观测
        observation = self._get_obs()

        # 构造 info
        info = self._get_info()

        return observation, float(reward), terminated, truncated, info

    def render(self):
        """
        简单打印当前状态。
        """
        print(
            f"[Render] step={self.current_step}, "
            f"risk={self.risk_level:.3f}, "
            f"actions={self.action_counts}"
        )