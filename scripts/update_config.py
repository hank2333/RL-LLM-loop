# scripts/update_config.py

# =========================
# 1. 标准库导入
# =========================
import argparse
from pathlib import Path
import sys
import copy


# =========================
# 2. 加入项目根目录到 sys.path
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入 JSON 工具函数
from scripts.utils_io import load_json, save_json


def parse_args():
    """
    解析命令行参数。
    
    返回:
    - 包含 config、search_space、llm_response、output 四个路径参数
    """
    parser = argparse.ArgumentParser(description="Update training config based on LLM response.")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to current train_config.json"
    )

    parser.add_argument(
        "--search-space",
        type=str,
        required=True,
        help="Path to search_space.json"
    )

    parser.add_argument(
        "--llm-response",
        type=str,
        required=True,
        help="Path to llm_response.json"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to next_config.json"
    )

    return parser.parse_args()


def validate_decision(llm_response: dict) -> str:
    """
    检查 decision 字段是否合法。
    
    合法值:
    - adjust
    - keep
    - stop
    
    如果缺失或非法，默认按 keep 处理。
    """
    decision = llm_response.get("decision", "keep")

    if decision not in ["adjust", "keep", "stop"]:
        return "keep"

    return decision


def normalize_param_value(param_name: str, value, rule: dict):
    """
    根据 search_space 中的规则校验参数值是否合法。
    
    当前 Day 3 只支持 type = "choice"
    
    参数:
    - param_name: 参数名
    - value: LLM 建议的新值
    - rule: 对应参数规则，例如
      {
        "type": "choice",
        "choices": [0.0001, 0.0003, 0.0005, 0.001]
      }
    
    返回:
    - 合法值（如果通过）
    - None（如果非法）
    """
    rule_type = rule.get("type")

    # 当前只实现 choice 类型
    if rule_type == "choice":
        choices = rule.get("choices", [])

        # 直接判断是否在候选集合中
        if value in choices:
            return value
        else:
            return None

    # 未来如果扩展其他类型，这里再补
    return None


def extract_valid_updates(llm_response: dict, search_space: dict) -> dict:
    """
    从 LLM 响应中提取合法的参数更新。
    
    规则：
    1. 只能修改 allowed_params 中列出的参数
    2. 每轮最多修改 max_params_to_change 个参数
    3. 参数值必须通过 normalize_param_value 校验
    
    返回:
    - valid_updates: dict
    """
    allowed_params = search_space["allowed_params"]
    max_params_to_change = search_space["max_params_to_change"]
    param_rules = search_space["param_rules"]

    # 读取 LLM 建议中的 parameter_updates
    raw_updates = llm_response.get("parameter_updates", {})

    valid_updates = {}

    # 按 LLM 给出的顺序逐个检查
    for param_name, proposed_value in raw_updates.items():
        # 如果参数不在允许修改列表中，跳过
        if param_name not in allowed_params:
            continue

        # 如果已经达到本轮允许修改参数个数上限，停止继续收集
        if len(valid_updates) >= max_params_to_change:
            break

        # 读取对应参数规则
        rule = param_rules.get(param_name)
        if rule is None:
            continue

        # 校验参数值是否合法
        normalized_value = normalize_param_value(param_name, proposed_value, rule)

        # 合法就收下
        if normalized_value is not None:
            valid_updates[param_name] = normalized_value

    return valid_updates


def build_next_config(current_config: dict, valid_updates: dict, llm_response: dict) -> dict:
    """
    基于当前配置和合法更新，生成 next_config。
    
    规则：
    - 深拷贝当前配置，避免原地修改
    - round_id + 1
    - 仅更新 train_params 中允许修改的字段
    
    返回:
    - next_config: dict
    """
    # 深拷贝，避免影响原始 current_config
    next_config = copy.deepcopy(current_config)

    # 下一轮轮次号 = 当前轮次号 + 1
    next_config["round_id"] = int(current_config["round_id"]) + 1

    # 应用合法更新
    for param_name, new_value in valid_updates.items():
        next_config["train_params"][param_name] = new_value

    # 你也可以在这里记录来源，但第一版先保持 config 干净
    return next_config


def main():
    """
    主入口：
    - 读取当前配置
    - 读取搜索空间
    - 读取 LLM 响应
    - 生成合法的 next_config
    - 保存输出
    """
    # 解析命令行参数
    args = parse_args()

    # 读取输入文件
    current_config = load_json(args.config)
    search_space = load_json(args.search_space)
    llm_response = load_json(args.llm_response)

    # 先判断 LLM 决策类型
    decision = validate_decision(llm_response)

    # 三种决策分支
    if decision == "stop":
        # stop 表示不生成新的调参动作，沿用旧配置但 round_id 仍然 +1
        valid_updates = {}
    elif decision == "keep":
        # keep 表示不改参数，但进入下一轮时配置结构保持一致
        valid_updates = {}
    else:
        # adjust 才会真正尝试提取参数更新
        valid_updates = extract_valid_updates(llm_response, search_space)

    # 生成下一轮配置
    next_config = build_next_config(current_config, valid_updates, llm_response)

    # 保存输出
    save_json(next_config, args.output)

    # 控制台打印结果，方便调试
    print("\n===== Next Config Generated =====")
    print(f"Decision: {decision}")
    print(f"Applied updates: {valid_updates}")
    print(f"Output path: {args.output}")
    print(f"Next round_id: {next_config['round_id']}")
    print(f"Next train_params: {next_config['train_params']}")


if __name__ == "__main__":
    main()