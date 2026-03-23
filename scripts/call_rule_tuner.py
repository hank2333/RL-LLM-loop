# scripts/call_rule_tuner.py

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.utils_io import load_json, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Rule-based outer-loop tuner.")

    parser.add_argument("--summary", type=str, required=True, help="Path to summary.json")
    parser.add_argument("--search-space", type=str, required=True, help="Path to search_space.json")
    parser.add_argument("--output", type=str, required=True, help="Path to rule_response.json")

    return parser.parse_args()


def get_next_choice(choices, current_value, direction):
    """
    direction:
    +1 -> increase one level
    -1 -> decrease one level
    """
    if current_value not in choices:
        return current_value

    idx = choices.index(current_value)
    new_idx = max(0, min(len(choices) - 1, idx + direction))
    return choices[new_idx]


def main():
    args = parse_args()

    summary = load_json(args.summary)
    search_space = load_json(args.search_space)

    round_id = summary["round_id"]
    metrics = summary["metrics"]
    train_params = summary["train_params"]
    rules = search_space["param_rules"]

    parameter_updates = {}
    reason_parts = []

    # 规则1：动作过于集中 → 提高 ent_coef
    if metrics["dominant_action_ratio"] > 0.8:
        current_ent = train_params["ent_coef"]
        next_ent = get_next_choice(rules["ent_coef"]["choices"], current_ent, +1)
        if next_ent != current_ent:
            parameter_updates["ent_coef"] = next_ent
            reason_parts.append("Increase ent_coef to reduce action concentration.")

    # 规则2：波动偏高 → 降低 learning_rate
    if metrics["return_std"] > 1.5 and "learning_rate" not in parameter_updates:
        current_lr = train_params["learning_rate"]
        next_lr = get_next_choice(rules["learning_rate"]["choices"], current_lr, -1)
        if next_lr != current_lr:
            parameter_updates["learning_rate"] = next_lr
            reason_parts.append("Decrease learning_rate to improve stability.")

    # 规则3：表现差且成功率低 → 增加 total_timesteps
    if metrics["avg_return"] < 0 and metrics["success_rate"] < 0.5 and len(parameter_updates) < 2:
        current_ts = train_params["total_timesteps"]
        next_ts = get_next_choice(rules["total_timesteps"]["choices"], current_ts, +1)
        if next_ts != current_ts:
            parameter_updates["total_timesteps"] = next_ts
            reason_parts.append("Increase total_timesteps because performance is weak.")

    if parameter_updates:
        decision = "adjust"
        should_continue = True
    else:
        decision = "keep"
        should_continue = True
        reason_parts.append("No rule condition strongly triggered; keep current config.")

    result = {
        "round_id": round_id,
        "decision": decision,
        "parameter_updates": parameter_updates,
        "reasoning": {
            "summary": " ".join(reason_parts),
            "risk_note": "Rule-based tuning is conservative and only reacts to simple summary signals."
        },
        "confidence": 0.6,
        "should_continue": should_continue
    }

    save_json(result, args.output)

    print("\n===== Rule Tuning Generated =====")
    print(f"Decision: {decision}")
    print(f"Updates: {parameter_updates}")


if __name__ == "__main__":
    main()