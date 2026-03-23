# scripts/log_experiment.py

import argparse
from pathlib import Path
import sys
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.utils_io import load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Append one experiment record to JSONL log.")

    parser.add_argument("--group-name", type=str, required=True, help="fixed / rule / llm")
    parser.add_argument("--seed", type=int, required=True, help="Experiment seed")
    parser.add_argument("--config", type=str, required=True, help="Path to current train_config.json")
    parser.add_argument("--summary", type=str, required=True, help="Path to summary.json")
    parser.add_argument("--next-config", type=str, required=True, help="Path to next_config.json")
    parser.add_argument("--log-file", type=str, required=True, help="Path to experiment_history.jsonl")

    # decision 文件对 fixed/rule/llm 都允许传，但 fixed 可能没有真实 llm_response
    parser.add_argument("--decision-file", type=str, default=None, help="Path to llm_response.json or rule_decision.json")

    return parser.parse_args()


def append_jsonl(record: dict, path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    current_config = load_json(args.config)
    summary = load_json(args.summary)
    next_config = load_json(args.next_config)

    if args.decision_file:
        decision_data = load_json(args.decision_file)
        decision = decision_data.get("decision", "unknown")
        parameter_updates = decision_data.get("parameter_updates", {})
        reasoning = decision_data.get("reasoning", {})
    else:
        decision = "keep"
        parameter_updates = {}
        reasoning = {}

    record = {
        "timestamp": datetime.now().isoformat(),
        "group_name": args.group_name,
        "seed": args.seed,
        "round_id": current_config["round_id"],
        "current_train_params": current_config["train_params"],
        "metrics": summary["metrics"],
        "diagnostics": summary["diagnostics"],
        "decision": decision,
        "parameter_updates": parameter_updates,
        "reasoning": reasoning,
        "next_train_params": next_config["train_params"]
    }

    append_jsonl(record, args.log_file)

    print("\n===== Experiment Logged =====")
    print(f"Log file: {args.log_file}")
    print(f"group={args.group_name}, seed={args.seed}, round={current_config['round_id']}")


if __name__ == "__main__":
    main()