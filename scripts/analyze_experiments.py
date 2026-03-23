# scripts/analyze_experiments.py

import json
from pathlib import Path
import pandas as pd


def load_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def flatten_record(record: dict):
    metrics = record["metrics"]
    current_params = record["current_train_params"]
    next_params = record["next_train_params"]

    row = {
        "timestamp": record["timestamp"],
        "group_name": record["group_name"],
        "seed": record["seed"],
        "round_id": record["round_id"],
        "decision": record["decision"],

        "current_learning_rate": current_params["learning_rate"],
        "current_n_steps": current_params["n_steps"],
        "current_ent_coef": current_params["ent_coef"],
        "current_total_timesteps": current_params["total_timesteps"],

        "avg_return": metrics["avg_return"],
        "return_std": metrics["return_std"],
        "avg_episode_length": metrics["avg_episode_length"],
        "success_rate": metrics["success_rate"],
        "dominant_action": metrics["dominant_action"],
        "dominant_action_ratio": metrics["dominant_action_ratio"],
        "stability_flag": metrics["stability_flag"],

        "next_learning_rate": next_params["learning_rate"],
        "next_n_steps": next_params["n_steps"],
        "next_ent_coef": next_params["ent_coef"],
        "next_total_timesteps": next_params["total_timesteps"],
    }
    return row


def main():
    log_path = "logs/experiment_history.jsonl"
    output_csv = "logs/experiment_table.csv"
    output_summary_csv = "logs/experiment_summary_by_group_round.csv"

    records = load_jsonl(log_path)
    rows = [flatten_record(r) for r in records]
    df = pd.DataFrame(rows)

    # 保存明细表
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # 计算 group + round 的均值汇总
    summary_df = (
        df.groupby(["group_name", "round_id"], as_index=False)
        .agg({
            "avg_return": "mean",
            "return_std": "mean",
            "success_rate": "mean",
            "dominant_action_ratio": "mean"
        })
    )
    summary_df.to_csv(output_summary_csv, index=False, encoding="utf-8-sig")

    print("\n===== Experiment Analysis Finished =====")
    print(f"Detail table saved to: {output_csv}")
    print(f"Summary table saved to: {output_summary_csv}")
    print("\nPreview of detail table:")
    print(df.head())
    print("\nPreview of summary table:")
    print(summary_df)


if __name__ == "__main__":
    main()