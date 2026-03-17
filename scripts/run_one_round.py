# scripts/run_one_round.py

# =========================
# 1. 标准库导入
# =========================
import argparse
from pathlib import Path
import subprocess
import sys


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Run one full RL-LLM tuning round."
    )

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
        "--system-prompt",
        type=str,
        required=True,
        help="Path to system prompt txt"
    )

    parser.add_argument(
        "--user-prompt-template",
        type=str,
        required=True,
        help="Path to user prompt template txt"
    )

    parser.add_argument(
        "--llm-script",
        type=str,
        required=True,
        help="Path to LLM call script, e.g. scripts/call_llm.py or scripts/call_llm_local.py"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store current round outputs"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model name passed to LLM script"
    )

    return parser.parse_args()


def run_command(cmd: list, step_name: str):
    """
    执行子命令，并在失败时直接抛错。
    
    参数:
    - cmd: 命令列表
    - step_name: 当前步骤名称，用于打印日志
    """
    print(f"\n===== Running Step: {step_name} =====")
    print("Command:", " ".join(cmd))

    # check=True 表示命令失败时直接抛 CalledProcessError
    subprocess.run(cmd, check=True)


def main():
    """
    主入口：
    按顺序执行：
    1. 训练
    2. 摘要
    3. LLM 调用
    4. 配置更新
    """
    args = parse_args()

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 统一约定本轮几个关键文件路径
    raw_metrics_path = str(output_dir / "raw_metrics.json")
    summary_path = str(output_dir / "summary.json")
    llm_response_path = str(output_dir / "llm_response.json")
    next_config_path = str(output_dir / "next_config.json")

    # 当前 Python 解释器路径
    python_executable = sys.executable

    # =========================
    # Step 1: 训练
    # =========================
    run_command(
        [
            python_executable,
            "scripts/run_training.py",
            "--config",
            args.config
        ],
        step_name="Training"
    )

    # =========================
    # Step 2: 生成摘要
    # =========================
    run_command(
        [
            python_executable,
            "scripts/summarize_results.py",
            "--input",
            raw_metrics_path,
            "--output",
            summary_path
        ],
        step_name="Summarization"
    )

    # =========================
    # Step 3: 调用 LLM
    # =========================
    llm_cmd = [
        python_executable,
        args.llm_script,
        "--summary",
        summary_path,
        "--search-space",
        args.search_space,
        "--system-prompt",
        args.system_prompt,
        "--user-prompt-template",
        args.user_prompt_template,
        "--output",
        llm_response_path
    ]

    # 如果传了 model，就一起传给 LLM 脚本
    if args.model:
        llm_cmd.extend(["--model", args.model])

    run_command(
        llm_cmd,
        step_name="LLM Decision"
    )

    # =========================
    # Step 4: 更新配置
    # =========================
    run_command(
        [
            python_executable,
            "scripts/update_config.py",
            "--config",
            args.config,
            "--search-space",
            args.search_space,
            "--llm-response",
            llm_response_path,
            "--output",
            next_config_path
        ],
        step_name="Config Update"
    )

    print("\n===== One Round Finished Successfully =====")
    print(f"Raw metrics:   {raw_metrics_path}")
    print(f"Summary:       {summary_path}")
    print(f"LLM response:  {llm_response_path}")
    print(f"Next config:   {next_config_path}")


if __name__ == "__main__":
    main()