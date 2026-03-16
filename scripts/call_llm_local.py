# scripts/call_llm.py

# =========================
# 1. 标准库导入
# =========================
import argparse
from pathlib import Path
import sys
import json


# =========================
# 2. 第三方库导入
# =========================
import requests


# =========================
# 3. 加入项目根目录到 sys.path
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.utils_io import load_json, save_json


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Call local LLM and generate llm_response.json")

    parser.add_argument(
        "--summary",
        type=str,
        required=True,
        help="Path to summary.json"
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
        "--output",
        type=str,
        required=True,
        help="Path to llm_response.json"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5:4b",
        help="Ollama model name"
    )

    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama generate API URL"
    )

    return parser.parse_args()


def load_text(path: str) -> str:
    """
    读取纯文本文件。
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_user_prompt(template_text: str, search_space: dict, summary: dict) -> str:
    """
    用 search_space 和 summary 替换模板中的占位符。
    """
    search_space_json = json.dumps(search_space, ensure_ascii=False, indent=2)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)

    prompt = template_text.replace("{SEARCH_SPACE_JSON}", search_space_json)
    prompt = prompt.replace("{SUMMARY_JSON}", summary_json)

    return prompt


def call_ollama_generate(ollama_url: str, model: str, system_prompt: str, user_prompt: str) -> str:
    """
    调用 Ollama /api/generate 接口。
    
    返回:
    - 模型最终返回的文本字符串
    """
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False
    }

    response = requests.post(ollama_url, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()

    # Ollama /api/generate 的主要文本在 response 字段中
    model_text = data.get("response", "")

    return model_text


def extract_json_from_text(text: str) -> dict:
    """
    尝试把模型输出文本解析成 JSON。
    
    当前做法：
    1. 先直接 json.loads
    2. 如果失败，再尝试截取首个 { 到末尾 } 的内容
    
    返回:
    - Python dict
    
    如果解析失败，抛出异常
    """
    text = text.strip()

    # 先尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 如果模型前后多吐了点废话，尝试粗暴截取 JSON 主体
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("Failed to extract valid JSON from model output.")


def validate_llm_response_shape(llm_response: dict):
    """
    做一个最小结构校验，避免明显错误。
    """
    required_top_keys = [
        "round_id",
        "decision",
        "parameter_updates",
        "reasoning",
        "confidence",
        "should_continue"
    ]

    for key in required_top_keys:
        if key not in llm_response:
            raise ValueError(f"Missing key in llm_response: {key}")

    if llm_response["decision"] not in ["adjust", "keep", "stop"]:
        raise ValueError("Invalid decision field in llm_response.")

    if not isinstance(llm_response["parameter_updates"], dict):
        raise ValueError("parameter_updates must be a dict.")

    if not isinstance(llm_response["reasoning"], dict):
        raise ValueError("reasoning must be a dict.")


def main():
    """
    主入口：
    - 读取 prompt 与输入文件
    - 调用本地 LLM
    - 解析输出 JSON
    - 保存 llm_response.json
    """
    args = parse_args()

    # 读取输入文件
    summary = load_json(args.summary)
    search_space = load_json(args.search_space)
    system_prompt = load_text(args.system_prompt)
    user_prompt_template = load_text(args.user_prompt_template)

    # 构造 user prompt
    user_prompt = build_user_prompt(user_prompt_template, search_space, summary)

    # 调用本地模型
    model_text = call_ollama_generate(
        ollama_url=args.ollama_url,
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    # 提取 JSON
    llm_response = extract_json_from_text(model_text)

    # 最小结构校验
    validate_llm_response_shape(llm_response)

    # 保存结果
    save_json(llm_response, args.output)

    print("\n===== LLM Response Generated =====")
    print(f"Model: {args.model}")
    print(f"Saved to: {args.output}")
    print(f"Decision: {llm_response['decision']}")
    print(f"Parameter updates: {llm_response['parameter_updates']}")


if __name__ == "__main__":
    main()