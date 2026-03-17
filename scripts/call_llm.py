# scripts/call_llm.py

# =========================
# 1. 标准库导入
# =========================
import argparse
from pathlib import Path
import sys
import json
import os


# =========================
# 2. 第三方库导入
# =========================
import requests


# =========================
# 3. 加入项目根目录到 sys.path
# =========================
# 这样脚本就能找到 scripts/utils_io.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入我们自己写的 JSON 工具函数
from scripts.utils_io import load_json, save_json


def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="Call Gemini API and generate llm_response.json"
    )

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
        default="gemini-3-flash-preview",
        help="Gemini model name"
    )

    parser.add_argument(
        "--gemini-url",
        type=str,
        default="https://generativelanguage.googleapis.com/v1beta/models",
        help="Gemini API base URL"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key. If not provided, fallback to GEMINI_API_KEY env var."
    )
    return parser.parse_args()


def load_text(path: str) -> str:
    """
    读取文本文件。
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_user_prompt(template_text: str, search_space: dict, summary: dict) -> str:
    """
    用 search_space 和 summary 替换模板里的占位符。
    """
    search_space_json = json.dumps(search_space, ensure_ascii=False, indent=2)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)

    prompt = template_text.replace("{SEARCH_SPACE_JSON}", search_space_json)
    prompt = prompt.replace("{SUMMARY_JSON}", summary_json)

    return prompt


def call_gemini_generate(
    gemini_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None
) -> str:
    """
    调用 Gemini generateContent REST API。
    
    返回:
    - 模型输出的文本
    """
    # 从环境变量读取 API Key或者直接输入
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key is missing. Provide --api-key or set GEMINI_API_KEY.")
    # 拼接完整请求地址
    url = f"{gemini_url}/{model}:generateContent"

    # Gemini 的请求体
    payload = {
        "systemInstruction": {
            "parts": [
                {"text": system_prompt}
            ]
        },
        "contents": [
            {
                "parts": [
                    {"text": user_prompt}
                ]
            }
        ],
        "generationConfig": {
            # 降低随机性，让 JSON 输出更稳
            "temperature": 0.2,
            # 尽量要求模型直接返回 JSON
            "responseMimeType": "application/json"
        }
    }

    # 请求头
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    print("\n===== Calling Gemini API =====")
    print(f"Model: {model}")
    print(f"URL: {url}")
    print(f"System prompt length: {len(system_prompt)}")
    print(f"User prompt length: {len(user_prompt)}")

    # 发请求
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=180
    )

    # 如果 HTTP 状态码不是 2xx，这里会直接抛错
    response.raise_for_status()

    # 解析 Gemini 返回 JSON
    data = response.json()

    # Gemini 返回文本通常在 candidates[0].content.parts[0].text
    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"No candidates returned by Gemini API. Raw response: {data}")

    content = candidates[0].get("content", {})
    parts = content.get("parts", [])
    if not parts:
        raise ValueError(f"No content parts returned by Gemini API. Raw response: {data}")

    model_text = parts[0].get("text", "")
    if not model_text:
        raise ValueError(f"Gemini returned empty text. Raw response: {data}")

    return model_text


def extract_json_from_text(text: str) -> dict:
    """
    尝试把模型输出解析成 JSON。
    
    处理逻辑:
    1. 先直接 json.loads
    2. 如果失败，再尝试截取第一个 { 到最后一个 } 之间的内容
    """
    text = text.strip()

    # 先尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 如果模型前后多吐了额外文本，尝试截 JSON 主体
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("Failed to extract valid JSON from model output.")


def validate_llm_response_shape(llm_response: dict):
    """
    做一个最小结构校验。
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
    - 读取 summary / search_space / prompts
    - 调用 Gemini API
    - 解析模型输出
    - 保存 llm_response.json
    """
    args = parse_args()

    # 读取输入 JSON
    summary = load_json(args.summary)
    search_space = load_json(args.search_space)

    # 读取 prompt 文本
    system_prompt = load_text(args.system_prompt)
    user_prompt_template = load_text(args.user_prompt_template)

    # 构造 user prompt
    user_prompt = build_user_prompt(
        template_text=user_prompt_template,
        search_space=search_space,
        summary=summary
    )

    # 调 Gemini API
    model_text = call_gemini_generate(
        gemini_url=args.gemini_url,
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_key=args.api_key
    )

    # 可选：把原始文本也保存一下，便于调试
    raw_output_path = str(Path(args.output).with_name("llm_raw_output.txt"))
    with open(raw_output_path, "w", encoding="utf-8") as f:
        f.write(model_text)

    # 提取 JSON
    llm_response = extract_json_from_text(model_text)

    # 最小结构校验
    validate_llm_response_shape(llm_response)

    # 保存正式 JSON
    save_json(llm_response, args.output)

    print("\n===== LLM Response Generated =====")
    print(f"Model: {args.model}")
    print(f"Saved JSON to: {args.output}")
    print(f"Saved raw text to: {raw_output_path}")
    print(f"Decision: {llm_response['decision']}")
    print(f"Parameter updates: {llm_response['parameter_updates']}")


if __name__ == "__main__":
    main()