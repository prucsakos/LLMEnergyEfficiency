"""
uv run single_infer.py \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --tokenizer nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
  --message "What is 779,678 * 866,978?" \
  --max-thinking-budget 1024 \
  --max-tokens 2048 \
  --temperature 0.0 \
  --top-p 0.95
"""

import argparse
from typing import Any, Dict, List

import openai
from transformers import AutoTokenizer


class ThinkingBudgetClient:
    def __init__(self, base_url: str, api_key: str, tokenizer_name_or_path: str):
        self.base_url = base_url
        self.api_key = api_key
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        max_thinking_budget: int = 512,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        assert (
            max_tokens > max_thinking_budget
        ), f"thinking budget must be smaller than maximum new tokens. Given {max_tokens=} and {max_thinking_budget=}"

        # Step 1: reasoning content
        response = self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_thinking_budget, **kwargs
        )
        content = response.choices[0].message.content

        reasoning_content = content
        if "</think>" not in reasoning_content:
            reasoning_content = f"{reasoning_content}.\n</think>\n\n"

        reasoning_tokens_len = len(
            self.tokenizer.encode(reasoning_content, add_special_tokens=False)
        )
        remaining_tokens = max_tokens - reasoning_tokens_len
        assert (
            remaining_tokens > 0
        ), f"remaining tokens must be positive. Given {remaining_tokens=}. Increase max_tokens or lower max_thinking_budget."

        # Step 2: append reasoning and final completion
        messages.append({"role": "assistant", "content": reasoning_content})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=True
        )

        response = self.client.completions.create(
            model=model, prompt=prompt, max_tokens=remaining_tokens, **kwargs
        )

        return {
            "reasoning_content": reasoning_content.strip().strip("</think>").strip(),
            "content": response.choices[0].text,
            "finish_reason": response.choices[0].finish_reason,
        }


def main():
    parser = argparse.ArgumentParser(description="ThinkingBudgetClient CLI")
    parser.add_argument("--base-url", type=str, required=True, help="Base URL of vLLM server")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API key (default: EMPTY)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer name/path")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--message", type=str, required=True, help="User message")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant. /think", help="System prompt")
    parser.add_argument("--max-thinking-budget", type=int, default=512, help="Max thinking tokens")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max total tokens")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p sampling")
    args = parser.parse_args()

    client = ThinkingBudgetClient(args.base_url, args.api_key, args.tokenizer)
    result = client.chat_completion(
        model=args.model,
        messages=[
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.message},
        ],
        max_thinking_budget=args.max_thinking_budget,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("=== Reasoning ===")
    print(result["reasoning_content"])
    print("\n=== Final Answer ===")
    print(result["content"])
    print("\nFinish reason:", result["finish_reason"])


if __name__ == "__main__":
    main()
