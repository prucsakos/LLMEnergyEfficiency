from __future__ import annotations
import time
from openai import OpenAI
from ..interfaces import GenerationParams, GenerationResult
from .base import BaseEngine

class VLLMOpenAIServerEngine(BaseEngine):
    """Engine for a running vLLM server (OpenAI-compatible).

    See vLLM docs for the /v1 (Completions/Chat) API. :contentReference[oaicite:8]{index=8}

    Args:
        base_url: e.g., "http://127.0.0.1:8000/v1"
        model: model id served by vLLM
        api_key: any non-empty string if your server enforces a key; "EMPTY" otherwise
        use_chat: if True, call /chat/completions; else /completions
    """
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", use_chat: bool = False):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.use_chat = use_chat

    def generate(self, prompt: str, params: GenerationParams) -> GenerationResult:
        t0 = time.time()
        # Build request parameters with only non-None values
        request_kwargs = {
            "model": self.model,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_new_tokens,
            "stop": params.stop or None,
            "seed": params.seed,
        }
        
        # Only add top_k if it's not None
        if params.top_k is not None:
            request_kwargs["top_k"] = params.top_k

        if self.use_chat:
            request_kwargs["messages"] = [{"role": "user", "content": prompt}]
            resp = self.client.chat.completions.create(**request_kwargs)
            text = resp.choices[0].message.content or ""
        else:
            request_kwargs["prompt"] = prompt
            resp = self.client.completions.create(**request_kwargs)
            text = resp.choices[0].text or ""

        t1 = time.time()
        usage = getattr(resp, "usage", None)
        prompt_tok = getattr(usage, "prompt_tokens", None) if usage else None
        comp_tok = getattr(usage, "completion_tokens", None) if usage else None
        total_tok = getattr(usage, "total_tokens", None) if usage else None

        # vLLM exposes TTFT in server metrics; here we record end-to-end latency in ms
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=total_tok,
            ttft_ms=None,
            latency_ms=(t1 - t0) * 1000.0,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else resp.__dict__,
        )
