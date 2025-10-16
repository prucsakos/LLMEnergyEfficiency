from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol, Dict

__all__ = ["GenerationParams", "GenerationResult", "TextEngine"]

@dataclass
class GenerationParams:
    """Canonical text-generation parameters.

    Args:
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p nucleus sampling.
        top_k: Top-k sampling (optional).
        stop: Optional list of stop strings.
        seed: Optional RNG seed for diversity (used by SC).
        dtype: 'auto'|'bfloat16'|'float16' etc. (forwarded if backend supports).
        use_kv_cache: Whether to use KV cache (if backend supports).
    """
    max_new_tokens: int = 128
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None  # If None, auto-determined by temperature > 0
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    dtype: str = "auto"
    use_kv_cache: bool = True

@dataclass
class GenerationResult:
    """Normalized result for a single prompt."""
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    ttft_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    raw: Optional[Dict] = None  # backend-specific payload
    formatted_input: Optional[str] = None  # formatted input text (for chat mode)

class TextEngine(Protocol):
    """Minimal protocol for text engines."""
    def generate(self, prompt: str, params: GenerationParams) -> GenerationResult: ...
