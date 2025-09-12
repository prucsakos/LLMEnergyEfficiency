from __future__ import annotations
import time, uuid, gc
from dataclasses import dataclass
from typing import Optional, List
import torch
from vllm import LLM, SamplingParams
from ..interfaces import GenerationParams, GenerationResult
from .base import BaseEngine

class VLLMLocalEngine(BaseEngine):
    """In-process vLLM engine for offline benchmarking."""
    def __init__(self,
                 model_id: str,
                 dtype: str = "auto",
                 gpu_memory_utilization: float = 0.90,
                 enforce_eager: bool = True,
                 tensor_parallel_size: int = 1):
        # Construct once per run
        self.llm = LLM(
            model=model_id,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )

    def generate(self, prompt: str, params: GenerationParams) -> GenerationResult:
        sp = SamplingParams(
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_new_tokens,
            stop=params.stop or None,
            seed=params.seed,
        )
        t0 = time.time()
        outs = self.llm.generate([prompt], sp, use_tqdm=False)
        t1 = time.time()
        out = outs[0]
        # Count tokens from RequestOutput (prompt_token_ids & outputs[0].token_ids) :contentReference[oaicite:2]{index=2}
        prompt_tok = len(out.prompt_token_ids or [])
        comp_tok = len(out.outputs[0].token_ids or []) if out.outputs else 0
        return GenerationResult(
            text=(out.outputs[0].text if out.outputs else ""),
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=prompt_tok + comp_tok,
            ttft_ms=None,  # not exposed directly offline; can be measured with streaming
            latency_ms=(t1 - t0) * 1000.0,
            raw=out.dict() if hasattr(out, "dict") else None,
        )

    def close(self):
        """Tear down and free GPU memory after each benchmark run."""
        try:
            del self.llm
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # release unoccupied cached memory :contentReference[oaicite:3]{index=3}
