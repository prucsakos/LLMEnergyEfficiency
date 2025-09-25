from __future__ import annotations
import time, gc
from typing import Optional, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from ..interfaces import GenerationParams, GenerationResult
from .base import BaseEngine


def _map_dtype(dtype_str: str) -> torch.dtype | None:
    s = (dtype_str or "").lower()
    if s in ("auto", ""):
        return None
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("float32", "fp32"):
        return torch.float32
    return None


class _StringStopping(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_ids = [tokenizer.encode(s, add_special_tokens=False) for s in stop_strings if s]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if not self.stop_ids:
            return False
        generated = input_ids[0].tolist()
        for seq in self.stop_ids:
            if len(seq) > 0 and generated[-len(seq):] == seq:
                return True
        return False


class TransformersLocalEngine(BaseEngine):
    """Local engine using Hugging Face Transformers with batched generate()."""

    def __init__(self, model_id: str, dtype: str = "auto"):
        torch_dtype = _map_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        # Ensure left padding for batched generation
        self.tokenizer.padding_side = "left"
        
        # Ensure a real PAD token exists
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                # If EOS exists (many models), reuse it for padding (OK for inference).
                self.tokenizer.pad_token = self.tokenizer.eos_token  # :contentReference[oaicite:2]{index=2}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

    def generate(self, prompt: str, params: GenerationParams) -> GenerationResult:
        return self.generate_batch([prompt], params)[0]

    def generate_batch(self, prompts: List[str], params: GenerationParams) -> List[GenerationResult]:
        if not prompts:
            return []
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        stopping = None
        if params.stop:
            stopping = StoppingCriteriaList([_StringStopping(self.tokenizer, params.stop)])

        gen_kwargs = {
            "max_new_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "do_sample": params.do_sample if params.do_sample is not None else (params.temperature > 0),
            "use_cache": params.use_kv_cache,
            "stopping_criteria": stopping,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if params.top_k is not None:
            gen_kwargs["top_k"] = params.top_k

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        t1 = time.time()

        # Separate generated continuations for each item
        results: List[GenerationResult] = []
        input_len = inputs["input_ids"].shape[1]
        for i in range(outputs.shape[0]):
            gen_ids = outputs[i][input_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            prompt_tokens = int(inputs["input_ids"][i].shape[-1])
            completion_tokens = int(gen_ids.shape[-1])
            results.append(GenerationResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=(t1 - t0) * 1000.0,
                raw=None,
            ))
        return results

    def close(self):
        try:
            del self.model
            del self.tokenizer
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


