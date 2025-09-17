from __future__ import annotations

from .vllm_local import VLLMLocalEngine
from .transformers_local import TransformersLocalEngine

def create_engine(engine_name: str, model_id: str, *, dtype: str, gpu_memory_utilization: float | None = None, enforce_eager: bool | None = None):
    name = (engine_name or "vllm").lower()
    if name == "vllm":
        print("Detected VLLM")
        return VLLMLocalEngine(
            model_id=model_id,
            dtype=dtype,
            gpu_memory_utilization=float(gpu_memory_utilization or 0.9),
            enforce_eager=bool(enforce_eager if enforce_eager is not None else True),
        )
    if name in ("hf", "transformers"):
        print("Detected Transformers")
        return TransformersLocalEngine(model_id=model_id, dtype=dtype)
    raise ValueError(f"Unknown engine: {engine_name}")


