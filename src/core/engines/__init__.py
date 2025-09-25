from __future__ import annotations

from .vllm_local import VLLMLocalEngine
from .transformers_local import TransformersLocalEngine
from .deepspeed_local import DeepSpeedLocalEngine
from .openai_api import OpenAIInferenceEngine, create_openai_engine

def create_engine(engine_name: str, model_id: str, *, dtype: str, gpu_memory_utilization: float | None = None, enforce_eager: bool | None = None,
                 # Quantization parameters
                 quantization: str | None = None, quantization_param_path: str | None = None,
                 # CPU offloading parameters
                 cpu_offload_gb: float | None = None, swap_space: int | None = None,
                 # Additional memory optimization
                 max_model_len: int | None = None, block_size: int | None = None):
    name = (engine_name or "vllm").lower()
    if name == "vllm":
        print("Detected VLLM")
        return VLLMLocalEngine(
            model_id=model_id,
            dtype=dtype,
            gpu_memory_utilization=float(gpu_memory_utilization or 0.9),
            enforce_eager=bool(enforce_eager if enforce_eager is not None else True),
            # Pass quantization parameters
            quantization=quantization,
            quantization_param_path=quantization_param_path,
            # Pass CPU offloading parameters
            cpu_offload_gb=cpu_offload_gb,
            swap_space=swap_space,
            # Pass additional memory optimization parameters
            max_model_len=max_model_len,
            block_size=block_size,
        )
    if name in ("hf", "transformers"):
        print("Detected Transformers")
        return TransformersLocalEngine(model_id=model_id, dtype=dtype)
    if name == "deepspeed":
        print("Detected DeepSpeed")
        return DeepSpeedLocalEngine(
            model_id=model_id,
            dtype=dtype,
            gpu_memory_utilization=float(gpu_memory_utilization or 0.9),
            enforce_eager=bool(enforce_eager if enforce_eager is not None else True),
            enable_flop_profiling=True,
        )
    if name == "openai":
        print("Detected OpenAI API")
        return create_openai_engine()
    raise ValueError(f"Unknown engine: {engine_name}")


