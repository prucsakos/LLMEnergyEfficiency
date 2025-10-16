from __future__ import annotations
import time, uuid, gc
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import torch
from vllm import LLM, SamplingParams
from ..interfaces import GenerationParams, GenerationResult
from .base import BaseEngine
from ...logs.benchmark_logger import get_logger

class VLLMLocalEngine(BaseEngine):
    """In-process vLLM engine for offline benchmarking."""
    def __init__(self,
                 model_id: str,
                 dtype: str = "auto",
                 gpu_memory_utilization: float = 0.90,
                 enforce_eager: bool = True,
                 tensor_parallel_size: int = 1,
                 # Quantization parameters
                 quantization: Optional[str] = None,
                 quantization_param_path: Optional[str] = None,
                 # CPU offloading parameters
                 cpu_offload_gb: Optional[float] = None,
                 swap_space: Optional[int] = None,
                 # Additional memory optimization
                 max_model_len: Optional[int] = None,
                 block_size: Optional[int] = None,
                 # Generation mode
                 generation_mode: str = "casual",
                 # System prompt for chat mode
                 system_prompt: Optional[str] = None,
                 # Chat template parameters
                 chat_template_kwargs: Optional[Dict[str, Any]] = None):
        # Build LLM initialization parameters
        llm_kwargs = {
            "model": model_id,
            "dtype": dtype,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
            "trust_remote_code": True,
        }
        
        # Add quantization parameters if specified
        if quantization is not None:
            llm_kwargs["quantization"] = quantization
            if quantization_param_path is not None:
                llm_kwargs["quantization_param_path"] = quantization_param_path
        
        # Add CPU offloading parameters if specified
        if cpu_offload_gb is not None:
            llm_kwargs["cpu_offload_gb"] = cpu_offload_gb
        if swap_space is not None:
            llm_kwargs["swap_space"] = swap_space
            
        # Add additional memory optimization parameters
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if block_size is not None:
            llm_kwargs["block_size"] = block_size
        
        # Store generation mode, system prompt, and chat template parameters for use in generation methods
        self.generation_mode = generation_mode.lower()
        if self.generation_mode not in ["casual", "chat"]:
            raise ValueError(f"Invalid generation_mode: {generation_mode}. Must be 'casual' or 'chat'")
        
        self.system_prompt = system_prompt
        self.chat_template_kwargs = chat_template_kwargs or {}
        
        # Log vLLM model loading with quantization info
        logger = get_logger()
        logger.info(f"🚀 Loading vLLM model: {model_id}")
        logger.info(f"   dtype: {dtype}")
        logger.info(f"   gpu_memory_utilization: {gpu_memory_utilization}")
        logger.info(f"   enforce_eager: {enforce_eager}")
        logger.info(f"   generation_mode: {self.generation_mode}")
        if self.system_prompt is not None:
            logger.info(f"   system_prompt: {self.system_prompt[:50]}...")
        if self.chat_template_kwargs:
            logger.info(f"   chat_template_kwargs: {self.chat_template_kwargs}")
        if quantization is not None:
            logger.info(f"   quantization: {quantization}")
            if quantization_param_path is not None:
                logger.info(f"   quantization_param_path: {quantization_param_path}")
        else:
            logger.info(f"   quantization: None (no quantization)")
        
        # Construct once per run
        self.llm = LLM(**llm_kwargs)
        logger.info(f"✅ vLLM model loaded successfully: {model_id}")

    def generate(self, prompt: str, params: GenerationParams) -> GenerationResult:
        return self.generate_batch([prompt], params)[0]

    def generate_batch(self, prompts: List[str], params: GenerationParams) -> List[GenerationResult]:
        if not prompts:
            return []
        # Build SamplingParams with only non-None values
        sampling_kwargs = {
            "temperature": params.temperature if params.temperature is not None else 1.0,
            "top_p": params.top_p if params.top_p is not None else 1.0,
            "max_tokens": params.max_new_tokens,
            "stop": params.stop or None,
            "seed": params.seed,
        }
        
        # Only add top_k if it's not None
        if params.top_k is not None:
            sampling_kwargs["top_k"] = params.top_k
            
        sp = SamplingParams(**sampling_kwargs)
        t0 = time.time()
        
        # Use appropriate generation method based on mode
        if self.generation_mode == "chat":
            # For chat mode, convert prompts to chat messages format
            # Each prompt becomes a conversation with system prompt (if provided) and user message
            chat_messages = []
            formatted_texts = []  # Store formatted texts for wandb logging
            
            for i, prompt in enumerate(prompts):
                conversation = []
                # Add system prompt if provided
                if self.system_prompt is not None:
                    conversation.append({"role": "system", "content": self.system_prompt})
                # Add user message
                conversation.append({"role": "user", "content": prompt})
                chat_messages.append(conversation)
                
                # Apply chat template to get formatted text for logging
                try:
                    # Apply chat template with custom parameters if provided
                    template_kwargs = {
                        "tokenize": False,
                        "add_generation_prompt": True
                    }
                    # Add custom chat template parameters
                    template_kwargs.update(self.chat_template_kwargs)
                    
                    formatted_text = self.llm.get_tokenizer().apply_chat_template(
                        conversation, 
                        **template_kwargs
                    )
                    formatted_texts.append(formatted_text)
                except Exception as e:
                    # If template application fails, use the raw conversation
                    logger = get_logger()
                    logger.warning(f"Chat template formatting failed for prompt {i+1}: {e}. Falling back to raw conversation format.")
                    formatted_texts.append(str(conversation))
            
            # Call vLLM chat with chat_template_kwargs
            outs = self.llm.chat(
                chat_messages, 
                sp, 
                use_tqdm=False,
                chat_template_kwargs=self.chat_template_kwargs
            )
        else:
            # For casual mode, use standard generate API
            outs = self.llm.generate(prompts, sp, use_tqdm=False)
            
        t1 = time.time()
        results: List[GenerationResult] = []
        for i, out in enumerate(outs):
            prompt_tok = len(out.prompt_token_ids or [])
            comp_tok = len(out.outputs[0].token_ids or []) if out.outputs else 0
            
            # Include formatted input text for chat mode
            formatted_input = formatted_texts[i] if self.generation_mode == "chat" and i < len(formatted_texts) else None
            
            results.append(GenerationResult(
                text=(out.outputs[0].text if out.outputs else ""),
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                total_tokens=prompt_tok + comp_tok,
                ttft_ms=None,
                latency_ms=(t1 - t0) * 1000.0,
                raw=out.dict() if hasattr(out, "dict") else None,
                formatted_input=formatted_input,
            ))
        return results

    def close(self):
        """Tear down and free GPU memory after each benchmark run."""
        try:
            del self.llm
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # release unoccupied cached memory :contentReference[oaicite:3]{index=3}
