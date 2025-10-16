from __future__ import annotations
import time, uuid, gc
from dataclasses import dataclass
from typing import Optional, List
import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..interfaces import GenerationParams, GenerationResult
from .base import BaseEngine
from ...logs.benchmark_logger import get_logger

def debug_gpu_memory():
    """Debug function to help identify what's using GPU memory."""
    if not torch.cuda.is_available():
        return
    
    print("=== GPU Memory Debug Info ===")
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"Max reserved: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
    
    # Try to get memory summary
    try:
        memory_summary = torch.cuda.memory_summary()
        print("Memory summary:")
        print(memory_summary)
    except Exception as e:
        print(f"Could not get memory summary: {e}")
    
    # Check for any tensors on GPU
    try:
        gpu_tensors = []
        for obj in gc.get_objects():
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                gpu_tensors.append((type(obj).__name__, obj.shape, obj.dtype, obj.device))
        
        if gpu_tensors:
            print(f"Found {len(gpu_tensors)} tensors on GPU:")
            for tensor_info in gpu_tensors[:10]:  # Show first 10
                print(f"  {tensor_info}")
        else:
            print("No tensors found on GPU")
    except Exception as e:
        print(f"Could not scan for GPU tensors: {e}")
    
    print("=== End GPU Memory Debug ===")

@dataclass
class DeepSpeedFLOPs:
    """Container for DeepSpeed FLOP measurements."""
    forward_flops: int
    backward_flops: int
    total_flops: int
    flops_per_token: float

class DeepSpeedLocalEngine(BaseEngine):
    """DeepSpeed engine with FLOP measurement capabilities."""
    
    def __init__(self,
                 model_id: str,
                 dtype: str = "auto",
                 gpu_memory_utilization: float = 0.90,
                 enforce_eager: bool = True,
                 enable_flop_profiling: bool = True,
                 # Quantization parameters
                 quantization: Optional[str] = None,
                 quantization_param_path: Optional[str] = None):
        """
        Initialize DeepSpeed engine with FLOP profiling.
        
        Args:
            model_id: HuggingFace model identifier
            dtype: Model precision (auto, bfloat16, float16, etc.)
            gpu_memory_utilization: GPU memory utilization fraction
            enforce_eager: Whether to use eager execution
            enable_flop_profiling: Whether to enable FLOP measurement
            quantization: Quantization method (deepspeedfp, int8, int4, etc.)
            quantization_param_path: Path to quantization parameters
        """
        self.model_id = model_id
        self.dtype = dtype
        self.enable_flop_profiling = enable_flop_profiling
        
        # Log DeepSpeed calibration model loading with quantization info
        logger = get_logger()
        logger.info(f"🔧 Loading DeepSpeed calibration model: {model_id}")
        logger.info(f"   dtype: {dtype}")
        logger.info(f"   gpu_memory_utilization: {gpu_memory_utilization}")
        logger.info(f"   enforce_eager: {enforce_eager}")
        logger.info(f"   enable_flop_profiling: {enable_flop_profiling} (always enabled for calibration)")
        if quantization is not None:
            logger.info(f"   quantization: {quantization}")
            if quantization_param_path is not None:
                logger.info(f"   quantization_param_path: {quantization_param_path}")
        else:
            logger.info(f"   quantization: None (no quantization)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set left padding for decoder-only models
        self.tokenizer.padding_side = "left"
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=self._get_torch_dtype(dtype),
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Initialize DeepSpeed with simpler configuration
        self.ds_engine = deepspeed.init_inference(
            model=model,
            mp_size=1,  # Single GPU for now
            dtype=self._get_torch_dtype(dtype),
            replace_with_kernel_inject=False,  # Disable kernel injection to avoid tuple issues
            max_out_tokens=2048,  # Reasonable default
        )
        
        # Store device for later use
        self.device = next(self.ds_engine.parameters()).device if hasattr(self.ds_engine, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DeepSpeed device: {self.device}")

        # Enable FLOP profiling if requested
        if self.enable_flop_profiling:
            self._setup_flop_profiling()
        
        logger.info(f"✅ DeepSpeed calibration model loaded successfully: {model_id}")
    
    def _get_torch_dtype(self, dtype: str) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        if dtype == "auto":
            return torch.float16  # Default for inference
        elif dtype == "bfloat16":
            return torch.bfloat16
        elif dtype == "float16":
            return torch.float16
        elif dtype == "float32":
            return torch.float32
        else:
            return torch.float16
    
    def _setup_flop_profiling(self):
        """Setup FLOP profiling for the model."""
        try:
            # Try to use DeepSpeed's built-in FLOP profiling
            if hasattr(self.ds_engine, 'module'):
                from deepspeed.profiling.flops_profiler import FlopsProfiler
                self.flops_profiler = FlopsProfiler(self.ds_engine.module)
                self.flops_profiler.start_profile()
            else:
                # Fallback: try to access the model directly
                from deepspeed.profiling.flops_profiler import FlopsProfiler
                self.flops_profiler = FlopsProfiler(self.ds_engine)
                self.flops_profiler.start_profile()
        except Exception as e:
            print(f"Warning: Could not setup FLOP profiling: {e}")
            self.enable_flop_profiling = False
            self.flops_profiler = None
    
    def _measure_flops(self, input_ids: torch.Tensor, generated_tokens: int) -> Optional[DeepSpeedFLOPs]:
        """Measure FLOPs for the generation."""
        if not self.enable_flop_profiling or not hasattr(self, 'flops_profiler') or self.flops_profiler is None:
            return None
        
        try:
            # Get FLOP measurements
            forward_flops = self.flops_profiler.get_total_flops()
            
            # Estimate FLOPs per token (rough approximation)
            total_tokens = input_ids.shape[1] + generated_tokens
            flops_per_token = forward_flops / max(total_tokens, 1)
            
            return DeepSpeedFLOPs(
                forward_flops=int(forward_flops),
                backward_flops=0,  # No backward pass in inference
                total_flops=int(forward_flops),
                flops_per_token=flops_per_token
            )
        except Exception as e:
            print(f"Warning: Could not measure FLOPs: {e}")
            return None
    
    def generate(self, prompt: str, params: GenerationParams) -> GenerationResult:
        """Generate text for a single prompt."""
        return self.generate_batch([prompt], params)[0]
    
    def generate_batch(self, prompts: List[str], params: GenerationParams) -> List[GenerationResult]:
        """Generate text for a batch of prompts."""
        if not prompts:
            return []
        
        # Tokenize inputs with dynamic padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False  # Don't truncate - let the model handle long sequences
        )
        
        # Use stored device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Prepare generation parameters
        generation_kwargs = {
            "max_new_tokens": params.max_new_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "do_sample": params.do_sample if params.do_sample is not None else (params.temperature is not None and params.temperature > 0),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        if params.top_k is not None:
            generation_kwargs["top_k"] = params.top_k
        
        if params.stop is not None:
            # Convert stop strings to token IDs
            stop_token_ids = []
            for stop_str in params.stop:
                stop_tokens = self.tokenizer.encode(stop_str, add_special_tokens=False)
                stop_token_ids.extend(stop_tokens)
            generation_kwargs["eos_token_id"] = stop_token_ids
        
        # Generate
        t0 = time.time()
        
        with torch.no_grad():
            outputs = self.ds_engine.generate(
                input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
        t1 = time.time()
        
        # Process results
        results: List[GenerationResult] = []
        
        for i, (input_seq, output_seq) in enumerate(zip(input_ids, outputs)):
            # Extract generated tokens (remove input tokens)
            generated_tokens = output_seq[len(input_seq):]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Calculate token counts
            prompt_tokens = len(input_seq)
            completion_tokens = len(generated_tokens)
            total_tokens = prompt_tokens + completion_tokens
            
            # Measure FLOPs if enabled
            flops_data = None
            if self.enable_flop_profiling:
                flops_data = self._measure_flops(input_seq.unsqueeze(0), completion_tokens)
            
            # Create result
            result_data = {
                "text": generated_text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "ttft_ms": None,  # Not easily measurable with current setup
                "latency_ms": (t1 - t0) * 1000.0,
                "raw": {
                    "input_ids": input_seq.cpu().tolist(),
                    "output_ids": output_seq.cpu().tolist(),
                    "flops": flops_data.__dict__ if flops_data else None
                }
            }
            
            results.append(GenerationResult(**result_data))
        
        return results
    
    def close(self):
        """Tear down and free GPU memory after each benchmark run."""
        print(f"Starting DeepSpeed cleanup...")
        print(f"Pre-cleanup GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
        
        try:
            # Step 1: Clean up FLOP profiler
            if hasattr(self, 'flops_profiler') and self.flops_profiler is not None:
                try:
                    print("Cleaning up FLOP profiler...")
                    self.flops_profiler.end_profile()
                    del self.flops_profiler
                    self.flops_profiler = None
                except Exception as e:
                    print(f"Warning: FLOP profiler cleanup failed: {e}")
            
            # Step 2: More aggressive cleanup for DeepSpeed engine
            if hasattr(self, 'ds_engine') and self.ds_engine is not None:
                try:
                    print("Cleaning up DeepSpeed engine...")
                    
                    # Try to access the underlying model and clean it up
                    if hasattr(self.ds_engine, 'module'):
                        print("Cleaning up DeepSpeed module...")
                        # Move model to CPU first to free GPU memory
                        if hasattr(self.ds_engine.module, 'cpu'):
                            self.ds_engine.module.cpu()
                        del self.ds_engine.module
                    
                    # Try to destroy the DeepSpeed engine
                    if hasattr(self.ds_engine, 'destroy'):
                        print("Calling DeepSpeed destroy...")
                        self.ds_engine.destroy()
                    
                    # Try to destroy DeepSpeed context
                    try:
                        import deepspeed
                        if hasattr(deepspeed, 'destroy'):
                            print("Calling DeepSpeed global destroy...")
                            deepspeed.destroy()
                    except Exception as e:
                        print(f"DeepSpeed global destroy failed: {e}")
                    
                    del self.ds_engine
                    self.ds_engine = None
                    
                except Exception as e:
                    print(f"Warning: DeepSpeed engine cleanup failed: {e}")
            
            # Step 3: Clean up tokenizer
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                try:
                    print("Cleaning up tokenizer...")
                    del self.tokenizer
                    self.tokenizer = None
                except Exception as e:
                    print(f"Warning: Tokenizer cleanup failed: {e}")
            
            # Step 4: Clean up device reference
            if hasattr(self, 'device'):
                del self.device
                
        except Exception as e:
            print(f"Warning: General cleanup failed: {e}")
                
        finally:
            # Step 5: Force garbage collection multiple times
            print("Running garbage collection...")
            for i in range(5):
                collected = gc.collect()
                print(f"GC cycle {i+1}: collected {collected} objects")
            
            if torch.cuda.is_available():
                # Step 6: Clear CUDA cache multiple times
                print("Clearing CUDA cache...")
                for i in range(5):
                    torch.cuda.empty_cache()
                    print(f"CUDA cache clear {i+1}: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
                
                # Step 7: Force synchronization to ensure cleanup is complete
                torch.cuda.synchronize()
                
                # Step 8: Check process memory usage
                try:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    memory_info = process.memory_info()
                    print(f"Process memory usage: {memory_info.rss / 1024 / 1024 / 1024:.2f} GB RSS, {memory_info.vms / 1024 / 1024 / 1024:.2f} GB VMS")
                except Exception as e:
                    print(f"Could not get process memory info: {e}")
                
                # Step 9: Final memory check
                final_allocated = torch.cuda.memory_allocated()/1e9
                final_reserved = torch.cuda.memory_reserved()/1e9
                print(f"Final GPU memory after cleanup: {final_allocated:.2f} GB allocated, {final_reserved:.2f} GB reserved")
                
                # Step 10: Additional cleanup if memory is still high
                if final_allocated > 1.0 or final_reserved > 10.0:  # If more than 1GB allocated or 10GB reserved
                    print(f"High memory usage detected (allocated: {final_allocated:.2f} GB, reserved: {final_reserved:.2f} GB), attempting additional cleanup...")
                    
                    # Debug what's using memory
                    debug_gpu_memory()
                    
                    # More aggressive cleanup for reserved memory
                    if final_reserved > 10.0:
                        print("High reserved memory detected, attempting aggressive cleanup...")
                        for i in range(10):  # More aggressive cleanup cycles
                            gc.collect()
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            
                            current_allocated = torch.cuda.memory_allocated()/1e9
                            current_reserved = torch.cuda.memory_reserved()/1e9
                            print(f"Aggressive cleanup cycle {i+1}: {current_allocated:.2f} GB allocated, {current_reserved:.2f} GB reserved")
                            
                            # If we've made good progress, break early
                            if current_reserved < 5.0:
                                print("Reserved memory successfully reduced, stopping aggressive cleanup")
                                break
                    
                    # Try to reset CUDA context (nuclear option)
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print(f"After CUDA reset: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
                    except Exception as e:
                        print(f"Warning: CUDA reset failed: {e}")
                else:
                    print("Memory cleanup successful - low memory usage detected.")
                
                print("DeepSpeed cleanup completed.")
