from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import traceback
import argparse, math, time, json, os, copy, subprocess
from typing import Iterable, List, Optional, Tuple, Callable, Dict, Any
import itertools
from tqdm.auto import tqdm
import numpy as np
import torch

# Load environment variables from .env file
from ..utils import load_env_variables
load_env_variables()

# Import logging system
from ..logs.benchmark_logger import setup_logging, get_logger

from ..config.bench_config import load_bench_config, expand_runs, RunSpec, Prompts
from ..core.interfaces import GenerationParams
from ..core.engines import create_engine
from ..reasoning.aggregators import majority_vote
from ..reasoning.controller import self_evaluate_batched, self_consistency_batch, two_pass_batch, single_pass_batch  # use batched judge
from ..data.adapters import load_gsm8k, load_mmlu, load_csqa, exact_match, Sample, iter_dataset
from ..metrics.flop_estimation import to_tflops
from ..metrics.inequality import gini
from ..metrics.energy import EnergyMeter
from ..logs.wandb_logger import WandbRunLogger

# Import calibration system
from ..calibration import (
    CalibrationPoint, CalibrationDataset,
    NextTokenFLOPModel,
    load_calibration_dataset
)

# Optional direct import from vLLM for batched generation
from vllm import SamplingParams


# ============================================================================
# BENCHMARKING FUNCTIONS
# ============================================================================


# ============================================================================
# ENHANCED BENCHMARKING WITH CALIBRATION
# ============================================================================



def run_calibration_subprocess(spec: RunSpec, 
                             calibration_file: str,
                             prefill_ranges: List[int],
                             config_file: str,
                             estimation_points: int = 64) -> Optional[CalibrationDataset]:
    """
    Run calibration in a separate subprocess to ensure proper GPU cleanup.
    
    Args:
        spec: Model specification
        calibration_file: Path to save calibration results
        prefill_ranges: List of prefill token ranges to test
        estimation_points: Number of estimation points for extrapolation evaluation
        
    Returns:
        CalibrationDataset if successful, None if failed
    """
    logger = get_logger()
    
    # Get the path to the calibration script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_script = os.path.join(script_dir, "calibration", "run_calibration_deepspeed.py")
    
    if not os.path.exists(calibration_script):
        logger.error(f"❌ Calibration script not found: {calibration_script}")
        return None
    
    logger.info(f"🔧 Starting calibration subprocess for {spec.model_name}")
    logger.info(f"Calibration script: {calibration_script}")
    logger.info(f"Calibration file: {calibration_file}")
    
    try:
        # Prepare subprocess command
        cmd = [
            "python3", calibration_script,
            "--config", config_file,
            "--model_name", spec.model_name,
            "--calibration_file", calibration_file,
            "--prefill_ranges"] + [str(x) for x in prefill_ranges] + [
            "--estimation_points", str(estimation_points),
            "--log_level", "INFO"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run the subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=os.getcwd()  # Run in current working directory
        )
        
        # Log subprocess output
        if result.stdout:
            logger.info("Calibration subprocess stdout:")
            for line in result.stdout.strip().split('\n'):
                logger.info(f"  {line}")

        if result.stderr:
            logger.warning("Calibration subprocess stderr:")
            for line in result.stderr.strip().split('\n'):
                logger.warning(f"  {line}")
        
        if result.returncode == 0:
            logger.info("✓ Calibration subprocess completed successfully")
            
            # Load the calibration data
            if os.path.exists(calibration_file):
                try:
                    calibration_dataset = load_calibration_dataset(calibration_file)
                    logger.info(f"✓ Loaded calibration data with {len(calibration_dataset.points)} points")
                    return calibration_dataset
                except Exception as e:
                    logger.error(f"❌ Failed to load calibration data: {e}")
                    return None
            else:
                logger.error(f"❌ Calibration file not found: {calibration_file}")
                return None
        else:
            logger.error(f"❌ Calibration subprocess failed with return code: {result.returncode}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Calibration subprocess timed out after 1 hour")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to run calibration subprocess: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def run_one_with_calibration(spec: RunSpec,
                           calibration_dataset: Optional[CalibrationDataset] = None,
                           batch_size: Optional[int] = None,
                           wandb_project: str | None = None,
                           notes: str = "",
                           measure_energy: bool = False,
                           energy_device: int = 0,
                           energy_sample_interval_ms: float = 20.0,
                           skip_evaluation: bool = False) -> None:
    """
    Run a single benchmark with optional FLOP calibration.

    Args:
        spec: Benchmark specification
        calibration_dataset: Optional calibration data for FLOP extrapolation
        batch_size: Override batch size from config
        wandb_project: W&B project name
        notes: Additional notes
        measure_energy: Whether to measure GPU energy
        energy_device: GPU device index for energy measurement
        energy_sample_interval_ms: Sampling interval for energy measurement
        skip_evaluation: Whether to skip evaluation entirely
    """
    # Use configured batch size unless overridden
    bs = int(batch_size or getattr(spec, "batch_size", 1) or 1)

    # Build engine (one per run), then tear down afterwards
    logger = get_logger()
    logger.info(f"Specified engine: {spec.engine}")
    
    # Check GPU memory before creating main engine
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated()/1e9
        reserved_gb = torch.cuda.memory_reserved()/1e9
        logger.log_gpu_memory("Pre-benchmark", allocated_gb, reserved_gb)
    
    engine = create_engine(
        spec.engine,
        model_id=spec.hf_repo,
        dtype=spec.backend.dtype,
        gpu_memory_utilization=spec.backend.gpu_memory_utilization,
        enforce_eager=spec.backend.enforce_eager,
        # Pass quantization parameters
        quantization=spec.backend.quantization,
        quantization_param_path=spec.backend.quantization_param_path,
        # Pass CPU offloading parameters
        cpu_offload_gb=spec.backend.cpu_offload_gb,
        swap_space=spec.backend.swap_space,
        # Pass additional memory optimization parameters
        max_model_len=spec.backend.max_model_len,
        block_size=spec.backend.block_size,
        # Pass generation mode
        generation_mode=spec.generation.generation_mode,
        # Pass system prompt
        system_prompt=spec.generation.system_prompt,
        # Pass chat template parameters
        chat_template_kwargs=spec.generation.chat_template_kwargs,
    )

    gen = GenerationParams(
        max_new_tokens=spec.generation.max_new_tokens,
        temperature=spec.generation.temperature,
        top_p=spec.generation.top_p,
        top_k=spec.generation.top_k,
        frequency_penalty=spec.generation.frequency_penalty,
        do_sample=spec.generation.do_sample,
        stop=spec.generation.stop,
        seed=spec.generation.seed,
        use_kv_cache=spec.generation.use_kv_cache,
        dtype=spec.backend.dtype,
    )

    # W&B: one run per (model, dataset, budget, style, k, batch, prompt_set)
    wb = None
    run_name = f"{spec.model_name}|{spec.dataset}|style={spec.reasoning.style}|B={spec.think_budget}|K={spec.reasoning.self_consistency_k}|bs={bs}|prompt={spec.prompt_set_name}"
    if wandb_project:
        cfg = {
            "model": spec.hf_repo,
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "dataset": spec.dataset,
            "style": spec.reasoning.style,
            "think_budget": spec.think_budget,
            "K": spec.reasoning.self_consistency_k,
            "dtype": spec.backend.dtype,
            "prompt_set": spec.prompt_set_name,
            "config_name": spec.config_name,
            # Generation parameters
            "temperature": spec.generation.temperature,
            "top_p": spec.generation.top_p,
            "top_k": spec.generation.top_k,
            "do_sample": spec.generation.do_sample,
            "seed": spec.generation.seed,
            "generation_mode": spec.generation.generation_mode,
            "system_prompt": spec.generation.system_prompt or "none",
            "chat_template_kwargs": spec.generation.chat_template_kwargs or {},
            # Benchmark backend (vLLM) parameters
            "benchmark_engine": "vllm",
            "benchmark_dtype": spec.backend.dtype,
            "benchmark_gpu_memory_utilization": spec.backend.gpu_memory_utilization,
            "benchmark_enforce_eager": spec.backend.enforce_eager,
            "benchmark_quantization": spec.backend.quantization or "none",
            "benchmark_quantization_param_path": spec.backend.quantization_param_path or "none",
            "benchmark_cpu_offload_gb": spec.backend.cpu_offload_gb or "none",
            "benchmark_swap_space": spec.backend.swap_space or "none",
            "benchmark_max_model_len": spec.backend.max_model_len or "none",
            "benchmark_block_size": spec.backend.block_size or "none",
            # Calibration backend (DeepSpeed) parameters - using same backend config
            "calibration_engine": "deepspeed",
            "calibration_dtype": spec.backend.dtype,
            "calibration_gpu_memory_utilization": spec.backend.gpu_memory_utilization,
            "calibration_enforce_eager": spec.backend.enforce_eager,
            "calibration_quantization": spec.backend.quantization or "none",
            "calibration_quantization_param_path": spec.backend.quantization_param_path or "none",
            "calibration_enable_flop_profiling": True,  # Always enabled for calibration
            # Calibration info
            "has_calibration": calibration_dataset is not None,
            "calibration_points": len(calibration_dataset.points) if calibration_dataset else 0,
        }
        wb = WandbRunLogger(project=wandb_project, run_name=run_name, config=cfg)

    # Materialize dataset to know total for tqdm
    examples: List[Sample] = list(iter_dataset(spec.dataset))
    total_n = len(examples)

    # NEW – per-item accounting + energy
    per_item_total_tokens: List[float] = []
    per_item_generated_tokens: List[float] = []  # think + answer only (excludes prompt)
    per_item_latency_ms: List[float] = []
    per_item_correct_flags: List[int] = []

    total_energy_j = 0.0
    meter = EnergyMeter(
        device_index=energy_device,
        sample_interval_s=(energy_sample_interval_ms / 1000.0),
        enabled=measure_energy
    )

    # Iterate dataset in batches
    total, correct_self = 0, 0
    prompt_tok_sum, gen_tok_sum = 0, 0
    think_tok_sum, ans_tok_sum = 0, 0
    lat_ms_sum = 0.0

    pbar = tqdm(total=total_n, desc=run_name, unit="ex")

    # Collect all traces for table logging
    all_traces = []  # list of dicts with question, gold, answer_prompt, answer_text, judge_text
    judge_input_prompt = None
    judge_response = None
    
    # Collect per-datapoint token counts for FLOP calculation
    per_datapoint_flops = []  # list of FLOP calculations per datapoint

    for i in range(0, total_n, bs):
        batch = examples[i:i+bs]
        qs = [ex.question for ex in batch]
        gts = [ex.gold for ex in batch]

        # --- start energy window (no-op if not enabled)
        if meter.enabled: meter.start()

        if spec.reasoning.self_consistency_k and spec.reasoning.self_consistency_k > 1:
            outs = self_consistency_batch(
                engine, qs, gen, spec.think_budget, spec.reasoning.style, spec.prompts, spec.reasoning.self_consistency_k
            )
            preds = [o["chosen_answer"] for o in outs]
            # For metrics, sum tokens across all K paths (full generation cost)
            think_toks = [sum([p["think_tokens"] for p in o["paths"]]) for o in outs]
            ans_toks   = [sum([p["answer_tokens"] for p in o["paths"]]) for o in outs]
            lats       = [float(sum([p["latency_ms_think"] + p["latency_ms_answer"] for p in o["paths"]]) / len(o["paths"])) for o in outs]
        elif spec.reasoning.style == "single_pass":
            # Use single-pass method: just the question as prompt, using think_budget as max_new_tokens
            outs = single_pass_batch(engine, qs, gen, spec.think_budget)
            preds = [o["answer_text"] for o in outs]
            think_toks = [0 for _ in outs]  # No thinking tokens in single pass
            ans_toks   = [o["answer_tokens"] for o in outs]
            lats       = [o["latency_ms"] for o in outs]
        else:
            outs = two_pass_batch(engine, qs, gen, spec.think_budget, spec.reasoning.style, spec.prompts)
            preds = [o["answer_text"] for o in outs]
            think_toks = [o["think_tokens"] for o in outs]
            ans_toks   = [o["answer_tokens"] for o in outs]
            lats       = [o["latency_ms_think"] + o["latency_ms_answer"] for o in outs]

        # --- stop energy window
        if meter.enabled:
            batch_energy_j = meter.stop()
            total_energy_j += float(batch_energy_j)

        # Optional batched self-evaluation (YES/NO judge)
        judge_batch_results: Optional[List[Tuple[bool, str, str]]] = None
        if spec.reasoning.self_eval and not skip_evaluation:
            # Use OpenAI engine for evaluation if specified
            eval_engine = None
            if spec.reasoning.openai_eval:
                try:
                    from ..core.engines import create_openai_engine
                    eval_engine = create_openai_engine()
                    logger.info("Using OpenAI API for evaluation")
                except Exception as e:
                    logger.warning(f"Failed to create OpenAI evaluation engine: {e}")
                    logger.info("Falling back to main engine for evaluation")
            
            judge_batch_results = self_evaluate_batched(engine, qs, preds, gts, gen, spec.prompts, eval_engine)

        # Accumulate metrics
        for j, ex in enumerate(batch):
            total += 1

            if judge_batch_results is not None:
                judge_yes, judge_input_prompt, judge_response = judge_batch_results[j]
                correct_self += int(judge_yes)

            # Calculate tokens for this datapoint
            prompt_tokens = len(ex.question.split())  # Approximate prompt tokens
            # Include both think and answer tokens for complete generation cost
            generated_tokens = think_toks[j] + ans_toks[j]
            
            gen_tok_sum += generated_tokens
            think_tok_sum += think_toks[j]
            ans_tok_sum += ans_toks[j]
            prompt_tok_sum += prompt_tokens
            lat_ms_sum += float(lats[j])

            # NEW — per-item tracking for inequality/efficiency metrics
            total_tokens = float(prompt_tokens + generated_tokens)  # prompt + think + answer
            per_item_total_tokens.append(total_tokens)
            per_item_generated_tokens.append(float(generated_tokens))  # generated-only for clarity-first metrics
            per_item_latency_ms.append(float(lats[j]))
            # judged correctness (1/0)
            is_correct_flag = int(judge_batch_results[j][0]) if judge_batch_results is not None else 0
            per_item_correct_flags.append(is_correct_flag)

            # Calculate FLOPs for this individual datapoint
            datapoint_flops = {}
            
            # Use calibration-based FLOP estimation if available
            if calibration_dataset and calibration_dataset.extrapolation_model:
                try:
                    # Both models now have the same predict interface
                    extrapolated_flops = calibration_dataset.extrapolation_model.predict(
                        prompt_tokens, generated_tokens
                    )
                    datapoint_flops['extrapolated'] = extrapolated_flops
                except Exception as e:
                    logger.warning(f"FLOP extrapolation failed: {e}")
                    datapoint_flops['extrapolated'] = None
            else:
                datapoint_flops['extrapolated'] = None

            per_datapoint_flops.append(datapoint_flops)

            # Collect all traces for table logging
            # Determine if this trace is successful or failed using judge evaluation
            if judge_batch_results is not None:
                judge_yes, judge_input_prompt, judge_response = judge_batch_results[j]
                is_successful = judge_yes  # Use judge evaluation instead of exact match
            else:
                is_successful = False  # Default to failed if no judge results
            
            # Handle different output structures (self-consistency vs two-pass vs single-pass)
            if "think_text" in outs[j]:
                # Two-pass batch output
                think_text = outs[j]["think_text"]
                answer_text = outs[j]["answer_text"]
                think_tokens = outs[j].get("think_tokens", 0)
                answer_tokens = outs[j].get("answer_tokens", 0)
                trace = {
                    "question": ex.question,
                    "gold": ex.gold,
                    "think_text": think_text,
                    "answer_text": answer_text,
                    "think_tokens": think_tokens,
                    "answer_tokens": answer_tokens,
                    "think_formatted_input": outs[j].get("think_formatted_input", ""),
                    "answer_formatted_input": outs[j].get("answer_formatted_input", ""),
                    "judge_text": (judge_batch_results[j][2] if judge_batch_results is not None else None),
                    "is_successful": is_successful,
                }
            elif "chosen_answer" in outs[j]:
                # Self-consistency output - log all K paths
                paths = outs[j]["paths"] if outs[j]["paths"] else []
                chosen_answer = outs[j]["chosen_answer"]
                
                # Create trace with all K paths
                trace = {
                    "question": ex.question,
                    "gold": ex.gold,
                    "chosen_answer": chosen_answer,
                    "judge_text": (judge_batch_results[j][2] if judge_batch_results is not None else None),
                    "is_successful": is_successful,
                }
                
                # Add each path's generated text and token counts
                for k_idx, path in enumerate(paths):
                    trace[f"path_{k_idx+1}_think"] = path.get("think_text", "")
                    trace[f"path_{k_idx+1}_answer"] = path.get("answer_text", "")
                    trace[f"path_{k_idx+1}_think_tokens"] = path.get("think_tokens", 0)
                    trace[f"path_{k_idx+1}_answer_tokens"] = path.get("answer_tokens", 0)
            else:
                # Single-pass output - extracted solution, full answer text, and token count
                answer_text = outs[j]["answer_text"]  # extracted solution
                full_answer_text = outs[j].get("full_answer_text", answer_text)  # full text
                answer_tokens = outs[j].get("answer_tokens", 0)  # token count
                trace = {
                    "question": ex.question,
                    "gold": ex.gold,
                    "answer_text": answer_text,  # extracted solution
                    "full_answer_text": full_answer_text,  # full answer text
                    "answer_tokens": answer_tokens,  # token count
                    "formatted_input": outs[j].get("answer_formatted_input", ""),  # formatted input for chat mode
                    "judge_text": (judge_batch_results[j][2] if judge_batch_results is not None else None),
                    "is_successful": is_successful,
                }
            
            all_traces.append(trace)

        pbar.update(len(batch))

    pbar.close()

    # Compute FLOPs correctly by averaging per-datapoint calculations
    avg_extrapolated_flops = None
    if calibration_dataset and calibration_dataset.extrapolation_model:
        extrapolated_flops_list = [dp['extrapolated'] for dp in per_datapoint_flops if dp['extrapolated'] is not None]
        if extrapolated_flops_list:
            avg_extrapolated_flops = sum(extrapolated_flops_list) / len(extrapolated_flops_list)
    
    # Average tokens for reference
    avg_gen_tokens = (gen_tok_sum / max(total, 1))
    avg_think_tokens = (think_tok_sum / max(total, 1))
    avg_answer_tokens = (ans_tok_sum / max(total, 1))

    # === Centralized derived metrics (each computed exactly once) ===
    num_items = max(1, len(per_item_total_tokens))
    num_correct = max(0, int(sum(per_item_correct_flags)))
    correct_generated_tokens = [g for g, c in zip(per_item_generated_tokens, per_item_correct_flags) if c]
    acc_self_eval = (correct_self / max(total, 1)) if (spec.reasoning.self_eval and not skip_evaluation) else None

    metrics = {
        # tokens & latency aggregates
        "tokens_avg_prompt_length": (prompt_tok_sum / max(total, 1)),
        "tokens_avg_generated": avg_gen_tokens,  # renamed from tokens_avg_generated_total
        "tokens_avg_generated_correct": (float(sum(correct_generated_tokens)) / len(correct_generated_tokens)) if len(correct_generated_tokens) > 0 else None,
        "tokens_avg_thinking_phase": avg_think_tokens,
        "tokens_avg_answer_phase": avg_answer_tokens,

        # sum-based token metrics
        "tokens_generated": float(sum(per_item_generated_tokens)),
        "tokens_generated_correct": float(sum(correct_generated_tokens)),
        "performance_avg_latency_ms": (lat_ms_sum / max(total, 1)),
        "performance_generation_speed_tok_per_s": (gen_tok_sum / (lat_ms_sum / 1000.0)) if lat_ms_sum > 0 else None,

        # accuracy
        "evaluation_self_eval_accuracy": acc_self_eval,

        # efficiency-style aggregates (keep legacy names where applicable)
        "efficiency_tokens_per_correct_mean": (sum(per_item_total_tokens) / max(1, num_correct)),
        "efficiency_latency_per_correct_ms": (sum(per_item_latency_ms) / max(1, num_correct)),
        "efficiency_compute_gini_coefficient": gini(per_item_total_tokens),

        # clarity-first efficiency (generated-only)
        "efficiency_gini_generated_tokens_all_examples": gini(per_item_generated_tokens) if len(per_item_generated_tokens) > 0 else None,
        "efficiency_gini_generated_tokens_correct_only": gini(correct_generated_tokens) if len(correct_generated_tokens) > 0 else None,
    }

    # === Energy metrics (Feature 2) ===
    avg_energy_per_datapoint_j = (total_energy_j / num_items) if total_energy_j > 0 else None
    energy_per_correct_j = (total_energy_j / max(1, num_correct)) if total_energy_j > 0 else None

    # Row for logging (one row per run) - verbose parameter names with detailed comments
    row = {
        # === MODEL ARCHITECTURE & CONFIGURATION ===
        # model_huggingface_repo: HuggingFace model repository identifier (spec.hf_repo)
        "model_huggingface_repo": spec.hf_repo,
        # model_architecture: Transformer architecture type (hardcoded: decoder-only)
        "model_architecture": "decoder-only",
        # model_parameters_billions: Total model parameters in billions (spec.card.params_B)
        "model_parameters_billions": spec.card.params_B,
        # model_layers_count: Number of transformer layers (spec.card.layers)
        "model_layers_count": spec.card.layers,
        # model_hidden_dimension: Hidden state dimension (spec.card.hidden_dim)
        "model_hidden_dimension": spec.card.hidden_dim,
        # model_attention_heads: Number of attention heads (spec.card.heads)
        "model_attention_heads": spec.card.heads,
        # model_release_date: Date the model was released (spec.card.release_date)
        "model_release_date": spec.card.release_date,
        # model_precision_dtype: Data type used for model weights/computation (spec.backend.dtype)
        "model_precision_dtype": spec.backend.dtype,
        # model_quantization_scheme: Quantization method if any (currently None)
        "model_quantization_scheme": None,
        # hardware_platform: Target hardware platform identifier
        "hardware_platform": "NVIDIA RTX 6000 Pro Blackwell",
        # reasoning_style: Reasoning approach (single_pass, two_pass, self_consistency) (spec.reasoning.style)
        "reasoning_style": spec.reasoning.style,
        # prompt_template_set: Name of prompt template set used (spec.prompt_set_name)
        "prompt_template_set": spec.prompt_set_name,
        # batch_size: Number of examples processed simultaneously (spec.batch_size)
        "batch_size": spec.batch_size,

        # === TOKEN METRICS (AVERAGED PER DATAPOINT) ===
        # tokens_avg_prompt_length: Average number of prompt tokens per example (prompt_tok_sum / max(total, 1))
        "tokens_avg_prompt_length": metrics["tokens_avg_prompt_length"],
        # tokens_avg_generated: Average total generated tokens per example (avg_gen_tokens = gen_tok_sum / max(total, 1))
        "tokens_avg_generated": metrics["tokens_avg_generated"],
        # tokens_generated: Sum of tokens generated on all answers
        "tokens_generated": metrics["tokens_generated"],
        # tokens_generated_correct: Sum of tokens generated on correct answers
        "tokens_generated_correct": metrics["tokens_generated_correct"],
        # tokens_avg_generated_correct: Average generated tokens on correct examples only
        "tokens_avg_generated_correct": metrics["tokens_avg_generated_correct"],
        # tokens_avg_thinking_phase: Average tokens used in thinking/reasoning phase (avg_think_tokens = think_tok_sum / max(total, 1))
        "tokens_avg_thinking_phase": metrics["tokens_avg_thinking_phase"],
        # tokens_avg_answer_phase: Average tokens used in final answer phase (avg_answer_tokens = ans_tok_sum / max(total, 1))
        "tokens_avg_answer_phase": metrics["tokens_avg_answer_phase"],
        # tokens_budget_utilization_ratio: Ratio of average generated tokens to thinking budget (avg_gen_tokens / spec.think_budget)
        "tokens_budget_utilization_ratio": avg_gen_tokens / spec.think_budget,
        # inference_passes_count: Number of forward passes per example (hardcoded: 2 for two-pass reasoning)
        "inference_passes_count": 1,
        # self_consistency_samples_k: Number of samples for self-consistency (spec.reasoning.self_consistency_k)
        "self_consistency_samples_k": spec.reasoning.self_consistency_k,

        # === PERFORMANCE METRICS ===
        # performance_avg_latency_ms: Average latency per example in milliseconds (lat_ms_sum / max(total, 1))
        "performance_avg_latency_ms": metrics["performance_avg_latency_ms"],
        # performance_generation_speed_tok_per_s: Token generation speed in tokens/second ((gen_tok_sum / (lat_ms_sum / 1000.0)))
        "performance_generation_speed_tok_per_s": metrics["performance_generation_speed_tok_per_s"],

        # === TASK & DATASET INFORMATION ===
        # task_dataset_name: Name of the evaluation dataset (spec.dataset)
        "task_dataset_name": spec.dataset,

        # === EVALUATION METRICS ===
        # evaluation_self_eval_accuracy: Accuracy from self-evaluation judge (correct_self / max(total, 1)) if enabled
        "evaluation_self_eval_accuracy": metrics["evaluation_self_eval_accuracy"],

        # === COMPUTE EFFICIENCY METRICS ===
        # compute_flops_avg_extrapolated_tflops: Average extrapolated FLOPs in teraFLOPs (to_tflops(avg_extrapolated_flops))
        "compute_flops_avg_extrapolated_tflops": to_tflops(avg_extrapolated_flops) if avg_extrapolated_flops is not None else None,

        # === EFFICIENCY METRICS (PERFORMANCE-PER-COMPUTE) ===
        # efficiency_tokens_per_correct_mean: Mean total tokens used per correct answer (tokens_per_correct_mean)
        "efficiency_tokens_per_correct_mean": metrics["efficiency_tokens_per_correct_mean"],
        # efficiency_latency_per_correct_ms: Average latency per correct answer in ms (latency_per_correct_ms)
        "efficiency_latency_per_correct_ms": metrics["efficiency_latency_per_correct_ms"],
        # efficiency_compute_gini_coefficient: Gini coefficient measuring token allocation inequality (compute_gini_total_tokens)
        "efficiency_compute_gini_coefficient": metrics["efficiency_compute_gini_coefficient"],
        # efficiency_gini_generated_tokens_all_examples: Gini of generated token allocation across all items
        "efficiency_gini_generated_tokens_all_examples": metrics["efficiency_gini_generated_tokens_all_examples"],
        # efficiency_gini_generated_tokens_correct_only: Gini of generated tokens among correct items
        "efficiency_gini_generated_tokens_correct_only": metrics["efficiency_gini_generated_tokens_correct_only"],

        # === ENERGY EFFICIENCY METRICS ===
        # energy_avg_joules_per_datapoint: Average energy consumption per example in Joules (avg_energy_per_datapoint_j)
        "energy_avg_joules_per_datapoint": avg_energy_per_datapoint_j,
        # energy_joules_per_correct_answer: Energy consumption per correct answer in Joules (energy_per_correct_j)
        "energy_joules_per_correct_answer": energy_per_correct_j,
        # energy_total_joules_consumed: Total energy consumed during entire benchmark run (total_energy_j)
        "energy_total_joules_consumed": total_energy_j,

        # === PROMPT TEMPLATES ===
        # prompts_chain_of_thought_template: Template for thinking/reasoning phase (spec.prompts.cot_think)
        "prompts_chain_of_thought_template": spec.prompts.cot_think,
        # prompts_answer_generation_template: Template for final answer generation (spec.prompts.answer)
        "prompts_answer_generation_template": spec.prompts.answer,
        # prompts_llm_judge_template: Template for LLM-based evaluation/judging (spec.prompts.llm_judge)
        "prompts_llm_judge_template": spec.prompts.llm_judge,

        # === GENERATION PARAMETERS ===
        # generation_mode: Generation mode setting (spec.generation.generation_mode)
        "generation_mode": spec.generation.generation_mode,
        # generation_system_prompt: System prompt for chat-style generation (spec.generation.system_prompt or "none")
        "generation_system_prompt": spec.generation.system_prompt or "none",
        # generation_chat_template_kwargs: Additional chat template parameters (spec.generation.chat_template_kwargs or {})
        "generation_chat_template_kwargs": spec.generation.chat_template_kwargs or {},
        # generation_frequency_penalty: Frequency penalty parameter (spec.generation.frequency_penalty or 0.0)
        "generation_frequency_penalty": spec.generation.frequency_penalty or 0.0,

        # === EXPERIMENT METADATA ===
        # experiment_notes: User-provided notes for this experiment run
        "experiment_notes": f"{notes}",
    }

    # Add calibration info if available (appended to row dictionary)
    if calibration_dataset:
        # calibration_data_points_count: Number of calibration data points collected (len(calibration_dataset.points))
        row["calibration_data_points_count"] = len(calibration_dataset.points)
        if calibration_dataset.model_accuracy:
            # calibration_model_r_squared: R² score of calibration model fit (calibration_dataset.model_accuracy.get("r2_score"))
            row["calibration_model_r_squared"] = calibration_dataset.model_accuracy.get("r2_score")
            # calibration_model_mae_tflops: Mean Absolute Error in teraFLOPs (calibration_dataset.model_accuracy.get("mae", 0) / 1e12)
            row["calibration_model_mae_tflops"] = calibration_dataset.model_accuracy.get("mae", 0) / 1e12
            # calibration_model_rmse_tflops: Root Mean Square Error in teraFLOPs (calibration_dataset.model_accuracy.get("rmse", 0) / 1e12)
            row["calibration_model_rmse_tflops"] = calibration_dataset.model_accuracy.get("rmse", 0) / 1e12

    extrapolated_flops_str = f"{row['compute_flops_avg_extrapolated_tflops']:.2f}" if row['compute_flops_avg_extrapolated_tflops'] is not None else "N/A"
    
    flops_info = f"avg_extrapolated_tFLOPs≈{extrapolated_flops_str}"
    
    logger.log_metrics(row['evaluation_self_eval_accuracy'], avg_gen_tokens, row['performance_avg_latency_ms'], flops_info)
    logger.info(f"[RUN] {spec.model_name} | {spec.dataset} | style={spec.reasoning.style} | "
                f"B={spec.think_budget} | K={spec.reasoning.self_consistency_k} | bs={bs} | prompt={spec.prompt_set_name} | "
                f"avg_gen_tokens={avg_gen_tokens:.2f}, avg_think_tokens={avg_think_tokens:.2f}, avg_answer_tokens={avg_answer_tokens:.2f} | {flops_info}")

    if wb:
        wb.log_row(row)
        
        # Log all traces to separate tables
        wb.log_trace_tables(all_traces, spec)
        
        # Log calibration data and metrics to the same wandb run
        if calibration_dataset:
            logger.info("Logging calibration data to wandb...")
            try:
                # Pass estimation data if available
                estimation_data = getattr(calibration_dataset, 'estimation_data', None)
                wb.log_calibration_data(calibration_dataset, spec, estimation_data)
                wb.log_calibration_metrics(calibration_dataset)
                logger.info("✓ Calibration data logged to wandb successfully")
            except Exception as e:
                logger.warning(f"Failed to log calibration data to wandb: {e}")
        
        wb.finish()

    # Tear down engine and free memory between runs
    engine.close()

    # NEW – gracefully close NVML if we initialized it
    try:
        meter.close()
    except Exception:
        pass

def main():
    # Setup logging first
    logger = setup_logging(name="bench_with_calibration")
    
    ap = argparse.ArgumentParser(description="Benchmark with FLOP calibration (next-token calibration by default)")
    ap.add_argument("--config", required=True, help="Path to YAML bench config")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    
    # Calibration options
    ap.add_argument("--calibration_prefill_ranges", nargs="+", type=int,
                   default=np.geomspace(1, 2024, 64, dtype=int).tolist(),
                   help="Prefill token ranges for next-token calibration")
    ap.add_argument("--estimation_points", type=int, default=64,
                   help="Number of estimation points for extrapolation evaluation (default: 64)")
    ap.add_argument("--skip-calibration", action="store_true",
                   help="Skip calibration and run benchmark directly (will use fallback FLOP estimation if no calibration data exists)")

    # NEW – optional, off by default to avoid any sampling overhead
    ap.add_argument("--measure_energy", action="store_true", default=True,
                    help="Measure GPU energy via NVML around generation (adds slight overhead)")
    ap.add_argument("--energy_device", type=int, default=0,
                    help="GPU index for NVML energy metering")
    ap.add_argument("--energy_sample_interval_ms", type=float, default=20.0,
                    help="Sampling interval (ms) when NVML energy counter is unavailable")
    ap.add_argument("--skip-evaluation", action="store_true",
                    help="Skip evaluation entirely (no correctness checking)")

    args = ap.parse_args()
    
    logger.info(f"Starting benchmark with next-token calibration")
    logger.info(f"Config: {args.config}")
    logger.info(f"W&B Project: {args.wandb_project}")
    logger.info(f"Notes: {args.notes}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Skip calibration: {args.skip_calibration}")
    logger.info(f"Skip evaluation: {args.skip_evaluation}")
    logger.info(f"Calibration prefill ranges: {args.calibration_prefill_ranges}")
    logger.info("Using next-token calibration (fast, efficient method)")

    cfg = load_bench_config(args.config)

    for spec in expand_runs(cfg):
        try:
            calibration_dataset = None
            
            if args.skip_calibration:
                logger.info(f"Skipping calibration for {spec.model_name} - running benchmark directly")
                logger.info("Will use fallback FLOP estimation if no existing calibration data found")
            else:
                # Check for existing calibration
                calibration_base_dir = "calibration_models"
                model_dir = spec.hf_repo.replace("/", "_")
                calibration_dir = os.path.join(calibration_base_dir, model_dir)
                
                # Use next-token calibration file
                calibration_file = os.path.join(calibration_dir, "next_token_calibration.json")
                
                if os.path.exists(calibration_file):
                    logger.info(f"Loading existing calibration from: {calibration_file}")
                    try:
                        calibration_dataset = load_calibration_dataset(calibration_file)
                        logger.info(f"✓ Loaded calibration with {len(calibration_dataset.points)} points")
                        if calibration_dataset.model_accuracy:
                            logger.info(f"  Model R²: {calibration_dataset.model_accuracy.get('r2_score', 'N/A'):.4f}")
                    except Exception as e:
                        logger.warning(f"Failed to load calibration: {e}")
                        logger.warning("Will attempt to run new calibration...")
                        calibration_dataset = None
                        
                        # Log the loading error to error log
                        with open("error.log", "a", encoding="utf-8") as f:
                            f.write(f"[CALIBRATION_LOAD_ERROR] model={spec.model_name}: {e}\n")
                            f.write(traceback.format_exc() + "\n")
                
                if calibration_dataset is None:
                    # Run new calibration in subprocess
                    logger.info(f"Running new calibration in subprocess for {spec.model_name}")
                    
                    calibration_dataset = run_calibration_subprocess(
                        spec=spec,
                        calibration_file=calibration_file,
                        prefill_ranges=args.calibration_prefill_ranges,
                        config_file=args.config,
                        estimation_points=args.estimation_points
                    )
                    
                    if calibration_dataset is None:
                        logger.warning("Calibration subprocess failed - continuing benchmark without calibration data...")
                        
                        # Log the calibration failure to error log
                        with open("error.log", "a", encoding="utf-8") as f:
                            f.write(f"[CALIBRATION_SUBPROCESS_ERROR] model={spec.model_name}: Subprocess failed\n")
                            f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("=" * 50 + "\n")
            
            # Log calibration status before running benchmark
            if calibration_dataset is not None:
                logger.info(f"🚀 Starting benchmark with calibration data ({len(calibration_dataset.points)} points)")
            else:
                logger.info("🚀 Starting benchmark without calibration data (FLOP extrapolation will be disabled)")
            
            # Run the benchmark with calibration (or without if skipped)
            run_one_with_calibration(
                spec,
                calibration_dataset=calibration_dataset,
                batch_size=args.batch_size,
                wandb_project=args.wandb_project,
                notes=args.notes,
                measure_energy=args.measure_energy,
                energy_device=args.energy_device,
                energy_sample_interval_ms=args.energy_sample_interval_ms,
                skip_evaluation=args.skip_evaluation
            )
            
        except Exception as e:
            logger = get_logger()
            err_msg = f"[ERROR] model={spec.model_name}, dataset={spec.dataset}: {e}"
            logger.log_error(err_msg, e)
            
            # Write to error log file
            with open("error.log", "a", encoding="utf-8") as f:
                f.write(err_msg + "\n")
                f.write(traceback.format_exc() + "\n")

if __name__ == "__main__":
    main()
