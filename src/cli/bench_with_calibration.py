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
from ..reasoning.controller import self_evaluate_batched, self_consistency_batch, two_pass_batch  # use batched judge
from ..data.adapters import load_gsm8k, load_mmlu, load_csqa, exact_match, Sample, iter_dataset
from ..metrics.flop_estimation import to_tflops
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

def _build_sample_trace_logging(sample_traces: List[dict], sample_idx: int) -> dict:
    """Build logging fields for a sample trace, handling both two-pass and self-consistency formats."""
    if sample_idx >= len(sample_traces):
        return {
            f"sample{sample_idx+1}_question": None,
            f"sample{sample_idx+1}_golden_answer": None,
            f"sample{sample_idx+1}_judge_answer": None,
        }
    
    trace = sample_traces[sample_idx]
    sample_prefix = f"sample{sample_idx+1}"
    
    # Base fields
    result = {
        f"{sample_prefix}_question": trace.get("question"),
        f"{sample_prefix}_golden_answer": trace.get("gold"),
        f"{sample_prefix}_judge_answer": trace.get("judge_text"),
    }
    
    # Check if this is self-consistency (has chosen_answer) or two-pass (has think_text/answer_text)
    if "chosen_answer" in trace:
        # Self-consistency format - log chosen answer and all K paths
        result[f"{sample_prefix}_chosen_answer"] = trace.get("chosen_answer")
        
        # Log up to 5 paths (assuming max K=5)
        for k in range(1, 6):
            think_key = f"path_{k}_think"
            answer_key = f"path_{k}_answer"
            if think_key in trace:
                result[f"{sample_prefix}_path{k}_think"] = trace.get(think_key, "")
                result[f"{sample_prefix}_path{k}_answer"] = trace.get(answer_key, "")
            else:
                result[f"{sample_prefix}_path{k}_think"] = None
                result[f"{sample_prefix}_path{k}_answer"] = None
    else:
        # Two-pass format - use original fields
        result[f"{sample_prefix}_first_pass"] = trace.get("think_text")
        result[f"{sample_prefix}_second_pass"] = trace.get("answer_text")
    
    return result


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
                           notes: str = "") -> None:
    """
    Run a single benchmark with optional FLOP calibration.
    
    Args:
        spec: Benchmark specification
        calibration_dataset: Optional calibration data for FLOP extrapolation
        batch_size: Override batch size from config
        wandb_project: W&B project name
        notes: Additional notes
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
    )

    gen = GenerationParams(
        max_new_tokens=spec.generation.max_new_tokens,
        temperature=spec.generation.temperature,
        top_p=spec.generation.top_p,
        top_k=spec.generation.top_k,
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
            "batch_size": bs,
            "prompt_set": spec.prompt_set_name,
            "config_name": spec.config_name,
            # Generation parameters
            "temperature": spec.generation.temperature,
            "top_p": spec.generation.top_p,
            "top_k": spec.generation.top_k,
            "do_sample": spec.generation.do_sample,
            "max_new_tokens": spec.generation.max_new_tokens,
            "stop": spec.generation.stop,
            "seed": spec.generation.seed,
            "use_kv_cache": spec.generation.use_kv_cache,
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

    # Iterate dataset in batches
    total, correct_measured, correct_self = 0, 0, 0
    prompt_tok_sum, gen_tok_sum = 0, 0
    lat_ms_sum = 0.0

    pbar = tqdm(total=total_n, desc=run_name, unit="ex")

    # Collect up to 3 sample traces for logging/inspection
    sample_traces = []  # list of dicts with question, gold, answer_prompt, answer_text, judge_text
    judge_input_prompt = None
    judge_response = None
    
    # Collect per-datapoint token counts for FLOP calculation
    per_datapoint_flops = []  # list of FLOP calculations per datapoint

    for i in range(0, total_n, bs):
        batch = examples[i:i+bs]
        qs = [ex.question for ex in batch]
        gts = [ex.gold for ex in batch]

        if spec.reasoning.self_consistency_k and spec.reasoning.self_consistency_k > 1:
            outs = self_consistency_batch(
                engine, qs, gen, spec.think_budget, spec.reasoning.style, spec.prompts, spec.reasoning.self_consistency_k
            )
            preds = [o["chosen_answer"] for o in outs]
            # For metrics, sum tokens across all K paths (full generation cost)
            think_toks = [sum([p["think_tokens"] for p in o["paths"]]) for o in outs]
            ans_toks   = [sum([p["answer_tokens"] for p in o["paths"]]) for o in outs]
            lats       = [float(sum([p["latency_ms_think"] + p["latency_ms_answer"] for p in o["paths"]]) / len(o["paths"])) for o in outs]
        else:
            outs = two_pass_batch(engine, qs, gen, spec.think_budget, spec.reasoning.style, spec.prompts)
            preds = [o["answer_text"] for o in outs]
            think_toks = [o["think_tokens"] for o in outs]
            ans_toks   = [o["answer_tokens"] for o in outs]
            lats       = [o["latency_ms_think"] + o["latency_ms_answer"] for o in outs]

        # Optional batched self-evaluation (YES/NO judge)
        judge_batch_results: Optional[List[Tuple[bool, str, str]]] = None
        if spec.reasoning.self_eval:
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
            ok = exact_match(preds[j], gts[j])
            correct_measured += int(ok)

            if judge_batch_results is not None:
                judge_yes, judge_input_prompt, judge_response = judge_batch_results[j]
                correct_self += int(judge_yes)

            # Calculate tokens for this datapoint
            prompt_tokens = len(ex.question.split())  # Approximate prompt tokens
            generated_tokens = think_toks[j] + ans_toks[j]
            
            gen_tok_sum += generated_tokens
            prompt_tok_sum += prompt_tokens
            lat_ms_sum += float(lats[j])

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

            # Save up to 3 sample traces
            if len(sample_traces) < 3:
                # Handle different output structures (self-consistency vs two-pass)
                if "think_text" in outs[j]:
                    # Two-pass batch output
                    think_text = outs[j]["think_text"]
                    answer_text = outs[j]["answer_text"]
                    trace = {
                        "question": ex.question,
                        "gold": ex.gold,
                        "think_text": think_text,
                        "answer_text": answer_text,
                        "judge_text": (judge_batch_results[j][2] if judge_batch_results is not None else None),
                    }
                else:
                    # Self-consistency output - log all K paths
                    paths = outs[j]["paths"] if outs[j]["paths"] else []
                    chosen_answer = outs[j]["chosen_answer"]
                    
                    # Create trace with all K paths
                    trace = {
                        "question": ex.question,
                        "gold": ex.gold,
                        "chosen_answer": chosen_answer,
                        "judge_text": (judge_batch_results[j][2] if judge_batch_results is not None else None),
                    }
                    
                    # Add each path's generated text
                    for k_idx, path in enumerate(paths):
                        trace[f"path_{k_idx+1}_think"] = path.get("think_text", "")
                        trace[f"path_{k_idx+1}_answer"] = path.get("answer_text", "")
                
                sample_traces.append(trace)

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

    # Row for logging (one row per run)
    row = {
        # Model/setup
        "model": spec.hf_repo,
        "arch": "decoder-only",
        "params_B": spec.card.params_B,
        "layers": spec.card.layers,
        "d_model": spec.card.hidden_dim,
        "heads": spec.card.heads,
        "precision": spec.backend.dtype,
        "quant": None,
        "hardware": "NVIDIA RTX 6000 Pro Blackwell",
        "batch_size": bs,
        "use_kv_cache": spec.generation.use_kv_cache,
        "reasoning_style": spec.reasoning.style,
        "prompt_set": spec.prompt_set_name,

        # Tokens (averaged per datapoint)
        "avg_prompt_tokens": prompt_tok_sum / max(total, 1),
        "avg_gen_tokens": avg_gen_tokens,
        "passes": 2,
        "beam_width": 1,
        "self_consistency_k": spec.reasoning.self_consistency_k,

        # Measured
        "latency_ms": lat_ms_sum / max(total, 1),
        "speed_tok_per_s": (gen_tok_sum / (lat_ms_sum / 1000.0)) if lat_ms_sum > 0 else None,
        "energy_j": None,

        # Task metric
        "dataset": spec.dataset,
        "metric_name": "exact_match",
        "accuracy": correct_measured / max(total, 1),

        # Extras (self-eval + FLOPs)
        "self_eval_acc": (correct_self / max(total, 1)) if spec.reasoning.self_eval else None,

        # FLOP estimates (averaged per datapoint)
        "avg_flops_extrapolated_tflops": to_tflops(avg_extrapolated_flops) if avg_extrapolated_flops is not None else None,

        # Sample traces (up to 3) - handles both two-pass and self-consistency
        **_build_sample_trace_logging(sample_traces, 0),
        **_build_sample_trace_logging(sample_traces, 1),
        **_build_sample_trace_logging(sample_traces, 2),
        "prompt_cot_think": spec.prompts.cot_think,
        "prompt_answer": spec.prompts.answer,
        "prompt_direct": spec.prompts.direct,
        "prompt_plan_think": spec.prompts.plan_think,
        "prompt_self_eval": spec.prompts.self_eval,

        # Notes
        "notes": f"{notes}",
    }

    # Add calibration info if available
    if calibration_dataset:
        row["calibration_points"] = len(calibration_dataset.points)
        if calibration_dataset.model_accuracy:
            row["calibration_r2"] = calibration_dataset.model_accuracy.get("r2_score")
            row["calibration_mae_tflops"] = calibration_dataset.model_accuracy.get("mae", 0) / 1e12
            row["calibration_rmse_tflops"] = calibration_dataset.model_accuracy.get("rmse", 0) / 1e12

    extrapolated_flops_str = f"{row['avg_flops_extrapolated_tflops']:.2f}" if row['avg_flops_extrapolated_tflops'] is not None else "N/A"
    
    flops_info = f"avg_extrapolated_tFLOPs≈{extrapolated_flops_str}"
    
    logger.log_metrics(row['self_eval_acc'], avg_gen_tokens, row['latency_ms'], flops_info)
    logger.info(f"[RUN] {spec.model_name} | {spec.dataset} | style={spec.reasoning.style} | "
                f"B={spec.think_budget} | K={spec.reasoning.self_consistency_k} | bs={bs} | prompt={spec.prompt_set_name} | "
                f"acc={row['accuracy']:.3f} | avg_gen_tokens={avg_gen_tokens:.2f} | {flops_info}")

    if wb:
        wb.log_row(row)
        
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
    
    args = ap.parse_args()
    
    logger.info(f"Starting benchmark with next-token calibration")
    logger.info(f"Config: {args.config}")
    logger.info(f"W&B Project: {args.wandb_project}")
    logger.info(f"Notes: {args.notes}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Skip calibration: {args.skip_calibration}")
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
                notes=args.notes
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
