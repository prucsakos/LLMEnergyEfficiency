from __future__ import annotations
import traceback
import argparse, math, statistics, time
from typing import Iterable, List, Optional, Tuple, Callable
import itertools
from tqdm.auto import tqdm

# Load environment variables from .env file
from ..utils import load_env_variables
load_env_variables()

from ..config.bench_config import load_bench_config, expand_runs, RunSpec, Prompts
from ..core.interfaces import GenerationParams
from ..core.engines import create_engine
from ..reasoning.aggregators import majority_vote
from ..reasoning.controller import self_evaluate_batched, self_consistency_batch, two_pass_batch  # use batched judge
from ..data.adapters import load_gsm8k, load_mmlu, load_csqa, exact_match, Sample, iter_dataset
from ..metrics.flop_estimation import flops_dense, flops_attention_kv, to_tflops
from ..logs.wandb_logger import WandbRunLogger

# Optional direct import from vLLM for batched generation
from vllm import SamplingParams

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

def run_one(spec: RunSpec, batch_size: Optional[int] = None, wandb_project: str | None = None, notes: str = "", verbose: bool = False) -> None:
    # Use configured batch size unless overridden
    bs = int(batch_size or getattr(spec, "batch_size", 1) or 1)

    # Build engine (one per run), then tear down afterwards
    print("Specified engine: ", spec.engine)
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
            "max_new_tokens": spec.generation.max_new_tokens,
            "stop": spec.generation.stop,
            "seed": spec.generation.seed,
            "use_kv_cache": spec.generation.use_kv_cache,
        }
        wb = WandbRunLogger(project=wandb_project, run_name=run_name, config=cfg)

    # Materialize dataset to know total for tqdm
    examples: List[Sample] = list(iter_dataset(spec.dataset))
    total_n = len(examples)

    # Process single batch only
    total, correct_measured, correct_self = 0, 0, 0
    prompt_tok_sum, gen_tok_sum = 0, 0
    lat_ms_sum = 0.0

    pbar = tqdm(total=bs, desc=run_name, unit="ex")

    # Collect up to 3 sample traces for logging/inspection
    sample_traces = []  # list of dicts with question, gold, answer_prompt, answer_text, judge_text
    judge_input_prompt = None
    judge_response = None
    
    # Collect per-datapoint token counts for correct FLOP calculation
    per_datapoint_flops = []  # list of FLOP calculations per datapoint

    # Process only the first batch (single batch mode)
    batch = examples[0:bs]
    qs = [ex.question for ex in batch]
    gts = [ex.gold for ex in batch]

    print(f"Processing single batch of {len(batch)} examples...")

    if spec.reasoning.self_consistency_k and spec.reasoning.self_consistency_k > 1:
        outs = self_consistency_batch(
            engine, qs, gen, spec.think_budget, spec.reasoning.style, spec.prompts, spec.reasoning.self_consistency_k
        )
        preds = [o["chosen_answer"] for o in outs]
        # For metrics, sum tokens across all K paths (full generation cost)
        think_toks = [sum([p["think_tokens"] for p in o["paths"]]) for o in outs]
        ans_toks   = [sum([p["answer_tokens"] for p in o["paths"]]) for o in outs]
        lats       = [float(statistics.mean([p["latency_ms_think"] + p["latency_ms_answer"] for p in o["paths"]])) for o in outs]
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
                print("Using OpenAI API for evaluation")
            except Exception as e:
                print(f"Failed to create OpenAI evaluation engine: {e}")
                print("Falling back to main engine for evaluation")
        
        judge_batch_results = self_evaluate_batched(engine, qs, preds, gts, gen, spec.prompts, eval_engine)

    # Accumulate metrics for the single batch
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
        num_params = spec.card.params_B * 1e9
        
        # Always calculate theoretical dense and attention FLOP estimates
        datapoint_dense_flops = flops_dense(num_params=num_params, num_tokens=generated_tokens)
        
        datapoint_attn_flops = None
        if spec.card.layers and spec.card.hidden_dim:
            datapoint_attn_flops = flops_attention_kv(
                num_layers=spec.card.layers,
                hidden_dim=spec.card.hidden_dim,
                num_prompt_tokens=prompt_tokens,
                num_generated_tokens=generated_tokens,
                num_params=num_params,
                include_dense_anchor=True,
            )
        
        # Only get DeepSpeed FLOP measurements if we're using DeepSpeed engine
        deepspeed_flops = None
        if spec.engine == "deepspeed":
            total_deepspeed_flops = 0
            
            # Handle self-consistency (multiple paths) vs two-pass (single path)
            if "paths" in outs[j]:
                # Self-consistency: sum FLOPs across all K paths
                for path in outs[j]["paths"]:
                    if "raw" in path and path["raw"]:
                        think_raw = path["raw"].get("think_raw")
                        answer_raw = path["raw"].get("answer_raw")
                        
                        if think_raw and think_raw.get("flops"):
                            total_deepspeed_flops += think_raw["flops"].get("total_flops", 0)
                        
                        if answer_raw and answer_raw.get("flops"):
                            total_deepspeed_flops += answer_raw["flops"].get("total_flops", 0)
            elif "raw" in outs[j] and outs[j]["raw"]:
                # Two-pass: single path
                think_raw = outs[j]["raw"].get("think_raw")
                answer_raw = outs[j]["raw"].get("answer_raw")
                
                if think_raw and think_raw.get("flops"):
                    total_deepspeed_flops += think_raw["flops"].get("total_flops", 0)
                
                if answer_raw and answer_raw.get("flops"):
                    total_deepspeed_flops += answer_raw["flops"].get("total_flops", 0)
            
            if total_deepspeed_flops > 0:
                deepspeed_flops = {"total_flops": total_deepspeed_flops}
        
        per_datapoint_flops.append({
            'dense': datapoint_dense_flops,
            'attention': datapoint_attn_flops,
            'deepspeed': deepspeed_flops
        })

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
    # This is necessary because attention FLOPs are non-linear with token count
    avg_dense_flops = sum(dp['dense'] for dp in per_datapoint_flops) / max(len(per_datapoint_flops), 1)
    
    avg_attn_flops = None
    if spec.card.layers and spec.card.hidden_dim:
        attn_flops_list = [dp['attention'] for dp in per_datapoint_flops if dp['attention'] is not None]
        if attn_flops_list:
            avg_attn_flops = sum(attn_flops_list) / len(attn_flops_list)
    
    # Handle DeepSpeed FLOP measurements
    avg_deepspeed_flops = None
    if spec.engine == "deepspeed":
        deepspeed_flops_list = [dp['deepspeed'] for dp in per_datapoint_flops if dp['deepspeed'] is not None]
        if deepspeed_flops_list:
            # Calculate average FLOPs per datapoint from DeepSpeed measurements
            total_deepspeed_flops = sum(flops.get('total_flops', 0) for flops in deepspeed_flops_list)
            avg_deepspeed_flops = total_deepspeed_flops / len(deepspeed_flops_list)
    
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

        # Corrected per-datapoint averaged FLOPs
        "avg_flops_dense_tflops": to_tflops(avg_dense_flops),
        "avg_flops_attention_kv_tflops": to_tflops(avg_attn_flops) if avg_attn_flops is not None else None,
        "avg_flops_deepspeed_tflops": to_tflops(avg_deepspeed_flops) if avg_deepspeed_flops is not None else None,

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

    attn_flops_str = f"{row['avg_flops_attention_kv_tflops']:.2f}" if row['avg_flops_attention_kv_tflops'] is not None else "N/A"
    deepspeed_flops_str = f"{row['avg_flops_deepspeed_tflops']:.2f}" if row['avg_flops_deepspeed_tflops'] is not None else "N/A"
    
    flops_info = f"avg_dense_tFLOPs≈{row['avg_flops_dense_tflops']:.2f} | avg_attn_tFLOPs≈{attn_flops_str}"
    if spec.engine == "deepspeed":
        flops_info += f" | avg_deepspeed_tFLOPs≈{deepspeed_flops_str}"
    
    print(f"[RUN] {spec.model_name} | {spec.dataset} | style={spec.reasoning.style} | "
          f"B={spec.think_budget} | K={spec.reasoning.self_consistency_k} | bs={bs} | prompt={spec.prompt_set_name} | "
          f"acc={row['accuracy']:.3f} | avg_gen_tokens={avg_gen_tokens:.2f} | {flops_info}")

    if wb:
        wb.log_row(row)
        wb.finish()

    # Tear down engine and free memory between runs
    engine.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML bench config")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    args = ap.parse_args()

    cfg = load_bench_config(args.config)

    for spec in expand_runs(cfg):
        try:
            run_one(spec, batch_size=args.batch_size, wandb_project=args.wandb_project, notes=args.notes)
        except Exception as e:
            err_msg = f"[ERROR] model={spec.model_name}, dataset={spec.dataset}: {e}\n"
            traceback_str = traceback.format_exc()
            
            # Write to error log file
            with open("error.log", "a", encoding="utf-8") as f:
                f.write(err_msg)
                f.write(traceback_str + "\n")
            
            # Print error message and full traceback to stdout
            print(err_msg.strip())
            print(traceback_str)

if __name__ == "__main__":
    main()
