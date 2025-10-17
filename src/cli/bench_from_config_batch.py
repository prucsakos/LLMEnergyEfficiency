from __future__ import annotations
import traceback
import argparse, math, statistics, time
from typing import Iterable, List, Optional, Tuple, Callable
import itertools
from tqdm.auto import tqdm

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
from ..metrics.flop_estimation import flops_dense, flops_attention_kv, to_tflops
from ..logs.wandb_logger import WandbRunLogger

# Optional direct import from vLLM for batched generation
from vllm import SamplingParams


def run_one(spec: RunSpec, batch_size: Optional[int] = None, wandb_project: str | None = None, notes: str = "") -> None:
    # Use configured batch size unless overridden
    bs = int(batch_size or getattr(spec, "batch_size", 1) or 1)

    # Build engine (one per run), then tear down afterwards
    logger = get_logger()
    logger.info(f"Specified engine: {spec.engine}")
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
            # Backend parameters
            "engine": spec.engine,
            "gpu_memory_utilization": spec.backend.gpu_memory_utilization,
            "enforce_eager": spec.backend.enforce_eager,
            "quantization": spec.backend.quantization or "none",
            "quantization_param_path": spec.backend.quantization_param_path or "none",
            "cpu_offload_gb": spec.backend.cpu_offload_gb or "none",
            "swap_space": spec.backend.swap_space or "none",
            "max_model_len": spec.backend.max_model_len or "none",
            "block_size": spec.backend.block_size or "none",
        }
        wb = WandbRunLogger(project=wandb_project, run_name=run_name, config=cfg)

    # Materialize dataset to know total for tqdm
    examples: List[Sample] = list(iter_dataset(spec.dataset))
    total_n = len(examples)

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
    
    # Collect per-datapoint token counts for correct FLOP calculation
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
            lats       = [float(statistics.mean([p["latency_ms_think"] + p["latency_ms_answer"] for p in o["paths"]])) for o in outs]
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

            if judge_batch_results is not None:
                judge_yes, judge_input_prompt, judge_response = judge_batch_results[j]
                correct_self += int(judge_yes)

            # Calculate tokens for this datapoint
            prompt_tokens = len(ex.question.split())  # Approximate prompt tokens
            generated_tokens = think_toks[j] + ans_toks[j]
            
            gen_tok_sum += generated_tokens
            think_tok_sum += think_toks[j]
            ans_tok_sum += ans_toks[j]
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
            
            # TODO: This logic shall be handles by the controller functions I think
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
    avg_think_tokens = (think_tok_sum / max(total, 1))
    avg_answer_tokens = (ans_tok_sum / max(total, 1))

    # Row for logging (one row per run)
    row = {
        # Model/setup
        "model": spec.hf_repo,
        "arch": "decoder-only",
        "params_B": spec.card.params_B,
        "layers": spec.card.layers,
        "d_model": spec.card.hidden_dim,
        "heads": spec.card.heads,
        "release_date": spec.card.release_date,
        "precision": spec.backend.dtype,
        "quant": None,
        "hardware": "NVIDIA RTX 6000 Pro Blackwell",
        "reasoning_style": spec.reasoning.style,
        "prompt_set": spec.prompt_set_name,

        # Tokens (averaged per datapoint)
        "avg_prompt_tokens": prompt_tok_sum / max(total, 1),
        "avg_gen_tokens": avg_gen_tokens,
        "avg_think_tokens": avg_think_tokens,
        "avg_answer_tokens": avg_answer_tokens,
        "budget_utilization_ratio": avg_gen_tokens / spec.think_budget,
        "passes": 2,
        "self_consistency_k": spec.reasoning.self_consistency_k,

        # Measured
        "latency_ms": lat_ms_sum / max(total, 1),
        "speed_tok_per_s": (gen_tok_sum / (lat_ms_sum / 1000.0)) if lat_ms_sum > 0 else None,

        # Task metric
        "dataset": spec.dataset,

        # Extras (self-eval + FLOPs)
        "self_eval_acc": (correct_self / max(total, 1)) if spec.reasoning.self_eval else None,

        # Corrected per-datapoint averaged FLOPs
        "avg_flops_dense_tflops": to_tflops(avg_dense_flops),
        "avg_flops_attention_kv_tflops": to_tflops(avg_attn_flops) if avg_attn_flops is not None else None,
        "avg_flops_deepspeed_tflops": to_tflops(avg_deepspeed_flops) if avg_deepspeed_flops is not None else None,

        "prompt_cot_think": spec.prompts.cot_think,
        "prompt_answer": spec.prompts.answer,
        "prompt_llm_judge": spec.prompts.llm_judge,

        # Generation parameters
        "generation_mode": spec.generation.generation_mode,
        "system_prompt": spec.generation.system_prompt or "none",
        "chat_template_kwargs": spec.generation.chat_template_kwargs or {},
        "frequency_penalty": spec.generation.frequency_penalty or 0.0,

        # Notes
        "notes": f"{notes}",
    }

    attn_flops_str = f"{row['avg_flops_attention_kv_tflops']:.2f}" if row['avg_flops_attention_kv_tflops'] is not None else "N/A"
    deepspeed_flops_str = f"{row['avg_flops_deepspeed_tflops']:.2f}" if row['avg_flops_deepspeed_tflops'] is not None else "N/A"
    
    flops_info = f"avg_dense_tFLOPs≈{row['avg_flops_dense_tflops']:.2f} | avg_attn_tFLOPs≈{attn_flops_str}"
    if spec.engine == "deepspeed":
        flops_info += f" | avg_deepspeed_tFLOPs≈{deepspeed_flops_str}"
    
    logger.log_metrics(row['self_eval_acc'], avg_gen_tokens, row['latency_ms'], flops_info)
    logger.info(f"[RUN] {spec.model_name} | {spec.dataset} | style={spec.reasoning.style} | "
                f"B={spec.think_budget} | K={spec.reasoning.self_consistency_k} | bs={bs} | prompt={spec.prompt_set_name} | "
                f"avg_gen_tokens={avg_gen_tokens:.2f}, avg_think_tokens={avg_think_tokens:.2f}, avg_answer_tokens={avg_answer_tokens:.2f} | {flops_info}")

    if wb:
        wb.log_row(row)
        # Log all traces to separate tables
        wb.log_trace_tables(all_traces, spec)
        wb.finish()

    # Tear down engine and free memory between runs
    engine.close()

def main():
    # Setup logging first
    logger = setup_logging(name="bench_from_config_batch")
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML bench config")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    args = ap.parse_args()
    
    logger.info(f"Starting batch benchmark from config")
    logger.info(f"Config: {args.config}")
    logger.info(f"W&B Project: {args.wandb_project}")
    logger.info(f"Notes: {args.notes}")
    logger.info(f"Batch size: {args.batch_size}")

    cfg = load_bench_config(args.config)

    for spec in expand_runs(cfg):
        try:
            run_one(spec, batch_size=args.batch_size, wandb_project=args.wandb_project, notes=args.notes)
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
