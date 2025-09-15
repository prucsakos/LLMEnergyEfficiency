from __future__ import annotations
import argparse, math, statistics, time
from typing import Iterable, List, Optional, Tuple, Callable
import itertools
from tqdm.auto import tqdm

from ..config.bench_config import load_bench_config, expand_runs, RunSpec, Prompts
from ..core.interfaces import GenerationParams
from ..core.engines.vllm_local import VLLMLocalEngine
from ..reasoning.aggregators import majority_vote
from ..reasoning.controller import self_evaluate, self_consistency_batch, two_pass_batch  # keep using the existing single-example judge
from ..data.adapters import load_gsm8k, load_mmlu, load_csqa, exact_match, Sample, iter_dataset
from ..metrics.flop_estimation import flops_dense, flops_attention_kv, to_tflops
from ..logs.wandb_logger import WandbRunLogger

# Optional direct import from vLLM for batched generation
from vllm import SamplingParams

# ------------------------------
# Batched inference helpers (vLLM)
# ------------------------------
def _combine_stops(base: Optional[List[str]], extra: Optional[str]) -> Optional[List[str]]:
    stops: List[str] = list(base) if base else []
    if extra and extra not in stops:
        stops.append(extra)
    return stops or None

def _build_think_prompts(questions: List[str], style: str, prompts: Prompts) -> Tuple[List[str], str, str]:
    """Return (think_prompts, open_tag, close_tag) for style."""
    if style == "plan":
        open_tag, close = "<plan>", "</plan>"
        think_prompts = [prompts.plan_think.format(question=q) for q in questions]
    elif style == "cot":
        open_tag, close = "<scratchpad>", "</scratchpad>"
        think_prompts = [prompts.cot_think.format(question=q) for q in questions]
    else:
        # style "none" -> no think prompts
        open_tag, close, think_prompts = "", "", []
    return think_prompts, open_tag, close

def _vllm_generate_batch(engine: VLLMLocalEngine,
                         prompts: List[str],
                         params: GenerationParams,
                         extra_stop: Optional[str] = None) -> Tuple[List[str], List[int], float]:
    """
    Run vLLM in batch for a list of prompts.
    Returns (texts, completion_token_counts, wall_ms).
    """
    if not prompts:
        return [], [], 0.0
    sp = SamplingParams(
        temperature=params.temperature,
        top_p=params.top_p,
        max_tokens=params.max_new_tokens,
        stop=_combine_stops(params.stop, extra_stop),
    )
    t0 = time.time()
    outs = engine.llm.generate(prompts, sp, use_tqdm=False)
    wall_ms = (time.time() - t0) * 1000.0
    texts: List[str] = []
    comp_tokens: List[int] = []
    for out in outs:
        text = out.outputs[0].text if out.outputs else ""
        toks = len(out.outputs[0].token_ids or []) if out.outputs else 0
        texts.append(text)
        comp_tokens.append(toks)
    return texts, comp_tokens, wall_ms

def run_one(spec: RunSpec, batch_size: Optional[int] = None, wandb_project: str | None = None, notes: str = "") -> None:
    # Use configured batch size unless overridden
    bs = int(batch_size or getattr(spec, "batch_size", 1) or 1)

    # Build engine (one per run), then tear down afterwards
    engine = VLLMLocalEngine(
        model_id=spec.hf_repo,
        dtype=spec.backend.dtype,
        gpu_memory_utilization=spec.backend.gpu_memory_utilization,
        enforce_eager=spec.backend.enforce_eager,
    )

    gen = GenerationParams(
        max_new_tokens=spec.generation.max_new_tokens,
        temperature=spec.generation.temperature,
        top_p=spec.generation.top_p,
        stop=spec.generation.stop,
        seed=spec.generation.seed,
        use_kv_cache=spec.generation.use_kv_cache,
        dtype=spec.backend.dtype,
    )

    # W&B: one run per (model, dataset, budget, style, k, batch)
    wb = None
    run_name = f"{spec.model_name}|{spec.dataset}|style={spec.reasoning.style}|B={spec.think_budget}|K={spec.reasoning.self_consistency_k}|bs={bs}"
    if wandb_project:
        cfg = {
            "model": spec.hf_repo,
            "dataset": spec.dataset,
            "style": spec.reasoning.style,
            "think_budget": spec.think_budget,
            "K": spec.reasoning.self_consistency_k,
            "dtype": spec.backend.dtype,
            "batch_size": bs,
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

    for i in range(0, total_n, bs):
        batch = examples[i:i+bs]
        qs = [ex.question for ex in batch]
        gts = [ex.gold for ex in batch]

        if spec.reasoning.self_consistency_k and spec.reasoning.self_consistency_k > 1:
            outs = self_consistency_batch(
                engine, qs, gen, spec.think_budget, spec.reasoning.style, spec.prompts, spec.reasoning.self_consistency_k
            )
            preds = [o["chosen_answer"] for o in outs]
            # For metrics, approximate tokens/latency as means over K paths
            think_toks = [int(statistics.mean([p["think_tokens"] for p in o["paths"]])) for o in outs]
            ans_toks   = [int(statistics.mean([p["answer_tokens"] for p in o["paths"]])) for o in outs]
            lats       = [float(statistics.mean([p["latency_ms_think"] + p["latency_ms_answer"] for p in o["paths"]])) for o in outs]
        else:
            outs = two_pass_batch(engine, qs, gen, spec.think_budget, spec.reasoning.style, spec.prompts)
            preds = [o["answer_text"] for o in outs]
            think_toks = [o["think_tokens"] for o in outs]
            ans_toks   = [o["answer_tokens"] for o in outs]
            lats       = [o["latency_ms_think"] + o["latency_ms_answer"] for o in outs]

        # Accumulate metrics
        for j, ex in enumerate(batch):
            total += 1
            ok = exact_match(preds[j], gts[j])
            correct_measured += int(ok)

            # Optional self-evaluation (YES/NO judge) — sequential is fine
            if spec.reasoning.self_eval:
                judge_yes, judge_response = self_evaluate(engine, ex.question, preds[j], ex.gold, gen, spec.prompts)
                correct_self += int(judge_yes)

            gen_tok_sum += (think_toks[j] + ans_toks[j])
            lat_ms_sum += float(lats[j])

        pbar.update(len(batch))

    pbar.close()

    # Compute FLOPs with your estimator(s)
    num_params = spec.card.params_B * 1e9
    # Dense-only estimate
    flops_est = flops_dense(num_params=num_params, num_tokens=gen_tok_sum)
    # Attention-aware (only if dims provided) — for your reference, not required
    flops_attn = None
    if spec.card.layers and spec.card.hidden_dim:
        flops_attn = flops_attention_kv(
            num_layers=spec.card.layers,
            hidden_dim=spec.card.hidden_dim,
            num_prompt_tokens=0,
            num_generated_tokens=gen_tok_sum,
            num_params=num_params,
            include_dense_anchor=True,
        )

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
        "hardware": "GPU-unknown",
        "batch_size": bs,
        "use_kv_cache": spec.generation.use_kv_cache,
        "reasoning_style": spec.reasoning.style,

        # Tokens
        "prompt_tokens": prompt_tok_sum,
        "gen_tokens": gen_tok_sum,
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
        "flops_dense_tflops": to_tflops(flops_est),
        "flops_attention_kv_tflops": to_tflops(flops_attn) if flops_attn is not None else None,

        # Notes
        "notes": f"{notes}",
    }

    print(f"[RUN] {spec.model_name} | {spec.dataset} | style={spec.reasoning.style} | "
          f"B={spec.think_budget} | K={spec.reasoning.self_consistency_k} | bs={bs} | "
          f"acc={row['accuracy']:.3f} | gen_tokens={gen_tok_sum} | "
          f"dense_tFLOPs≈{row['flops_dense_tflops']:.2f}")

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
        run_one(spec, batch_size=args.batch_size, wandb_project=args.wandb_project, notes=args.notes)

if __name__ == "__main__":
    main()
