from __future__ import annotations
import argparse, math, statistics
from typing import Iterable, List

from tqdm import tqdm
from ..config.bench_config import load_bench_config, expand_runs, RunSpec
from ..core.interfaces import GenerationParams
from ..core.engines import create_engine
from ..reasoning.controller import two_pass, self_consistency, self_evaluate
from ..data.adapters import Sample, load_gsm8k, load_mmlu, load_csqa, exact_match, iter_dataset
from ..metrics.flop_estimation import flops_dense, flops_attention_kv, to_tflops
from ..logs.wandb_logger import WandbRunLogger


def run_one(spec: RunSpec, wandb_project: str | None = None, notes: str = "", verbose: bool = False) -> None:
    # Build engine (one per run), then tear down afterwards
    engine = create_engine(
        spec.engine,
        model_id=spec.hf_repo,
        dtype=spec.backend.dtype,
        gpu_memory_utilization=spec.backend.gpu_memory_utilization,
        enforce_eager=spec.backend.enforce_eager,
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

    # W&B: one run per (model, dataset, budget, style, k)
    wb = None
    if wandb_project:
        run_name = f"{spec.model_name}|{spec.dataset}|style={spec.reasoning.style}|B={spec.think_budget}|K={spec.reasoning.self_consistency_k}"
        cfg = {
            "model": spec.hf_repo,
            "dataset": spec.dataset,
            "style": spec.reasoning.style,
            "think_budget": spec.think_budget,
            "K": spec.reasoning.self_consistency_k,
            "dtype": spec.backend.dtype,
        }
        wb = WandbRunLogger(project=wandb_project, run_name=run_name, config=cfg)

    # Materialize dataset to know total for tqdm
    examples: List[Sample] = list(iter_dataset(spec.dataset))
    total_n = len(examples)

    # Iterate dataset (MVP: batch_size=1 for strict control)
    total, correct_measured, correct_self = 0, 0, 0
    prompt_tok_sum, gen_tok_sum = 0, 0
    lat_ms_sum = 0.0

    for ex in iter_dataset(spec.dataset):
        total += 1
        if spec.reasoning.self_consistency_k > 1:
            out = self_consistency(
                engine, ex.question, gen, spec.think_budget, spec.reasoning.style,
                spec.prompts, spec.reasoning.self_consistency_k)
            pred = out["chosen_answer"]
            # Approximate tokens/latency as avg over paths (MVP)
            think = [p["think_tokens"] for p in out["paths"]]
            answ  = [p["answer_tokens"] for p in out["paths"]]
            lats  = [p["latency_ms_think"] + p["latency_ms_answer"] for p in out["paths"]]
            think_tok = int(statistics.mean(think))
            ans_tok = int(statistics.mean(answ))
            lat_ms = float(statistics.mean(lats))
        else:
            out = two_pass(engine, ex.question, gen, spec.think_budget, spec.reasoning.style, spec.prompts, verbose=verbose)
            pred = out["answer_text"]
            think_tok = int(out["think_tokens"])
            ans_tok   = int(out["answer_tokens"])
            lat_ms    = float(out["latency_ms_think"] + out["latency_ms_answer"])

        # Measured EM accuracy (ground truth)
        ok = exact_match(pred, ex.gold)
        correct_measured += int(ok)

        # Optional self-evaluation (YES/NO judge)
        if spec.reasoning.self_eval:
            judge_yes, judge_resp = self_evaluate(engine, ex.question, pred, ex.gold, gen, spec.prompts)
            correct_self += int(judge_yes)

        

        prompt_tok_sum += 0  # we only counted generated tokens; prompt length not measured here
        gen_tok_sum += (think_tok + ans_tok)
        lat_ms_sum += lat_ms

        if verbose:
            print(f"""
Exact match with golden answer: {ok}
Judge yes: {judge_yes}, resp: {judge_resp}
True Correct answer: {ex.gold}
                  """)

        break

    # TODO: gen_tok_sum -> we rather want to calculate an average FLOP per a reasoning process.

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
        "batch_size": 1,
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
          f"B={spec.think_budget} | K={spec.reasoning.self_consistency_k} | "
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
    args = ap.parse_args()

    cfg = load_bench_config(args.config)
    for spec in expand_runs(cfg):
        run_one(spec, wandb_project=args.wandb_project, notes=args.notes, verbose=True)

if __name__ == "__main__":
    main()
