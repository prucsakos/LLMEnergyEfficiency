from __future__ import annotations
import argparse, time
from typing import Iterable, Dict, Any
from ..core.interfaces import GenerationParams
from ..core.engines.vllm_server import VLLMOpenAIServerEngine
from ..reasoning.controller import run_two_pass, run_self_consistency
from ..data.adapters import load_gsm8k, load_mmlu, load_csqa, exact_match, Sample
from ..metrics.flop_estimation import ModelCard, estimate_inference_flops, gflops

def iter_dataset(name: str, split: str, **kw) -> Iterable[Sample]:
    if name == "gsm8k": return load_gsm8k(split)
    if name == "mmlu":  return load_mmlu(split=split, subjects=kw.get("subjects"))
    if name == "csqa":  return load_csqa(split)
    raise ValueError(f"Unknown dataset {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["gsm8k","mmlu","csqa"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--style", default="cot", choices=["none","cot","plan_solve"])
    ap.add_argument("--think-budget", type=int, default=128)
    ap.add_argument("--self-consistency-k", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--params-b", type=float, required=True, help="Model size in billions (non-embed if known)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="llm-bench")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    engine = VLLMOpenAIServerEngine(base_url=args.base_url, model=args.model)
    gparams = GenerationParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype=args.dtype,
        use_kv_cache=True,
    )

    # Optional W&B
    wb = None
    if args.wandb:
        from ..logs.wandb_logger import WandbLogger
        wb = WandbLogger(project=args.wandb_project, run_name=f"{args.model}-{args.dataset}-{args.style}")

    card = ModelCard(params_non_embed=float(args.params_b) * 1e9)

    # Iterate dataset
    total = 0
    correct = 0
    for ex in iter_dataset(args.dataset, args.split):
        if args.self_consistency_k > 1:
            out = run_self_consistency(engine, ex.question, gparams, args.think_budget, args.style, args.self_consistency_k)
            pred = out["chosen_answer"]
            paths = out["paths"]
            think_tok = sum(p["think_tokens"] for p in paths) / len(paths)
            ans_tok = sum(p["answer_tokens"] for p in paths) / len(paths)
            lat_ms = sum(p["latency_ms_think"] + p["latency_ms_answer"] for p in paths) / len(paths)
        else:
            out = run_two_pass(engine, ex.question, gparams, args.think_budget, args.style)
            pred = out["answer_text"]
            think_tok = out["think_tokens"]
            ans_tok = out["answer_tokens"]
            lat_ms = out["latency_ms_think"] + out["latency_ms_answer"]

        is_correct = exact_match(pred, ex.gold)
        correct += int(is_correct)
        total += 1

        # Estimate FLOPs (simple)
        flops = estimate_inference_flops(card, prompt_tokens=0, gen_tokens=int(think_tok + ans_tok), method="simple")
        speed = (think_tok + ans_tok) / (lat_ms / 1000.0) if lat_ms > 0 else None

        row = {
          # Model/setup
          "model": args.model, "arch": "decoder-only", "params_B": args.params_b,
          "layers": None, "d_model": None, "heads": None,
          "precision": args.dtype, "quant": None, "hardware": "unknown",
          "batch_size": 1, "use_kv_cache": True, "reasoning_style": args.style,
          # Tokens
          "prompt_tokens": 0, "gen_tokens": int(think_tok + ans_tok), "passes": 2,
          "beam_width": 1, "self_consistency_k": args.self_consistency_k,
          # Measured
          "latency_ms": float(lat_ms), "speed_tok_per_s": speed, "energy_j": None,
          # Task metric
          "dataset": args.dataset, "metric_name": "exact_match", "accuracy": correct/total,
          # Notes
          "notes": args.notes,
        }
        print(f"[{total}] acc={correct/total:.3f}  gflops~{gflops(flops):.1f}  tokens={think_tok+ans_tok:.0f}")

        if wb:
            wb.log_row(row)

    if wb: wb.finish()

if __name__ == "__main__":
    main()
