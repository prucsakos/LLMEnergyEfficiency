from __future__ import annotations
import argparse, math, statistics, time
from typing import Iterable, List, Optional, Tuple
import itertools
from tqdm.auto import tqdm

from ..config.bench_config import load_bench_config, expand_runs, RunSpec, Prompts
from ..core.interfaces import GenerationParams
from ..core.engines.vllm_local import VLLMLocalEngine
from ..reasoning.aggregators import majority_vote
from ..reasoning.controller import self_evaluate  # keep using the existing single-example judge
from ..data.adapters import load_gsm8k, load_mmlu, load_csqa, exact_match, Sample
from ..metrics.flop_estimation import flops_dense, flops_attention_kv, to_tflops
from ..logging.wandb_logger import WandbRunLogger

# Optional direct import from vLLM for batched generation
from vllm import SamplingParams

def iter_dataset(name: str, split: str = "test"):
    if name == "gsm8k": return load_gsm8k(split)
    if name == "mmlu":  return load_mmlu(split=split)
    if name == "csqa":  return load_csqa(split="validation" if split == "test" else split)
    raise ValueError(f"Unknown dataset {name}")

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

def two_pass_batch(engine: VLLMLocalEngine,
                   questions: List[str],
                   gen: GenerationParams,
                   think_budget: int,
                   style: str,
                   prompts: Prompts):
    """
    Batched variant of the original two_pass(). Returns a list of dicts, one per question:
    {
      "answer_text": str,
      "think_tokens": int,
      "answer_tokens": int,
      "latency_ms_think": float,
      "latency_ms_answer": float,
    }
    """
    n = len(questions)
    results = [{} for _ in range(n)]
    # Pass 1: "think" (optional)
    if style in ("cot", "plan") and think_budget > 0:
        think_prompts, open_tag, close_tag_ = _build_think_prompts(questions, style, prompts)
        think_texts, think_tok_counts, think_ms = _vllm_generate_batch(
            engine,
            think_prompts,
            GenerationParams(**{**gen.__dict__, "max_new_tokens": int(think_budget)}),
            extra_stop=close_tag_,
        )
        # Build deliberate blocks with tags
        deliberate_blocks = [f"{open_tag}{t}{close_tag_}" for t in think_texts]
        think_lat_per_item = think_ms / max(n, 1)
    else:
        # No deliberate pass
        deliberate_blocks = ["" for _ in questions]
        think_tok_counts = [0 for _ in questions]
        think_lat_per_item = 0.0

    # Pass 2: answer
    answer_prompts: List[str] = []
    for q, deliberate in zip(questions, deliberate_blocks):
        if deliberate:
            answer_prompts.append(prompts.answer.format(question=q, deliberate=deliberate))
        else:
            # "direct" answering when no deliberate content
            answer_prompts.append(prompts.direct.format(question=q))
    answer_texts, answer_tok_counts, answer_ms = _vllm_generate_batch(
        engine,
        answer_prompts,
        GenerationParams(**{**gen.__dict__, "max_new_tokens": int(gen.max_new_tokens)}),
        extra_stop="</final>",
    )
    ans_lat_per_item = answer_ms / max(n, 1)

    for i in range(n):
        results[i] = {
            "answer_text": answer_texts[i].strip(),
            "think_tokens": int(think_tok_counts[i]),
            "answer_tokens": int(answer_tok_counts[i]),
            "latency_ms_think": float(think_lat_per_item),
            "latency_ms_answer": float(ans_lat_per_item),
        }
    return results

def self_consistency_batch(engine: VLLMLocalEngine,
                           questions: List[str],
                           gen: GenerationParams,
                           think_budget: int,
                           style: str,
                           prompts: Prompts,
                           k: int):
    """
    Batched self-consistency: for each question, run K independent samples and majority-vote.
    Returns a list of dicts:
      { "chosen_answer": str, "paths": [ {think_tokens, answer_tokens, latency_ms_think, latency_ms_answer}, ...] }
    """
    n = len(questions)
    if k <= 1:
        # Defer to two-pass-batch shape for convenience
        outs = two_pass_batch(engine, questions, gen, think_budget, style, prompts)
        return [{"chosen_answer": o["answer_text"], "paths": [ {
                    "think_tokens": o["think_tokens"],
                    "answer_tokens": o["answer_tokens"],
                    "latency_ms_think": o["latency_ms_think"],
                    "latency_ms_answer": o["latency_ms_answer"],
                } ]} for o in outs]

    # Build repeated lists of prompts
    # Think stage
    have_think = (style in ("cot", "plan") and think_budget > 0)
    if have_think:
        think_prompts, open_tag, close_tag_ = _build_think_prompts(questions, style, prompts)
        think_prompts_rep = list(itertools.chain.from_iterable([[p]*k for p in think_prompts]))
        think_texts_rep, think_tok_rep, think_ms = _vllm_generate_batch(
            engine,
            think_prompts_rep,
            GenerationParams(**{**gen.__dict__, "max_new_tokens": int(think_budget)}),
            extra_stop=close_tag_,
        )
        # Build deliberate blocks for answer prompts
        deliberate_rep = [f"{open_tag}{t}{close_tag_}" for t in think_texts_rep]
        # Group deliberate blocks by question
        deliberate_groups = [deliberate_rep[i*k:(i+1)*k] for i in range(n)]
        think_tok_groups = [think_tok_rep[i*k:(i+1)*k] for i in range(n)]
        think_lat_per_item = (think_ms / max(n*k, 1))
    else:
        deliberate_groups = [[""]*k for _ in range(n)]
        think_tok_groups = [[0]*k for _ in range(n)]
        think_lat_per_item = 0.0

    # Answer stage
    answer_prompts_rep: List[str] = []
    for i in range(n):
        q = questions[i]
        dels = deliberate_groups[i]
        if dels and dels[0]:
            answer_prompts_rep.extend([prompts.answer.format(question=q, deliberate=d) for d in dels])
        else:
            answer_prompts_rep.extend([prompts.direct.format(question=q) for _ in range(k)])
    answer_texts_rep, answer_tok_rep, answer_ms = _vllm_generate_batch(
        engine,
        answer_prompts_rep,
        GenerationParams(**{**gen.__dict__, "max_new_tokens": int(gen.max_new_tokens)}),
        extra_stop="</final>",
    )
    ans_lat_per_item = (answer_ms / max(n*k, 1))

    # Collate per-question
    outs = []
    for i in range(n):
        # Slice this question's K paths
        think_toks = think_tok_groups[i]
        ans_toks = answer_tok_rep[i*k:(i+1)*k]
        texts = [t.strip() for t in answer_texts_rep[i*k:(i+1)*k]]
        chosen = majority_vote(texts)
        paths = []
        for j in range(k):
            paths.append({
                "think_tokens": int(think_toks[j]),
                "answer_tokens": int(ans_toks[j]),
                "latency_ms_think": float(think_lat_per_item),
                "latency_ms_answer": float(ans_lat_per_item),
            })
        outs.append({"chosen_answer": chosen, "paths": paths})
    return outs

# ------------------------------

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
                judge_yes = self_evaluate(engine, ex.question, preds[j], ex.gold, gen, spec.prompts)
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
