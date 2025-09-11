from __future__ import annotations
import re
import time
from typing import Dict, Any, Tuple, List, Optional
from ..core.interfaces import TextEngine, GenerationParams, GenerationResult
from .prompts import build_prompt_cot, build_prompt_plan, build_prompt_answer, closing_tag
from .aggregators import majority_vote
from ..config.bench_config import Prompts

SCRATCH_TAG = ("<scratchpad>", "</scratchpad>")
PLAN_TAG    = ("<plan>", "</plan>")
FINAL_TAG   = ("<final>", "</final>")

def _between(s: str, start: str, end: str | None) -> str:
    pat = re.escape(start) + (r"(.*?)" + re.escape(end) if end else r"(.*)")
    m = re.search(pat, s, flags=re.S | re.I)
    return (m.group(1) if m else "").strip()

def two_pass(
    engine: TextEngine,
    question: str,
    gen: GenerationParams,
    think_budget: int,
    style: str,
    prompts: Prompts,
) -> Dict[str, Any]:
    """Run (thinking -> answer). style in {'none','cot','plan_solve'}."""
    deliberate_tagged = ""

    # Pass 1: optional thinking
    think_tokens = 0
    t_think = 0.0
    think_text = ""
    if style != "none" and think_budget > 0:
        if style == "cot":
            p1 = prompts.cot_think
            start_tag, end_tag = "<scratchpad>", "</scratchpad>"
        else:
            p1 = prompts.plan_think
            start_tag, end_tag = "<plan>", "</plan>"
        prompt1 = p1.format(question=question)
        res1 = engine.generate(prompt1, GenerationParams(**{**gen.__dict__, "max_new_tokens": think_budget, "stop": [end_tag]}))
        t_think = res1.latency_ms or 0.0
        raw_block = _between(res1.text, start_tag, end_tag)
        think_text = raw_block
        think_tokens = res1.completion_tokens or 0
        deliberate_tagged = f"{start_tag}{raw_block}{end_tag}"

    # Pass 2: answer-only
    ans_prompt = prompts.answer.format(question=question, deliberate=deliberate_tagged)
    res2 = engine.generate(ans_prompt, gen)
    t_ans = res2.latency_ms or 0.0
    answer_text = _between(res2.text, "<final>", "<final/>") or res2.text.strip()

    return {
        "think_text": think_text,
        "answer_text": answer_text,
        "think_tokens": think_tokens,
        "answer_tokens": res2.completion_tokens or 0,
        "latency_ms_think": t_think,
        "latency_ms_answer": t_ans,
        "raw_pass2": res2.raw,
    }

def self_consistency(
    engine: TextEngine,
    question: str,
    base_gen: GenerationParams,
    think_budget: int,
    style: str,
    prompts: Prompts,
    k: int,
    seed: int = 1234,
) -> Dict[str, Any]:
    """K independent paths; majority-vote final answer (Wang et al., 2022)."""
    # Ensure sampling diversity
    answers: List[str] = []
    paths: List[Dict[str, Any]] = []
    for i in range(k):
        p = GenerationParams(**{**base_gen.__dict__, "seed": seed + i, "temperature": max(0.6, base_gen.temperature)})
        out = two_pass(engine, question, p, think_budget, style, prompts)
        answers.append(out["answer_text"])
        paths.append(out)
    # majority vote
    norm = [a.strip().lower() for a in answers]
    from collections import Counter
    chosen_norm, _ = Counter(norm).most_common(1)[0]
    chosen = next(a for a in answers if a.strip().lower() == chosen_norm)
    return {"chosen_answer": chosen, "paths": paths}

def self_evaluate(
    engine: TextEngine,
    question: str,
    candidate: str,
    gold: str,
    gen: GenerationParams,
    prompts: Prompts,
) -> bool:
    """Model judges correctness (YES/NO). Returns True if model says YES."""
    judge_prompt = prompts.self_eval.format(question=question, candidate=candidate, gold=gold)
    res = engine.generate(judge_prompt, GenerationParams(**{**gen.__dict__, "max_new_tokens": 4, "temperature": 0.0}))
    text = res.text.strip().lower()
    return "yes" in text or text.strip() == "1"