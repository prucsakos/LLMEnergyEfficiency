from __future__ import annotations
import re
import time
from typing import Dict, Any, Tuple, List, Optional
from ..core.interfaces import TextEngine, GenerationParams, GenerationResult
from .prompts import build_prompt_cot, build_prompt_plan, build_prompt_answer, closing_tag
from .aggregators import majority_vote

SCRATCH_TAG = ("<scratchpad>", "</scratchpad>")
PLAN_TAG    = ("<plan>", "</plan>")
FINAL_TAG   = ("<final>", "</final>")

def _between(s: str, start: str, end: Optional[str]) -> str:
    if end:
        m = re.search(re.escape(start) + r"(.*?)" + re.escape(end), s, flags=re.S | re.I)
    else:
        m = re.search(re.escape(start) + r"(.*)", s, flags=re.S | re.I)
    return m.group(1) if m else ""

def run_two_pass(
    engine: TextEngine,
    question: str,
    params: GenerationParams,
    think_budget: int,
    style: str = "cot",  # 'cot' | 'plan_solve' | 'none'
) -> Dict[str, Any]:
    """Run (optional) thinking pass then an answer pass.

    Args:
        engine: TextEngine to call (e.g., VLLMOpenAIServerEngine).
        question: Natural-language question/problem.
        params: GenerationParams (temperature, etc.).
        think_budget: Max new tokens for pass-1 thinking (0 disables thinking).
        style: 'cot' (scratchpad) | 'plan_solve' (plan) | 'none' (direct answer).

    Returns:
        Dict with fields:
          think_text, answer_text, think_tokens, answer_tokens,
          latency_ms_think, latency_ms_answer, raw_pass1, raw_pass2
    """
    # Pass 1: optional thinking
    think_text = ""
    pass1_res: Optional[GenerationResult] = None
    if style != "none" and think_budget > 0:
        prompt1 = build_prompt_cot(question) if style == "cot" else build_prompt_plan(question)
        p1 = GenerationParams(**{**params.__dict__, "max_new_tokens": think_budget, "stop": [closing_tag(style)]})
        t0 = time.time()
        pass1_res = engine.generate(prompt1, p1)
        t1 = time.time()
        # Extract content within tags
        if style == "cot":
            think_text = _between(pass1_res.text, SCRATCH_TAG[0], None)
            tagged = f"{SCRATCH_TAG[0]}{think_text}{SCRATCH_TAG[1]}"
        else:
            think_text = _between(pass1_res.text, PLAN_TAG[0], None)
            tagged = f"{PLAN_TAG[0]}{think_text}{PLAN_TAG[1]}"
        latency_ms_think = (t1 - t0) * 1000.0
        think_tokens = pass1_res.completion_tokens
    else:
        tagged = ""
        latency_ms_think, think_tokens = 0.0, 0

    # Pass 2: answer-only
    prompt2 = build_prompt_answer(question, tagged)
    t0 = time.time()
    pass2_res = engine.generate(prompt2, params)
    t1 = time.time()
    ans_text = _between(pass2_res.text, FINAL_TAG[0], None).strip() or pass2_res.text.strip()
    latency_ms_answer = (t1 - t0) * 1000.0

    return {
        "think_text": think_text.strip(),
        "answer_text": ans_text,
        "think_tokens": think_tokens or 0,
        "answer_tokens": pass2_res.completion_tokens or 0,
        "latency_ms_think": latency_ms_think,
        "latency_ms_answer": latency_ms_answer,
        "raw_pass1": getattr(pass1_res, "raw", None),
        "raw_pass2": pass2_res.raw,
    }

# TODO: Metric on how effective the current type of majority voting is. - Distribution of the answers. Unique ratio etc... -> Relevant at benchmarks where the answer is not trivial
# TODO: Use LLM for self-evaluation instead of majority voting
def run_self_consistency(
    engine: TextEngine,
    question: str,
    params: GenerationParams,
    think_budget: int,
    style: str,
    k: int,
    seed_base: int = 1234,
) -> Dict[str, Any]:
    """Run K independent (think->answer) paths and majority-vote the final.

    Implements Self-Consistency as in Wang et al., 2022. :contentReference[oaicite:11]{index=11}
    """
    answers: List[str] = []
    paths: List[Dict[str, Any]] = []
    for i in range(k):
        p = GenerationParams(**{**params.__dict__, "seed": (seed_base + i), "temperature": max(params.temperature, 0.6)})
        out = run_two_pass(engine, question, p, think_budget, style)
        answers.append(out["answer_text"])
        paths.append(out)
    chosen = majority_vote(answers)
    return {"chosen_answer": chosen, "paths": paths}
