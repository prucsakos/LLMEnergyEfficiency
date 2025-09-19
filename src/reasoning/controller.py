from __future__ import annotations
import itertools
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

def _llm_consistency_eval(engine: TextEngine, question: str, candidate_answers: List[str], prompts: Prompts, gen: GenerationParams) -> str:
    """Use LLM to perform majority vote by counting and selecting the most frequent answer."""
    # Format candidate answers for the prompt
    formatted_answers = "\n".join([f"Answer {i+1}: {ans}" for i, ans in enumerate(candidate_answers)])
    
    # Create the consistency evaluation prompt
    eval_prompt = prompts.consistency_eval.format(
        question=question,
        candidate_answers=formatted_answers
    )
    
    # Generate evaluation
    result = engine.generate(eval_prompt, gen)
    
    # Extract the chosen answer
    chosen = _between(result.text, "<chosen>", "</chosen>")
    
    # If no valid choice found, fall back to majority vote
    if not chosen:
        return majority_vote(candidate_answers)
    
    return chosen.strip()

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

def _engine_generate_batch(engine: TextEngine,
                           prompts: List[str],
                           params: GenerationParams,
                           extra_stop: Optional[str] = None) -> Tuple[List[str], List[int], float]:
    if not prompts:
        return [], [], 0.0
    merged = GenerationParams(**{**params.__dict__, "stop": _combine_stops(params.stop, extra_stop)})
    t0 = time.time()
    outs = engine.generate_batch(prompts, merged)  # type: ignore[attr-defined]
    wall_ms = (time.time() - t0) * 1000.0
    texts = [(o.text if o and o.text is not None else "") for o in outs]
    comp_tokens = [int(o.completion_tokens or 0) for o in outs]
    return texts, comp_tokens, wall_ms

def two_pass(
    engine: TextEngine,
    question: str,
    gen: GenerationParams,
    think_budget: int,
    style: str,
    prompts: Prompts,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run (thinking -> answer). style in {'none','cot','plan_solve'}."""
    deliberate_tagged = ""

    # Pass 1: optional thinking
    think_tokens = 0
    t_think = 0.0
    think_text = ""
    prompt1, res1 = "", ""
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
    answer_text = _between(res2.text, "<final>", "</final>") or res2.text.strip()

    if verbose:
        print(f"""Two_Pass Verbos info:\n
\nFirst prompt:\n{prompt1}
\nGeneration Result:\n{res1}
\nAnswer prompt:\n{ans_prompt}
\nResult:\n{answer_text}
              """)

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
) -> Tuple[bool, str, str]:
    """Model judges correctness (YES/NO). Returns True if model says YES."""
    judge_prompt = prompts.self_eval.format(question=question, candidate=candidate, gold=gold)
    res = engine.generate(judge_prompt, GenerationParams(**{**gen.__dict__, "max_new_tokens": 4, "temperature": 0.0}))
    text = res.text.strip().lower()
    return "yes" in text or text.strip() == "1", judge_prompt, res.text

def self_evaluate_batched(
    engine: TextEngine,
    questions: List[str],
    candidates: List[str],
    golds: List[str],
    gen: GenerationParams,
    prompts: Prompts,
) -> List[Tuple[bool, str, str]]:
    """Batched judge for correctness. Returns list of (is_yes, prompt, raw_text)."""
    n = len(questions)
    if not (len(candidates) == n and len(golds) == n):
        raise ValueError("self_evaluate_batched: inputs must have equal length")

    judge_prompts = [
        prompts.self_eval.format(question=q, candidate=c, gold=g)
        for q, c, g in zip(questions, candidates, golds)
    ]
    texts, _tok, _ms = _engine_generate_batch(
        engine,
        judge_prompts,
        GenerationParams(**{**gen.__dict__, "max_new_tokens": 4, "temperature": 0.0}),
        extra_stop=None,
    )
    outs: List[Tuple[bool, str, str]] = []
    for prompt, txt in zip(judge_prompts, texts):
        low = (txt or "").strip().lower()
        is_yes = ("yes" in low) or (low == "1")
        outs.append((is_yes, prompt, txt))
    return outs

def two_pass_batch(engine: TextEngine,
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
    have_think = (style in ("cot", "plan") and think_budget > 0)
    if have_think:
        think_prompts, open_tag, close_tag_ = _build_think_prompts(questions, style, prompts)
        think_texts, think_tok_counts, think_ms = _engine_generate_batch(
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
        think_texts = ["" for _ in questions]
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
    answer_texts, answer_tok_counts, answer_ms = _engine_generate_batch(
        engine,
        answer_prompts,
        GenerationParams(**{**gen.__dict__, "max_new_tokens": int(gen.max_new_tokens)}),
        extra_stop="</final>",
    )
    ans_lat_per_item = answer_ms / max(n, 1)

    for i in range(n):
        results[i] = {
            "answer_prompt": answer_prompts[i].strip(),
            "answer_text": answer_texts[i].strip(),
            "think_text": think_texts[i].strip() if have_think else "",
            "think_tokens": int(think_tok_counts[i]),
            "answer_tokens": int(answer_tok_counts[i]),
            "latency_ms_think": float(think_lat_per_item),
            "latency_ms_answer": float(ans_lat_per_item),
        }
    return results

def self_consistency_batch(engine: TextEngine,
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

    # Build repeated lists of prompts
    # Think stage
    have_think = (style in ("cot", "plan") and think_budget > 0)
    if have_think:
        think_prompts, open_tag, close_tag_ = _build_think_prompts(questions, style, prompts)
        # Define sampling portfolio for diversity
        # Temperature ladder: [0.2, 0.4, 0.6, 0.8, 1.0] - increasing creativity/randomness
        # Top-p ladder: [0.85, 0.9, 0.92, 0.95, 0.97] - paired with temperatures
        temperature_ladder = [0.2, 0.4, 0.6, 0.8, 1.0]
        topp_ladder = [0.85, 0.9, 0.92, 0.95, 0.97]
        
        # Create portfolio configurations
        portfolio_configs = []
        
        # Add temperature + top-p ladder combinations
        for i, (temp, top_p) in enumerate(zip(temperature_ladder, topp_ladder)):
            portfolio_configs.append({
                "temperature": temp,
                "top_p": top_p
            })
        
        # Add additional top-p variants for more diversity
        additional_topp_variants = [0.8, 0.75, 0.7]
        for top_p in additional_topp_variants:
            portfolio_configs.append({
                "temperature": gen.temperature,  # Keep original temperature
                "top_p": top_p
            })
        
        # Generate with sampling portfolio for diversity
        think_texts_rep = []
        think_tok_rep = []
        think_ms = 0.0
        
        for j in range(k):
            # Select portfolio configuration for this path
            config = portfolio_configs[j % len(portfolio_configs)]
            
            # Create generation parameters with portfolio configuration
            varied_gen = GenerationParams(
                **{**gen.__dict__, 
                   "max_new_tokens": int(think_budget),
                   "temperature": config["temperature"],
                   "top_p": config["top_p"]}
            )
            
            # Create batch of prompts for this portfolio configuration
            batch_prompts = []
            for i in range(n):
                # Use standard prompt for all paths (no prompt variants)
                prompt = think_prompts[i]
                batch_prompts.append(prompt)
            
            # Generate batch for this portfolio configuration
            batch_results = _engine_generate_batch(
                engine,
                batch_prompts,
                varied_gen,
                extra_stop=close_tag_,
            )
            
            # Extract results and add to overall results
            for i in range(n):
                think_text = _between(batch_results[0][i], open_tag, close_tag_)
                think_texts_rep.append(think_text)
                think_tok_rep.append(batch_results[1][i])
                think_ms += batch_results[2] / max(n*k, 1) # Average latency per item
        # Build deliberate blocks for answer prompts
        deliberate_rep = [f"{open_tag}{t}{close_tag_}" for t in think_texts_rep]
        # Group deliberate blocks by question
        deliberate_groups = [deliberate_rep[i::n][:k] for i in range(n)]
        think_tok_groups = [think_tok_rep[i::n][:k] for i in range(n)]
        think_lat_per_item = (think_ms)
    else:
        deliberate_groups = [[""]*k for _ in range(n)]
        think_texts_rep = [""] * (n * k)
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
    answer_texts_rep, answer_tok_rep, answer_ms = _engine_generate_batch(
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
        # Use LLM to perform majority vote by counting frequencies
        chosen = _llm_consistency_eval(engine, questions[i], texts, prompts, gen)
        paths = []
        for j in range(k):
            # Get the actual generated text for this path
            think_text = think_texts_rep[i*k + j] if have_think else ""
            answer_text = answer_texts_rep[i*k + j]
            paths.append({
                "think_text": think_text.strip(),
                "answer_text": answer_text.strip(),
                "think_tokens": int(think_toks[j]),
                "answer_tokens": int(ans_toks[j]),
                "latency_ms_think": float(think_lat_per_item),
                "latency_ms_answer": float(ans_lat_per_item),
            })
        outs.append({"chosen_answer": chosen, "paths": paths})
    return outs