from __future__ import annotations
import itertools
import re
import time
from typing import Dict, Any, Tuple, List, Optional

from ..core.interfaces import TextEngine, GenerationParams, GenerationResult
from .aggregators import majority_vote
from ..config.bench_config import Prompts


def _cut_at_end_tag(s: str, end_tag: str | None) -> str:
    """Extract content from generated text, cutting at end tag if present.
    
    Handles two cases:
    1. If text contains both start and end tags (like <think>content</think>), extract content between them
    2. If only end tag is present, cut everything after it
    3. If no end tag, return entire text
    
    This handles cases where thinking budget is cut and model is force-stopped.
    """
    if not end_tag:
        return s.strip()
    
    # First try to extract content between tags (case 1)
    # Look for pattern: <tag>content</tag>
    start_tag = end_tag.replace("</", "<")
    if start_tag in s and end_tag in s:
        start_pos = s.find(start_tag) + len(start_tag)
        end_pos = s.find(end_tag)
        if start_pos < end_pos:
            return s[start_pos:end_pos].strip()
    
    # Fallback: cut at end tag if present (case 2)
    end_pos = s.find(end_tag)
    if end_pos != -1:
        return s[:end_pos].strip()
    
    # No end tag found, return entire text (case 3)
    return s.strip()

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
    chosen = _cut_at_end_tag(result.text, "</chosen>")
    
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
    if style in ("plan", "cot"):
        open_tag, close = "<think>", "</think>"
        if style == "plan":
            think_prompts = [prompts.plan_think.format(question=q) for q in questions]
        else:  # style == "cot"
            think_prompts = [prompts.cot_think.format(question=q) for q in questions]
    else:
        # style "none" -> no think prompts
        open_tag, close, think_prompts = "", "", []
    return think_prompts, open_tag, close

def _engine_generate_batch(engine: TextEngine,
                           prompts: List[str],
                           params: GenerationParams,
                           extra_stop: Optional[str] = None) -> Tuple[List[str], List[int], float, List[Any]]:
    if not prompts:
        return [], [], 0.0, []
    merged = GenerationParams(**{**params.__dict__, "stop": _combine_stops(params.stop, extra_stop)})
    t0 = time.time()
    outs = engine.generate_batch(prompts, merged)  # type: ignore[attr-defined]
    wall_ms = (time.time() - t0) * 1000.0
    texts = [(o.text if o and o.text is not None else "") for o in outs]
    comp_tokens = [int(o.completion_tokens or 0) for o in outs]
    raw_data = [o.raw if o and hasattr(o, 'raw') else None for o in outs] # DeepSpeed raw data containing FLOPs
    return texts, comp_tokens, wall_ms, raw_data


def self_evaluate_batched(
    engine: TextEngine,
    questions: List[str],
    candidates: List[str],
    golds: List[str],
    gen: GenerationParams,
    prompts: Prompts,
    eval_engine: Optional[TextEngine] = None,
) -> List[Tuple[bool, str, str]]:
    """Batched judge for correctness. Returns list of (is_yes, prompt, raw_text)."""
    n = len(questions)
    if not (len(candidates) == n and len(golds) == n):
        raise ValueError("self_evaluate_batched: inputs must have equal length")

    judge_prompts = [
        prompts.self_eval.format(question=q, candidate=c, gold=g)
        for q, c, g in zip(questions, candidates, golds)
    ]
    
    # Use evaluation engine if provided, otherwise use main engine
    evaluation_engine = eval_engine if eval_engine is not None else engine
    
    try:
        texts, _tok, _ms, _raw = _engine_generate_batch(
            evaluation_engine,
            judge_prompts,
            GenerationParams(**{**gen.__dict__, "max_new_tokens": 150, "temperature": 1.0}),
            extra_stop=None,
        )
    except Exception as e:
        # Check if this is a quota exceeded error from OpenAI
        error_str = str(e).lower()
        if ("quota" in error_str or "insufficient_quota" in error_str or 
            "429" in error_str or "quotaexceedederror" in error_str):
            print(f"OpenAI quota exceeded during evaluation: {e}")
            
            # Log additional quota information if available
            if hasattr(e, 'quota_info') and e.quota_info:
                quota_info = e.quota_info
                print("📊 Quota Details:")
                if quota_info.get('reset_time'):
                    from datetime import datetime
                    if isinstance(quota_info['reset_time'], datetime):
                        now = datetime.now()
                        time_diff = quota_info['reset_time'] - now
                        if time_diff.total_seconds() > 0:
                            hours = int(time_diff.total_seconds() // 3600)
                            minutes = int((time_diff.total_seconds() % 3600) // 60)
                            print(f"   Quota Resets In: {hours}h {minutes}m")
                        else:
                            print(f"   Quota may have already reset")
                if quota_info.get('remaining_requests') is not None:
                    print(f"   Remaining Requests: {quota_info['remaining_requests']}")
                if quota_info.get('remaining_tokens') is not None:
                    print(f"   Remaining Tokens: {quota_info['remaining_tokens']}")
            
            print("Falling back to main engine for self-evaluation")
            # Fallback to main engine
            texts, _tok, _ms, _raw = _engine_generate_batch(
                engine,
                judge_prompts,
                GenerationParams(**{**gen.__dict__, "max_new_tokens": 150, "temperature": 1.0}),
                extra_stop=None,
            )
        else:
            # Re-raise if it's not a quota error
            raise e
    
    outs: List[Tuple[bool, str, str]] = []
    for prompt, txt in zip(judge_prompts, texts):
        # Extract judgement from response (expecting YES or NO)
        judgement = txt.strip().lower()
        is_yes = ("yes" in judgement) or (judgement == "1")
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
        think_texts, think_tok_counts, think_ms, think_raw_data = _engine_generate_batch(
            engine,
            think_prompts,
            GenerationParams(**{**gen.__dict__, "max_new_tokens": int(think_budget), "stop": [close_tag_]}),
            extra_stop=None,
        )
        # Extract think content by cutting at end tag
        think_contents = [_cut_at_end_tag(t, close_tag_) for t in think_texts]
        deliberate_blocks = [f"{open_tag}{t}{close_tag_}" for t in think_contents]
        think_lat_per_item = think_ms / max(n, 1)
    else:
        # No deliberate pass
        deliberate_blocks = ["" for _ in questions]
        think_contents = ["" for _ in questions]
        think_tok_counts = [0 for _ in questions]
        think_lat_per_item = 0.0
        think_raw_data = [None for _ in questions]

    # Pass 2: answer
    answer_prompts: List[str] = []
    for q, deliberate in zip(questions, deliberate_blocks):
        if deliberate:
            answer_prompts.append(prompts.answer.format(question=q, deliberate=deliberate))
        else:
            # "direct" answering when no deliberate content
            answer_prompts.append(prompts.direct.format(question=q))
    MAX_ANSWER_TOKENS = 32
    answer_texts, answer_tok_counts, answer_ms, answer_raw_data = _engine_generate_batch(
        engine,
        answer_prompts,
        GenerationParams(**{**gen.__dict__, "max_new_tokens": int(MAX_ANSWER_TOKENS), "stop": ["</final>"]}),
        extra_stop=None,
    )

    ans_lat_per_item = answer_ms / max(n, 1)

    for i in range(n):
        # Extract content from tags like the original two_pass function
        think_text = think_contents[i].strip() if have_think else ""
        answer_text = _cut_at_end_tag(answer_texts[i], "</final>")
        
        results[i] = {
            "answer_prompt": answer_prompts[i].strip(),
            "answer_text": answer_text,
            "think_text": think_text,
            "think_tokens": int(think_tok_counts[i]),
            "answer_tokens": int(answer_tok_counts[i]),
            "latency_ms_think": float(think_lat_per_item),
            "latency_ms_answer": float(ans_lat_per_item),
            "raw": {
                "think_raw": think_raw_data[i] if i < len(think_raw_data) else None,
                "answer_raw": answer_raw_data[i] if i < len(answer_raw_data) else None,
            }
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
        think_raw_rep = []
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
                GenerationParams(**{**varied_gen.__dict__, "stop": [close_tag_]}),
                extra_stop=None,
            )
            
            # Extract results and add to overall results
            for i in range(n):
                think_text = _cut_at_end_tag(batch_results[0][i], close_tag_)
                think_texts_rep.append(think_text)
                think_tok_rep.append(batch_results[1][i])
                think_raw_rep.append(batch_results[3][i])  # Add raw data
                think_ms += batch_results[2] / max(n*k, 1) # Average latency per item
        # Build deliberate blocks for answer prompts
        deliberate_rep = [f"{open_tag}{t}{close_tag_}" for t in think_texts_rep]
        # Group deliberate blocks by question
        deliberate_groups = [deliberate_rep[i::n][:k] for i in range(n)]
        think_tok_groups = [think_tok_rep[i::n][:k] for i in range(n)]
        think_raw_groups = [think_raw_rep[i::n][:k] for i in range(n)]
        think_lat_per_item = (think_ms)
    else:
        deliberate_groups = [[""]*k for _ in range(n)]
        think_texts_rep = [""] * (n * k)
        think_tok_groups = [[0]*k for _ in range(n)]
        think_raw_groups = [[None]*k for _ in range(n)]
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
    answer_texts_rep, answer_tok_rep, answer_ms, answer_raw_rep = _engine_generate_batch(
        engine,
        answer_prompts_rep,
        GenerationParams(**{**gen.__dict__, "max_new_tokens": int(gen.max_new_tokens), "stop": ["</final>"]}),
        extra_stop=None,
    )
    ans_lat_per_item = (answer_ms / max(n*k, 1))

    # Collate per-question
    outs = []
    for i in range(n):
        # Slice this question's K paths
        think_toks = think_tok_groups[i]
        think_raws = think_raw_groups[i]
        ans_toks = answer_tok_rep[i*k:(i+1)*k]
        texts = [_cut_at_end_tag(t, "</final>") for t in answer_texts_rep[i*k:(i+1)*k]]
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
                "raw": {
                    "think_raw": think_raws[j] if j < len(think_raws) else None,
                    "answer_raw": answer_raw_rep[i*k + j] if i*k + j < len(answer_raw_rep) else None,
                }
            })
        outs.append({"chosen_answer": chosen, "paths": paths})
    return outs