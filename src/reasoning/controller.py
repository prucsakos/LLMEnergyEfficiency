from __future__ import annotations
import itertools
import re
import time
from typing import Dict, Any, Tuple, List, Optional

from ..core.interfaces import TextEngine, GenerationParams, GenerationResult
from .aggregators import majority_vote
from ..config.bench_config import Prompts



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
    
    # Return the full generated text as the chosen answer
    chosen = result.text.strip()
    
    # If no valid choice found, fall back to majority vote
    if not chosen:
        return majority_vote(candidate_answers)
    
    return chosen



def _engine_generate_batch(engine: TextEngine,
                           prompts: List[str],
                           params: GenerationParams) -> Tuple[List[str], List[int], float, List[Any]]:
    if not prompts:
        return [], [], 0.0, []
    t0 = time.time()
    outs = engine.generate_batch(prompts, params)  # type: ignore[attr-defined]
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
        prompts.llm_judge.format(question=q, candidate=c, gold=g)
        for q, c, g in zip(questions, candidates, golds)
    ]
    
    # Use evaluation engine if provided, otherwise use main engine
    evaluation_engine = eval_engine if eval_engine is not None else engine
    
    try:
        texts, _tok, _ms, _raw = _engine_generate_batch(
            evaluation_engine,
            judge_prompts,
            GenerationParams(**{**gen.__dict__, "max_new_tokens": 150, "temperature": 1.0}),
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
                GenerationParams(**{**gen.__dict__, "max_new_tokens": 20, "temperature": 0., "top_p": 1.0}),
            )
        else:
            # Re-raise if it's not a quota error
            raise e
    
    outs: List[Tuple[bool, str, str]] = []
    for prompt, txt in zip(judge_prompts, texts):
        # Extract judgement from response (expecting CORRECT or INCORRECT)
        judgement = txt.strip().upper()
        
        # Find positions of both keywords
        correct_pos = judgement.find("CORRECT")
        incorrect_pos = judgement.find("INCORRECT")
        
        if correct_pos != -1 and incorrect_pos != -1:
            # Both present - use the one that appears first
            is_correct = correct_pos < incorrect_pos
        elif incorrect_pos != -1:
            # Only INCORRECT present
            is_correct = False
        elif correct_pos != -1:
            # Only CORRECT present
            is_correct = True
        else:
            # Neither present - default to False
            is_correct = False
            
        outs.append((is_correct, prompt, txt))

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
    have_think = (style == "cot" and think_budget > 0)
    if have_think:
        # Build thinking prompts directly
        think_prompts = [prompts.cot_think.format(question=q) for q in questions]
        think_texts, think_tok_counts, think_ms, think_raw_data = _engine_generate_batch(
            engine,
            think_prompts,
            GenerationParams(**{**gen.__dict__, "max_new_tokens": int(think_budget)}),
        )
        # Use full generated text as thinking content
        think_contents = [t.strip() for t in think_texts]
        deliberate_blocks = think_contents
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
        # Always use answer format, with deliberate being empty string if no thinking content
        # Format the prompt to avoid trailing newlines when deliberate is empty
        if deliberate.strip():
            answer_prompts.append(prompts.answer.format(question=q, deliberate=deliberate))
        else:
            # Remove the trailing newline from deliberate when it's empty
            base_prompt = prompts.answer.replace("\n{deliberate}", "")
            answer_prompts.append(base_prompt.format(question=q))
    MAX_ANSWER_TOKENS = 32
    answer_texts, answer_tok_counts, answer_ms, answer_raw_data = _engine_generate_batch(
        engine,
        answer_prompts,
        GenerationParams(**{**gen.__dict__, "max_new_tokens": int(MAX_ANSWER_TOKENS), "temperature": 0.0, "top_p": 1.0}),
    )

    ans_lat_per_item = answer_ms / max(n, 1)

    for i in range(n):
        # Use full generated text without cutting
        think_text = think_contents[i].strip() if have_think else ""
        answer_text = answer_texts[i].strip()
        
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
    have_think = (style == "cot" and think_budget > 0)
    if have_think:
        # Build thinking prompts directly
        think_prompts = [prompts.cot_think.format(question=q) for q in questions]
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
                varied_gen,
            )
            
            # Extract results and add to overall results
            for i in range(n):
                think_text = batch_results[0][i].strip()  # Use full generated text
                think_texts_rep.append(think_text)
                think_tok_rep.append(batch_results[1][i])
                think_raw_rep.append(batch_results[3][i])  # Add raw data
                think_ms += batch_results[2] / max(n*k, 1) # Average latency per item
        # Build deliberate blocks for answer prompts
        deliberate_rep = think_texts_rep  # Use full thinking text directly
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
        # Always use answer format, with deliberate being empty string if no thinking content
        # Format the prompt to avoid trailing newlines when deliberate is empty
        for d in dels:
            if d.strip():
                answer_prompts_rep.append(prompts.answer.format(question=q, deliberate=d))
            else:
                # Remove the trailing newline from deliberate when it's empty
                base_prompt = prompts.answer.replace("\n{deliberate}", "")
                answer_prompts_rep.append(base_prompt.format(question=q))
    answer_texts_rep, answer_tok_rep, answer_ms, answer_raw_rep = _engine_generate_batch(
        engine,
        answer_prompts_rep,
        GenerationParams(**{**gen.__dict__, "max_new_tokens": int(gen.max_new_tokens)}),
    )
    ans_lat_per_item = (answer_ms / max(n*k, 1))

    # Collate per-question
    outs = []
    for i in range(n):
        # Slice this question's K paths
        think_toks = think_tok_groups[i]
        think_raws = think_raw_groups[i]
        ans_toks = answer_tok_rep[i*k:(i+1)*k]
        texts = [t.strip() for t in answer_texts_rep[i*k:(i+1)*k]]  # Use full generated text
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