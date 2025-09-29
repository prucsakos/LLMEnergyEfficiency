# mvp_benchmark.py
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

# Optional vLLM import guarded by backend arg
try:
    from vllm import LLM, SamplingParams  # noqa
    _VLLM_AVAILABLE = True
except Exception:
    _VLLM_AVAILABLE = False

import mlflow

# Import logging system
from src.logs.benchmark_logger import setup_logging, get_logger

# ----------------------------
# Toy dataset (tiny, deterministic)
# Multiple-choice with single correct option letter (A/B/C/D)
# ----------------------------
ToyExample = Tuple[str, List[str], str]  # (question, choices, correct_letter)

TOY_DATA: List[ToyExample] = [
    ("What is 17 + 28?", ["A) 44", "B) 45", "C) 46", "D) 47"], "D"),
    ("If all bloops are razzies, and all razzies are lazzies, are all bloops lazzies?",
     ["A) Yes", "B) No", "C) Cannot be determined", "D) Only sometimes"], "A"),
    ("Largest prime less than 20?", ["A) 13", "B) 17", "C) 19", "D) 23"], "C"),
    ("Which is heavier: 1 kg of feathers or 1 kg of iron?",
     ["A) Feathers", "B) Iron", "C) Same", "D) Depends on volume"], "C"),
    ("Solve for x: 2x + 6 = 14",
     ["A) x=2", "B) x=3", "C) x=4", "D) x=5"], "C"),
]

# ----------------------------
# CoT chat helpers
# ----------------------------
SYS_PROMPT = (
    "You are a careful reasoning assistant. Solve the problem step by step. "
    "At the end, answer with the letter of the correct choice ONLY."
)

def build_messages(question: str, choices: List[str]) -> List[Dict[str, str]]:
    # Assemble a single-turn chat with CoT hint.
    user_text = (
        f"{question}\n\nChoices:\n" + "\n".join(choices) +
        "\n\nLet's think step by step. Conclude with just the letter."
    )
    return [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_text},
    ]

LETTER_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)

def extract_choice_letter(text: str) -> Optional[str]:
    # Take the last standalone A-D letter in the output.
    matches = LETTER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()

# ----------------------------
# FLOPs estimators
# ----------------------------
@dataclass
class ModelShape:
    params: int
    layers: int
    d_model: int

def estimate_flops_dense_only(params: int, total_tokens: int, c: float = 2.0) -> float:
    # FLOPs ≈ c * params * tokens
    return float(c) * float(params) * float(total_tokens)

def estimate_flops_attention_aware(P: int, G: int, shape: ModelShape) -> float:
    # Prefill: L(8 P d^2 + 2 P^2 d)
    # Decode:  L(8 G d^2 + 4 d (G P + G(G-1)/2))
    L, d = shape.layers, shape.d_model
    prefill = L * (8.0 * P * d * d + 2.0 * (P ** 2) * d)
    decode = L * (8.0 * G * d * d + 4.0 * d * (G * P + (G * (G - 1)) / 2.0))
    anchor = shape.params * (P + G)  # small calibration term
    return prefill + decode + anchor

def get_model_shape_from_config(cfg: AutoConfig) -> ModelShape:
    # Try common attribute names; fallback heuristics if missing.
    layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    d_model = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    n_params = getattr(cfg, "_name_or_path", None)  # dummy for mypy
    # We'll compute n_params via approximate formula if not available.
    # Better: use HF model.num_parameters() after load for accuracy.
    return ModelShape(params=0, layers=int(layers or 0), d_model=int(d_model or 0))

# ----------------------------
# Inference backends
# ----------------------------
def run_hf_generate(model_name: str, device: str, messages: List[Dict[str, str]],
                    max_new_tokens=128, temperature=0.7, top_p=0.9, seed=1234) -> Tuple[str, int, int]:
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    is_chat = hasattr(tokenizer, "apply_chat_template")
    if is_chat:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback: simple concatenation
        prompt = f"{SYS_PROMPT}\n\nUser: {messages[-1]['content']}\nAssistant:"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32, device_map="auto")
    model.eval()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    P = input_ids.shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
    )

    torch.cuda.synchronize() if "cuda" in device else None
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, **gen_kwargs)
    torch.cuda.synchronize() if "cuda" in device else None
    dt = time.perf_counter() - t0

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract only the new text after the prompt
    gen_text = text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
    G = tokenizer(gen_text, return_tensors="pt").input_ids.shape[1]
    return gen_text.strip(), P, G

def run_vllm_generate(model_name: str, messages: List[Dict[str, str]],
                      max_new_tokens=128, temperature=0.7, top_p=0.9, seed=1234) -> Tuple[str, int, int]:
    if not _VLLM_AVAILABLE:
        raise RuntimeError("vLLM not installed/available, choose --backend hf")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    is_chat = hasattr(tokenizer, "apply_chat_template")
    if is_chat:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{SYS_PROMPT}\n\nUser: {messages[-1]['content']}\nAssistant:"

    llm = LLM(model=model_name, dtype="half", gpu_memory_utilization=0.9)
    sampling = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, seed=seed
    )

    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sampling)
    dt = time.perf_counter() - t0
    out_text = outputs[0].outputs[0].text

    P = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
    G = tokenizer(out_text, return_tensors="pt").input_ids.shape[1]
    return out_text.strip(), P, G

# ----------------------------
# Evaluation
# ----------------------------
def evaluate_run(model_name: str, backend: str, device: str, max_new_tokens=128) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    cfg = AutoConfig.from_pretrained(model_name)
    shape = get_model_shape_from_config(cfg)

    # We will compute param count more accurately when using HF (post-load); for vLLM we approximate via config.
    params_count = None

    correct = 0
    total_P = 0
    total_G = 0
    times = []

    # Warmup/device sync helper for fair timing
    if backend == "hf":
        # Do a short warmup to trigger cudagraphs/kernels
        _ = run_hf_generate(model_name, device, build_messages("2+2?", ["A)3","B)4","C)5","D)6"]), max_new_tokens=16)
    else:
        _ = run_vllm_generate(model_name, build_messages("2+2?", ["A)3","B)4","C)5","D)6"]), max_new_tokens=16)

    for q, choices, gold in TOY_DATA:
        msgs = build_messages(q, choices)
        if backend == "hf":
            t0 = time.perf_counter()
            out_text, P, G = run_hf_generate(model_name, device, msgs, max_new_tokens=max_new_tokens)
            dt = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            out_text, P, G = run_vllm_generate(model_name, msgs, max_new_tokens=max_new_tokens)
            dt = time.perf_counter() - t0

        pred = extract_choice_letter(out_text) or ""  # handle None
        correct += int(pred == gold)
        total_P += P
        total_G += G
        times.append(dt)

    accuracy = correct / len(TOY_DATA)
    latency_s = sum(times)
    tok_per_s = (total_P + total_G) / latency_s if latency_s > 0 else 0.0

    # Accurately get params if possible by loading HF (quick lightweight way: model.num_parameters)
    if backend == "hf":
        # Load tiny headless model for counting already done; reuse from earlier generate is heavy to pass here.
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if "cuda" in device else torch.float32, device_map="auto")
        params_count = sum(p.numel() for p in model.parameters())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        # fallback: try from cfg if available; many configs expose 'vocab_size' etc but not params; set 0 if unknown.
        params_count = 0

    # FLOPs estimates (aggregate across questions)
    dense_flops = estimate_flops_dense_only(params_count or 0, total_P + total_G)
    attn_flops  = estimate_flops_attention_aware(total_P, total_G, ModelShape(params=params_count or 0, layers=shape.layers, d_model=shape.d_model))

    return {
        "n_examples": len(TOY_DATA),
        "correct": correct,
        "accuracy": accuracy,
        "prompt_tokens": total_P,
        "gen_tokens": total_G,
        "total_tokens": total_P + total_G,
        "latency_s": latency_s,
        "speed_tok_per_s": tok_per_s,
        "params": int(params_count or 0),
        "layers": shape.layers,
        "d_model": shape.d_model,
        "flops_dense": float(dense_flops),
        "flops_attn": float(attn_flops),
    }

# ----------------------------
# MLflow logging
# ----------------------------
def log_to_mlflow(model_name: str, backend: str, results: Dict, run_name: Optional[str] = None):
    mlflow.set_experiment("reasoning_pareto_mvp")
    with mlflow.start_run(run_name=run_name or f"{model_name}-{backend}-cot-toy"):
        mlflow.log_params({
            "model_name": model_name,
            "backend": backend,
            "reasoning_style": "cot_single",
            "use_kv_cache": True,
            "batch_size": 1,
        })
        # model/config params
        mlflow.log_params({
            "params": results["params"],
            "layers": results["layers"],
            "d_model": results["d_model"],
        })
        # metrics
        mlflow.log_metrics({
            "accuracy": results["accuracy"],
            "n_examples": results["n_examples"],
            "prompt_tokens": results["prompt_tokens"],
            "gen_tokens": results["gen_tokens"],
            "total_tokens": results["total_tokens"],
            "latency_s": results["latency_s"],
            "speed_tok_per_s": results["speed_tok_per_s"],
            "flops_dense": results["flops_dense"],
            "flops_attn": results["flops_attn"],
        })

        # Save raw results as artifact
        os.makedirs("artifacts", exist_ok=True)
        path = os.path.join("artifacts", f"{model_name.replace('/','_')}_{backend}_toy_results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        mlflow.log_artifact(path)

# ----------------------------
# Main
# ----------------------------
def main():
    # Setup logging first
    logger = setup_logging(name="mvp_benchmark")
    
    parser = argparse.ArgumentParser(description="MVP CoT evaluator with MLflow logging")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="HF model id (chat/instruct model recommended)")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"],
                        help="Inference backend: hf or vllm")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    logger.info(f"Starting MVP benchmark")
    logger.info(f"Model: {args.model}")
    logger.info(f"Backend: {args.backend}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")

    if args.backend == "vllm" and not _VLLM_AVAILABLE:
        logger.error("vLLM backend selected but vLLM is not available. Install vllm or use --backend hf.")
        raise SystemExit("vLLM backend selected but vLLM is not available. Install vllm or use --backend hf.")

    logger = get_logger()
    logger.info(f"Running MVP: model={args.model} backend={args.backend} device={args.device}")
    results = evaluate_run(args.model, args.backend, args.device, max_new_tokens=args.max_new_tokens)
    logger.info("Results: " + json.dumps(results, indent=2))

    # Log to MLflow
    log_to_mlflow(args.model, args.backend, results, run_name=f"{args.model}-{args.backend}-cot-toy")

if __name__ == "__main__":
    main()
