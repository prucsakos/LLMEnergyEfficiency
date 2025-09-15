from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any, Optional
from datasets import load_dataset
from .normalize import normalize_freeform, extract_gsm8k_final

@dataclass
class Sample:
    id: str
    question: str
    gold: str
    choices: Optional[List[str]] = None  # for MCQ datasets
    meta: Optional[Dict[str, Any]] = None

def iter_dataset(name: str, split: str = "test"):
    if name == "gsm8k": return load_gsm8k(split)
    if name == "mmlu":  return load_mmlu(split=split)
    if name == "csqa":  return load_csqa(split="validation" if split == "test" else split)
    raise ValueError(f"Unknown dataset {name}")

# ---------- GSM8K (free-form numeric/text answers) ----------
def load_gsm8k(split: str = "test") -> Iterable[Sample]:
    """Yield GSM8K samples. Fields: 'question', 'answer'. :contentReference[oaicite:14]{index=14}"""
    ds = load_dataset("openai/gsm8k", "main")[split]
    for i, row in enumerate(ds):
        q = row["question"]
        gold = extract_gsm8k_final(row["answer"])
        yield Sample(id=f"gsm8k-{split}-{i}", question=q, gold=normalize_freeform(gold))

# ---------- MMLU (MCQ over 57 subjects) ----------
def load_mmlu(subjects: Optional[List[str]] = None, split: str = "test") -> Iterable[Sample]:
    """Yield MMLU samples. Fields include 'question', 'choices', 'answer' (label). :contentReference[oaicite:15]{index=15}"""
    ds = load_dataset("cais/mmlu", "all")[split]
    subj_filter = set(subjects) if subjects else None
    for i, row in enumerate(ds):
        if subj_filter and row.get("subject") not in subj_filter:
            continue
        q = row["question"]
        choices = list(row["choices"]) if isinstance(row["choices"], (list, tuple)) else row["choices"]
        gold_label = row["answer"]  # typically an integer index (0..n-1) or char label in some configs
        if isinstance(gold_label, int):
            gold = choices[gold_label]
        else:
            # map 'A'.. to index
            idx = ord(str(gold_label).strip().upper()) - ord('A')
            gold = choices[idx]
        yield Sample(id=f"mmlu-{split}-{i}", question=q, gold=normalize_freeform(gold), choices=choices)

# ---------- CommonSenseQA (MCQ) ----------
def load_csqa(split: str = "validation") -> Iterable[Sample]:
    """Yield CSQA samples. Fields: 'question', 'choices.text', 'answerKey'. :contentReference[oaicite:16]{index=16}"""
    ds = load_dataset("tau/commonsense_qa")[split]
    for i, row in enumerate(ds):
        q = row["question"]
        # structure: {'label': [...], 'text': [...]}
        texts = row["choices"]["text"]
        labels = row["choices"]["label"]
        gold_key = row["answerKey"].strip().upper()
        # Map key 'A'.. to text
        idx = labels.index(gold_key)
        gold = texts[idx]
        yield Sample(id=f"csqa-{split}-{i}", question=q, gold=normalize_freeform(gold), choices=list(texts))

# ---------- Accuracy helpers ----------
def exact_match(pred: str, gold: str) -> bool:
    return normalize_freeform(pred) == normalize_freeform(gold)
