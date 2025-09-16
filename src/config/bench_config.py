from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable
import itertools, copy, yaml, pathlib

@dataclass
class Card:
    params_B: float
    layers: Optional[int] = None
    hidden_dim: Optional[int] = None
    heads: Optional[int] = None
    arch: str = None

@dataclass
class BackendDefaults:
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = True  # safer with some torch stacks; can disable later

@dataclass
class GenDefaults:
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 128
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    use_kv_cache: bool = True

@dataclass
class ReasoningDefaults:
    style: str = "cot"          # none | cot | plan_solve
    self_consistency_k: int = 1
    self_eval: bool = False     # model judges YES/NO after answering

@dataclass
class Prompts:
    # Default prompt templates; can be overridden per model in YAML
    cot_think: str = (
        "Solve the problem. Write reasoning ONLY inside <scratchpad>...</scratchpad>.\n"
        "<question>\n{question}\n</question>\n\n<scratchpad>"
    )
    plan_think: str = (
        "Devise a brief plan. Write it ONLY inside <plan>...</plan>.\n"
        "<question>\n{question}\n</question>\n\n<plan>"
    )
    answer: str = (
        "Using the information below, produce ONLY the final answer inside <final>...</final>.\n"
        "<question>\n{question}\n</question>\n"
        "{deliberate}\n\n<final>"
    )
    self_eval: str = (
        "You are a strict judge. Given the question, gold answer, and a candidate answer,\n"
        "reply with ONLY 'YES' if the candidate is correct, otherwise 'NO'.\n\n"
        "<question>\n{question}\n</question>\n"
        "<gold>\n{gold}\n</gold>\n"
        "<candidate>\n{candidate}\n</candidate>\n"
        "Judgement:"
    )
    direct: str = (
        "Answer the question. Output ONLY the final answer inside <final>...</final>.\n"
        "<question>\n{question}\n</question>\n<final>"
    )

@dataclass
class ModelSpec:
    name: str
    hf_repo: str
    card: Card
    think_budgets: List[int]
    datasets: List[str]
    batch_size: int = 1
    backend: BackendDefaults = field(default_factory=BackendDefaults)
    generation: GenDefaults = field(default_factory=GenDefaults)
    reasoning: ReasoningDefaults = field(default_factory=ReasoningDefaults)
    prompts_override: Dict[str, str] = field(default_factory=dict)

@dataclass
class BenchConfig:
    models: List[ModelSpec]
    prompts: Prompts = field(default_factory=Prompts)

def _dict_to_dataclass(cls, d):
    # simple helper for nested dataclasses
    fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
    init = {}
    for k, v in d.items():
        if k in fields:
            init[k] = v
    return cls(**init)

def load_bench_config(path: str | pathlib.Path) -> BenchConfig:
    data = yaml.safe_load(open(path, "r", encoding="utf-8"))
    prompts = _dict_to_dataclass(Prompts, data.get("prompts", {}))
    models = []
    for m in data["models"]:
        card = _dict_to_dataclass(Card, m["card"])
        backend = _dict_to_dataclass(BackendDefaults, m.get("backend", {}))
        gen = _dict_to_dataclass(GenDefaults, m.get("generation", {}))
        reason = _dict_to_dataclass(ReasoningDefaults, m.get("reasoning", {}))
        spec = ModelSpec(
            name=m["name"],
            hf_repo=m["hf_repo"],
            card=card,
            think_budgets=m["think_budgets"],
            datasets=m["datasets"],
            batch_size=m.get("batch_size", 1),
            backend=backend,
            generation=gen,
            reasoning=reason,
            prompts_override=m.get("prompts_override", {}),
        )
        models.append(spec)
    return BenchConfig(models=models, prompts=prompts)

@dataclass
class RunSpec:
    model_name: str
    hf_repo: str
    card: Card
    dataset: str
    think_budget: int
    batch_size: int
    backend: BackendDefaults
    generation: GenDefaults
    reasoning: ReasoningDefaults
    prompts: Prompts

def expand_runs(cfg: BenchConfig) -> Iterable[RunSpec]:
    """
    Note: yield is actually perfect for grid search
    """
    for m in cfg.models:
        # merge default prompts with per-model overrides
        prompts = copy.deepcopy(cfg.prompts)
        for k, v in m.prompts_override.items():
            setattr(prompts, k, v)
        for dataset, budget in itertools.product(m.datasets, m.think_budgets):
            yield RunSpec(
                model_name=m.name,
                hf_repo=m.hf_repo,
                card=m.card,
                dataset=dataset,
                think_budget=budget,
                batch_size=m.batch_size,
                backend=m.backend,
                generation=m.generation,
                reasoning=m.reasoning,
                prompts=prompts,
            )
