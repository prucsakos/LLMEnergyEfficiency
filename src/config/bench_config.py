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
    
    # Quantization settings
    quantization: Optional[str] = None  # "awq", "gptq", "squeezellm", "fp8", "marlin", "exl2", "gptq_marlin", "aqlm", "gguf"
    quantization_param_path: Optional[str] = None  # Path to quantization parameters
    
    # CPU offloading settings
    cpu_offload_gb: Optional[float] = None  # Amount of model to offload to CPU in GB
    swap_space: Optional[int] = None  # Swap space in GB for CPU offloading
    
    # Additional memory optimization
    max_model_len: Optional[int] = None  # Maximum sequence length
    block_size: Optional[int] = None  # Block size for PagedAttention

@dataclass
class GenDefaults:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    do_sample: Optional[bool] = None  # If None, auto-determined by temperature > 0
    max_new_tokens: int = 128
    stop: Optional[List[str]] = None
    seed: Optional[int] = None
    use_kv_cache: bool = True

@dataclass
class ReasoningDefaults:
    style: str = "cot"          # none | cot | plan_solve
    self_consistency_k: int = 1
    self_eval: bool = True      # model judges YES/NO after answering (always true)
    openai_eval: bool = False   # use OpenAI API for evaluation instead of model

@dataclass
class Prompts:
    # Default prompt templates; can be overridden per model in YAML
    cot_think: str = (
        "Solve the problem. Write reasoning ONLY inside <think>...</think>.\n"
        "<question>\n{question}\n</question>\n\n<think>"
    )
    plan_think: str = (
        "Devise a brief plan. Write it ONLY inside <think>...</think>.\n"
        "<question>\n{question}\n</question>\n\n<think>"
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
        "Judgement: "
    )
    direct: str = (
        "Answer the question. Output ONLY the final answer inside <final>...</final>.\n"
        "<question>\n{question}\n</question>\n<final>"
    )
    consistency_eval: str = (
        "You are a majority vote counter. Given a question and multiple candidate answers, "
        "identify which answer appears most frequently and select that one.\n\n"
        "<question>\n{question}\n</question>\n"
        "<candidate_answers>\n{candidate_answers}\n</candidate_answers>\n\n"
        "Count the frequency of each answer and select the most popular one by outputting ONLY the chosen answer inside <chosen>...</chosen>.\n"
        "<chosen>"
    )
    
    def format_prompts(self):
        """Format prompt templates - no longer needed since we use actual tags directly."""
        return self

@dataclass
class ModelSpec:
    name: str
    hf_repo: str
    card: Card
    think_budgets: List[int]
    model_family: str = "unknown"  # Model family (e.g., "llama", "gemma", "phi", "qwen")
    engine: str = "vllm"  # 'vllm' | 'transformers'
    batch_size: int = 1
    backend: BackendDefaults = field(default_factory=BackendDefaults)
    generation: GenDefaults = field(default_factory=GenDefaults)
    reasoning: ReasoningDefaults = field(default_factory=ReasoningDefaults)
    prompts_override: Dict[str, str] = field(default_factory=dict)

@dataclass
class BenchConfig:
    models: List[ModelSpec]
    datasets: List[str] = field(default_factory=list)
    prompts: Prompts = field(default_factory=Prompts)
    prompt_sets: List[Dict[str, str]] = field(default_factory=list)  # List of prompt set dictionaries
    config_name: str = "default"  # Global config name for the entire run

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
    prompts = prompts.format_prompts()  # Format prompts with tag values
    datasets = list(data.get("datasets", []))
    config_name = data.get("config_name", "default")  # Read global config name
    
    # Handle multiple prompt sets
    prompt_sets = data.get("prompt_sets", [])
    if not prompt_sets:
        # If no prompt_sets specified, use the default prompts as a single set
        prompt_sets = [data.get("prompts", {})]
    
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
            model_family=m.get("model_family", "unknown"),
            engine=m.get("engine", "vllm"),
            batch_size=m.get("batch_size", 1),
            backend=backend,
            generation=gen,
            reasoning=reason,
            prompts_override=m.get("prompts_override", {}),
        )
        models.append(spec)
    return BenchConfig(models=models, datasets=datasets, prompts=prompts, prompt_sets=prompt_sets, config_name=config_name)

@dataclass
class RunSpec:
    model_name: str
    hf_repo: str
    card: Card
    model_family: str
    engine: str
    dataset: str
    think_budget: int
    batch_size: int
    backend: BackendDefaults
    generation: GenDefaults
    reasoning: ReasoningDefaults
    prompts: Prompts
    prompt_set_name: str = "default"
    config_name: str = "default"

def expand_runs(cfg: BenchConfig) -> Iterable[RunSpec]:
    """
    Note: yield is actually perfect for grid search
    Now also iterates over prompt sets
    """
    for m in cfg.models:
        for prompt_set_idx, prompt_set in enumerate(cfg.prompt_sets):
            # Create prompts object from this prompt set
            prompts = _dict_to_dataclass(Prompts, prompt_set)
            
            # merge with per-model overrides
            for k, v in m.prompts_override.items():
                setattr(prompts, k, v)
            
            # Format prompts with tag values
            prompts = prompts.format_prompts()
            
            # Generate prompt set name
            prompt_set_name = prompt_set.get("name", f"prompt_set_{prompt_set_idx}")
            
            for dataset, budget in itertools.product(cfg.datasets, m.think_budgets):
                yield RunSpec(
                    model_name=m.name,
                    hf_repo=m.hf_repo,
                    card=m.card,
                    model_family=m.model_family,
                    engine=m.engine,
                    dataset=dataset,
                    think_budget=budget,
                    batch_size=m.batch_size,
                    backend=m.backend,
                    generation=m.generation,
                    reasoning=m.reasoning,
                    prompts=prompts,
                    prompt_set_name=prompt_set_name,
                    config_name=cfg.config_name,  # Use global config name
                )
