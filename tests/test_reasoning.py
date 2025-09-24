from dataclasses import dataclass
from src.core.interfaces import GenerationParams, GenerationResult
from src.reasoning.controller import two_pass_batch, self_consistency_batch, self_evaluate_batched
from src.config.bench_config import Prompts

# A tiny dummy engine to avoid vLLM in unit tests
class DummyEngine:
    def __init__(self): self.calls=[]
    def generate(self, prompt, params):
        self.calls.append(prompt)
        if "<think>" in prompt and "<final>" not in prompt:
            return GenerationResult(text="<think>think</think>", completion_tokens=5, latency_ms=10)
        if "Judgement:" in prompt:
            # say YES if candidate equals gold
            if "<candidate>\n42" in prompt and "<gold>\n42" in prompt:
                return GenerationResult(text="YES", completion_tokens=1, latency_ms=1)
            return GenerationResult(text="NO", completion_tokens=1, latency_ms=1)
        return GenerationResult(text="<final>42</final>", completion_tokens=2, latency_ms=5)
    
    def generate(self, prompt, params):
        self.calls.append(prompt)
        if "<think>" in prompt and "<final>" not in prompt:
            return GenerationResult(text="<think>think</think>", completion_tokens=5, latency_ms=10)
        elif "<judgement>" in prompt:
            # say YES if candidate equals gold
            if "<candidate>\n42" in prompt and "<gold>\n42" in prompt:
                return GenerationResult(text="<judgement>YES</judgement>", completion_tokens=3, latency_ms=1)
            else:
                return GenerationResult(text="<judgement>NO</judgement>", completion_tokens=3, latency_ms=1)
        elif "majority vote counter" in prompt:
            # Return the chosen answer for consistency evaluation
            return GenerationResult(text="<chosen>42</chosen>", completion_tokens=3, latency_ms=5)
        else:
            return GenerationResult(text="<final>42</final>", completion_tokens=2, latency_ms=5)
    
    def generate_batch(self, prompts, params):
        self.calls.extend(prompts)
        results = []
        for prompt in prompts:
            if "<think>" in prompt and "<final>" not in prompt:
                results.append(GenerationResult(text="<think>think</think>", completion_tokens=5, latency_ms=10))
            elif "<judgement>" in prompt:
                # say YES if candidate equals gold
                if "<candidate>\n42" in prompt and "<gold>\n42" in prompt:
                    results.append(GenerationResult(text="<judgement>YES</judgement>", completion_tokens=3, latency_ms=1))
                else:
                    results.append(GenerationResult(text="<judgement>NO</judgement>", completion_tokens=3, latency_ms=1))
            elif "majority vote counter" in prompt:
                # Return the chosen answer for consistency evaluation
                results.append(GenerationResult(text="<chosen>42</chosen>", completion_tokens=3, latency_ms=5))
            else:
                results.append(GenerationResult(text="<final>42</final>", completion_tokens=2, latency_ms=5))
        return results

def test_two_pass_cot():
    eng = DummyEngine()
    gen = GenerationParams(max_new_tokens=16)
    prompts = Prompts().format_prompts()
    results = two_pass_batch(eng, ["Q"], gen, think_budget=8, style="cot", prompts=prompts)
    out = results[0]  # Get first (and only) result
    assert out["think_text"] == "think"
    assert out["answer_text"] == "42"
    assert out["think_tokens"] == 5
    assert out["answer_tokens"] == 2

def test_self_consistency_majority():
    eng = DummyEngine()
    gen = GenerationParams()
    prompts = Prompts().format_prompts()
    results = self_consistency_batch(eng, ["Q"], gen, think_budget=8, style="cot", prompts=prompts, k=3)
    res = results[0]  # Get first (and only) result
    assert res["chosen_answer"] == "42"
    assert len(res["paths"]) == 3

def test_self_eval_yes_no():
    eng = DummyEngine()
    gen = GenerationParams()
    prompts = Prompts().format_prompts()
    results = self_evaluate_batched(eng, ["Q", "Q"], ["42", "1"], ["42", "42"], gen, prompts)
    ok, _, _ = results[0]  # First evaluation (should be True)
    bad, _, _ = results[1]  # Second evaluation (should be False)
    assert ok is True and bad is False
