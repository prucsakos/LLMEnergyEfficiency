from dataclasses import dataclass
from src.core.interfaces import GenerationParams, GenerationResult
from src.reasoning.controller import two_pass, self_consistency, self_evaluate
from src.config.bench_config import Prompts

# A tiny dummy engine to avoid vLLM in unit tests
class DummyEngine:
    def __init__(self): self.calls=[]
    def generate(self, prompt, params):
        self.calls.append(prompt)
        if "<scratchpad>" in prompt:
            return GenerationResult(text="<scratchpad>think</scratchpad>", completion_tokens=5, latency_ms=10)
        if "<plan>" in prompt:
            return GenerationResult(text="<plan>plan</plan>", completion_tokens=4, latency_ms=8)
        if "Judgement:" in prompt:
            # say YES if candidate equals gold
            if "<candidate>\n42" in prompt and "<gold>\n42" in prompt:
                return GenerationResult(text="YES", completion_tokens=1, latency_ms=1)
            return GenerationResult(text="NO", completion_tokens=1, latency_ms=1)
        return GenerationResult(text="<final>42</final>", completion_tokens=2, latency_ms=5)

def test_two_pass_cot():
    eng = DummyEngine()
    gen = GenerationParams(max_new_tokens=16)
    out = two_pass(eng, "Q", gen, think_budget=8, style="cot", prompts=Prompts())
    assert out["think_text"] == "think"
    assert out["answer_text"] == "42"
    assert out["think_tokens"] == 5
    assert out["answer_tokens"] == 2

def test_self_consistency_majority():
    eng = DummyEngine()
    gen = GenerationParams()
    res = self_consistency(eng, "Q", gen, think_budget=8, style="cot", prompts=Prompts(), k=3)
    assert res["chosen_answer"] == "42"
    assert len(res["paths"]) == 3

def test_self_eval_yes_no():
    eng = DummyEngine()
    gen = GenerationParams()
    ok = self_evaluate(eng, "Q", candidate="42", gold="42", gen=gen, prompts=Prompts())
    bad = self_evaluate(eng, "Q", candidate="1", gold="42", gen=gen, prompts=Prompts())
    assert ok is True and bad is False
