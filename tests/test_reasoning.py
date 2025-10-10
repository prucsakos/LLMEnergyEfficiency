from dataclasses import dataclass
from src.core.interfaces import GenerationParams, GenerationResult
from src.reasoning.controller import two_pass_batch, self_consistency_batch, self_evaluate_batched, single_pass_batch, extract_solution_from_trace
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
        elif "Judgement:" in prompt:
            # say YES if candidate equals gold
            if "<candidate>\n42" in prompt and "<gold>\n42" in prompt:
                return GenerationResult(text="YES", completion_tokens=1, latency_ms=1)
            else:
                return GenerationResult(text="NO", completion_tokens=1, latency_ms=1)
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
            elif "Judgement:" in prompt:
                # say YES if candidate equals gold
                if "<candidate>\n42" in prompt and "<gold>\n42" in prompt:
                    results.append(GenerationResult(text="YES", completion_tokens=1, latency_ms=1))
                else:
                    results.append(GenerationResult(text="NO", completion_tokens=1, latency_ms=1))
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

def test_single_pass_batch():
    eng = DummyEngine()
    gen = GenerationParams(max_new_tokens=16)
    results = single_pass_batch(eng, ["What is 2+2?", "What is 3*3?"], gen, think_budget=8)
    
    # Check first result
    out1 = results[0]
    assert out1["answer_text"] == "<final>42</final>"  # Extracted solution (short text, so full text returned)
    assert out1["full_answer_text"] == "<final>42</final>"  # Full text preserved
    assert out1["answer_tokens"] == 2
    assert "latency_ms" in out1
    assert "raw" in out1
    
    # Check second result
    out2 = results[1]
    assert out2["answer_text"] == "<final>42</final>"  # Extracted solution (short text, so full text returned)
    assert out2["full_answer_text"] == "<final>42</final>"  # Full text preserved
    assert out2["answer_tokens"] == 2
    assert "latency_ms" in out2
    assert "raw" in out2
    
    # Verify the engine was called with the questions directly
    assert "What is 2+2?" in eng.calls
    assert "What is 3*3?" in eng.calls

def test_extract_solution_from_trace():
    # Test case 1: Boxed answer present
    trace1 = "Let me think about this. First I need to calculate 2+2. That equals 4. So the answer is \\boxed{4}."
    result1 = extract_solution_from_trace(trace1)
    assert result1 == "4"
    
    # Test case 2: Multiple boxed answers - should return last one
    trace2 = "I think the answer might be \\boxed{3} but actually \\boxed{5} is correct."
    result2 = extract_solution_from_trace(trace2)
    assert result2 == "5"
    
    # Test case 3: No boxed answer - should return last 50 tokens
    trace3 = "This is a long trace without any boxed answers. " * 10 + "The final answer is 42."
    result3 = extract_solution_from_trace(trace3)
    assert "42" in result3
    assert len(result3.split()) <= 50
    
    # Test case 4: Short trace without boxed answer
    trace4 = "The answer is 42."
    result4 = extract_solution_from_trace(trace4)
    assert result4 == "The answer is 42."
    
    # Test case 5: Empty trace
    result5 = extract_solution_from_trace("")
    assert result5 == ""
    
    # Test case 6: Whitespace only
    result6 = extract_solution_from_trace("   \n  \t  ")
    assert result6 == ""
