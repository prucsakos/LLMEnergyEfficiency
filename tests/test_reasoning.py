from dataclasses import dataclass
from src.core.interfaces import GenerationParams, GenerationResult
from src.reasoning.controller import two_pass_batch, self_consistency_batch, self_evaluate_batched, single_pass_batch, extract_solution_from_trace
from src.config.bench_config import Prompts

# A tiny dummy engine to avoid vLLM in unit tests
class DummyEngine:
    def __init__(self): self.calls=[]
    def generate(self, prompt, params):
        self.calls.append(prompt)
        if "Think through this problem" in prompt and "Do not provide the final answer" in prompt:
            return GenerationResult(text="think", completion_tokens=5, latency_ms=10)
        elif "You are an expert mathematics evaluator" in prompt:
            # say CORRECT if candidate equals gold
            if "Student answer: 42" in prompt and "Gold standard: 42" in prompt:
                return GenerationResult(text="CORRECT", completion_tokens=1, latency_ms=1)
            else:
                return GenerationResult(text="INCORRECT", completion_tokens=1, latency_ms=1)
        elif "majority vote counter" in prompt:
            # Return the chosen answer for consistency evaluation
            return GenerationResult(text="42", completion_tokens=3, latency_ms=5)
        else:
            return GenerationResult(text="42", completion_tokens=2, latency_ms=5)
    
    def generate_batch(self, prompts, params):
        self.calls.extend(prompts)
        results = []
        for prompt in prompts:
            if "Think through this problem" in prompt and "Do not provide the final answer" in prompt:
                results.append(GenerationResult(text="think", completion_tokens=5, latency_ms=10))
            elif "You are an expert mathematics evaluator" in prompt:
                # say CORRECT if candidate equals gold
                if "Student answer: 42" in prompt and "Gold standard: 42" in prompt:
                    results.append(GenerationResult(text="CORRECT", completion_tokens=1, latency_ms=1))
                else:
                    results.append(GenerationResult(text="INCORRECT", completion_tokens=1, latency_ms=1))
            elif "majority vote counter" in prompt:
                # Return the chosen answer for consistency evaluation
                results.append(GenerationResult(text="42", completion_tokens=3, latency_ms=5))
            else:
                results.append(GenerationResult(text="42", completion_tokens=2, latency_ms=5))
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
    assert out1["answer_text"] == "42"  # Extracted solution (short text, so full text returned)
    assert out1["full_answer_text"] == "42"  # Full text preserved
    assert out1["answer_tokens"] == 2
    assert "latency_ms" in out1
    assert "raw" in out1
    
    # Check second result
    out2 = results[1]
    assert out2["answer_text"] == "42"  # Extracted solution (short text, so full text returned)
    assert out2["full_answer_text"] == "42"  # Full text preserved
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
    
    # Test case 3: No boxed answer - should return last 25 words (not 50)
    trace3 = "This is a long trace without any boxed answers. " * 10 + "The final answer is 42."
    result3 = extract_solution_from_trace(trace3)
    assert "42" in result3
    assert len(result3.split()) <= 25
    
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


def test_extract_solution_from_trace_complex_nested():
    """Test complex nested bracket scenarios that the new depth-tracking method handles"""
    
    # Test case 1: Simple nested brackets
    trace1 = "The solution is \\boxed{2 + 3 = {5}}."
    result1 = extract_solution_from_trace(trace1)
    assert result1 == "2 + 3 = {5}"
    
    # Test case 2: Multiple levels of nesting
    trace2 = "The result is \\boxed{{a + b} * {c + d} = {result}}."
    result2 = extract_solution_from_trace(trace2)
    assert result2 == "{a + b} * {c + d} = {result}"
    
    # Test case 3: Complex nested with multiple boxed (should get last one)
    trace3 = "Wrong: \\boxed{2 + {3}} but right: \\boxed{{1 + 2} * {3 + 4} = 21}."
    result3 = extract_solution_from_trace(trace3)
    assert result3 == "{1 + 2} * {3 + 4} = 21"
    
    # Test case 4: Very complex nesting with sets and dictionaries
    trace4 = "The answer is \\boxed{{{x: {1, 2, 3}}, {y: {4, 5, 6}}} = {result}}."
    result4 = extract_solution_from_trace(trace4)
    assert result4 == "{{x: {1, 2, 3}}, {y: {4, 5, 6}}} = {result}"
    
    # Test case 5: fbox instead of boxed
    trace5 = "The solution is \\fbox{42}."
    result5 = extract_solution_from_trace(trace5)
    assert result5 == "42"
    
    # Test case 6: fbox with nested brackets
    trace6 = "The result is \\fbox{{a} + {b} = {c}}."
    result6 = extract_solution_from_trace(trace6)
    assert result6 == "{a} + {b} = {c}"
    
    # Test case 7: Edge case - unmatched brackets (should fallback)
    trace7 = "The answer is \\boxed{2 + 3 = 5."
    result7 = extract_solution_from_trace(trace7)
    assert "2 + 3 = 5" in result7
    
    # Test case 8: Multiple fbox and boxed mixed (should get the last boxed, not fbox)
    trace8 = "First: \\fbox{3} then \\boxed{7} and finally \\fbox{{1 + 2} = 3}."
    result8 = extract_solution_from_trace(trace8)
    assert result8 == "7"  # Gets the last \boxed, not the last \fbox
    
    # Test case 9: Deep nesting (5+ levels)
    trace9 = "The answer is \\boxed{{{a: {b: {c: {d: {e: 5}}}}}}}."
    result9 = extract_solution_from_trace(trace9)
    assert result9 == "{{a: {b: {c: {d: {e: 5}}}}}}"
    
    # Test case 10: Mathematical expressions with nested brackets
    trace10 = "The solution is \\boxed{\\frac{{a + b}}{{c + d}} = \\frac{{{1 + 2}}}{{{3 + 4}}} = \\frac{3}{7}}."
    result10 = extract_solution_from_trace(trace10)
    assert result10 == "\\frac{{a + b}}{{c + d}} = \\frac{{{1 + 2}}}{{{3 + 4}}} = \\frac{3}{7}"


def test_last_boxed_only_string_function():
    """Test the last_boxed_only_string function directly"""
    from src.reasoning.controller import last_boxed_only_string
    
    # Test case 1: Simple case
    test1 = "The answer is \\boxed{42}."
    result1 = last_boxed_only_string(test1)
    assert result1 == "\\boxed{42}"
    
    # Test case 2: Nested brackets
    test2 = "The solution is \\boxed{2 + 3 = {5}}."
    result2 = last_boxed_only_string(test2)
    assert result2 == "\\boxed{2 + 3 = {5}}"
    
    # Test case 3: Multiple boxed
    test3 = "First: \\boxed{3} but \\boxed{7} is correct."
    result3 = last_boxed_only_string(test3)
    assert result3 == "\\boxed{7}"
    
    # Test case 4: fbox conversion
    test4 = "The result is \\fbox{42}."
    result4 = last_boxed_only_string(test4)
    assert result4 == "\\boxed{42}"
    
    # Test case 5: Complex nested fbox
    test5 = "The answer is \\fbox{{a} + {b} = {c}}."
    result5 = last_boxed_only_string(test5)
    assert result5 == "\\boxed{{a} + {b} = {c}}"
    
    # Test case 6: No boxed found
    test6 = "No boxed answer here."
    result6 = last_boxed_only_string(test6)
    assert result6 is None
    
    # Test case 7: Unmatched brackets
    test7 = "The answer is \\boxed{2 + 3 = 5."
    result7 = last_boxed_only_string(test7)
    assert result7 is None


def test_remove_boxed_function():
    """Test the remove_boxed function"""
    from src.reasoning.controller import remove_boxed
    
    # Test case 1: Normal boxed
    test1 = "\\boxed{42}"
    result1 = remove_boxed(test1)
    assert result1 == "42"
    
    # Test case 2: Boxed with space
    test2 = "\\boxed 42"
    result2 = remove_boxed(test2)
    assert result2 == "42"
    
    # Test case 3: Nested brackets
    test3 = "\\boxed{2 + 3 = {5}}"
    result3 = remove_boxed(test3)
    assert result3 == "2 + 3 = {5}"
    
    # Test case 4: Malformed (no closing brace)
    test4 = "\\boxed{42"
    result4 = remove_boxed(test4)
    assert result4 == "\\boxed{42"
    
    # Test case 5: Not boxed format
    test5 = "just some text"
    result5 = remove_boxed(test5)
    assert result5 == "just some text"
