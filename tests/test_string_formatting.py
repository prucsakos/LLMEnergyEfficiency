#!/usr/bin/env python3
"""
Test script to demonstrate how {{}} and {} work with Python's string.format() method.

This script helps understand the formatting issue that was causing the IndexError
in the benchmark system when using \boxed{{}} in prompt templates.
"""

# import pytest  # Not needed for basic testing


def test_single_braces_formatting():
    """Test basic string formatting with single braces {}."""
    template = "Hello {name}, you have {count} messages."
    result = template.format(name="Alice", count=5)
    assert result == "Hello Alice, you have 5 messages."
    print(f"✅ Single braces: {result}")


def test_double_braces_literal():
    """Test that double braces {{}} become literal braces {}."""
    template = "The answer is {{}}"
    result = template.format()
    assert result == "The answer is {}"
    print(f"✅ Double braces literal: {result}")


def test_mixed_braces_formatting():
    """Test mixing format placeholders {} with literal braces {{}}."""
    template = "Question: {question}\nAnswer: {{}}"
    result = template.format(question="What is 2+2?")
    assert result == "Question: What is 2+2?\nAnswer: {}"
    print(f"✅ Mixed braces: {result}")


def test_boxed_formatting_correct():
    """Test the correct way to format \boxed{} in prompts."""
    # The issue is that \boxed{} contains a single brace which Python interprets as a format placeholder
    # We need to escape it properly
    template = "What is the final answer? Provide only the final answer in \\boxed{{}}, followed by a brief explanation.\n\n{question}\n\n{deliberate}\n\nFinal answer is"
    
    result = template.format(
        question="Solve for x: 2x + 6 = 14",
        deliberate="Let me think step by step..."
    )
    
    expected = "What is the final answer? Provide only the final answer in \\boxed{}, followed by a brief explanation.\n\nSolve for x: 2x + 6 = 14\n\nLet me think step by step...\n\nFinal answer is"
    assert result == expected
    print(f"✅ Correct \\boxed{{}} formatting: {result}")


def test_boxed_formatting_problematic():
    """Test the problematic way that causes IndexError."""
    # This is the problematic template that causes IndexError
    template = "What is the final answer? Provide only the final answer in \\boxed{}, followed by a brief explanation.\n\n{question}\n\n{deliberate}\n\nFinal answer is"
    
    # This should raise IndexError because \boxed{} has unmatched braces
    try:
        result = template.format(
            question="Solve for x: 2x + 6 = 14",
            deliberate="Let me think step by step..."
        )
        print(f"❌ Unexpected success: {result}")
    except IndexError as e:
        print(f"✅ Correctly caught IndexError: {e}")
    
    # The fix is to use double braces to escape the literal braces
    fixed_template = "What is the final answer? Provide only the final answer in \\boxed{{}}, followed by a brief explanation.\n\n{question}\n\n{deliberate}\n\nFinal answer is"
    
    result = fixed_template.format(
        question="Solve for x: 2x + 6 = 14",
        deliberate="Let me think step by step..."
    )
    
    expected = "What is the final answer? Provide only the final answer in \\boxed{}, followed by a brief explanation.\n\nSolve for x: 2x + 6 = 14\n\nLet me think step by step...\n\nFinal answer is"
    assert result == expected
    print(f"✅ Fixed \\boxed{{{{}}}} formatting: {result}")


def test_formatting_with_missing_placeholders():
    """Test what happens when format placeholders don't match arguments."""
    template = "Hello {name}, you have {count} messages."
    
    # This should work
    result1 = template.format(name="Alice", count=5)
    assert result1 == "Hello Alice, you have 5 messages."
    
    # This should raise KeyError
    try:
        template.format(name="Alice")  # Missing 'count'
        print("❌ Expected KeyError but didn't get one")
    except KeyError:
        print("✅ Correctly raised KeyError for missing 'count'")
    
    # This should raise KeyError  
    try:
        template.format(count=5)  # Missing 'name'
        print("❌ Expected KeyError but didn't get one")
    except KeyError:
        print("✅ Correctly raised KeyError for missing 'name'")
    
    print("✅ Missing placeholder tests passed")


def test_complex_formatting_scenarios():
    """Test various complex formatting scenarios."""
    
    # Scenario 1: Multiple placeholders
    template1 = "Model: {model}, Dataset: {dataset}, Accuracy: {accuracy:.2f}"
    result1 = template1.format(model="GPT-4", dataset="MMLU", accuracy=0.85)
    assert result1 == "Model: GPT-4, Dataset: MMLU, Accuracy: 0.85"
    print(f"✅ Complex formatting 1: {result1}")
    
    # Scenario 2: Nested braces in math expressions
    template2 = "The solution is \\boxed{{{answer}}} where {answer} = {value}"
    result2 = template2.format(answer="x", value=4)
    assert result2 == "The solution is \\boxed{x} where x = 4"
    print(f"✅ Complex formatting 2: {result2}")
    
    # Scenario 3: Empty deliberate content
    template3 = "Question: {question}\n\n{deliberate}\n\nAnswer:"
    result3 = template3.format(question="What is 1+1?", deliberate="")
    assert result3 == "Question: What is 1+1?\n\n\n\nAnswer:"
    print(f"✅ Complex formatting 3: {result3}")


def test_prompt_template_simulation():
    """Simulate the actual prompt template scenario from the benchmark."""
    
    # This simulates the prompts.answer template from hypothesis_main.yaml
    # The original had \boxed{} which causes IndexError, so we use \boxed{{}} to escape it
    answer_template = """What is the final answer? Provide only the final answer in \\boxed{{}}, followed by a brief explanation only after.

{question}

{deliberate}

Final answer is"""
    
    # Test with thinking content
    result_with_thinking = answer_template.format(
        question="Find the derivative of x^2",
        deliberate="Let me think step by step:\n1. The function is f(x) = x^2\n2. Using power rule: f'(x) = 2x"
    )
    
    print("✅ With thinking content:")
    print(result_with_thinking)
    print()
    
    # Test with empty thinking content
    result_empty_thinking = answer_template.format(
        question="Find the derivative of x^2", 
        deliberate=""
    )
    
    print("✅ With empty thinking content:")
    print(result_empty_thinking)
    print()
    
    # Test the problematic version that causes IndexError
    problematic_template = """What is the final answer? Provide only the final answer in \\boxed{}, followed by a brief explanation only after.

{question}

{deliberate}

Final answer is"""
    
    try:
        result_problematic = problematic_template.format(
            question="Find the derivative of x^2",
            deliberate="Let me think step by step..."
        )
        print("❌ Problematic version unexpectedly worked:")
        print(result_problematic)
    except IndexError as e:
        print(f"✅ Problematic version correctly failed with IndexError: {e}")
    print()


def test_benchmark_controller_simulation():
    """Simulate the exact scenario from the benchmark controller."""
    
    class MockPrompts:
        def __init__(self):
            # This is the template that was causing issues - using double braces to escape
            self.answer = """What is the final answer? Provide only the final answer in \\boxed{{}}, followed by a brief explanation only after.

{question}

{deliberate}

Final answer is"""
    
    prompts = MockPrompts()
    questions = ["What is 2+2?", "What is 3*4?"]
    deliberate_blocks = ["Let me think... 2+2=4", ""]
    
    answer_prompts = []
    for q, deliberate in zip(questions, deliberate_blocks):
        if deliberate.strip():
            # This line was causing the IndexError
            answer_prompts.append(prompts.answer.format(question=q, deliberate=deliberate))
        else:
            # Remove the deliberate placeholder when it's empty
            base_prompt = prompts.answer.replace("{deliberate}", "")
            answer_prompts.append(base_prompt.format(question=q))
    
    print("✅ Benchmark controller simulation:")
    for i, prompt in enumerate(answer_prompts):
        print(f"Prompt {i+1}:")
        print(prompt)
        print("-" * 50)


if __name__ == "__main__":
    print("🧪 Testing Python String Formatting with Braces")
    print("=" * 60)
    
    # Run all tests
    test_single_braces_formatting()
    test_double_braces_literal()
    test_mixed_braces_formatting()
    test_boxed_formatting_correct()
    test_boxed_formatting_problematic()
    test_formatting_with_missing_placeholders()
    test_complex_formatting_scenarios()
    test_prompt_template_simulation()
    test_benchmark_controller_simulation()
    
    print("\n🎉 All tests completed successfully!")
    print("\n📝 Key Takeaways:")
    print("1. Single braces {} are format placeholders")
    print("2. Double braces {{}} become literal braces {}")
    print("3. \\boxed{{}} becomes \\boxed{} (literal braces) - CORRECT")
    print("4. \\boxed{} causes IndexError - PROBLEMATIC")
    print("5. The issue was that \\boxed{} has unmatched braces that Python interprets as format placeholders!")
    print("\n🔧 The Fix:")
    print("- Use \\boxed{{}} in YAML templates to get \\boxed{} in the final output")
    print("- This escapes the braces so Python doesn't try to interpret them as format placeholders")
    print("- The original YAML files had \\boxed{{}} which was actually CORRECT")
    print("- The issue was likely elsewhere in the code, not in the prompt formatting!")
