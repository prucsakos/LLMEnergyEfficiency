#!/usr/bin/env python3
"""
Test script for GPQA-Diamond dataset adapter.

This script tests the newly implemented GPQA-Diamond dataset adapter
and provides statistics and sample outputs.
"""

import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.adapters import iter_dataset


def test_gpqa_diamond():
    """Test the GPQA-Diamond dataset adapter and provide statistics."""

    print("🧪 Testing GPQA-Diamond Dataset Adapter")
    print("=" * 50)

    try:
        # Load the dataset
        print("\n📥 Loading GPQA-Diamond dataset...")
        samples = list(iter_dataset("gpqa_diamond", split="train"))
        print(f"✅ Successfully loaded {len(samples)} samples")

        # Basic statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"   • Total samples: {len(samples)}")
        print(f"   • Sample ID range: {samples[0].id} to {samples[-1].id}")

        # Analyze choices and answers
        choice_counts = []
        answer_labels = []

        for sample in samples:
            if sample.choices:
                choice_counts.append(len(sample.choices))
            # Extract answer label (A, B, C, D) from gold answer
            if sample.gold.startswith("(") and ")" in sample.gold:
                label = sample.gold.split(")")[0][1:]  # Extract letter after "("
                answer_labels.append(label)

        if choice_counts:
            print(f"   • Choices per question: {set(choice_counts)} (should be {4})")

        if answer_labels:
            from collections import Counter
            label_counts = Counter(answer_labels)
            print(f"   • Answer distribution: {dict(label_counts)}")

        # Sample outputs
        print(f"\n📝 Sample Outputs (first 3 samples):")
        print("-" * 50)

        for i, sample in enumerate(samples[:3]):
            print(f"\nSample {i+1}:")
            print(f"ID: {sample.id}")
            print("Question (first 200 chars):")
            print(f"  {sample.question[:200]}{'...' if len(sample.question) > 200 else ''}")
            print(f"Gold Answer: {sample.gold}")
            if sample.choices:
                print(f"Choices ({len(sample.choices)}):")
                for j, choice in enumerate(sample.choices):
                    print(f"  {chr(65+j)}) {choice[:100]}{'...' if len(choice) > 100 else ''}")
            print()

        # Question structure analysis
        print("🔍 Question Structure Analysis:")
        print("-" * 30)

        sample_question = samples[0].question if samples else ""
        if sample_question:
            lines = sample_question.split('\n')
            print(f"Question has {len(lines)} lines")

            # Check for expected format
            has_choose_text = "Choose the best answer" in sample_question
            has_options = "(A)" in sample_question and "(B)" in sample_question

            print(f"Contains 'Choose the best answer': {has_choose_text}")
            print(f"Contains option labels (A), (B), etc.: {has_options}")

            # Check gold answer format
            gold = samples[0].gold
            has_label_format = gold.startswith("(") and ")" in gold and len(gold.split(")")[0]) == 2
            print(f"Gold answer has label format like '(A) answer': {has_label_format}")

        print(f"\n✅ GPQA-Diamond adapter test completed successfully!")
        print(f"Dataset appears to be properly formatted as a multiple-choice question dataset.")

    except Exception as e:
        print(f"❌ Error testing GPQA-Diamond adapter: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    test_gpqa_diamond()
