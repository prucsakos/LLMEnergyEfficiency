"""
Important note on benchmarks: Each dataset have its unique properties, evaluation methods
and structure. Thats why each dataset used in the project shall have their unique adapters.
"""

# huggingface
from datasets import load_dataset

dataset = load_dataset("hotpotqa/hotpot_qa")

