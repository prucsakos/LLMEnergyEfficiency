from __future__ import annotations
from collections import Counter
from typing import List

def majority_vote(answers: List[str]) -> str:
    """Pick the most frequent normalized answer."""
    norm = [a.strip().lower() for a in answers]
    [(ans, _)] = Counter(norm).most_common(1)
    # Return the original casing of the first match
    for a in answers:
        if a.strip().lower() == ans:
            return a.strip()
    return ans
