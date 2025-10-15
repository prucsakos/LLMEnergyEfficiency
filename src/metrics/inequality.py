from __future__ import annotations
from typing import Iterable
import numpy as np

def gini(values: Iterable[float]) -> float:
    """Compute the Gini coefficient of non‑negative values.
    Returns 0.0 for perfect equality, approaches 1.0 as inequality increases.
    """
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return 0.0
    if np.any(arr < 0):
        raise ValueError("Gini only defined for non‑negative values")
    total = arr.sum()
    if total <= 0:
        return 0.0
    sorted_arr = np.sort(arr)
    index = np.arange(1, n + 1, dtype=float)  # 1-indexed
    g = 1.0 + (1.0 / n) - 2.0 * np.sum((n - index + 1.0) * sorted_arr) / (n * total)
    return float(max(0.0, min(1.0, g)))
