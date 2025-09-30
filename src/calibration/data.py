"""
Data structures for calibration measurements and datasets.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class CalibrationPoint:
    """Single calibration measurement point."""
    prefill_tokens: int
    generation_tokens: int
    measured_flops: float
    latency_ms: float
    timestamp: float


@dataclass
class CalibrationDataset:
    """Complete calibration dataset for a model."""
    model_name: str
    model_config: Dict[str, Any]
    points: List[CalibrationPoint]
    extrapolation_model: Optional[Any] = None
    model_accuracy: Optional[Dict[str, float]] = None
    estimation_data: Optional[List[List[Any]]] = None
