"""
Utility functions for calibration.
"""

import json
import numpy as np

from .data import CalibrationPoint, CalibrationDataset
from .models import FLOPExtrapolationModel, NextTokenFLOPModel


def load_calibration_dataset(filepath: str) -> CalibrationDataset:
    """Load calibration dataset from disk."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Reconstruct calibration points
    points = [
        CalibrationPoint(
            prefill_tokens=p["prefill_tokens"],
            generation_tokens=p["generation_tokens"],
            measured_flops=p["measured_flops"],
            latency_ms=p["latency_ms"],
            timestamp=p["timestamp"]
        )
        for p in data["points"]
    ]
    
    # Create dataset
    dataset = CalibrationDataset(
        model_name=data["model_name"],
        model_config=data["model_config"],
        points=points,
        model_accuracy=data.get("model_accuracy")
    )
    
    # Reconstruct extrapolation model if available
    if "extrapolation_model" in data and data["extrapolation_model"]:
        model_info = data["extrapolation_model"]
        
        # Determine model type based on feature names
        if "include_interaction" in model_info:
            # Full FLOP model
            model = FLOPExtrapolationModel(
                degree=model_info["degree"],
                include_interaction=model_info["include_interaction"]
            )
        else:
            # Next-token FLOP model
            model = NextTokenFLOPModel(degree=model_info["degree"])
        
        model.feature_names = np.array(model_info["feature_names"])
        model.regressor.coef_ = np.array(model_info["coefficients"])
        model.regressor.intercept_ = model_info["intercept"]
        model.r2_score = model_info["r2_score"]
        model.mae = model_info["mae"]
        model.rmse = model_info["rmse"]
        model.is_fitted = True
        dataset.extrapolation_model = model
    
    return dataset
