from __future__ import annotations
import os
import time
from typing import Dict, Any, Optional, List
import wandb
import numpy as np

class WandbRunLogger:
    """Thin W&B logger for one-row-per-run semantics.

    Call `log_row(row)` once per (model x dataset x config) run, then `finish()`.
    """
    def __init__(self, project: str, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.run = wandb.init(project=project, name=run_name, config=config or {}, reinit=True)

    def log_row(self, row: Dict[str, Any]) -> None:
        """Log a single row (your schema below)."""
        self.run.log(row)
        # Also record as summary so it's visible at-a-glance
        for k, v in row.items():
            if isinstance(v, (int, float, str, bool)) and k not in ("notes",):
                self.run.summary[k] = v

    def log_table(self, table_name: str, columns: List[str], data: List[List[Any]]) -> None:
        """Log a table to wandb."""
        table = wandb.Table(columns=columns, data=data)
        self.run.log({table_name: table})
    
    def log_calibration_data(self, calibration_dataset, model_spec) -> None:
        """Log calibration dataset and model estimations to wandb."""
        if not calibration_dataset or not calibration_dataset.points:
            return
        
        # Log actual calibration datapoints
        calibration_data = []
        for point in calibration_dataset.points:
            calibration_data.append([
                point.prefill_tokens,
                point.generation_tokens,
                point.measured_flops,
                point.latency_ms,
                point.timestamp
            ])
        
        self.log_table(
            "calibration_datapoints",
            ["prefill_tokens", "generation_tokens", "measured_flops", "latency_ms", "timestamp"],
            calibration_data
        )
        
        # Generate linspace with exponential steps for model estimations
        if calibration_dataset.extrapolation_model:
            # Create exponential steps from 1 to 16000
            prefill_steps = np.geomspace(1, 16000, num=50, dtype=int)
            generation_steps = np.geomspace(1, 16000, num=50, dtype=int)
            
            # Create all combinations
            estimation_data = []
            for prefill in prefill_steps:
                for gen in generation_steps:
                    try:
                        estimated_flops = calibration_dataset.extrapolation_model.predict(prefill, gen)
                        estimation_data.append([prefill, gen, estimated_flops])
                    except Exception as e:
                        print(f"Warning: Failed to predict FLOPs for P={prefill}, G={gen}: {e}")
                        continue
            
            self.log_table(
                "calibration_model_estimations",
                ["prefill_tokens", "generation_tokens", "estimated_flops"],
                estimation_data
            )
    
    def log_calibration_metrics(self, calibration_dataset) -> None:
        """Log calibration metrics to wandb."""
        if not calibration_dataset or not calibration_dataset.model_accuracy:
            return
        
        metrics = {
            "calibration_r2_score": calibration_dataset.model_accuracy.get("r2_score"),
            "calibration_mae": calibration_dataset.model_accuracy.get("mae"),
            "calibration_rmse": calibration_dataset.model_accuracy.get("rmse"),
            "calibration_n_points": calibration_dataset.model_accuracy.get("n_points"),
            "calibration_total_points": len(calibration_dataset.points)
        }
        
        # Log metrics
        self.run.log(metrics)
        
        # Also add to summary
        for k, v in metrics.items():
            if v is not None:
                self.run.summary[k] = v

    def finish(self) -> None:
        self.run.finish()
