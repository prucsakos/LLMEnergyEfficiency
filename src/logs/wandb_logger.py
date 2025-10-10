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
    
    def log_calibration_data(self, calibration_dataset, model_spec, estimation_data=None) -> None:
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
        
        # Log pre-generated estimation data if provided
        if estimation_data:
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

    def log_trace_tables(self, all_traces: List[Dict[str, Any]], spec) -> None:
        """Log all traces to separate wandb tables for successful and failed traces."""
        if not all_traces:
            return
        
        # Separate successful and failed traces
        successful_traces = [trace for trace in all_traces if trace.get("is_successful", False)]
        failed_traces = [trace for trace in all_traces if not trace.get("is_successful", True)]
        
        # Log successful traces table
        if successful_traces:
            self._log_trace_table("successful_traces", successful_traces, spec)
        
        # Log failed traces table
        if failed_traces:
            self._log_trace_table("failed_traces", failed_traces, spec)
    
    def _log_trace_table(self, table_name: str, traces: List[Dict[str, Any]], spec) -> None:
        """Log a single trace table to wandb."""
        if not traces:
            return
        
        # Determine the trace type: self-consistency, two-pass, or single-pass
        is_sc = any("chosen_answer" in trace for trace in traces)
        is_single_pass = spec.reasoning.style == "single_pass"
        max_k = spec.reasoning.self_consistency_k or 1
        
        # Build columns based on the trace type
        columns = ["datapoint_id", "formatted_question", "golden_answer"]
        
        if is_sc:
            # Self-consistency: add columns for each path
            for k in range(1, max_k + 1):
                columns.extend([f"path_{k}_think_text", f"path_{k}_answer_text", f"path_{k}_think_tokens", f"path_{k}_answer_tokens"])
        elif is_single_pass:
            # Single-pass: extracted solution, full answer text, and token counts
            columns.extend(["extracted_solution", "full_answer_text", "answer_tokens"])
        else:
            # Two-pass: single think and answer with token counts
            columns.extend(["think_text", "answer_text", "think_tokens", "answer_tokens"])
        
        # Build data rows
        data = []
        for i, trace in enumerate(traces):
            row = [
                i,  # datapoint_id
                trace.get("question", ""),  # formatted_question
                trace.get("gold", ""),  # golden_answer
            ]
            
            if is_sc:
                # Self-consistency: add each path with token counts
                for k in range(1, max_k + 1):
                    think_key = f"path_{k}_think"
                    answer_key = f"path_{k}_answer"
                    think_tokens_key = f"path_{k}_think_tokens"
                    answer_tokens_key = f"path_{k}_answer_tokens"
                    row.extend([
                        trace.get(think_key, ""),
                        trace.get(answer_key, ""),
                        trace.get(think_tokens_key, 0),
                        trace.get(answer_tokens_key, 0)
                    ])
            elif is_single_pass:
                # Single-pass: extracted solution, full answer text, and token count
                row.extend([
                    trace.get("answer_text", ""),  # extracted solution
                    trace.get("full_answer_text", ""),  # full answer text
                    trace.get("answer_tokens", 0)  # token count
                ])
            else:
                # Two-pass: single think and answer with token counts
                row.extend([
                    trace.get("think_text", ""),
                    trace.get("answer_text", ""),
                    trace.get("think_tokens", 0),
                    trace.get("answer_tokens", 0)
                ])
            
            data.append(row)
        
        # Log the table
        self.log_table(table_name, columns, data)

    def finish(self) -> None:
        self.run.finish()
