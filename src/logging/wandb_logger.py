from __future__ import annotations
import os
import time
from typing import Dict, Any, Optional
import wandb

class WandbLogger:
    """Thin W&B logger for one-row-per-run semantics.

    Call `log_row(row)` once per (model x dataset x config) run, then `finish()`.
    """
    def __init__(self, project: str, run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.run = wandb.init(project=project, name=run_name, config=config or {}, reinit=True)

    def log_row(self, row: Dict[str, Any]) -> None:
        """Log a single row (your schema below)."""
        self.run.log(row)

    def finish(self) -> None:
        self.run.finish()
