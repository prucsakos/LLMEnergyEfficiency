#!/usr/bin/env python3
"""
Standalone calibration runner for DeepSpeed-based FLOP estimation.

This script runs calibration in a separate process to ensure proper GPU cleanup
when the calibration process ends.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import argparse
import json
import traceback
import time
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from src.utils import load_env_variables
load_env_variables()

# Import logging system
from src.logs.benchmark_logger import setup_logging, get_logger

# Import calibration system
from src.calibration import (
    CalibrationDataset,
    NextTokenCalibrationRunner,
    load_calibration_dataset
)

# Import config system
from src.config.bench_config import load_bench_config, expand_runs


def run_calibration_for_model(model_spec, 
                             calibration_file: str,
                             prefill_ranges: list,
                             estimation_points: int = 64) -> bool:
    """
    Run calibration for a specific model.
    
    Args:
        model_spec: Model specification
        calibration_file: Path to save calibration results
        prefill_ranges: List of prefill token ranges to test
        estimation_points: Number of estimation points for extrapolation evaluation
        
    Returns:
        bool: True if calibration succeeded, False otherwise
    """
    logger = get_logger()
    
    try:
        logger.info(f"Starting calibration for {model_spec.model_name}")
        logger.info(f"Model: {model_spec.hf_repo}")
        logger.info(f"Calibration file: {calibration_file}")
        
        # Create calibration runner
        calibration_runner = NextTokenCalibrationRunner(
            prefill_ranges=prefill_ranges,
            generation_tokens=1
        )
        
        # Run calibration
        calibration_dataset = calibration_runner.run_calibration(
            model_spec, 
            save_path=calibration_file, 
            estimation_points=estimation_points
        )
        
        logger.info(f"✓ Calibration completed successfully with {len(calibration_dataset.points)} points")
        
        if calibration_dataset.model_accuracy:
            logger.info(f"  Model R²: {calibration_dataset.model_accuracy.get('r2_score', 'N/A'):.4f}")
            logger.info(f"  MAE: {calibration_dataset.model_accuracy.get('mae', 0) / 1e12:.2f} TFLOPs")
            logger.info(f"  RMSE: {calibration_dataset.model_accuracy.get('rmse', 0) / 1e12:.2f} TFLOPs")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Calibration failed for {model_spec.model_name}: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        
        # Log the calibration error to error log
        try:
            with open("error.log", "a", encoding="utf-8") as f:
                f.write(f"\n=== Calibration Error for {model_spec.model_name} ===\n")
                f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {e}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
                f.write("=" * 50 + "\n")
        except Exception as log_error:
            logger.warning(f"Failed to write to error log: {log_error}")
        
        return False


def main():
    """Main entry point for calibration subprocess."""
    parser = argparse.ArgumentParser(description="Run DeepSpeed calibration for a model")
    
    # Model specification
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark config file")
    parser.add_argument("--model_name", type=str, required=True, help="Model name to calibrate")
    
    # Calibration options
    parser.add_argument("--calibration_file", type=str, required=True, help="Path to save calibration results")
    parser.add_argument("--prefill_ranges", nargs="+", type=int, 
                       default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
                       help="Prefill token ranges for calibration")
    parser.add_argument("--estimation_points", type=int, default=64,
                       help="Number of estimation points for extrapolation evaluation")
    
    # Logging options
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger()
    
    logger.info("=" * 60)
    logger.info("🔧 DEEPSPEED CALIBRATION SUBPROCESS")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Calibration file: {args.calibration_file}")
    logger.info(f"Prefill ranges: {args.prefill_ranges}")
    logger.info(f"Estimation points: {args.estimation_points}")
    
    try:
        # Load configuration
        cfg = load_bench_config(args.config)
        
        # Find the model specification
        model_spec = None
        for spec in expand_runs(cfg):
            if spec.model_name == args.model_name:
                model_spec = spec
                break
        
        if model_spec is None:
            logger.error(f"❌ Model '{args.model_name}' not found in config")
            sys.exit(1)
        
        # Run calibration
        success = run_calibration_for_model(
            model_spec=model_spec,
            calibration_file=args.calibration_file,
            prefill_ranges=args.prefill_ranges,
            estimation_points=args.estimation_points
        )
        
        if success:
            logger.info("🎉 Calibration completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Calibration failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Fatal error in calibration subprocess: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()
