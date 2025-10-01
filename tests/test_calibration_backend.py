"""
Test suite for separate calibration backend configuration.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from src.config.bench_config import (
    load_bench_config, 
    BackendDefaults, 
    CalibrationBackendDefaults,
    ModelSpec,
    RunSpec
)


def test_calibration_backend_defaults():
    """Test that CalibrationBackendDefaults has correct default values."""
    backend = CalibrationBackendDefaults()
    
    assert backend.dtype == "auto"
    assert backend.gpu_memory_utilization == 0.90
    assert backend.enforce_eager == True
    assert backend.quantization is None
    assert backend.quantization_param_path is None


def test_model_spec_with_calibration_backend():
    """Test that ModelSpec includes calibration_backend field."""
    from src.config.bench_config import Card
    
    card = Card(params_B=1.0, arch="test")
    backend = BackendDefaults()
    calibration_backend = CalibrationBackendDefaults()
    
    model_spec = ModelSpec(
        name="test-model",
        hf_repo="test/repo",
        card=card,
        think_budgets=[128],
        backend=backend,
        calibration_backend=calibration_backend
    )
    
    assert hasattr(model_spec, 'calibration_backend')
    assert isinstance(model_spec.calibration_backend, CalibrationBackendDefaults)


def test_config_loading_with_calibration_backend():
    """Test loading a config file with calibration_backend specified."""
    config_data = {
        "datasets": ["test_dataset"],
        "config_name": "test_config",
        "models": [
            {
                "name": "test-model",
                "hf_repo": "test/repo",
                "card": {"params_B": 1.0, "arch": "test"},
                "think_budgets": [128],
                "backend": {
                    "dtype": "float16",
                    "gpu_memory_utilization": 0.8,
                    "quantization": "bitsandbytes"
                },
                "calibration_backend": {
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.7,
                    "quantization": "deepspeedfp"
                }
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config = load_bench_config(config_path)
        
        assert len(config.models) == 1
        model = config.models[0]
        
        # Check benchmark backend
        assert model.backend.dtype == "float16"
        assert model.backend.gpu_memory_utilization == 0.8
        assert model.backend.quantization == "bitsandbytes"
        
        # Check calibration backend
        assert model.calibration_backend.dtype == "bfloat16"
        assert model.calibration_backend.gpu_memory_utilization == 0.7
        assert model.calibration_backend.quantization == "deepspeedfp"
        
    finally:
        Path(config_path).unlink()


def test_config_loading_without_calibration_backend():
    """Test loading a config file without calibration_backend (should use defaults)."""
    config_data = {
        "datasets": ["test_dataset"],
        "config_name": "test_config",
        "models": [
            {
                "name": "test-model",
                "hf_repo": "test/repo",
                "card": {"params_B": 1.0, "arch": "test"},
                "think_budgets": [128],
                "backend": {
                    "dtype": "float16",
                    "quantization": "bitsandbytes"
                }
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config = load_bench_config(config_path)
        
        assert len(config.models) == 1
        model = config.models[0]
        
        # Check that calibration_backend uses defaults
        assert hasattr(model, 'calibration_backend')
        assert model.calibration_backend.dtype == "auto"
        assert model.calibration_backend.gpu_memory_utilization == 0.90
        assert model.calibration_backend.quantization is None
        
    finally:
        Path(config_path).unlink()


def test_run_spec_includes_calibration_backend():
    """Test that RunSpec includes calibration_backend field."""
    from src.config.bench_config import Card, GenDefaults, ReasoningDefaults, Prompts
    
    card = Card(params_B=1.0, arch="test")
    backend = BackendDefaults()
    calibration_backend = CalibrationBackendDefaults()
    generation = GenDefaults()
    reasoning = ReasoningDefaults()
    prompts = Prompts()
    
    run_spec = RunSpec(
        model_name="test-model",
        hf_repo="test/repo",
        card=card,
        model_family="test",
        engine="vllm",
        dataset="test_dataset",
        think_budget=128,
        batch_size=1,
        backend=backend,
        calibration_backend=calibration_backend,
        generation=generation,
        reasoning=reasoning,
        prompts=prompts
    )
    
    assert hasattr(run_spec, 'calibration_backend')
    assert isinstance(run_spec.calibration_backend, CalibrationBackendDefaults)


def test_expand_runs_includes_calibration_backend():
    """Test that expand_runs includes calibration_backend in RunSpec."""
    from src.config.bench_config import BenchConfig, Card, GenDefaults, ReasoningDefaults, Prompts, expand_runs
    
    card = Card(params_B=1.0, arch="test")
    backend = BackendDefaults()
    calibration_backend = CalibrationBackendDefaults()
    generation = GenDefaults()
    reasoning = ReasoningDefaults()
    prompts = Prompts()
    
    model_spec = ModelSpec(
        name="test-model",
        hf_repo="test/repo",
        card=card,
        think_budgets=[128],
        backend=backend,
        calibration_backend=calibration_backend,
        generation=generation,
        reasoning=reasoning
    )
    
    config = BenchConfig(
        models=[model_spec],
        datasets=["test_dataset"],
        prompts=prompts,
        prompt_sets=[{"name": "default"}],
        config_name="test"
    )
    
    runs = list(expand_runs(config))
    assert len(runs) == 1
    
    run = runs[0]
    assert hasattr(run, 'calibration_backend')
    assert isinstance(run.calibration_backend, CalibrationBackendDefaults)


def test_calibration_backend_quantization_validation():
    """Test that calibration backend only allows deepspeedfp or None for quantization."""
    # Test with deepspeedfp (should work)
    backend1 = CalibrationBackendDefaults(quantization="deepspeedfp")
    assert backend1.quantization == "deepspeedfp"
    
    # Test with None (should work)
    backend2 = CalibrationBackendDefaults(quantization=None)
    assert backend2.quantization is None
    
    # Test with other quantization methods (should still work, but not recommended)
    backend3 = CalibrationBackendDefaults(quantization="bitsandbytes")
    assert backend3.quantization == "bitsandbytes"


if __name__ == "__main__":
    pytest.main([__file__])
