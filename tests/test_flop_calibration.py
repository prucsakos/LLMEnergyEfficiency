"""
Comprehensive test suite for FLOP calibration system.
"""
import pytest
import numpy as np
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict, Any

# Import the classes we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.calibration.data import CalibrationPoint, CalibrationDataset
from src.calibration.models import FLOPExtrapolationModel
from src.calibration.runners import FLOPCalibrationRunner

# ============================================================================
# Test Data and Fixtures
# ============================================================================

@pytest.fixture
def sample_calibration_points():
    """Create sample calibration points for testing."""
    return [
        CalibrationPoint(
            prefill_tokens=64,
            generation_tokens=32,
            measured_flops=1.2e12,
            latency_ms=150.0,
            timestamp=1000.0
        ),
        CalibrationPoint(
            prefill_tokens=128,
            generation_tokens=64,
            measured_flops=2.4e12,
            latency_ms=280.0,
            timestamp=1001.0
        ),
        CalibrationPoint(
            prefill_tokens=256,
            generation_tokens=128,
            measured_flops=4.8e12,
            latency_ms=520.0,
            timestamp=1002.0
        ),
        CalibrationPoint(
            prefill_tokens=512,
            generation_tokens=256,
            measured_flops=9.6e12,
            latency_ms=980.0,
            timestamp=1003.0
        ),
        CalibrationPoint(
            prefill_tokens=1024,
            generation_tokens=512,
            measured_flops=19.2e12,
            latency_ms=1850.0,
            timestamp=1004.0
        ),
    ]

@pytest.fixture
def mock_model_spec():
    """Create a mock model specification for testing."""
    mock_spec = Mock()
    mock_spec.model_name = "test-model"
    mock_spec.hf_repo = "test/model"
    mock_spec.engine = "deepspeed"
    mock_spec.backend = Mock()
    mock_spec.backend.dtype = "bfloat16"
    mock_spec.backend.gpu_memory_utilization = 0.9
    mock_spec.backend.enforce_eager = True
    mock_spec.card = Mock()
    mock_spec.card.params_B = 7
    mock_spec.card.layers = 32
    mock_spec.card.hidden_dim = 4096
    mock_spec.card.heads = 32
    return mock_spec

# ============================================================================
# Test CalibrationPoint
# ============================================================================

def test_calibration_point_creation():
    """Test CalibrationPoint creation and attributes."""
    point = CalibrationPoint(
        prefill_tokens=128,
        generation_tokens=64,
        measured_flops=2.4e12,
        latency_ms=280.0,
        timestamp=1000.0
    )
    
    assert point.prefill_tokens == 128
    assert point.generation_tokens == 64
    assert point.measured_flops == 2.4e12
    assert point.latency_ms == 280.0
    assert point.timestamp == 1000.0

# ============================================================================
# Test CalibrationDataset
# ============================================================================

def test_calibration_dataset_creation():
    """Test CalibrationDataset creation."""
    points = [
        CalibrationPoint(64, 32, 1.2e12, 150.0, 1000.0),
        CalibrationPoint(128, 64, 2.4e12, 280.0, 1001.0)
    ]
    
    dataset = CalibrationDataset(
        model_name="test-model",
        model_config={"params_B": 7, "layers": 32},
        points=points
    )
    
    assert dataset.model_name == "test-model"
    assert dataset.model_config["params_B"] == 7
    assert len(dataset.points) == 2
    assert dataset.extrapolation_model is None
    assert dataset.model_accuracy is None

# ============================================================================
# Test FLOPExtrapolationModel
# ============================================================================

class TestFLOPExtrapolationModel:
    """Test suite for FLOPExtrapolationModel."""
    
    def test_model_initialization(self):
        """Test model initialization with different parameters."""
        # Test default initialization
        model = FLOPExtrapolationModel()
        assert model.degree == 2
        assert model.include_interaction == True
        assert model.is_fitted == False
        
        # Test custom initialization
        model = FLOPExtrapolationModel(degree=3, include_interaction=False)
        assert model.degree == 3
        assert model.include_interaction == False
        assert model.is_fitted == False
    
    def test_fit_with_sufficient_data(self, sample_calibration_points):
        """Test model fitting with sufficient calibration data."""
        model = FLOPExtrapolationModel(degree=2, include_interaction=True)
        
        accuracy_metrics = model.fit(sample_calibration_points)
        
        # Check that model is fitted
        assert model.is_fitted == True
        assert model.feature_names is not None
        assert model.r2_score is not None
        assert model.mae is not None
        assert model.rmse is not None
        
        # Check accuracy metrics
        assert "r2_score" in accuracy_metrics
        assert "mae" in accuracy_metrics
        assert "rmse" in accuracy_metrics
        assert "n_points" in accuracy_metrics
        assert accuracy_metrics["n_points"] == len(sample_calibration_points)
        
        # R² should be reasonable for linear relationship
        assert accuracy_metrics["r2_score"] > 0.8
    
    def test_fit_with_insufficient_data(self):
        """Test model fitting with insufficient data."""
        model = FLOPExtrapolationModel()
        
        # Test with too few points
        insufficient_points = [
            CalibrationPoint(64, 32, 1.2e12, 150.0, 1000.0),
            CalibrationPoint(128, 64, 2.4e12, 280.0, 1001.0)
        ]
        
        with pytest.raises(ValueError, match="Need at least 3 calibration points"):
            model.fit(insufficient_points)
    
    def test_predict_single(self, sample_calibration_points):
        """Test single prediction after fitting."""
        model = FLOPExtrapolationModel()
        model.fit(sample_calibration_points)
        
        # Test prediction
        prediction = model.predict(256, 128)
        
        assert isinstance(prediction, float)
        assert prediction > 0  # Should be positive
        
        # Test prediction before fitting
        unfitted_model = FLOPExtrapolationModel()
        with pytest.raises(ValueError, match="Model must be fitted"):
            unfitted_model.predict(256, 128)
    
    def test_predict_batch(self, sample_calibration_points):
        """Test batch prediction after fitting."""
        model = FLOPExtrapolationModel()
        model.fit(sample_calibration_points)
        
        prefill_tokens = [64, 128, 256]
        generation_tokens = [32, 64, 128]
        
        predictions = model.predict_batch(prefill_tokens, generation_tokens)
        
        assert len(predictions) == 3
        assert all(isinstance(p, float) for p in predictions)
        assert all(p > 0 for p in predictions)
    
    def test_model_info(self, sample_calibration_points):
        """Test getting model information."""
        model = FLOPExtrapolationModel(degree=2, include_interaction=True)
        
        # Test unfitted model
        info = model.get_model_info()
        assert info["fitted"] == False
        
        # Test fitted model
        model.fit(sample_calibration_points)
        info = model.get_model_info()
        
        assert info["fitted"] == True
        assert info["degree"] == 2
        assert info["include_interaction"] == True
        assert "feature_names" in info
        assert "coefficients" in info
        assert "intercept" in info
        assert "r2_score" in info
        assert "mae" in info
        assert "rmse" in info
    
    def test_save_and_load(self, sample_calibration_points):
        """Test saving and loading model."""
        model = FLOPExtrapolationModel(degree=2, include_interaction=True)
        model.fit(sample_calibration_points)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Test saving
            model.save(temp_path)
            assert os.path.exists(temp_path)
            
            # Test loading
            loaded_model = FLOPExtrapolationModel.load(temp_path)
            
            assert loaded_model.degree == model.degree
            assert loaded_model.include_interaction == model.include_interaction
            assert loaded_model.is_fitted == True
            assert loaded_model.r2_score == model.r2_score
            assert loaded_model.mae == model.mae
            assert loaded_model.rmse == model.rmse
            
            # Test that loaded model can make predictions
            original_pred = model.predict(256, 128)
            loaded_pred = loaded_model.predict(256, 128)
            assert abs(original_pred - loaded_pred) < 1e-6
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_polynomial_features(self):
        """Test polynomial feature generation."""
        model = FLOPExtrapolationModel(degree=2, include_interaction=True)
        
        prefill_tokens = [64, 128, 256]
        generation_tokens = [32, 64, 128]
        
        features = model._prepare_features(prefill_tokens, generation_tokens)
        
        assert features.shape[0] == 3  # 3 samples
        assert features.shape[1] >= 3  # At least [1, P, G] features
        
        # Check that features include interaction terms
        feature_names = model.poly_features.get_feature_names_out(['P', 'G'])
        assert any('P G' in name or 'P*G' in name for name in feature_names)

# ============================================================================
# Test FLOPCalibrationRunner
# ============================================================================

class TestFLOPCalibrationRunner:
    """Test suite for FLOPCalibrationRunner."""
    
    def test_initialization(self):
        """Test calibration runner initialization."""
        runner = FLOPCalibrationRunner(
            prefill_ranges=[64, 128, 256],
            generation_ranges=[32, 64]
        )
        
        assert runner.prefill_ranges == [64, 128, 256]
        assert runner.generation_ranges == [32, 64]
        # Calculate combinations manually since it's no longer stored as an attribute
        import itertools
        combinations = list(itertools.product(runner.prefill_ranges, runner.generation_ranges))
        assert len(combinations) == 6  # 3 * 2
    
    def test_initialization_defaults(self):
        """Test calibration runner with default parameters."""
        runner = FLOPCalibrationRunner()
        
        assert len(runner.prefill_ranges) > 0
        assert len(runner.generation_ranges) > 0
        # Calculate combinations manually since it's no longer stored as an attribute
        import itertools
        combinations = list(itertools.product(runner.prefill_ranges, runner.generation_ranges))
        assert len(combinations) > 0
    
    def test_generate_test_prompts(self):
        """Test test prompt generation."""
        runner = FLOPCalibrationRunner()
        
        prompt = runner._generate_test_prompts(128, 64)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
    
    @patch('src.calibration.runners.create_engine')
    @patch('transformers.AutoTokenizer')
    def test_run_calibration_mock(self, mock_tokenizer, mock_create_engine, mock_model_spec):
        """Test calibration run with mocked engine."""
        # Mock the tokenizer with proper methods
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.get_vocab.return_value = {str(i): i for i in range(1000)}
        mock_tokenizer_instance.bos_token_id = None
        mock_tokenizer_instance.eos_token_id = None
        mock_tokenizer_instance.pad_token_id = None
        mock_tokenizer_instance.unk_token_id = None
        mock_tokenizer_instance.decode.return_value = "test prompt"
        mock_tokenizer_instance.encode.return_value = [1, 2, 3, 4, 5]  # Mock token IDs
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock the engine and its methods
        mock_engine = Mock()
        mock_engine.generate.return_value = Mock(
            raw={"flops": {"total_flops": 1.2e12}},
            latency_ms=150.0
        )
        mock_create_engine.return_value = mock_engine
        
        # Create a small calibration runner for testing
        runner = FLOPCalibrationRunner(
            prefill_ranges=[64, 128],
            generation_ranges=[32, 64]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_calibration.json")
            
            # Run calibration
            dataset = runner.run_calibration(mock_model_spec, save_path=save_path)
            
            # Verify results
            assert isinstance(dataset, CalibrationDataset)
            assert dataset.model_name == "test-model"
            assert len(dataset.points) == 4  # 2 * 2 combinations
            assert dataset.extrapolation_model is not None
            assert dataset.model_accuracy is not None
            
            # Verify save file exists
            assert os.path.exists(save_path)
            
            # Verify engine was called correctly
            assert mock_create_engine.called
            mock_engine.close.assert_called()
    
    def test_save_calibration_dataset(self, sample_calibration_points):
        """Test saving calibration dataset."""
        dataset = CalibrationDataset(
            model_name="test-model",
            model_config={"params_B": 7, "layers": 32},
            points=sample_calibration_points
        )
        
        # Fit a model to the dataset
        model = FLOPExtrapolationModel()
        model.fit(sample_calibration_points)
        dataset.extrapolation_model = model
        dataset.model_accuracy = {"r2_score": 0.95, "mae": 0.1, "rmse": 0.15}
        
        runner = FLOPCalibrationRunner()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            runner._save_calibration_dataset(dataset, temp_path)
            
            # Verify file was created
            assert os.path.exists(temp_path)
            
            # Verify file contents
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert data["model_name"] == "test-model"
            assert data["model_config"]["params_B"] == 7
            assert len(data["points"]) == len(sample_calibration_points)
            assert "extrapolation_model" in data
            assert "model_accuracy" in data
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

# ============================================================================
# Integration Tests
# ============================================================================

class TestCalibrationIntegration:
    """Integration tests for the complete calibration system."""
    
    def test_end_to_end_calibration_workflow(self):
        """Test complete calibration workflow."""
        # Create synthetic calibration data that follows a predictable pattern
        synthetic_points = []
        for i, (P, G) in enumerate([(64, 32), (128, 64), (256, 128), (512, 256)]):
            # Create a simple linear relationship: FLOPs = P * G * 1e9
            synthetic_flops = P * G * 1e9
            synthetic_points.append(CalibrationPoint(
                prefill_tokens=P,
                generation_tokens=G,
                measured_flops=synthetic_flops,
                latency_ms=100.0 + i * 50.0,
                timestamp=1000.0 + i
            ))
        
        # Test model fitting
        model = FLOPExtrapolationModel(degree=2, include_interaction=True)
        accuracy_metrics = model.fit(synthetic_points)
        
        # Should have high accuracy for synthetic linear data
        assert accuracy_metrics["r2_score"] > 0.99
        
        # Test predictions
        test_cases = [(100, 50), (200, 100), (300, 150)]
        for P, G in test_cases:
            predicted = model.predict(P, G)
            expected = P * G * 1e9
            # Should be very close for linear relationship
            assert abs(predicted - expected) / expected < 0.01
    
    def test_model_robustness(self):
        """Test model robustness with noisy data."""
        # Create data with some noise
        base_points = []
        np.random.seed(42)  # For reproducible tests
        
        for i, (P, G) in enumerate([(64, 32), (128, 64), (256, 128), (512, 256), (1024, 512)]):
            base_flops = P * G * 1e9
            # Add 10% noise
            noise = np.random.normal(0, 0.1 * base_flops)
            noisy_flops = base_flops + noise
            
            base_points.append(CalibrationPoint(
                prefill_tokens=P,
                generation_tokens=G,
                measured_flops=noisy_flops,
                latency_ms=100.0 + i * 50.0,
                timestamp=1000.0 + i
            ))
        
        # Test model fitting with noisy data
        model = FLOPExtrapolationModel(degree=2, include_interaction=True)
        accuracy_metrics = model.fit(base_points)
        
        # Should still have reasonable accuracy
        assert accuracy_metrics["r2_score"] > 0.8
        assert accuracy_metrics["mae"] > 0  # Should have some error due to noise

# ============================================================================
# Performance Tests
# ============================================================================

class TestCalibrationPerformance:
    """Performance tests for calibration system."""
    
    def test_large_dataset_performance(self):
        """Test performance with larger calibration datasets."""
        # Create a larger synthetic dataset
        large_points = []
        for P in range(64, 2049, 64):  # 32 points
            for G in range(32, 513, 32):  # 16 points
                flops = P * G * 1e9
                large_points.append(CalibrationPoint(
                    prefill_tokens=P,
                    generation_tokens=G,
                    measured_flops=flops,
                    latency_ms=100.0,
                    timestamp=1000.0
                ))
        
        # Test fitting performance
        import time
        start_time = time.time()
        
        model = FLOPExtrapolationModel(degree=2, include_interaction=True)
        accuracy_metrics = model.fit(large_points)
        
        fit_time = time.time() - start_time
        
        # Should complete in reasonable time (< 1 second for 512 points)
        assert fit_time < 1.0
        assert accuracy_metrics["r2_score"] > 0.99
        
        # Test prediction performance
        start_time = time.time()
        predictions = model.predict_batch([256, 512, 1024], [128, 256, 512])
        predict_time = time.time() - start_time
        
        # Should be very fast (< 0.01 seconds for 3 predictions)
        assert predict_time < 0.01
        assert len(predictions) == 3

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
