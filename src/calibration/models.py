"""
FLOP estimation models for calibration.
"""

from __future__ import annotations
import json
from typing import List, Dict, Any
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from .data import CalibrationPoint


class FLOPExtrapolationModel:
    """
    Modular extrapolation model for converting DeepSpeed FLOP measurements to VLLM estimates.
    
    This class implements a polynomial regression model that learns the relationship
    between (prefill_tokens, generation_tokens) and measured FLOPs from DeepSpeed,
    then provides extrapolation to VLLM scenarios.
    """
    
    def __init__(self, degree: int = 2, include_interaction: bool = True):
        """
        Initialize the extrapolation model.
        
        Args:
            degree: Polynomial degree for features (1=linear, 2=quadratic, etc.)
            include_interaction: Whether to include interaction terms (P*G)
        """
        self.degree = degree
        self.include_interaction = include_interaction
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=True, interaction_only=False)
        self.regressor = LinearRegression()
        self.is_fitted = False
        self.feature_names = None
        self.r2_score = None
        self.mae = None
        self.rmse = None
        
    def _prepare_features(self, prefill_tokens: List[int], generation_tokens: List[int]) -> np.ndarray:
        """Prepare polynomial features from token counts."""
        # Create base features: [P, G, P*G, P^2, G^2, ...]
        X = np.column_stack([prefill_tokens, generation_tokens])
        return self.poly_features.fit_transform(X)
    
    def fit(self, calibration_points: List[CalibrationPoint]) -> Dict[str, float]:
        """
        Fit the extrapolation model to calibration data.
        
        Args:
            calibration_points: List of calibration measurements
            
        Returns:
            Dictionary with model accuracy metrics
        """
        if len(calibration_points) < 3:
            raise ValueError(f"Need at least 3 calibration points, got {len(calibration_points)}")
        
        # Extract features and targets
        prefill_tokens = [p.prefill_tokens for p in calibration_points]
        generation_tokens = [p.generation_tokens for p in calibration_points]
        flops = [p.measured_flops for p in calibration_points]
        
        # Prepare polynomial features
        X = self._prepare_features(prefill_tokens, generation_tokens)
        y = np.array(flops)
        
        # Fit the model
        self.regressor.fit(X, y)
        self.is_fitted = True
        self.feature_names = self.poly_features.get_feature_names_out(['P', 'G'])
        
        # Calculate accuracy metrics
        y_pred = self.regressor.predict(X)
        self.r2_score = r2_score(y, y_pred)
        self.mae = mean_absolute_error(y, y_pred)
        self.rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        return {
            'r2_score': self.r2_score,
            'mae': self.mae,
            'rmse': self.rmse,
            'n_points': len(calibration_points)
        }
    
    def predict(self, prefill_tokens: int, generation_tokens: int) -> float:
        """
        Predict FLOPs for given token counts.
        
        Args:
            prefill_tokens: Number of prefill tokens
            generation_tokens: Number of generation tokens
            
        Returns:
            Predicted FLOPs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self._prepare_features([prefill_tokens], [generation_tokens])
        return float(self.regressor.predict(X)[0])
    
    def predict_batch(self, prefill_tokens: List[int], generation_tokens: List[int]) -> List[float]:
        """Predict FLOPs for multiple token count pairs."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self._prepare_features(prefill_tokens, generation_tokens)
        return self.regressor.predict(X).tolist()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model."""
        if not self.is_fitted:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "degree": self.degree,
            "include_interaction": self.include_interaction,
            "feature_names": self.feature_names.tolist(),
            "coefficients": self.regressor.coef_.tolist(),
            "intercept": float(self.regressor.intercept_),
            "r2_score": self.r2_score,
            "mae": self.mae,
            "rmse": self.rmse
        }
    
    def save(self, filepath: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            "degree": self.degree,
            "include_interaction": self.include_interaction,
            "feature_names": self.feature_names.tolist(),
            "coefficients": self.regressor.coef_.tolist(),
            "intercept": float(self.regressor.intercept_),
            "r2_score": self.r2_score,
            "mae": self.mae,
            "rmse": self.rmse
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'FLOPExtrapolationModel':
        """Load a fitted model from disk."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        model = cls(degree=model_data["degree"], include_interaction=model_data["include_interaction"])
        model.feature_names = np.array(model_data["feature_names"])
        model.regressor.coef_ = np.array(model_data["coefficients"])
        model.regressor.intercept_ = model_data["intercept"]
        model.r2_score = model_data["r2_score"]
        model.mae = model_data["mae"]
        model.rmse = model_data["rmse"]
        model.is_fitted = True
        
        return model


class NextTokenFLOPModel:
    """
    Next-token FLOP estimation model that learns the cost of generating a single token
    based on the current prefill length.
    
    This model is more efficient than the full (P,G) model because:
    1. It only needs to learn a 1D function: flops = f(prefill_tokens)
    2. For generation_tokens > 1, we sum the cost of each individual token
    3. Faster calibration since we only need to measure single-token generation
    """
    
    def __init__(self, degree: int = 2):
        """
        Initialize the next-token FLOP model.
        
        Args:
            degree: Polynomial degree for prefill_tokens (1=linear, 2=quadratic, etc.)
        """
        self.degree = degree
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=True)
        self.regressor = LinearRegression()
        self.is_fitted = False
        self.feature_names = None
        self.r2_score = None
        self.mae = None
        self.rmse = None
        
    def _prepare_features(self, prefill_tokens: List[int]) -> np.ndarray:
        """Prepare polynomial features from prefill token counts."""
        # Create features: [1, P, P^2, P^3, ...] for degree=2: [1, P, P^2]
        X = np.array(prefill_tokens).reshape(-1, 1)
        return self.poly_features.fit_transform(X)
    
    def fit(self, calibration_points: List[CalibrationPoint]) -> Dict[str, float]:
        """
        Fit the next-token model to calibration data.
        
        Args:
            calibration_points: List of calibration measurements (should be single-token generation)
            
        Returns:
            Dictionary with model accuracy metrics
        """
        if len(calibration_points) < 3:
            raise ValueError(f"Need at least 3 calibration points, got {len(calibration_points)}")
        
        # Extract features and targets
        prefill_tokens = [p.prefill_tokens for p in calibration_points]
        flops_per_token = [p.measured_flops / p.generation_tokens for p in calibration_points]
        
        # Prepare polynomial features
        X = self._prepare_features(prefill_tokens)
        y = np.array(flops_per_token)
        
        # Fit the model
        self.regressor.fit(X, y)
        self.is_fitted = True
        self.feature_names = self.poly_features.get_feature_names_out(['P'])
        
        # Calculate accuracy metrics
        y_pred = self.regressor.predict(X)
        self.r2_score = r2_score(y, y_pred)
        self.mae = mean_absolute_error(y, y_pred)
        self.rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        return {
            'r2_score': self.r2_score,
            'mae': self.mae,
            'rmse': self.rmse,
            'n_points': len(calibration_points)
        }
    
    def predict_next_token_flops(self, prefill_tokens: int) -> float:
        """
        Predict FLOPs for generating the next single token.
        
        Args:
            prefill_tokens: Number of prefill tokens
            
        Returns:
            Predicted FLOPs for the next token
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self._prepare_features([prefill_tokens])
        return float(self.regressor.predict(X)[0])
    
    def predict_total_flops(self, prefill_tokens: int, generation_tokens: int) -> float:
        """
        Predict total FLOPs for generating multiple tokens by summing individual token costs.
        
        Args:
            prefill_tokens: Initial number of prefill tokens
            generation_tokens: Number of tokens to generate
            
        Returns:
            Total predicted FLOPs
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        total_flops = 0.0
        
        # For each token to generate, calculate the cost based on current prefill length
        current_prefill = prefill_tokens
        for _ in range(generation_tokens):
            token_flops = self.predict_next_token_flops(current_prefill)
            total_flops += token_flops
            current_prefill += 1  # Each generated token becomes part of the context
        
        return total_flops
    
    def predict(self, prefill_tokens: int, generation_tokens: int) -> float:
        """
        Predict total FLOPs for given token counts (drop-in replacement for FLOPExtrapolationModel).
        
        Args:
            prefill_tokens: Number of prefill tokens
            generation_tokens: Number of generation tokens
            
        Returns:
            Predicted total FLOPs
        """
        return self.predict_total_flops(prefill_tokens, generation_tokens)
    
    def predict_batch(self, prefill_tokens: List[int], generation_tokens: List[int]) -> List[float]:
        """Predict total FLOPs for multiple (prefill, generation) pairs."""
        return [self.predict_total_flops(p, g) for p, g in zip(prefill_tokens, generation_tokens)]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model."""
        if not self.is_fitted:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "degree": self.degree,
            "feature_names": self.feature_names.tolist(),
            "coefficients": self.regressor.coef_.tolist(),
            "intercept": float(self.regressor.intercept_),
            "r2_score": self.r2_score,
            "mae": self.mae,
            "rmse": self.rmse
        }
    
    def save(self, filepath: str) -> None:
        """Save the fitted model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            "degree": self.degree,
            "feature_names": self.feature_names.tolist(),
            "coefficients": self.regressor.coef_.tolist(),
            "intercept": float(self.regressor.intercept_),
            "r2_score": self.r2_score,
            "mae": self.mae,
            "rmse": self.rmse
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'NextTokenFLOPModel':
        """Load a fitted model from disk."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        model = cls(degree=model_data["degree"])
        model.feature_names = np.array(model_data["feature_names"])
        model.regressor.coef_ = np.array(model_data["coefficients"])
        model.regressor.intercept_ = model_data["intercept"]
        model.r2_score = model_data["r2_score"]
        model.mae = model_data["mae"]
        model.rmse = model_data["rmse"]
        model.is_fitted = True
        
        return model
