from __future__ import annotations
import traceback
import argparse, math, time, json, os, copy
from typing import Iterable, List, Optional, Tuple, Callable, Dict, Any
from dataclasses import dataclass
import itertools
from tqdm.auto import tqdm
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load environment variables from .env file
from ..utils import load_env_variables
load_env_variables()

from ..config.bench_config import load_bench_config, expand_runs, RunSpec, Prompts
from ..core.interfaces import GenerationParams
from ..core.engines import create_engine
from ..reasoning.aggregators import majority_vote
from ..reasoning.controller import self_evaluate_batched, self_consistency_batch, two_pass_batch  # use batched judge
from ..data.adapters import load_gsm8k, load_mmlu, load_csqa, exact_match, Sample, iter_dataset
from ..metrics.flop_estimation import to_tflops
from ..logs.wandb_logger import WandbRunLogger

# Optional direct import from vLLM for batched generation
from vllm import SamplingParams

# ============================================================================
# FLOP CALIBRATION SYSTEM
# ============================================================================

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

class FLOPCalibrationRunner:
    """
    Systematic calibration runner that tests different token combinations
    and measures actual FLOPs using DeepSpeed.
    """
    
    def __init__(self, 
                 prefill_ranges: List[int] = None,
                 generation_ranges: List[int] = None):
        """
        Initialize the calibration runner.
        
        Args:
            prefill_ranges: List of prefill token counts to test
            generation_ranges: List of generation token counts to test  
        """
        self.prefill_ranges = prefill_ranges or [16, 32, 64, 128, 256, 512, 1024]
        self.generation_ranges = generation_ranges or [16, 32, 64, 128, 256, 512]
        
        # Create all combinations
        self.combinations = list(itertools.product(self.prefill_ranges, self.generation_ranges))
        print(f"Calibration will test {len(self.combinations)} combinations")
    
    def _generate_test_prompts(self, prefill_tokens: int, generation_tokens: int, tokenizer=None) -> str:
        """
        Generate test prompts with exact target token counts using tokenizer.
        
        Args:
            prefill_tokens: Target prefill token count
            generation_tokens: Target generation token count (unused, kept for compatibility)
            tokenizer: HuggingFace tokenizer for exact token counting
            
        Returns:
            The generated prompt string
        """
        if tokenizer is not None:
            # Use tokenizer for exact token counting
            return self._generate_prompt_with_exact_tokens(prefill_tokens, tokenizer)
        else:
            # Fallback to estimation method
            return self._generate_prompt_with_estimation(prefill_tokens)
    
    def _generate_prompt_with_exact_tokens(self, target_tokens: int, tokenizer) -> str:
        """Generate a prompt with exactly the target number of random tokens using tokenizer."""
        # Method: Work directly with token IDs to ensure exact token count
        vocab_size = len(tokenizer.get_vocab())
        
        # Get special token IDs to avoid them
        special_token_ids = set()
        if tokenizer.bos_token_id is not None:
            special_token_ids.add(tokenizer.bos_token_id)
        if tokenizer.eos_token_id is not None:
            special_token_ids.add(tokenizer.eos_token_id)
        if tokenizer.pad_token_id is not None:
            special_token_ids.add(tokenizer.pad_token_id)
        if tokenizer.unk_token_id is not None:
            special_token_ids.add(tokenizer.unk_token_id)
        
        # Create a list of valid token IDs (excluding special tokens)
        valid_token_ids = [i for i in range(vocab_size) if i not in special_token_ids]
        
        # Select random token IDs
        selected_token_ids = np.random.choice(valid_token_ids, size=target_tokens, replace=True)
        
        # Decode the token IDs
        prompt = tokenizer.decode(selected_token_ids, skip_special_tokens=True)
        
        # Verify the token count
        actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        
        # If it doesn't match, keep trying until we get exact count
        max_attempts = 3
        attempt = 0
        
        while actual_tokens != target_tokens and attempt < max_attempts:
            selected_token_ids = np.random.choice(valid_token_ids, size=target_tokens, replace=True)
            prompt = tokenizer.decode(selected_token_ids, skip_special_tokens=True)
            actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            attempt += 1
        
        return prompt
    
    def _generate_prompt_with_estimation(self, target_tokens: int) -> str:
        """Fallback method using word estimation with random tokens."""
        # Scale the prompt to approximate the target token count
        # Rough estimate: 1 token ≈ 0.75 words
        target_words = int(target_tokens * 0.75)
        
        # Generate random words to reach target length
        random_words = []
        
        # Common words for random selection
        common_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "through", "during", "before", "after", "above", "below", "between", "among", "within", "without", "against", "toward", "upon", "across", "behind", "beyond", "under", "over", "around", "near", "far", "here", "there", "where", "when", "why", "how", "what", "which", "who", "whom", "whose", "this", "that", "these", "those", "some", "any", "many", "much", "few", "little", "more", "most", "less", "least", "all", "both", "each", "every", "either", "neither", "one", "two", "three", "first", "second", "last", "next", "other", "another", "such", "same", "different", "new", "old", "good", "bad", "big", "small", "long", "short", "high", "low", "fast", "slow", "hot", "cold", "dry", "wet", "clean", "dirty", "full", "empty", "open", "closed", "right", "wrong", "true", "false", "yes", "no", "always", "never", "sometimes", "often", "rarely", "usually", "probably", "maybe", "certainly", "definitely", "absolutely", "completely", "totally", "entirely", "partly", "mostly", "mainly", "especially", "particularly", "specifically", "generally", "basically", "essentially", "fundamentally", "primarily", "secondarily", "additionally", "furthermore", "moreover", "however", "therefore", "consequently", "thus", "hence", "accordingly", "meanwhile", "simultaneously", "previously", "subsequently", "initially", "finally", "ultimately", "eventually", "immediately", "instantly", "quickly", "slowly", "gradually", "suddenly", "carefully", "carelessly", "easily", "difficultly", "simply", "complexly", "clearly", "obviously", "apparently", "evidently", "supposedly", "allegedly", "reportedly", "presumably", "hopefully", "unfortunately", "fortunately", "luckily", "unluckily", "surprisingly", "unexpectedly", "predictably", "inevitably", "necessarily", "sufficiently", "adequately", "properly", "correctly", "accurately", "precisely", "exactly", "approximately", "roughly", "nearly", "almost", "quite", "very", "extremely", "highly", "greatly", "significantly", "considerably", "substantially", "dramatically", "slightly", "somewhat", "rather", "fairly", "pretty", "quite", "really", "truly", "actually", "literally", "figuratively", "metaphorically", "symbolically", "representatively", "typically", "normally", "usually", "commonly", "frequently", "regularly", "occasionally", "rarely", "seldom", "hardly", "barely", "scarcely", "almost", "nearly", "practically", "virtually", "essentially", "basically", "fundamentally", "primarily", "mainly", "mostly", "largely", "partly", "partially", "completely", "totally", "entirely", "fully", "wholly", "absolutely", "definitely", "certainly", "surely", "undoubtedly", "indeed", "truly", "really", "actually", "genuinely", "honestly", "sincerely", "seriously"]
        
        for _ in range(target_words):
            random_word = np.random.choice(common_words)
            random_words.append(random_word)
        
        return " ".join(random_words)
    
    def run_calibration(self, 
                       model_spec: RunSpec,
                       save_path: Optional[str] = None) -> CalibrationDataset:
        """
        Run the complete calibration process for a model.
        
        Args:
            model_spec: Model specification for calibration
            save_path: Optional path to save calibration results
            
        Returns:
            CalibrationDataset with all measurements and fitted model
        """
        print(f"\n=== Starting FLOP Calibration for {model_spec.model_name} ===")
        print(f"Model: {model_spec.hf_repo}")
        print(f"Engine: {model_spec.engine}")
        
        # Create DeepSpeed engine for calibration
        calibration_spec = copy.deepcopy(model_spec)
        calibration_spec.engine = "deepspeed"
        
        # Check GPU memory before creating engine
        if torch.cuda.is_available():
            print(f"Pre-calibration GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
        
        engine = create_engine(
            calibration_spec.engine,
            model_id=calibration_spec.hf_repo,
            dtype=calibration_spec.backend.dtype,
            gpu_memory_utilization=calibration_spec.backend.gpu_memory_utilization,
            enforce_eager=calibration_spec.backend.enforce_eager,
        )
        
        # Try to load tokenizer for exact token counting
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(calibration_spec.hf_repo)
            print(f"✓ Loaded tokenizer for exact token counting")
        except Exception as e:
            print(f"⚠ Could not load tokenizer ({e}), falling back to estimation")
            tokenizer = None
        
        calibration_points = []
        
        # Progress bar for calibration
        pbar = tqdm(total=len(self.combinations), desc="Calibration Progress", unit="run")
        
        try:
            for prefill_tokens, generation_tokens in self.combinations:
                full_prompt = self._generate_test_prompts(
                    prefill_tokens, generation_tokens, tokenizer
                )
                
                # Log actual vs target token counts if using tokenizer
                if tokenizer is not None:
                    actual_tokens = len(tokenizer.encode(full_prompt))
                    # Always use actual token count for dataset accuracy
                    prefill_tokens = actual_tokens
                    pbar.set_postfix({
                        'target_prefill': prefill_tokens,
                        'actual_tokens': actual_tokens,
                        'target_gen': generation_tokens
                    })
                
                # Generation parameters
                gen_params = GenerationParams(
                    max_new_tokens=generation_tokens,
                    temperature=0.1,  # Low temperature for consistent results
                    top_p=0.9,
                    use_kv_cache=True,
                    dtype=calibration_spec.backend.dtype,
                )
                
                # Generate and measure
                result = engine.generate(full_prompt, gen_params)
                
                # Extract FLOP measurements
                if result.raw and result.raw.get("flops"):
                    flops_data = result.raw["flops"]
                    total_flops = flops_data.get("total_flops", 0)
                    
                    calibration_point = CalibrationPoint(
                        prefill_tokens=prefill_tokens,
                        generation_tokens=generation_tokens,
                        measured_flops=total_flops,
                        latency_ms=result.latency_ms or 0.0,
                        timestamp=time.time()
                    )
                    calibration_points.append(calibration_point)
                    
                    pbar.set_postfix({
                        'P': prefill_tokens,
                        'G': generation_tokens,
                        'FLOPs': f"{total_flops/1e12:.2f}T"
                    })
                else:
                    print(f"Warning: No FLOP data for {prefill_tokens}P/{generation_tokens}G")
                
                pbar.update(1)
        
        finally:
            pbar.close()
            engine.close()
        
        # Create calibration dataset
        calibration_dataset = CalibrationDataset(
            model_name=model_spec.model_name,
            model_config={
                "hf_repo": model_spec.hf_repo,
                "params_B": model_spec.card.params_B,
                "layers": model_spec.card.layers,
                "hidden_dim": model_spec.card.hidden_dim,
                "heads": model_spec.card.heads,
                "dtype": model_spec.backend.dtype,
            },
            points=calibration_points
        )
        
        # Fit extrapolation model
        if len(calibration_points) >= 3:
            extrapolation_model = FLOPExtrapolationModel(degree=2, include_interaction=True)
            accuracy_metrics = extrapolation_model.fit(calibration_points)
            calibration_dataset.extrapolation_model = extrapolation_model
            calibration_dataset.model_accuracy = accuracy_metrics
            
            print(f"\n=== Calibration Complete ===")
            print(f"Valid measurements: {len(calibration_points)}")
            print(f"Model R²: {accuracy_metrics['r2_score']:.4f}")
            print(f"Model MAE: {accuracy_metrics['mae']/1e12:.2f} TFLOPs")
            print(f"Model RMSE: {accuracy_metrics['rmse']/1e12:.2f} TFLOPs")
        else:
            print(f"Warning: Insufficient calibration points ({len(calibration_points)}) for model fitting")
        
        # Save calibration data if requested
        if save_path:
            self._save_calibration_dataset(calibration_dataset, save_path)
        
        return calibration_dataset
    
    def _save_calibration_dataset(self, dataset: CalibrationDataset, save_path: str) -> None:
        """Save calibration dataset to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Prepare data for JSON serialization
        data = {
            "model_name": dataset.model_name,
            "model_config": dataset.model_config,
            "points": [
                {
                    "prefill_tokens": p.prefill_tokens,
                    "generation_tokens": p.generation_tokens,
                    "measured_flops": p.measured_flops,
                    "latency_ms": p.latency_ms,
                    "timestamp": p.timestamp
                }
                for p in dataset.points
            ],
            "model_accuracy": dataset.model_accuracy
        }
        
        # Add model info if available
        if dataset.extrapolation_model:
            data["extrapolation_model"] = dataset.extrapolation_model.get_model_info()
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Calibration data saved to: {save_path}")

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
        model = FLOPExtrapolationModel(
            degree=model_info["degree"],
            include_interaction=model_info["include_interaction"]
        )
        model.feature_names = np.array(model_info["feature_names"])
        model.regressor.coef_ = np.array(model_info["coefficients"])
        model.regressor.intercept_ = model_info["intercept"]
        model.r2_score = model_info["r2_score"]
        model.mae = model_info["mae"]
        model.rmse = model_info["rmse"]
        model.is_fitted = True
        dataset.extrapolation_model = model
    
    return dataset

# ============================================================================
# ENHANCED BENCHMARKING WITH CALIBRATION
# ============================================================================

def _build_sample_trace_logging(sample_traces: List[dict], sample_idx: int) -> dict:
    """Build logging fields for a sample trace, handling both two-pass and self-consistency formats."""
    if sample_idx >= len(sample_traces):
        return {
            f"sample{sample_idx+1}_question": None,
            f"sample{sample_idx+1}_golden_answer": None,
            f"sample{sample_idx+1}_judge_answer": None,
        }
    
    trace = sample_traces[sample_idx]
    sample_prefix = f"sample{sample_idx+1}"
    
    # Base fields
    result = {
        f"{sample_prefix}_question": trace.get("question"),
        f"{sample_prefix}_golden_answer": trace.get("gold"),
        f"{sample_prefix}_judge_answer": trace.get("judge_text"),
    }
    
    # Check if this is self-consistency (has chosen_answer) or two-pass (has think_text/answer_text)
    if "chosen_answer" in trace:
        # Self-consistency format - log chosen answer and all K paths
        result[f"{sample_prefix}_chosen_answer"] = trace.get("chosen_answer")
        
        # Log up to 5 paths (assuming max K=5)
        for k in range(1, 6):
            think_key = f"path_{k}_think"
            answer_key = f"path_{k}_answer"
            if think_key in trace:
                result[f"{sample_prefix}_path{k}_think"] = trace.get(think_key, "")
                result[f"{sample_prefix}_path{k}_answer"] = trace.get(answer_key, "")
            else:
                result[f"{sample_prefix}_path{k}_think"] = None
                result[f"{sample_prefix}_path{k}_answer"] = None
    else:
        # Two-pass format - use original fields
        result[f"{sample_prefix}_first_pass"] = trace.get("think_text")
        result[f"{sample_prefix}_second_pass"] = trace.get("answer_text")
    
    return result

def run_one_with_calibration(spec: RunSpec, 
                           calibration_dataset: Optional[CalibrationDataset] = None,
                           batch_size: Optional[int] = None, 
                           wandb_project: str | None = None, 
                           notes: str = "") -> None:
    """
    Run a single benchmark with optional FLOP calibration.
    
    Args:
        spec: Benchmark specification
        calibration_dataset: Optional calibration data for FLOP extrapolation
        batch_size: Override batch size from config
        wandb_project: W&B project name
        notes: Additional notes
    """
    # Use configured batch size unless overridden
    bs = int(batch_size or getattr(spec, "batch_size", 1) or 1)

    # Build engine (one per run), then tear down afterwards
    print("Specified engine: ", spec.engine)
    
    # Check GPU memory before creating main engine
    if torch.cuda.is_available():
        print(f"Pre-benchmark GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved")
    
    engine = create_engine(
        spec.engine,
        model_id=spec.hf_repo,
        dtype=spec.backend.dtype,
        gpu_memory_utilization=spec.backend.gpu_memory_utilization,
        enforce_eager=spec.backend.enforce_eager,
        # Pass quantization parameters
        quantization=spec.backend.quantization,
        quantization_param_path=spec.backend.quantization_param_path,
        # Pass CPU offloading parameters
        cpu_offload_gb=spec.backend.cpu_offload_gb,
        swap_space=spec.backend.swap_space,
        # Pass additional memory optimization parameters
        max_model_len=spec.backend.max_model_len,
        block_size=spec.backend.block_size,
    )

    gen = GenerationParams(
        max_new_tokens=spec.generation.max_new_tokens,
        temperature=spec.generation.temperature,
        top_p=spec.generation.top_p,
        top_k=spec.generation.top_k,
        stop=spec.generation.stop,
        seed=spec.generation.seed,
        use_kv_cache=spec.generation.use_kv_cache,
        dtype=spec.backend.dtype,
    )

    # W&B: one run per (model, dataset, budget, style, k, batch, prompt_set)
    wb = None
    run_name = f"{spec.model_name}|{spec.dataset}|style={spec.reasoning.style}|B={spec.think_budget}|K={spec.reasoning.self_consistency_k}|bs={bs}|prompt={spec.prompt_set_name}"
    if wandb_project:
        cfg = {
            "model": spec.hf_repo,
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "dataset": spec.dataset,
            "style": spec.reasoning.style,
            "think_budget": spec.think_budget,
            "K": spec.reasoning.self_consistency_k,
            "dtype": spec.backend.dtype,
            "batch_size": bs,
            "prompt_set": spec.prompt_set_name,
            "config_name": spec.config_name,
            # Generation parameters
            "temperature": spec.generation.temperature,
            "top_p": spec.generation.top_p,
            "top_k": spec.generation.top_k,
            "max_new_tokens": spec.generation.max_new_tokens,
            "stop": spec.generation.stop,
            "seed": spec.generation.seed,
            "use_kv_cache": spec.generation.use_kv_cache,
            # Calibration info
            "has_calibration": calibration_dataset is not None,
            "calibration_points": len(calibration_dataset.points) if calibration_dataset else 0,
        }
        wb = WandbRunLogger(project=wandb_project, run_name=run_name, config=cfg)

    # Materialize dataset to know total for tqdm
    examples: List[Sample] = list(iter_dataset(spec.dataset))
    total_n = len(examples)

    # Iterate dataset in batches
    total, correct_measured, correct_self = 0, 0, 0
    prompt_tok_sum, gen_tok_sum = 0, 0
    lat_ms_sum = 0.0

    pbar = tqdm(total=total_n, desc=run_name, unit="ex")

    # Collect up to 3 sample traces for logging/inspection
    sample_traces = []  # list of dicts with question, gold, answer_prompt, answer_text, judge_text
    judge_input_prompt = None
    judge_response = None
    
    # Collect per-datapoint token counts for FLOP calculation
    per_datapoint_flops = []  # list of FLOP calculations per datapoint

    for i in range(0, total_n, bs):
        batch = examples[i:i+bs]
        qs = [ex.question for ex in batch]
        gts = [ex.gold for ex in batch]

        if spec.reasoning.self_consistency_k and spec.reasoning.self_consistency_k > 1:
            outs = self_consistency_batch(
                engine, qs, gen, spec.think_budget, spec.reasoning.style, spec.prompts, spec.reasoning.self_consistency_k
            )
            preds = [o["chosen_answer"] for o in outs]
            # For metrics, sum tokens across all K paths (full generation cost)
            think_toks = [sum([p["think_tokens"] for p in o["paths"]]) for o in outs]
            ans_toks   = [sum([p["answer_tokens"] for p in o["paths"]]) for o in outs]
            lats       = [float(sum([p["latency_ms_think"] + p["latency_ms_answer"] for p in o["paths"]]) / len(o["paths"])) for o in outs]
        else:
            outs = two_pass_batch(engine, qs, gen, spec.think_budget, spec.reasoning.style, spec.prompts)
            preds = [o["answer_text"] for o in outs]
            think_toks = [o["think_tokens"] for o in outs]
            ans_toks   = [o["answer_tokens"] for o in outs]
            lats       = [o["latency_ms_think"] + o["latency_ms_answer"] for o in outs]

        # Optional batched self-evaluation (YES/NO judge)
        judge_batch_results: Optional[List[Tuple[bool, str, str]]] = None
        if spec.reasoning.self_eval:
            # Use OpenAI engine for evaluation if specified
            eval_engine = None
            if spec.reasoning.openai_eval:
                try:
                    from ..core.engines import create_openai_engine
                    eval_engine = create_openai_engine()
                    print("Using OpenAI API for evaluation")
                except Exception as e:
                    print(f"Failed to create OpenAI evaluation engine: {e}")
                    print("Falling back to main engine for evaluation")
            
            judge_batch_results = self_evaluate_batched(engine, qs, preds, gts, gen, spec.prompts, eval_engine)

        # Accumulate metrics
        for j, ex in enumerate(batch):
            total += 1
            ok = exact_match(preds[j], gts[j])
            correct_measured += int(ok)

            if judge_batch_results is not None:
                judge_yes, judge_input_prompt, judge_response = judge_batch_results[j]
                correct_self += int(judge_yes)

            # Calculate tokens for this datapoint
            prompt_tokens = len(ex.question.split())  # Approximate prompt tokens
            generated_tokens = think_toks[j] + ans_toks[j]
            
            gen_tok_sum += generated_tokens
            prompt_tok_sum += prompt_tokens
            lat_ms_sum += float(lats[j])

            # Calculate FLOPs for this individual datapoint
            datapoint_flops = {}
            
            # Use calibration-based FLOP estimation if available
            if calibration_dataset and calibration_dataset.extrapolation_model:
                try:
                    extrapolated_flops = calibration_dataset.extrapolation_model.predict(
                        prompt_tokens, generated_tokens
                    )
                    datapoint_flops['extrapolated'] = extrapolated_flops
                except Exception as e:
                    print(f"Warning: FLOP extrapolation failed: {e}")
                    datapoint_flops['extrapolated'] = None
            else:
                datapoint_flops['extrapolated'] = None

            per_datapoint_flops.append(datapoint_flops)

            # Save up to 3 sample traces
            if len(sample_traces) < 3:
                # Handle different output structures (self-consistency vs two-pass)
                if "think_text" in outs[j]:
                    # Two-pass batch output
                    think_text = outs[j]["think_text"]
                    answer_text = outs[j]["answer_text"]
                    trace = {
                        "question": ex.question,
                        "gold": ex.gold,
                        "think_text": think_text,
                        "answer_text": answer_text,
                        "judge_text": (judge_batch_results[j][2] if judge_batch_results is not None else None),
                    }
                else:
                    # Self-consistency output - log all K paths
                    paths = outs[j]["paths"] if outs[j]["paths"] else []
                    chosen_answer = outs[j]["chosen_answer"]
                    
                    # Create trace with all K paths
                    trace = {
                        "question": ex.question,
                        "gold": ex.gold,
                        "chosen_answer": chosen_answer,
                        "judge_text": (judge_batch_results[j][2] if judge_batch_results is not None else None),
                    }
                    
                    # Add each path's generated text
                    for k_idx, path in enumerate(paths):
                        trace[f"path_{k_idx+1}_think"] = path.get("think_text", "")
                        trace[f"path_{k_idx+1}_answer"] = path.get("answer_text", "")
                
                sample_traces.append(trace)

        pbar.update(len(batch))

    pbar.close()

    # Compute FLOPs correctly by averaging per-datapoint calculations
    avg_extrapolated_flops = None
    if calibration_dataset and calibration_dataset.extrapolation_model:
        extrapolated_flops_list = [dp['extrapolated'] for dp in per_datapoint_flops if dp['extrapolated'] is not None]
        if extrapolated_flops_list:
            avg_extrapolated_flops = sum(extrapolated_flops_list) / len(extrapolated_flops_list)
    
    # Average tokens for reference
    avg_gen_tokens = (gen_tok_sum / max(total, 1))

    # Row for logging (one row per run)
    row = {
        # Model/setup
        "model": spec.hf_repo,
        "arch": "decoder-only",
        "params_B": spec.card.params_B,
        "layers": spec.card.layers,
        "d_model": spec.card.hidden_dim,
        "heads": spec.card.heads,
        "precision": spec.backend.dtype,
        "quant": None,
        "hardware": "NVIDIA RTX 6000 Pro Blackwell",
        "batch_size": bs,
        "use_kv_cache": spec.generation.use_kv_cache,
        "reasoning_style": spec.reasoning.style,
        "prompt_set": spec.prompt_set_name,

        # Tokens (averaged per datapoint)
        "avg_prompt_tokens": prompt_tok_sum / max(total, 1),
        "avg_gen_tokens": avg_gen_tokens,
        "passes": 2,
        "beam_width": 1,
        "self_consistency_k": spec.reasoning.self_consistency_k,

        # Measured
        "latency_ms": lat_ms_sum / max(total, 1),
        "speed_tok_per_s": (gen_tok_sum / (lat_ms_sum / 1000.0)) if lat_ms_sum > 0 else None,
        "energy_j": None,

        # Task metric
        "dataset": spec.dataset,
        "metric_name": "exact_match",
        "accuracy": correct_measured / max(total, 1),

        # Extras (self-eval + FLOPs)
        "self_eval_acc": (correct_self / max(total, 1)) if spec.reasoning.self_eval else None,

        # FLOP estimates (averaged per datapoint)
        "avg_flops_extrapolated_tflops": to_tflops(avg_extrapolated_flops) if avg_extrapolated_flops is not None else None,

        # Sample traces (up to 3) - handles both two-pass and self-consistency
        **_build_sample_trace_logging(sample_traces, 0),
        **_build_sample_trace_logging(sample_traces, 1),
        **_build_sample_trace_logging(sample_traces, 2),
        "prompt_cot_think": spec.prompts.cot_think,
        "prompt_answer": spec.prompts.answer,
        "prompt_direct": spec.prompts.direct,
        "prompt_plan_think": spec.prompts.plan_think,
        "prompt_self_eval": spec.prompts.self_eval,

        # Notes
        "notes": f"{notes}",
    }

    # Add calibration info if available
    if calibration_dataset:
        row["calibration_points"] = len(calibration_dataset.points)
        if calibration_dataset.model_accuracy:
            row["calibration_r2"] = calibration_dataset.model_accuracy.get("r2_score")
            row["calibration_mae_tflops"] = calibration_dataset.model_accuracy.get("mae", 0) / 1e12
            row["calibration_rmse_tflops"] = calibration_dataset.model_accuracy.get("rmse", 0) / 1e12

    extrapolated_flops_str = f"{row['avg_flops_extrapolated_tflops']:.2f}" if row['avg_flops_extrapolated_tflops'] is not None else "N/A"
    
    flops_info = f"avg_extrapolated_tFLOPs≈{extrapolated_flops_str}"
    
    print(f"[RUN] {spec.model_name} | {spec.dataset} | style={spec.reasoning.style} | "
          f"B={spec.think_budget} | K={spec.reasoning.self_consistency_k} | bs={bs} | prompt={spec.prompt_set_name} | "
          f"acc={row['accuracy']:.3f} | avg_gen_tokens={avg_gen_tokens:.2f} | {flops_info}")

    if wb:
        wb.log_row(row)
        
        # Log calibration data and metrics to the same wandb run
        if calibration_dataset:
            print("Logging calibration data to wandb...")
            try:
                wb.log_calibration_data(calibration_dataset, spec)
                wb.log_calibration_metrics(calibration_dataset)
                print("✓ Calibration data logged to wandb successfully")
            except Exception as e:
                print(f"Warning: Failed to log calibration data to wandb: {e}")
        
        wb.finish()

    # Tear down engine and free memory between runs
    engine.close()

def main():
    ap = argparse.ArgumentParser(description="Benchmark with FLOP calibration")
    ap.add_argument("--config", required=True, help="Path to YAML bench config")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    
    # Calibration options
    ap.add_argument("--calibration_prefill_ranges", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                   help="Prefill token ranges for calibration")
    ap.add_argument("--calibration_generation_ranges", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32, 64],
                   help="Generation token ranges for calibration")
    
    args = ap.parse_args()

    cfg = load_bench_config(args.config)

    for spec in expand_runs(cfg):
        try:
            # Check for existing calibration
            calibration_base_dir = "calibration_models"
            model_dir = spec.hf_repo.replace("/", "_")
            calibration_dir = os.path.join(calibration_base_dir, model_dir)
            calibration_file = os.path.join(calibration_dir, "calibration.json")
            
            calibration_dataset = None
            
            if os.path.exists(calibration_file):
                print(f"Loading existing calibration from: {calibration_file}")
                try:
                    calibration_dataset = load_calibration_dataset(calibration_file)
                    print(f"✓ Loaded calibration with {len(calibration_dataset.points)} points")
                    if calibration_dataset.model_accuracy:
                        print(f"  Model R²: {calibration_dataset.model_accuracy.get('r2_score', 'N/A'):.4f}")
                except Exception as e:
                    print(f"Warning: Failed to load calibration: {e}")
                    calibration_dataset = None
            
            if calibration_dataset is None:
                # Run new calibration
                print(f"Running new calibration for {spec.model_name}")
                calibration_runner = FLOPCalibrationRunner(
                    prefill_ranges=args.calibration_prefill_ranges,
                    generation_ranges=args.calibration_generation_ranges
                )
                
                calibration_dataset = calibration_runner.run_calibration(spec, save_path=calibration_file)
            
            # Add 5-second sleep between calibration and run to ensure GPU memory is freed
            print("Waiting 5 seconds for GPU memory cleanup...")
            time.sleep(5)
            
            # Run the benchmark with calibration
            run_one_with_calibration(
                spec, 
                calibration_dataset=calibration_dataset,
                batch_size=args.batch_size, 
                wandb_project=args.wandb_project, 
                notes=args.notes
            )
            
        except Exception as e:
            err_msg = f"[ERROR] model={spec.model_name}, dataset={spec.dataset}: {e}\n"
            traceback_str = traceback.format_exc()
            
            # Write to error log file
            with open("error.log", "a", encoding="utf-8") as f:
                f.write(err_msg)
                f.write(traceback_str + "\n")
            
            # Print error message and full traceback to stdout
            print(err_msg.strip())
            print(traceback_str)

if __name__ == "__main__":
    main()
