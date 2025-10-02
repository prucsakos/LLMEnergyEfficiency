"""
Calibration runners for different FLOP estimation methods.
"""

from __future__ import annotations
import os
import json
import time
import copy
import itertools
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from tqdm.auto import tqdm

from .data import CalibrationPoint, CalibrationDataset
from .models import FLOPExtrapolationModel, NextTokenFLOPModel
from ..logs.benchmark_logger import get_logger
from ..core.engines import create_engine
from ..config.bench_config import RunSpec
from ..core.interfaces import GenerationParams


class NextTokenCalibrationRunner:
    """
    Calibration runner for next-token FLOP estimation.
    
    This runner is more efficient than the full calibration because:
    1. Only measures single-token generation (faster)
    2. Uses a simpler 1D regression model
    3. Extrapolates to multi-token generation by summing individual costs
    """
    
    def __init__(self, 
                 prefill_ranges: List[int] = None,
                 generation_tokens: int = 1):
        """
        Initialize the next-token calibration runner.
        
        Args:
            prefill_ranges: List of prefill token counts to test
            generation_tokens: Number of tokens to generate per test (should be 1 for calibration)
        """
        self.prefill_ranges = prefill_ranges or [16, 32, 64, 128, 256, 512, 1024, 2048]
        self.generation_tokens = generation_tokens
        
        logger = get_logger()
        logger.info(f"Next-token calibration will test {len(self.prefill_ranges)} prefill lengths")
        logger.info(f"Generation tokens per test: {self.generation_tokens}")
    
    def _generate_test_prompts(self, prefill_tokens: int, generation_tokens: int, tokenizer=None) -> str:
        """
        Generate test prompts with exact target token counts using tokenizer.
        
        Args:
            prefill_tokens: Target prefill token count
            generation_tokens: Target generation token count (should be 1 for calibration)
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
                       save_path: Optional[str] = None,
                       estimation_points: int = 50) -> CalibrationDataset:
        """
        Run the next-token calibration process for a model.
        
        Args:
            model_spec: Model specification for calibration
            save_path: Optional path to save calibration results
            estimation_points: Number of estimation points for extrapolation evaluation
            
        Returns:
            CalibrationDataset with all measurements and fitted model
        """
        logger = get_logger()
        logger.info(f"\n=== Starting Next-Token FLOP Calibration for {model_spec.model_name} ===")
        logger.info(f"Model: {model_spec.hf_repo}")
        logger.info(f"Engine: {model_spec.engine}")
        
        # Create DeepSpeed engine for calibration
        calibration_spec = copy.deepcopy(model_spec)
        calibration_spec.engine = "deepspeed"
        
        # Check GPU memory before creating engine
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated()/1e9
            reserved_gb = torch.cuda.memory_reserved()/1e9
            logger.log_gpu_memory("Pre-calibration", allocated_gb, reserved_gb)
        
        engine = create_engine(
            "deepspeed",  # Always use DeepSpeed for calibration
            model_id=calibration_spec.hf_repo,
            dtype=calibration_spec.backend.dtype,
            gpu_memory_utilization=calibration_spec.backend.gpu_memory_utilization,
            enforce_eager=calibration_spec.backend.enforce_eager,
            quantization=calibration_spec.backend.quantization,
            quantization_param_path=calibration_spec.backend.quantization_param_path,
        )
        
        # Try to load tokenizer for exact token counting
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(calibration_spec.hf_repo)
            logger.info("✓ Loaded tokenizer for exact token counting")
        except Exception as e:
            logger.warning(f"⚠ Could not load tokenizer ({e}), falling back to estimation")
            tokenizer = None
        
        calibration_points = []
        
        # Progress bar for calibration
        pbar = tqdm(total=len(self.prefill_ranges), desc="Next-Token Calibration Progress", unit="run")
        
        try:
            for prefill_tokens in self.prefill_ranges:
                full_prompt = self._generate_test_prompts(
                    prefill_tokens, self.generation_tokens, tokenizer
                )
                
                # Log actual vs target token counts if using tokenizer
                if tokenizer is not None:
                    actual_tokens = len(tokenizer.encode(full_prompt))
                    # Always use actual token count for dataset accuracy
                    prefill_tokens = actual_tokens
                    pbar.set_postfix({
                        'target_prefill': prefill_tokens,
                        'actual_tokens': actual_tokens,
                        'gen_tokens': self.generation_tokens
                    })
                
                # Generation parameters - only generate 1 token for calibration
                gen_params = GenerationParams(
                    max_new_tokens=self.generation_tokens,
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
                        generation_tokens=self.generation_tokens,
                        measured_flops=total_flops,
                        latency_ms=result.latency_ms or 0.0,
                        timestamp=time.time()
                    )
                    calibration_points.append(calibration_point)
                    
                    pbar.set_postfix({
                        'P': prefill_tokens,
                        'G': self.generation_tokens,
                        'FLOPs': f"{total_flops/1e12:.2f}T"
                    })
                else:
                    logger.warning(f"No FLOP data for {prefill_tokens}P/{self.generation_tokens}G")
                
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
        
        # Fit next-token extrapolation model
        if len(calibration_points) >= 3:
            extrapolation_model = NextTokenFLOPModel(degree=2)
            accuracy_metrics = extrapolation_model.fit(calibration_points)
            calibration_dataset.extrapolation_model = extrapolation_model
            calibration_dataset.model_accuracy = accuracy_metrics
            
            logger.info(f"\n=== Next-Token Calibration Complete ===")
            logger.info(f"Valid measurements: {len(calibration_points)}")
            logger.info(f"Model R²: {accuracy_metrics['r2_score']:.4f}")
            logger.info(f"Model MAE: {accuracy_metrics['mae']/1e12:.2f} TFLOPs")
            logger.info(f"Model RMSE: {accuracy_metrics['rmse']/1e12:.2f} TFLOPs")
            
            # Generate estimation data to evaluate extrapolation performance
            logger.info("Generating estimation data to evaluate extrapolation performance...")
            estimation_data = self.generate_estimation_data(calibration_dataset, num_points=estimation_points)
            calibration_dataset.estimation_data = estimation_data
            logger.info(f"Generated {len(estimation_data)} estimation points for extrapolation evaluation")
        else:
            logger.warning(f"Insufficient calibration points ({len(calibration_points)}) for model fitting")
        
        # Save calibration data if requested
        if save_path:
            self._save_calibration_dataset(calibration_dataset, save_path)
        
        return calibration_dataset
    
    def generate_estimation_data(self, calibration_dataset: CalibrationDataset, num_points: int = 50) -> List[List[Any]]:
        """
        Generate estimation data for wandb logging.
        
        Args:
            calibration_dataset: The fitted calibration dataset
            num_points: Number of points per dimension for the estimation grid
            
        Returns:
            List of [prefill_tokens, generation_tokens, estimated_flops] tuples
        """
        logger = get_logger()
        
        if not calibration_dataset or not calibration_dataset.extrapolation_model:
            return []
        
        # Fixed prefill tokens = 50, variable generation tokens
        prefill_tokens = 50  # Fixed prefill
        generation_steps = np.linspace(1, 12000, num=num_points, dtype=int)  # Variable generation sizes
        
        # Create estimation data for each generation step with fixed prefill=50
        estimation_data = []
        for gen_tokens in tqdm(generation_steps, desc="Generating estimation data", unit="point"):
            try:
                estimated_flops = calibration_dataset.extrapolation_model.predict(prefill_tokens, gen_tokens)
                estimation_data.append([prefill_tokens, gen_tokens, estimated_flops])
            except Exception as e:
                logger.warning(f"Failed to predict FLOPs for P={prefill_tokens}, G={gen_tokens}: {e}")
                continue
        
        return estimation_data
    
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
        
        logger = get_logger()
        logger.info(f"Next-token calibration data saved to: {save_path}")

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
        self.prefill_ranges = prefill_ranges or [16, 32, 64, 128, 256, 512, 1024, 2048]
        self.generation_ranges = generation_ranges or [1, 2, 4, 8, 16, 32]
        
        logger = get_logger()
        total_combinations = len(self.prefill_ranges) * len(self.generation_ranges)
        logger.info(f"Calibration will test {total_combinations} combinations")
        logger.info(f"Prefill ranges: {self.prefill_ranges}")
        logger.info(f"Generation ranges: {self.generation_ranges}")
    
    def _generate_test_prompts(self, prefill_tokens: int, generation_tokens: int, tokenizer=None) -> str:
        """
        Generate test prompts with exact target token counts using tokenizer.
        
        Args:
            prefill_tokens: Target prefill token count
            generation_tokens: Target generation token count
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
                       save_path: Optional[str] = None,
                       estimation_points: int = 50) -> CalibrationDataset:
        """
        Run the complete calibration process for a model.
        
        Args:
            model_spec: Model specification for calibration
            save_path: Optional path to save calibration results
            estimation_points: Number of estimation points for extrapolation evaluation
            
        Returns:
            CalibrationDataset with all measurements and fitted model
        """
        logger = get_logger()
        logger.info(f"\n=== Starting FLOP Calibration for {model_spec.model_name} ===")
        logger.info(f"Model: {model_spec.hf_repo}")
        logger.info(f"Engine: {model_spec.engine}")
        
        # Create DeepSpeed engine for calibration
        calibration_spec = copy.deepcopy(model_spec)
        calibration_spec.engine = "deepspeed"
        
        # Check GPU memory before creating engine
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated()/1e9
            reserved_gb = torch.cuda.memory_reserved()/1e9
            logger.log_gpu_memory("Pre-calibration", allocated_gb, reserved_gb)
        
        engine = create_engine(
            "deepspeed",  # Always use DeepSpeed for calibration
            model_id=calibration_spec.hf_repo,
            dtype=calibration_spec.backend.dtype,
            gpu_memory_utilization=calibration_spec.backend.gpu_memory_utilization,
            enforce_eager=calibration_spec.backend.enforce_eager,
            quantization=calibration_spec.backend.quantization,
            quantization_param_path=calibration_spec.backend.quantization_param_path,
        )
        
        # Try to load tokenizer for exact token counting
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(calibration_spec.hf_repo)
            logger.info("✓ Loaded tokenizer for exact token counting")
        except Exception as e:
            logger.warning(f"⚠ Could not load tokenizer ({e}), falling back to estimation")
            tokenizer = None
        
        calibration_points = []
        
        # Create all combinations
        combinations = list(itertools.product(self.prefill_ranges, self.generation_ranges))
        
        # Progress bar for calibration
        pbar = tqdm(total=len(combinations), desc="Calibration Progress", unit="run")
        
        try:
            for prefill_tokens, generation_tokens in combinations:
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
                        'gen_tokens': generation_tokens
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
                    logger.warning(f"No FLOP data for {prefill_tokens}P/{generation_tokens}G")
                
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
            
            logger.info(f"\n=== Calibration Complete ===")
            logger.info(f"Valid measurements: {len(calibration_points)}")
            logger.info(f"Model R²: {accuracy_metrics['r2_score']:.4f}")
            logger.info(f"Model MAE: {accuracy_metrics['mae']/1e12:.2f} TFLOPs")
            logger.info(f"Model RMSE: {accuracy_metrics['rmse']/1e12:.2f} TFLOPs")
            
            # Generate estimation data to evaluate extrapolation performance
            logger.info("Generating estimation data to evaluate extrapolation performance...")
            estimation_data = self.generate_estimation_data(calibration_dataset, num_points=estimation_points)
            calibration_dataset.estimation_data = estimation_data
            logger.info(f"Generated {len(estimation_data)} estimation points for extrapolation evaluation")
        else:
            logger.warning(f"Insufficient calibration points ({len(calibration_points)}) for model fitting")
        
        # Save calibration data if requested
        if save_path:
            self._save_calibration_dataset(calibration_dataset, save_path)
        
        return calibration_dataset
    
    def generate_estimation_data(self, calibration_dataset: CalibrationDataset, num_points: int = 50) -> List[List[Any]]:
        """
        Generate estimation data for wandb logging.
        
        Args:
            calibration_dataset: The fitted calibration dataset
            num_points: Number of points per dimension for the estimation grid
            
        Returns:
            List of [prefill_tokens, generation_tokens, estimated_flops] tuples
        """
        logger = get_logger()
        
        if not calibration_dataset or not calibration_dataset.extrapolation_model:
            return []
        
        # Fixed prefill tokens = 50, variable generation tokens
        prefill_tokens = 50  # Fixed prefill
        generation_steps = np.linspace(1, 12000, num=num_points, dtype=int)  # Variable generation sizes
        
        # Create estimation data for each generation step with fixed prefill=50
        estimation_data = []
        for gen_tokens in tqdm(generation_steps, desc="Generating estimation data", unit="point"):
            try:
                estimated_flops = calibration_dataset.extrapolation_model.predict(prefill_tokens, gen_tokens)
                estimation_data.append([prefill_tokens, gen_tokens, estimated_flops])
            except Exception as e:
                logger.warning(f"Failed to predict FLOPs for P={prefill_tokens}, G={gen_tokens}: {e}")
                continue
        
        return estimation_data
    
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
        
        logger = get_logger()
        logger.info(f"Calibration data saved to: {save_path}")

