"""
OpenAI API inference engine for text generation.
Supports multiple models for rate limit handling and sequential usage.
"""

import os
import time
import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import openai
from openai import OpenAI

from .base import BaseEngine
from ..interfaces import GenerationParams, GenerationResult


class QuotaExceededError(Exception):
    """Custom exception for OpenAI quota exceeded errors."""
    def __init__(self, message, quota_info=None):
        super().__init__(message)
        self.quota_info = quota_info or {}


def parse_quota_info(error_message):
    """Parse quota information from OpenAI error message."""
    import json
    import re
    from datetime import datetime, timedelta
    
    quota_info = {
        'error_type': 'unknown',
        'message': '',
        'reset_time': None,
        'remaining_requests': None,
        'remaining_tokens': None,
        'limit_requests': None,
        'limit_tokens': None
    }
    
    try:
        error_str = str(error_message)
        
        # Try to extract dictionary from the error message
        dict_match = re.search(r'\{.*\}', error_str)
        if dict_match:
            dict_str = dict_match.group()
            try:
                # First try JSON parsing (double quotes)
                error_data = json.loads(dict_str)
            except json.JSONDecodeError:
                try:
                    # Try converting single quotes to double quotes
                    json_str = dict_str.replace("'", '"')
                    error_data = json.loads(json_str)
                except json.JSONDecodeError:
                    try:
                        # Try using ast.literal_eval for Python dict format
                        import ast
                        error_data = ast.literal_eval(dict_str)
                    except (ValueError, SyntaxError):
                        error_data = None
            
            if error_data and 'error' in error_data:
                error_obj = error_data['error']
                quota_info['error_type'] = error_obj.get('type', 'unknown')
                quota_info['message'] = error_obj.get('message', '')
        
        # Look for rate limit headers in the error (if available)
        # These are typically in the response headers but might be in error context
        if hasattr(error_message, 'response') and hasattr(error_message.response, 'headers'):
            headers = error_message.response.headers
            quota_info.update({
                'remaining_requests': headers.get('x-ratelimit-remaining-requests'),
                'remaining_tokens': headers.get('x-ratelimit-remaining-tokens'),
                'limit_requests': headers.get('x-ratelimit-limit-requests'),
                'limit_tokens': headers.get('x-ratelimit-limit-tokens'),
                'reset_time': headers.get('x-ratelimit-reset-requests')
            })
        
        # Parse reset time if available
        if quota_info['reset_time']:
            try:
                reset_timestamp = int(quota_info['reset_time'])
                quota_info['reset_time'] = datetime.fromtimestamp(reset_timestamp)
            except (ValueError, TypeError):
                pass
                
    except (AttributeError, KeyError):
        # If parsing fails, just store the raw message
        quota_info['message'] = str(error_message)
    
    return quota_info


@dataclass
class OpenAIEngineConfig:
    """Configuration for OpenAI API engine."""
    api_key: str
    base_url: str
    models: List[str]  # List of models to use sequentially
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 60.0


class OpenAIInferenceEngine(BaseEngine):
    """
    OpenAI API inference engine with support for multiple models and rate limit handling.
    
    Features:
    - Sequential model usage for rate limit management
    - Automatic retry with exponential backoff
    - Response header parsing for rate limit info
    - Support for all OpenAI API parameters
    """
    
    def __init__(self, config: OpenAIEngineConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
        self.current_model_index = 0
        self.rate_limit_info = {}
        
    def _get_current_model(self) -> str:
        """Get the current model for this request."""
        model = self.config.models[self.current_model_index]
        # Rotate to next model for next request
        self.current_model_index = (self.current_model_index + 1) % len(self.config.models)
        return model
    
    def _parse_rate_limit_headers(self, response_headers: Dict[str, str]) -> None:
        """Parse rate limit information from response headers."""
        self.rate_limit_info = {
            'requests_remaining': response_headers.get('x-ratelimit-remaining-requests'),
            'tokens_remaining': response_headers.get('x-ratelimit-remaining-tokens'),
            'requests_reset': response_headers.get('x-ratelimit-reset-requests'),
            'tokens_reset': response_headers.get('x-ratelimit-reset-tokens'),
            'all_headers': response_headers  # Store all headers for debugging
        }
        
        # Log rate limit info if available
        if self.rate_limit_info['requests_remaining']:
            print(f"Rate limit: {self.rate_limit_info['requests_remaining']} requests remaining")
        
        # Print all headers for debugging

    def _log_quota_info(self, quota_info: Dict[str, Any]) -> None:
        """Log detailed quota information."""
        print("📊 OpenAI Quota Information:")
        print(f"   Error Type: {quota_info.get('error_type', 'unknown')}")
        print(f"   Message: {quota_info.get('message', 'No message available')}")
        
        if quota_info.get('remaining_requests') is not None:
            print(f"   Remaining Requests: {quota_info['remaining_requests']}")
        if quota_info.get('remaining_tokens') is not None:
            print(f"   Remaining Tokens: {quota_info['remaining_tokens']}")
        if quota_info.get('limit_requests') is not None:
            print(f"   Request Limit: {quota_info['limit_requests']}")
        if quota_info.get('limit_tokens') is not None:
            print(f"   Token Limit: {quota_info['limit_tokens']}")
        
        if quota_info.get('reset_time'):
            if isinstance(quota_info['reset_time'], str):
                print(f"   Reset Time: {quota_info['reset_time']}")
            else:
                from datetime import datetime
                now = datetime.now()
                reset_time = quota_info['reset_time']
                if isinstance(reset_time, datetime):
                    time_diff = reset_time - now
                    if time_diff.total_seconds() > 0:
                        hours = int(time_diff.total_seconds() // 3600)
                        minutes = int((time_diff.total_seconds() % 3600) // 60)
                        print(f"   Quota Resets In: {hours}h {minutes}m ({reset_time.strftime('%Y-%m-%d %H:%M:%S')})")
                    else:
                        print(f"   Quota Reset Time: {reset_time.strftime('%Y-%m-%d %H:%M:%S')} (may have already reset)")
        
        print("   🔄 Falling back to main engine for evaluation...")
    
    def _make_request_with_retry(self, messages: List[Dict[str, str]], params: GenerationParams) -> Dict[str, Any]:
        """Make OpenAI API request with retry logic."""
        model = self._get_current_model()
        
        # Convert GenerationParams to OpenAI API format for GPT-5
        # Use direct prompt (no chat template structure)
        input_text = messages[0]['content'] if messages else ""
        
        openai_params = {
            'model': model,
            'input': input_text,
            'text': {'verbosity': 'low'},
            'max_output_tokens': 100,  # Higher value for better responses
            'reasoning': {'effort': 'minimal'}
        }
        
        # GPT-5-mini doesn't support temperature or top_p parameters
        # These are handled by the verbosity and max_output_tokens parameters
        
        # GPT-5-mini doesn't support stop or seed parameters
        
        # Remove None values
        openai_params = {k: v for k, v in openai_params.items() if v is not None}
        
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.responses.create(**openai_params)
                
                # Note: The newer OpenAI client library doesn't expose HTTP headers directly
                # in the response object. Headers are typically available in the underlying
                # HTTP response, but the client abstracts them away for security reasons.
                # 
                # Rate limit information is usually available in the response body or
                # through specific error responses when rate limits are hit.
                
                # Store basic response info for GPT-5
                self.rate_limit_info = {
                    'model_used': model,
                    'response_id': getattr(response, 'id', 'unknown'),
                    'created': getattr(response, 'created', None),
                    'note': 'GPT-5 responses.create() API'
                }
                
                return response
                
            except openai.RateLimitError as e:
                last_exception = e
                print(f"Rate limit hit (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                
                # Check if this is a quota exceeded error (429 with insufficient_quota)
                error_str = str(e).lower()
                if "insufficient_quota" in error_str or "quota" in error_str:
                    print("Quota exceeded - raising QuotaExceededError for fallback")
                    quota_info = parse_quota_info(e)
                    self._log_quota_info(quota_info)
                    raise QuotaExceededError(f"OpenAI quota exceeded: {e}", quota_info)
                
                # Try next model if available
                if len(self.config.models) > 1:
                    model = self._get_current_model()
                    openai_params['model'] = model
                    print(f"Switching to model: {model}")
                
                # Wait before retry
                wait_time = self.config.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Waiting {wait_time:.2f}s before retry...")
                time.sleep(wait_time)
                
            except openai.APIError as e:
                last_exception = e
                print(f"API error (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                
                # Check if this is a quota exceeded error (429 with insufficient_quota)
                error_str = str(e).lower()
                if "insufficient_quota" in error_str or "quota" in error_str or "429" in error_str:
                    print("Quota exceeded - raising QuotaExceededError for fallback")
                    quota_info = parse_quota_info(e)
                    self._log_quota_info(quota_info)
                    raise QuotaExceededError(f"OpenAI quota exceeded: {e}", quota_info)
                
                if attempt < self.config.max_retries - 1:
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    break
                    
            except Exception as e:
                last_exception = e
                print(f"Unexpected error (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                break
        
        # If all retries failed, raise the last exception
        raise last_exception or Exception("All retry attempts failed")
    
    def generate(self, prompt: str, params: GenerationParams) -> GenerationResult:
        """Generate text for a single prompt."""
        messages = [{"role": "user", "content": prompt}]
        
        start_time = time.time()
        response = self._make_request_with_retry(messages, params)
        end_time = time.time()
        
        # Extract response data for GPT-5
        text = getattr(response, 'output_text', '') or ""
        # GPT-5 might not have usage info in the same format
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        
        # Calculate timing
        latency_ms = (end_time - start_time) * 1000
        
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            raw={
                'model': self._get_current_model(),
                'finish_reason': getattr(response, 'finish_reason', 'stop'),
                'rate_limit_info': self.rate_limit_info.copy()
            }
        )
    
    def generate_batch(self, prompts: List[str], params: GenerationParams) -> List[GenerationResult]:
        """Generate text for multiple prompts sequentially."""
        results = []
        
        for i, prompt in enumerate(prompts):
            result = self.generate(prompt, params)
            results.append(result)
            
            # Small delay between requests to be respectful
            if i < len(prompts) - 1:
                time.sleep(0.05)
        
        return results
    
    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information."""
        return self.rate_limit_info.copy()


def create_openai_engine() -> OpenAIInferenceEngine:
    """Create OpenAI engine from environment variables."""
    from ...utils import get_env_var, check_required_env_vars, load_env_variables
    
    # Ensure environment variables are loaded
    load_env_variables()
    
    # Check required environment variables
    if not check_required_env_vars(['OPENAI_API_KEY']):
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    api_key = get_env_var('OPENAI_API_KEY', required=True)
    base_url = get_env_var('OPENAI_BASE_URL', default='https://api.openai.com/v1')
    models_str = get_env_var('OPENAI_MODELS', default='gpt-4o-mini')
    
    # Parse models (comma-separated, no whitespaces)
    models = [model.strip() for model in models_str.split(',') if model.strip()]
    
    if not models:
        raise ValueError("At least one model must be specified in OPENAI_MODELS")
    
    config = OpenAIEngineConfig(
        api_key=api_key,
        base_url=base_url,
        models=models
    )
    
    return OpenAIInferenceEngine(config)
