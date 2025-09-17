from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from ..interfaces import GenerationParams, GenerationResult

class BaseEngine(ABC):
    """Abstract base for engines."""
    @abstractmethod
    def generate(self, prompt: str, params: GenerationParams) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def generate_batch(self, prompts: List[str], params: GenerationParams) -> List[GenerationResult]:
        raise NotImplementedError
