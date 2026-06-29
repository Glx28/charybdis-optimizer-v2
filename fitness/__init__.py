"""Fitness factor base class."""
from abc import ABC, abstractmethod
from core import Layout

class FitnessFactor(ABC):
    """Base class for a single fitness factor."""
    name: str = ""
    
    @abstractmethod
    def compute(self, layout: Layout) -> float:
        """Return raw score. For penalties, lower is better. For rewards, higher is better."""
        pass
    
    @property
    def is_penalty(self) -> bool:
        return True

class RewardFactor(FitnessFactor):
    """Base class for reward factors (higher is better)."""
    @property
    def is_penalty(self) -> bool:
        return False
