"""LLM Interpretability Toolkit - Main Package"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.analyzer import InterpretabilityAnalyzer
from .core.config import Config

__all__ = ["InterpretabilityAnalyzer", "Config"]