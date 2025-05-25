"""
Core module for SCICanvas toolkit.

Contains base classes, configuration management, and shared utilities.
"""

from .base import BaseModel, BasePredictor
from .config import Config
from .trainer import Trainer

__all__ = [
    "BaseModel",
    "BasePredictor", 
    "Config",
    "Trainer",
] 