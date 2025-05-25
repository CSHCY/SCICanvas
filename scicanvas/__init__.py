"""
SCICanvas: Universal Deep Learning AI for Science Toolkit

A comprehensive PyTorch-based toolkit for AI applications in scientific research.
"""

__version__ = "0.1.0"
__author__ = "SCICanvas Team"
__email__ = "contact@scicanvas.org"

# Import core modules
from . import core
from . import utils

# Import domain-specific modules
from . import single_cell
from . import protein
from . import materials

# Make key classes available at package level
from .core.base import BaseModel, BasePredictor
from .core.config import Config

__all__ = [
    "core",
    "utils", 
    "single_cell",
    "protein",
    "materials",
    "BaseModel",
    "BasePredictor", 
    "Config",
] 