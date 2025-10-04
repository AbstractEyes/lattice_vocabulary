# src/geovocab2/__init__.py
"""
GeoVocab2 - Geometric vocabulary system
"""

# Expose main subpackages
from . import shapes
from . import fusion
from . import config
from . import data
from . import train

__all__ = ['shapes', 'fusion', 'config', 'data', 'train']