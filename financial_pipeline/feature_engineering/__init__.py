"""
Feature Engineering Module.

This package provides tools for creating and transforming features from financial data.
It includes a base transformer class for creating custom scikit-learn compatible
transformers, as well as pre-built transformers for common technical indicators
and placeholders for more advanced feature generation (e.g., from pretrained models).

Key components are organized into submodules:
- `base`: Contains `BaseFeatureTransformer`.
- `technical_indicators`: Includes transformers like `MovingAverageTransformer`,
  `RSITransformer`, `MACDTransformer`.
- `pretrained_features`: Placeholder for transformers using pretrained models.
"""

from .base import BaseFeatureTransformer
from .technical_indicators import MovingAverageTransformer, RSITransformer, MACDTransformer
from .pretrained_features import PretrainedModelFeatureTransformer

__all__ = [
    'BaseFeatureTransformer',
    'MovingAverageTransformer',
    'RSITransformer',
    'MACDTransformer',
    'PretrainedModelFeatureTransformer'
]
