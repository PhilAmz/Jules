"""
Models Module.

This package contains wrappers for various machine learning models, ensuring they
adhere to a scikit-learn compatible API (`fit`, `predict`, `get_params`, `set_params`).
This allows for seamless integration into the pipeline's cross-validation and
hyperparameter optimization workflows.

Key components:
- `base_model.py`: Defines `BaseModelWrapper`, the abstract base class for all
  model wrappers.
- `wrappers.py`: Provides concrete implementations for models like LightGBM
  (`LGBMWrapper`), simple Keras Neural Networks (`SimpleNNWrapper`), and
  scikit-learn's Ridge Regression (`RidgeWrapper`).
"""

from .base_model import BaseModelWrapper
from .wrappers import LGBMWrapper, SimpleNNWrapper, RidgeWrapper

__all__ = [
    'BaseModelWrapper',
    'LGBMWrapper',
    'SimpleNNWrapper',
    'RidgeWrapper'
]
