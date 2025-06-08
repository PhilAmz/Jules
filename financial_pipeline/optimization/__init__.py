"""
Optimization Module.

This package handles hyperparameter optimization for the financial pipeline,
primarily using the Optuna library. It provides functions to define search
spaces, run optimization studies, and manage trial evaluations.

Key components are found in `optimizer.py`.
"""

from .optimizer import (
    create_feature_pipeline,
    create_model,
    objective,
    run_optimization
)

__all__ = [
    'create_feature_pipeline',
    'create_model',
    'objective',
    'run_optimization'
]
