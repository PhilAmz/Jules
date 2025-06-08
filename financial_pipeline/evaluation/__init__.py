"""
Evaluation Module.

This package provides tools for evaluating model performance and visualizing results.
It includes:
- A collection of standard and custom metric functions (from `metrics.py`).
- Utilities for plotting, such as TensorBoard logging and reliability diagrams
  (from `plotting.py`).
"""

# Re-export metrics from metrics.py
from .metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    sharpe_ratio, # Placeholder
    adjusted_r2_score,
    directional_accuracy
)

# Import plotting functions
from .plotting import (
    setup_tensorboard_writer,
    log_metrics_to_tensorboard,
    plot_reliability_diagram
)

__all__ = [
    # Metrics
    'mean_squared_error',
    'mean_absolute_error',
    'mean_absolute_percentage_error',
    'r2_score',
    'accuracy_score',
    'f1_score',
    'precision_score',
    'recall_score',
    'roc_auc_score',
    'log_loss',
    'sharpe_ratio',
    'adjusted_r2_score',
    'directional_accuracy',
    # Plotting
    'setup_tensorboard_writer',
    'log_metrics_to_tensorboard',
    'plot_reliability_diagram'
]
