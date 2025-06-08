# This file will contain functions for plotting evaluation results.

import os
import numpy as np
import pandas as pd # Only for example usage, not strictly needed for functions

# Try to import TensorFlow for TensorBoard
try:
    import tensorflow as tf
    _TENSORFLOW_AVAILABLE_FOR_PLOTTING = True
except ImportError:
    _TENSORFLOW_AVAILABLE_FOR_PLOTTING = False
    # print("Warning: TensorFlow not found. TensorBoard logging utilities will not be usable.")

# Try to import Matplotlib and Scikit-learn calibration
try:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    # print("Warning: Matplotlib or scikit-learn not found. Plotting functions may not be usable.")


def setup_tensorboard_writer(log_dir='logs/pipeline_run'):
    """
    Sets up a TensorFlow SummaryWriter for TensorBoard logging.

    Args:
        log_dir (str, optional): Directory to save TensorBoard logs.
                                 Defaults to 'logs/pipeline_run'.

    Returns:
        tf.summary.SummaryWriter: A SummaryWriter instance.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    if not _TENSORFLOW_AVAILABLE_FOR_PLOTTING:
        raise ImportError("TensorFlow is required for TensorBoard logging but not installed.")

    try:
        os.makedirs(log_dir, exist_ok=True)
        writer = tf.summary.create_file_writer(log_dir)
        print(f"TensorBoard writer created. Logging to: {log_dir}")
        return writer
    except Exception as e:
        print(f"Error creating TensorBoard writer in {log_dir}: {e}")
        # Fallback or re-raise, depending on desired strictness
        # For now, let it raise if tf is available but writer creation fails
        raise


def log_metrics_to_tensorboard(writer, metrics_dict, step, prefix='Fold'):
    """
    Logs a dictionary of metrics to TensorBoard.

    Args:
        writer (tf.summary.SummaryWriter): The TensorFlow SummaryWriter instance.
        metrics_dict (dict): A dictionary of metrics, e.g., {'mse': 0.5, 'mae': 0.2}.
        step (int): The current step (e.g., fold number, epoch).
        prefix (str, optional): Prefix for the metric names in TensorBoard
                                (e.g., 'Fold_Train', 'Fold_Val', 'Epoch_Train').
                                Defaults to 'Fold'.

    Raises:
        ImportError: If TensorFlow is not installed.
        AttributeError: If writer is None or not a valid SummaryWriter.
    """
    if not _TENSORFLOW_AVAILABLE_FOR_PLOTTING:
        raise ImportError("TensorFlow is required for TensorBoard logging but not installed.")
    if writer is None:
        print("Warning: TensorBoard writer is None. Metrics will not be logged.")
        return

    try:
        with writer.as_default():
            for metric_name, metric_value in metrics_dict.items():
                tf.summary.scalar(f'{prefix}/{metric_name}', metric_value, step=step)
        writer.flush() # Ensure metrics are written
    except Exception as e:
        print(f"Error logging metrics to TensorBoard: {e}")
        # Decide if this should be fatal or just a warning. For now, print warning.


def plot_reliability_diagram(y_true, y_prob, n_bins=10, ax=None, title_suffix=""):
    """
    Plots a reliability diagram (calibration curve).

    Args:
        y_true (np.ndarray or pd.Series): True binary labels (0 or 1).
        y_prob (np.ndarray or pd.Series): Predicted probabilities for the positive class.
        n_bins (int, optional): Number of bins to discretize probabilities. Defaults to 10.
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on.
                                             If None, a new figure and axis are created.
                                             Defaults to None.
        title_suffix (str, optional): Suffix to add to the plot title. Defaults to "".

    Returns:
        matplotlib.axes.Axes: The axis on which the plot was drawn.

    Raises:
        ImportError: If Matplotlib or scikit-learn is not installed.
    """
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib and scikit-learn are required for plotting reliability diagrams.")

    if not isinstance(y_true, np.ndarray): y_true = np.array(y_true)
    if not isinstance(y_prob, np.ndarray): y_prob = np.array(y_prob)

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration Curve')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

    ax.set_xlabel("Mean Predicted Probability (per bin)")
    ax.set_ylabel("Fraction of Positives (per bin)")
    ax.set_title(f"Reliability Diagram {title_suffix}".strip())
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.7)

    # Add Brier score to plot as well? Could be an option.
    # from sklearn.metrics import brier_score_loss
    # brier = brier_score_loss(y_true, y_prob)
    # ax.text(0.05, 0.95, f"Brier Score: {brier:.3f}", transform=ax.transAxes, va='top')

    return ax


if __name__ == '__main__':
    # --- TensorBoard Logging Example ---
    if _TENSORFLOW_AVAILABLE_FOR_PLOTTING:
        print("--- TensorBoard Logging Utilities Example ---")
        example_log_dir = 'logs/plotting_example_run'
        try:
            tb_writer = setup_tensorboard_writer(log_dir=example_log_dir)
            if tb_writer:
                # Simulate logging metrics for a few folds/steps
                for step_i in range(5):
                    train_metrics_example = {'loss': 1.0 - 0.1*step_i, 'accuracy': 0.7 + 0.05*step_i}
                    val_metrics_example = {'loss': 0.9 - 0.08*step_i, 'accuracy': 0.72 + 0.04*step_i}

                    log_metrics_to_tensorboard(tb_writer, train_metrics_example, step=step_i, prefix='ExampleRun/Train')
                    log_metrics_to_tensorboard(tb_writer, val_metrics_example, step=step_i, prefix='ExampleRun/Validation')

                tb_writer.close() # Close the writer when done
                print(f"TensorBoard logs written to {example_log_dir}. Run 'tensorboard --logdir={os.path.abspath(example_log_dir)}' to view.")
            else:
                print("Failed to create TensorBoard writer for example.")
        except Exception as e:
            print(f"Error in TensorBoard example: {e}")
    else:
        print("Skipping TensorBoard Logging Utilities Example: TensorFlow not available.")

    # --- Reliability Plot Example ---
    if _MATPLOTLIB_AVAILABLE:
        print("\n--- Reliability Plot Example ---")
        # Generate some dummy data for classification
        np.random.seed(42)
        y_true_calib = np.random.randint(0, 2, size=1000)
        # Reasonably calibrated probabilities
        y_prob_calib_good = np.clip(y_true_calib * 0.6 + 0.2 + np.random.normal(0, 0.15, size=1000), 0, 1)
        # Poorly calibrated probabilities (overconfident)
        y_prob_calib_poor = np.clip(y_true_calib * 0.9 + 0.05 + np.random.normal(0, 0.05, size=1000), 0, 1)


        fig_calib, axes = plt.subplots(1, 2, figsize=(14, 6))

        plot_reliability_diagram(y_true_calib, y_prob_calib_good, n_bins=10, ax=axes[0], title_suffix="(Well Calibrated)")
        plot_reliability_diagram(y_true_calib, y_prob_calib_poor, n_bins=10, ax=axes[1], title_suffix="(Poorly Calibrated)")

        fig_calib.tight_layout()
        # To show plots in a non-interactive environment, you might need plt.show() or save them.
        # For automated runs, saving is better:
        example_plot_path = "example_reliability_plot.png"
        try:
            fig_calib.savefig(example_plot_path)
            print(f"Reliability plot example saved to {example_plot_path}")
        except Exception as e:
            print(f"Could not save reliability plot: {e}")
        # plt.show() # Uncomment if running interactively and want to see the plot
    else:
        print("Skipping Reliability Plot Example: Matplotlib/Scikit-learn not available.")
