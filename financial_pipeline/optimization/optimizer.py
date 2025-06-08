"""
Hyperparameter optimization module using Optuna.

This module provides functions to define search spaces for feature engineering
and model parameters, an objective function for Optuna to optimize, and a
runner function to execute the optimization study. It integrates with the
pipeline's `train_and_evaluate_model` function for evaluating each trial.
"""
import optuna
import numpy as np
import pandas as pd # For type hints and example data
from sklearn.pipeline import Pipeline
from sklearn.base import clone # To clone model_wrapper if needed, though train_and_evaluate does it.

# Project imports - assuming this file is run in an environment where financial_pipeline is accessible
# Adjust paths if necessary when running standalone or from different locations.
from financial_pipeline.main import train_and_evaluate_model # train_and_evaluate_model
from financial_pipeline.data_ingestion import fetch_financial_data # For example usage
from financial_pipeline.feature_engineering import MovingAverageTransformer # For example usage
from financial_pipeline.models import LGBMWrapper # For example usage
from financial_pipeline.cross_validation import BasicTimeSeriesSplit # For example usage
from financial_pipeline.evaluation.metrics import mean_squared_error # For example usage

import os # For tensorboard log dir creation

def _suggest_param(trial: optuna.Trial, param_config):
    """Helper to suggest a parameter based on its configuration."""
    param_type = param_config[0]
    param_name = param_config[1]

    if param_type == 'suggest_float':
        # ('suggest_float', name, low, high, {'log': True/False, 'step': val})
        options = param_config[4] if len(param_config) > 4 else {}
        return trial.suggest_float(param_name, param_config[2], param_config[3], **options)
    elif param_type == 'suggest_int':
        # ('suggest_int', name, low, high, {'step': val, 'log': True/False})
        options = param_config[4] if len(param_config) > 4 else {}
        return trial.suggest_int(param_name, param_config[2], param_config[3], **options)
    elif param_type == 'suggest_categorical':
        # ('suggest_categorical', name, choices_list)
        return trial.suggest_categorical(param_name, param_config[2])
    else:
        raise ValueError(f"Unsupported Optuna suggestion type: {param_type}")


def create_feature_pipeline(trial: optuna.Trial, feature_eng_config: list) -> Pipeline:
    """
    Creates a feature engineering pipeline with parameters suggested by Optuna.

    Args:
        trial (optuna.Trial): Optuna Trial object.
        feature_eng_config (list): List of dicts, where each dict defines a
            transformer class and its tunable parameters.
            Example: [{'class': MovingAverageTransformer,
                       'params': {'window_sizes': ('suggest_categorical', 'sma_wins', [[10], [20,30]])}}]

    Returns:
        sklearn.pipeline.Pipeline: The constructed feature engineering pipeline.
    """
    steps = []
    for i, transformer_conf in enumerate(feature_eng_config):
        TransformerClass = transformer_conf['class']
        transformer_params = {}
        if 'params' in transformer_conf:
            for param_key, param_config in transformer_conf['params'].items():
                # Ensure unique name for optuna if same param_key used in different transformers
                # Or assume param_names in config are already unique for the trial scope.
                # For now, assume param_config[1] (optuna param name) is unique across trial.
                transformer_params[param_key] = _suggest_param(trial, param_config)

        # Transformer name in pipeline (must be unique)
        transformer_name = f"transformer_{i}_{TransformerClass.__name__}"
        steps.append((transformer_name, TransformerClass(**transformer_params)))

    return Pipeline(steps)


def create_model(trial: optuna.Trial, model_config: dict):
    """
    Creates a model wrapper instance with parameters suggested by Optuna.

    Args:
        trial (optuna.Trial): Optuna Trial object.
        model_config (dict): Dict defining the model class and its tunable parameters.
            Example: {'class': LGBMWrapper,
                      'params': {'learning_rate': ('suggest_float', 'lr', 0.01, 0.1)}}

    Returns:
        An instance of the model wrapper (e.g., LGBMWrapper).
    """
    ModelClass = model_config['class']
    model_params = {}
    if 'params' in model_config:
        for param_key, param_config in model_config['params'].items():
            model_params[param_key] = _suggest_param(trial, param_config)

    # Add any static params defined in model_config that are not part of optuna search space
    if 'static_params' in model_config:
        model_params.update(model_config['static_params'])

    return ModelClass(**model_params)


def objective(
    trial: optuna.Trial,
    data_fetcher_params: dict,
    feature_engineering_config_search_space: list,
    model_config_search_space: dict,
    cv_splitter_config: dict,
    target_column_name: str,
    metric_to_optimize, # function, e.g., sklearn.metrics.mean_squared_error
    metric_greater_is_better: bool,
    walk_forward_val_test_scheme: bool,
    tensorboard_log_dir_trial_prefix: str = None # e.g., "logs/study_name/trial_"
) -> float:
    """
    Optuna objective function for optimizing the financial pipeline.

    This function is called by Optuna for each trial. It constructs a feature
    engineering pipeline and a model based on parameters suggested by the trial,
    then evaluates this setup using `train_and_evaluate_model`. The performance
    metric (e.g., MSE, F1-score) is returned for Optuna to minimize/maximize.

    Args:
        trial (optuna.Trial): Current Optuna trial object.
        data_fetcher_params (dict): Static parameters for data fetching.
        feature_engineering_config_search_space (list): Configuration defining
            the search space for feature engineering transformers and their parameters.
        model_config_search_space (dict): Configuration defining the search space
            for the model wrapper and its parameters.
        cv_splitter_config (dict): Static configuration for the cross-validation
            splitter (class and its parameters).
        target_column_name (str): Name of the target variable column.
        metric_to_optimize (callable): Metric function (e.g., mean_squared_error)
            used for evaluating pipeline performance. Optuna will optimize this metric.
        metric_greater_is_better (bool): True if a higher value of `metric_to_optimize`
            is better (e.g., R2 score, AUC), False otherwise (e.g., MSE).
        walk_forward_val_test_scheme (bool): Passed to `train_and_evaluate_model`.
            If True, uses validation set metrics for optimization; otherwise, uses test set metrics.
        tensorboard_log_dir_trial_prefix (str, optional): If provided, creates a unique
            TensorBoard log directory for each trial (e.g., "logs/study_name/trial_").
            Defaults to None.

    Returns:
        float: The value of the metric to be optimized by Optuna.
               If `metric_greater_is_better` is True, returns -metric to fit Optuna's
               default minimization behavior. Returns float('inf') or -float('inf')
               if the trial fails or metric is invalid.
    """
    try:
        # 1. Create feature engineering pipeline for this trial
        feature_pipeline = create_feature_pipeline(trial, feature_engineering_config_search_space)

        # 2. Create model for this trial
        model_wrapper = create_model(trial, model_config_search_space)

        # 3. Instantiate CV splitter
        CVClass = cv_splitter_config['class']
        cv_splitter = CVClass(**cv_splitter_config.get('params', {}))

        # 4. TensorBoard logging for this specific trial
        current_tb_log_dir = None
        if tensorboard_log_dir_trial_prefix:
            current_tb_log_dir = f"{tensorboard_log_dir_trial_prefix}trial_{trial.number}"
            # Create the specific trial log directory
            if not os.path.exists(current_tb_log_dir):
                os.makedirs(current_tb_log_dir, exist_ok=True)

        # 5. Train and evaluate
        # Pass only the single metric_to_optimize to simplify metric parsing later
        _, _, fold_val_metrics, fold_test_metrics = train_and_evaluate_model(
            data_fetcher_params=data_fetcher_params,
            feature_engineering_pipeline=feature_pipeline,
            model_wrapper=model_wrapper,
            cv_splitter=cv_splitter,
            target_column_name=target_column_name,
            metrics_to_calculate=[metric_to_optimize], # Only pass the one we care about for Optuna
            walk_forward_val_test_scheme=walk_forward_val_test_scheme,
            tensorboard_log_dir=current_tb_log_dir
        )

        # 6. Extract relevant metrics
        metric_name_to_extract = metric_to_optimize.__name__
        relevant_metrics_over_folds = []

        if walk_forward_val_test_scheme:
            if not fold_val_metrics: # Should not happen if WFS is true and runs
                print(f"Warning (Trial {trial.number}): WalkForwardSplit scheme selected, but no validation metrics returned.")
                return float('inf') if not metric_greater_is_better else -float('inf')
            relevant_metrics_over_folds = [m[metric_name_to_extract] for m in fold_val_metrics if metric_name_to_extract in m]
        else:
            if not fold_test_metrics:  # Should not happen if standard CV runs
                print(f"Warning (Trial {trial.number}): Standard scheme selected, but no test metrics returned.")
                return float('inf') if not metric_greater_is_better else -float('inf')
            relevant_metrics_over_folds = [m[metric_name_to_extract] for m in fold_test_metrics if metric_name_to_extract in m]

        if not relevant_metrics_over_folds:
            print(f"Warning (Trial {trial.number}): No metrics found for '{metric_name_to_extract}' in relevant folds.")
            return float('inf') if not metric_greater_is_better else -float('inf')

        # 7. Calculate average metric
        average_metric_value = np.mean(relevant_metrics_over_folds)

        # 8. Handle NaN or Inf (though np.mean of empty list would error earlier)
        if np.isnan(average_metric_value) or np.isinf(average_metric_value):
            print(f"Warning (Trial {trial.number}): Metric '{metric_name_to_extract}' is NaN or Inf. Returning worst value.")
            return float('inf') if not metric_greater_is_better else -float('inf')

        trial.set_user_attr("average_metric_value", average_metric_value) # Store for inspection

        # 9. Return metric (Optuna minimizes by default)
        return average_metric_value if not metric_greater_is_better else -average_metric_value

    except optuna.exceptions.TrialPruned as e:
        # If trial is pruned, Optuna handles it. Re-raise.
        raise e
    except Exception as e:
        print(f"Exception in trial {trial.number}: {e}")
        # Return a value indicating failure, so Optuna doesn't favor this trial.
        # This helps Optuna explore other parts of the search space.
        return float('inf') if not metric_greater_is_better else -float('inf')


def run_optimization(
    data_fetcher_params: dict,
    feature_engineering_config_search_space: list,
    model_config_search_space: dict,
    cv_splitter_config: dict,
    target_column_name: str,
    metric_to_optimize, # function, e.g., sklearn.metrics.mean_squared_error
    metric_greater_is_better: bool,
    walk_forward_val_test_scheme: bool,
    n_trials: int = 100,
    study_name: str = None,
    storage_url: str = None, # e.g., "sqlite:///my_optuna_study.db" for persistence
    tensorboard_log_dir_study_prefix: str = None # e.g., "logs/optuna_study"
):
    """
    Runs an Optuna hyperparameter optimization study for the financial pipeline.

    This function sets up an Optuna study and calls `study.optimize()` with the
    provided `objective` function and configurations.

    Args:
        data_fetcher_params (dict): Static parameters for data fetching.
        feature_engineering_config_search_space (list): Configuration for the
            feature engineering search space.
        model_config_search_space (dict): Configuration for the model search space.
        cv_splitter_config (dict): Static configuration for the CV splitter.
        target_column_name (str): Name of the target variable.
        metric_to_optimize (callable): Metric function to be optimized.
        metric_greater_is_better (bool): Direction of optimization for the metric.
        walk_forward_val_test_scheme (bool): Indicates if walk-forward validation
            (train/val/test) is used, affecting which metric set is used for optimization.
        n_trials (int, optional): Number of optimization trials to run. Defaults to 100.
        study_name (str, optional): Name for the Optuna study. Useful for organizing
            and resuming studies, especially with persistent storage. Defaults to None.
        storage_url (str, optional): URL for Optuna's study database (e.g.,
            "sqlite:///my_study.db"). If None, an in-memory study is used.
            Defaults to None.
        tensorboard_log_dir_study_prefix (str, optional): Base directory for saving
            TensorBoard logs for each trial. If provided, a subdirectory will be
            created for each trial under this path. Defaults to None.

    Returns:
        optuna.study.Study: The completed Optuna study object, containing information
                            about all trials, best parameters, etc.
    """
    direction = 'maximize' if metric_greater_is_better else 'minimize'
    study = optuna.create_study(direction=direction, study_name=study_name, storage=storage_url, load_if_exists=True)

    # Prepare tensorboard prefix for individual trials if specified
    trial_tb_prefix = None
    if tensorboard_log_dir_study_prefix:
        # Ensure the main study log directory exists
        if not os.path.exists(tensorboard_log_dir_study_prefix):
            os.makedirs(tensorboard_log_dir_study_prefix, exist_ok=True)
        trial_tb_prefix = os.path.join(tensorboard_log_dir_study_prefix, "") # Optuna adds "trial_X"

    objective_func_with_args = lambda trial: objective(
        trial,
        data_fetcher_params,
        feature_engineering_config_search_space,
        model_config_search_space,
        cv_splitter_config,
        target_column_name,
        metric_to_optimize,
        metric_greater_is_better, # Passed to objective for error handling, not for Optuna direction here
        walk_forward_val_test_scheme,
        tensorboard_log_dir_trial_prefix=trial_tb_prefix
    )

    study.optimize(objective_func_with_args, n_trials=n_trials, show_progress_bar=True)

    print("\n--- Optimization Finished ---")
    print(f"Study Name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best trial number: {best_trial.number}")
    print(f"Best value ({metric_to_optimize.__name__}): {best_trial.value if not metric_greater_is_better else -best_trial.value}")
    print("Best parameters found:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # You can also access user attributes stored in the trial
    if "average_metric_value" in best_trial.user_attrs:
        print(f"  (Raw average metric from objective: {best_trial.user_attrs['average_metric_value']})")

    return study


if __name__ == '__main__':
    print("--- Optuna Optimizer Demonstration ---")

    # 1. Data Fetcher Parameters (static for optimization)
    example_data_params = {
        'tickers': 'AAPL',
        'start_date': '2022-01-01',
        'end_date': '2023-01-01', # Shorter period for faster example
        'interval': '1d'
    }
    example_target = 'Target_Next_Close' # Will be created in train_and_evaluate_model

    # 2. Feature Engineering Search Space
    # Using actual class objects
    fe_search_space = [
        {
            'class': MovingAverageTransformer,
            'params': {
                # Optuna param name 'sma_w' should be unique within a trial if this transformer class was used multiple times
                # with different tunable window sizes. Here, it's one instance, so 'sma_w' is fine.
                'window_sizes': ('suggest_categorical', 'sma_w', [[10], [15], [10, 20]]),
                'column': ('suggest_categorical', 'sma_col', ['Close', 'Open'])
            }
        },
        # Could add more transformers here, e.g., RSITransformer with tunable window
    ]

    # 3. Model Configuration Search Space
    # Using actual class object for the model wrapper
    model_search_space = {
        'class': LGBMWrapper,
        'static_params': { # Params not tuned by Optuna but passed to constructor
            'objective': 'regression',
            'metric': 'rmse', # This will be used by LGBM internally. Optuna optimizes based on metric_to_optimize.
            'random_state': 42,
            'n_jobs': 1,
        },
        'params': { # Tunable params
            'n_estimators': ('suggest_int', 'lgbm_n_est', 50, 150, {'step': 25}),
            'learning_rate': ('suggest_float', 'lgbm_lr', 0.01, 0.2, {'log': True}),
            'num_leaves': ('suggest_int', 'lgbm_num_leaves', 10, 40)
        }
    }

    # 4. CV Splitter Configuration (static for optimization)
    cv_config = {
        'class': BasicTimeSeriesSplit,
        'params': {'n_splits': 3, 'test_size': 30} # Fewer splits/shorter test for faster example
    }

    # 5. Metric to Optimize (function) and its direction
    opt_metric_func = mean_squared_error
    opt_metric_higher_is_better = False # For MSE, lower is better

    # 6. Walk-forward scheme (False for BasicTimeSeriesSplit)
    use_wfs = False

    # 7. TensorBoard logging for study
    # Each trial will have its own sub-directory under this path.
    # This path itself is for the whole study, individual trial logs are managed by `objective`.
    study_tb_log_prefix = os.path.join("logs", "optuna_financial_study")


    # 8. Run Optimization
    print(f"Starting Optuna study. Optimizing for: {opt_metric_func.__name__} ({'Maximize' if opt_metric_higher_is_better else 'Minimize'})")

    # Optuna may take a while depending on n_trials and complexity of model training.
    # For a quick test, use a small n_trials, e.g., 5-10.
    # In a real scenario, n_trials would be much larger (e.g., 100+).
    try:
        study_results = run_optimization(
            data_fetcher_params=example_data_params,
            feature_engineering_config_search_space=fe_search_space,
            model_config_search_space=model_search_space,
            cv_splitter_config=cv_config,
            target_column_name=example_target,
            metric_to_optimize=opt_metric_func,
            metric_greater_is_better=opt_metric_higher_is_better,
            walk_forward_val_test_scheme=use_wfs,
            n_trials=5, # Small number for quick example run
            study_name="financial_pipeline_optimization_example",
            storage_url=None, # In-memory storage for this example
            tensorboard_log_dir_study_prefix=study_tb_log_prefix
        )
        print("\nAccess best parameters from study_results.best_trial.params")
        print("Access best value from study_results.best_trial.value")

    except ImportError as ie:
        print(f"ImportError during Optuna example: {ie}. Ensure all pipeline components are installed and accessible.")
    except Exception as e:
        import traceback
        print(f"An unexpected error occurred during the Optuna example run: {e}")
        print(traceback.format_exc())

    print("\n--- Optuna Optimizer Demonstration Finished ---")
