"""
Main orchestration module for the financial pipeline.

This module provides functions to run end-to-end model training and evaluation,
including data fetching, feature engineering, cross-validation, model training,
and final model preparation. It also includes an example `if __name__ == '__main__':`
block to demonstrate a typical pipeline execution flow.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import mean_squared_error # Example metric
import os # For TensorBoard log directory

# Project specific imports
from data_ingestion import fetch_financial_data
from feature_engineering import MovingAverageTransformer # Example feature transformer
from models import LGBMWrapper # Example model wrapper
from cross_validation import BasicTimeSeriesSplit, WalkForwardSplit # Example CV splitters
from evaluation.plotting import setup_tensorboard_writer, log_metrics_to_tensorboard # For TensorBoard

def train_and_evaluate_model(
    data_fetcher_params: dict,
    feature_engineering_pipeline: Pipeline,
    model_wrapper, # Should be an instance of a class derived from BaseModelWrapper
    cv_splitter, # Should be an instance of a class derived from BaseCrossValidator
    target_column_name: str,
    metrics_to_calculate: list,
    walk_forward_val_test_scheme: bool = False,
    tensorboard_log_dir: str = None # New parameter for TensorBoard log directory
):
    """
    Trains and evaluates a model using a specified cross-validation strategy.

    Args:
        data_fetcher_params (dict): Parameters dictionary for `fetch_financial_data`.
        feature_engineering_pipeline (sklearn.pipeline.Pipeline): An unfitted scikit-learn Pipeline
            object containing feature engineering transformers.
        model_wrapper: An instance of a model wrapper class (derived from `BaseModelWrapper`),
            e.g., `LGBMWrapper()`. This wrapper contains the model to be trained.
        cv_splitter: An instance of a cross-validation splitter class (derived from
            `sklearn.model_selection.BaseCrossValidator`), e.g., `BasicTimeSeriesSplit()`.
        target_column_name (str): The name of the target variable column in the DataFrame
            returned by `fetch_financial_data`.
        metrics_to_calculate (list): A list of callable metric functions. Each function
            should accept `(y_true, y_pred)` and return a scalar value.
        walk_forward_val_test_scheme (bool, optional): Set to `True` if the `cv_splitter`
            yields three sets of indices (train, validation, test), like `WalkForwardSplit`.
            Defaults to `False` (expects train, test indices).
        tensorboard_log_dir (str, optional): Path to the directory where TensorBoard logs
            for this run should be saved. If `None`, TensorBoard logging is skipped.
            Defaults to `None`.

    Returns:
        tuple: A tuple containing:
            - `trained_models` (list): List of model instances trained on each fold.
            - `fold_metrics_train` (list): List of dictionaries, each containing training
              metrics for a fold.
            - `fold_metrics_val` (list): List of dictionaries, each containing validation
              metrics for a fold (empty if `walk_forward_val_test_scheme` is False or
              no validation set is produced by the splitter).
            - `fold_metrics_test` (list): List of dictionaries, each containing test
              metrics for a fold.
    """
    print("1. Fetching data...")
    df = fetch_financial_data(**data_fetcher_params)
    if df.empty:
        print("No data fetched. Exiting.")
        return [], [], [], []

    # --- Helper to create target variable (Example: predict next day's Close) ---
    if target_column_name not in df.columns:
        if 'Close' in df.columns and target_column_name == 'Target_Next_Close':
            print(f"Creating target column '{target_column_name}' by shifting 'Close' by -1.")
            df[target_column_name] = df['Close'].shift(-1)
            df.dropna(subset=[target_column_name], inplace=True)
            if df.empty:
                print("DataFrame became empty after target creation and NaN drop. Exiting.")
                return [], [], [], []
        else:
            raise ValueError(f"Target column '{target_column_name}' not found in DataFrame and no default creation rule for it.")

    print(f"Data shape after target creation: {df.shape}")

    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    if X.empty or y.empty:
        print("Features (X) or target (y) are empty. Exiting.")
        return [], [], [], []

    fold_metrics_train = []
    fold_metrics_val = []
    fold_metrics_test = []
    trained_models = []

    tb_writer = None
    if tensorboard_log_dir:
        try:
            # Ensure the base log directory exists for timestamped subdirectories
            base_log_parent_dir = os.path.dirname(tensorboard_log_dir)
            if base_log_parent_dir and not os.path.exists(base_log_parent_dir):
                 os.makedirs(base_log_parent_dir, exist_ok=True)
            tb_writer = setup_tensorboard_writer(tensorboard_log_dir)
        except Exception as e:
            print(f"Warning: Failed to setup TensorBoard writer at {tensorboard_log_dir}. Error: {e}")
            tb_writer = None

    print(f"\n2. Starting Cross-Validation with {cv_splitter.__class__.__name__}...")

    n_splits_from_splitter = cv_splitter.get_n_splits(X,y) # Get total splits for progress display

    for fold, split_indices in enumerate(cv_splitter.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits_from_splitter} ---")

        if walk_forward_val_test_scheme:
            if len(split_indices) != 3:
                raise ValueError("WalkForwardSplitter expected to yield 3 sets of indices (train, val, test).")
            train_idx, val_idx, test_idx = split_indices
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        else:
            if len(split_indices) != 2:
                raise ValueError("Standard CV Splitter expected to yield 2 sets of indices (train, test).")
            train_idx, test_idx = split_indices
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            X_val, y_val = None, None
            print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

        print("   Applying feature engineering pipeline...")
        X_train_transformed = feature_engineering_pipeline.fit_transform(X_train.copy(), y_train.copy())
        X_test_transformed = feature_engineering_pipeline.transform(X_test.copy())

        X_val_transformed = None
        current_val_metrics = None # Initialize
        if X_val is not None and not X_val.empty:
            X_val_transformed = feature_engineering_pipeline.transform(X_val.copy())

        print("   Training model...")
        current_model = clone(model_wrapper)

        fit_params = {}
        if X_val_transformed is not None and not X_val_transformed.empty and y_val is not None and not y_val.empty:
            if hasattr(current_model, 'objective'):
                 fit_params['eval_set'] = [(X_val_transformed, y_val)]
            elif hasattr(current_model, 'layers_config'):
                fit_params['validation_data'] = (X_val_transformed, y_val)

        if hasattr(current_model, 'objective'):
            fit_params.setdefault('verbose', -1)
        elif hasattr(current_model, 'layers_config'):
            fit_params.setdefault('verbose', 0)

        current_model.fit(X_train_transformed, y_train, **fit_params)
        trained_models.append(current_model)

        print("   Evaluating model...")
        preds_train = current_model.predict(X_train_transformed)
        current_train_metrics = {m.__name__: m(y_train, preds_train) for m in metrics_to_calculate}
        fold_metrics_train.append(current_train_metrics)
        print(f"     Train Metrics: {current_train_metrics}")

        if X_val_transformed is not None and not X_val_transformed.empty and y_val is not None and not y_val.empty:
            preds_val = current_model.predict(X_val_transformed)
            current_val_metrics = {m.__name__: m(y_val, preds_val) for m in metrics_to_calculate}
            fold_metrics_val.append(current_val_metrics)
            print(f"     Val Metrics:   {current_val_metrics}")

        preds_test = current_model.predict(X_test_transformed)
        current_test_metrics = {m.__name__: m(y_test, preds_test) for m in metrics_to_calculate}
        fold_metrics_test.append(current_test_metrics)
        print(f"     Test Metrics:  {current_test_metrics}")

        if tb_writer:
            log_metrics_to_tensorboard(tb_writer, current_train_metrics, fold + 1, prefix=f'Fold/Train')
            if current_val_metrics: # Check if it was computed
                log_metrics_to_tensorboard(tb_writer, current_val_metrics, fold + 1, prefix=f'Fold/Validation')
            log_metrics_to_tensorboard(tb_writer, current_test_metrics, fold + 1, prefix=f'Fold/Test')

    if tb_writer:
        # Log average metrics across folds
        if fold_metrics_train:
            avg_train_fold_metrics = {k: np.mean([dic[k] for dic in fold_metrics_train]) for k in fold_metrics_train[0]}
            log_metrics_to_tensorboard(tb_writer, avg_train_fold_metrics, n_splits_from_splitter, prefix='Average/Train')
        if fold_metrics_val: # Only if there was validation data in any fold
            avg_val_fold_metrics = {k: np.mean([dic[k] for dic in fold_metrics_val]) for k in fold_metrics_val[0]}
            log_metrics_to_tensorboard(tb_writer, avg_val_fold_metrics, n_splits_from_splitter, prefix='Average/Validation')
        if fold_metrics_test:
            avg_test_fold_metrics = {k: np.mean([dic[k] for dic in fold_metrics_test]) for k in fold_metrics_test[0]}
            log_metrics_to_tensorboard(tb_writer, avg_test_fold_metrics, n_splits_from_splitter, prefix='Average/Test')

        tb_writer.close()
        abs_log_dir_path = os.path.abspath(tensorboard_log_dir)
        print(f"\nTensorBoard logging complete. Run 'tensorboard --logdir={abs_log_dir_path}' to view.")


    return trained_models, fold_metrics_train, fold_metrics_val, fold_metrics_test


def get_final_model_for_test_set(
    data_fetcher_params: dict,
    feature_engineering_pipeline: Pipeline,
    best_model_config: dict,
    ModelClass, # The class of the model wrapper, e.g., LGBMWrapper
    target_column_name: str,
    full_train_indices: np.ndarray # NumPy array of indices
):
    """
    Trains a final model on a specified combined training dataset (e.g., all train + val folds).

    This function is typically called after hyperparameter optimization or cross-validation
    to prepare a model for deployment or final evaluation on a hold-out set.

    Args:
        data_fetcher_params (dict): Parameters for `fetch_financial_data`.
        feature_engineering_pipeline (sklearn.pipeline.Pipeline): An unfitted scikit-learn Pipeline
            for feature engineering. It will be fitted on the `full_train_indices`.
        best_model_config (dict): Dictionary of parameters for the `ModelClass`. These are
            typically the best parameters found during optimization or a chosen configuration.
        ModelClass: The model wrapper class to be instantiated (e.g., `LGBMWrapper`).
        target_column_name (str): Name of the target variable column.
        full_train_indices (np.ndarray): A NumPy array of indices specifying which rows of the
            fetched data constitute the full training set.

    Returns:
        tuple: A tuple containing:
            - `final_model`: The trained model instance.
            - `fitted_feature_pipeline`: The feature engineering pipeline fitted on the
              `full_train_indices` data.

    Raises:
        ValueError: If data cannot be fetched, target column is not found, indices are
                    out of bounds, or the resulting training set is empty.
    """
    print("\n3. Training final model on full specified training data...")
    df = fetch_financial_data(**data_fetcher_params)
    if df.empty:
        raise ValueError("No data fetched for final model training.")

    if target_column_name not in df.columns:
         if 'Close' in df.columns and target_column_name == 'Target_Next_Close':
            print(f"Creating target column '{target_column_name}' by shifting 'Close' by -1 for final model.")
            df[target_column_name] = df['Close'].shift(-1)
            df.dropna(subset=[target_column_name], inplace=True)
            if df.empty:
                raise ValueError("DataFrame became empty after target creation for final model.")
         else:
            raise ValueError(f"Target column '{target_column_name}' not found in DataFrame for final model.")

    if not full_train_indices.size > 0 : # Check if array is not empty
        raise ValueError("full_train_indices is empty.")
    if max(full_train_indices) >= len(df):
        raise ValueError(f"Max index in full_train_indices ({max(full_train_indices)}) "
                         f"is out of bounds for the DataFrame of length {len(df)} "
                         "after target creation and NaN handling.")

    X_full = df.drop(columns=[target_column_name])
    y_full = df[target_column_name]

    X_full_train, y_full_train = X_full.iloc[full_train_indices], y_full.iloc[full_train_indices]

    if X_full_train.empty or y_full_train.empty:
        raise ValueError("Full training set (X_full_train or y_full_train) is empty.")

    print(f"   Full training data size: {len(X_full_train)}")
    print("   Fitting feature engineering pipeline on full training data...")
    fitted_feature_pipeline = clone(feature_engineering_pipeline)
    X_full_train_transformed = fitted_feature_pipeline.fit_transform(X_full_train, y_full_train)

    print("   Training final model...")
    final_model = ModelClass(**best_model_config)
    fit_params = {}
    if hasattr(final_model, 'objective'):
        fit_params.setdefault('verbose', -1)
    elif hasattr(final_model, 'layers_config'):
        fit_params.setdefault('verbose', 0)

    final_model.fit(X_full_train_transformed, y_full_train, **fit_params)

    return final_model, fitted_feature_pipeline


if __name__ == '__main__':
    print("--- Financial Pipeline Main Execution Example ---")

    ticker = 'AAPL'
    data_fetcher_params_main = {
        'tickers': ticker,
        'start_date': '2022-01-01',
        'end_date': '2023-06-01',
        'interval': '1d'
    }

    example_target_column = 'Target_Next_Close'

    fe_pipeline = Pipeline([
        ('sma_10', MovingAverageTransformer(window_sizes=[10], column='Close')),
        ('sma_30', MovingAverageTransformer(window_sizes=[30], column='Open'))
    ])

    model_wrap = LGBMWrapper(
        objective='regression',
        metric='rmse',
        n_estimators=50,
        learning_rate=0.05,
        num_leaves=20,
        random_state=42,
        n_jobs=1
    )

    cv_main_splitter = BasicTimeSeriesSplit(n_splits=4, test_size=30)
    metrics = [mean_squared_error, mean_absolute_error]

    # Define a unique log directory for each run using a timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    example_tb_log_dir = os.path.join('logs', f"{ticker}_run_{timestamp}") # Store in logs/AAPL_run_YYYYMMDD_HHMMSS


    trained_models_cv, train_metrics_cv, val_metrics_cv, test_metrics_cv = train_and_evaluate_model(
        data_fetcher_params=data_fetcher_params_main,
        feature_engineering_pipeline=fe_pipeline,
        model_wrapper=model_wrap,
        cv_splitter=cv_main_splitter,
        target_column_name=example_target_column,
        metrics_to_calculate=metrics,
        walk_forward_val_test_scheme=False,
        tensorboard_log_dir=example_tb_log_dir
    )

    if trained_models_cv:
        print("\n--- CV Results Summary ---")
        avg_test_metrics = {}
        if test_metrics_cv:
            for metric_name_key in test_metrics_cv[0].keys():
                avg_test_metrics[metric_name_key] = np.mean([m[metric_name_key] for m in test_metrics_cv])
            print(f"Average Test Metrics over {len(test_metrics_cv)} folds: {avg_test_metrics}")

        avg_train_metrics = {}
        if train_metrics_cv:
            for metric_name_key in train_metrics_cv[0].keys():
                avg_train_metrics[metric_name_key] = np.mean([m[metric_name_key] for m in train_metrics_cv])
            print(f"Average Train Metrics over {len(train_metrics_cv)} folds: {avg_train_metrics}")

        _temp_df_for_indices = fetch_financial_data(**data_fetcher_params_main)
        if not _temp_df_for_indices.empty:
            if 'Close' in _temp_df_for_indices.columns and example_target_column == 'Target_Next_Close':
                _temp_df_for_indices[example_target_column] = _temp_df_for_indices['Close'].shift(-1)
                _temp_df_for_indices.dropna(subset=[example_target_column], inplace=True)

            if not _temp_df_for_indices.empty:
                num_total_samples = len(_temp_df_for_indices)
                example_full_train_indices = np.arange(num_total_samples)

                print(f"\n--- Example: Training a 'final' model on all {len(example_full_train_indices)} available samples ---")
                best_config = model_wrap.get_params()

                try:
                    final_model_instance, final_fe_pipeline = get_final_model_for_test_set(
                        data_fetcher_params=data_fetcher_params_main,
                        feature_engineering_pipeline=fe_pipeline,
                        best_model_config=best_config,
                        ModelClass=LGBMWrapper,
                        target_column_name=example_target_column,
                        full_train_indices=example_full_train_indices
                    )
                    print(f"Final model trained: {type(final_model_instance.model_)}")
                    print(f"Final FE pipeline fitted steps: {final_fe_pipeline.steps}")

                    if len(example_full_train_indices) > 5:
                        X_for_final_pred = _temp_df_for_indices.drop(columns=[example_target_column]).iloc[example_full_train_indices[-5:]]
                        X_for_final_pred_transformed = final_fe_pipeline.transform(X_for_final_pred)
                        final_preds_example = final_model_instance.predict(X_for_final_pred_transformed)
                        print(f"Example predictions from final model on last 5 training samples: {final_preds_example}")

                except ValueError as e:
                    print(f"Error in get_final_model_for_test_set example: {e}")
            else:
                print("Could not prepare data for final model example (df empty after target creation).")
        else:
            print("Could not re-fetch data for final model example.")
    else:
        print("CV training did not produce any models.")

    print("\n--- Financial Pipeline Main Execution Example Finished ---")
