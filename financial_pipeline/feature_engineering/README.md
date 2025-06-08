# Feature Engineering Module

## Overview

The `feature_engineering` module is responsible for creating and transforming features from raw financial data. It is designed to be flexible, allowing for the use of built-in technical indicators and the easy integration of custom feature transformers. All transformers are intended to be compatible with scikit-learn's `Pipeline` objects.

Key components:
- `base.py`: Contains `BaseFeatureTransformer`, the base class all custom transformers should inherit from. This ensures scikit-learn compatibility.
- `technical_indicators.py`: Implements common financial technical indicators as transformers (e.g., SMA, RSI, MACD).
- `pretrained_features.py`: Placeholder for transformers that might leverage pretrained models to generate features (e.g., sentiment scores from news, embeddings).

## Adding Custom Feature Transformers

To add a new custom feature transformer, follow these steps:

1.  **Create or Choose a File:** You can add your transformer to an existing file (like `technical_indicators.py` if it's a TA indicator) or create a new Python file within the `feature_engineering` directory.

2.  **Inherit from `BaseFeatureTransformer`:**
    Your custom transformer class should inherit from `financial_pipeline.feature_engineering.base.BaseFeatureTransformer`. This base class itself inherits from `sklearn.base.BaseEstimator` and `sklearn.base.TransformerMixin`, providing standard scikit-learn methods like `get_params` and `set_params`.

    ```python
    from financial_pipeline.feature_engineering.base import BaseFeatureTransformer
    import pandas as pd

    class MyCustomTransformer(BaseFeatureTransformer):
        def __init__(self, custom_param1='default_value', custom_param2=10):
            super().__init__() # Optional if BaseFeatureTransformer.__init__ does nothing
            self.custom_param1 = custom_param1
            self.custom_param2 = custom_param2
            # Other initializations
    ```

3.  **Implement `fit(self, X, y=None)`:**
    The `fit` method is used to learn any parameters from the training data. For many feature transformers (like those that calculate technical indicators or lags), no actual "fitting" or learning from data is required. In such cases, the `fit` method should simply return `self`.

    ```python
    def fit(self, X: pd.DataFrame, y=None):
        # Example: Validate that necessary columns exist in X
        if 'RequiredColumn' not in X.columns:
            raise ValueError("DataFrame must contain 'RequiredColumn'")

        # If the transformer learns something (e.g., min/max for scaling)
        # self.min_ = X['SomeColumn'].min()
        # self.max_ = X['SomeColumn'].max()

        # It's good practice to store the names of columns that will be created
        self.new_columns_ = [f"new_feat_{self.custom_param2}"]
        return self
    ```

4.  **Implement `transform(self, X)`:**
    The `transform` method performs the actual feature creation or modification. It takes a DataFrame `X` as input and should return a new DataFrame with the transformed/added features.

    ```python
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy() # Work on a copy to avoid modifying the original DataFrame

        # Example: Create a new feature
        # new_feature_name = self.new_columns_[0] # From fit method
        new_feature_name = f"my_feature_{self.custom_param1}_{self.custom_param2}"
        X_transformed[new_feature_name] = X['SomeColumn'] * self.custom_param2
                                         # + self.min_ (if learned in fit)

        # Handle NaNs introduced by your transformation if necessary.
        # The BaseFeatureTransformer provides a helper: self._fillna_methods(X_transformed, [new_feature_name])
        # Or apply specific logic:
        # X_transformed[new_feature_name].fillna(0, inplace=True)
        X_transformed = self._fillna_methods(X_transformed, [new_feature_name])


        return X_transformed
    ```
    **Important:** Ensure your `transform` method does not use information from the target variable `y` unless it's explicitly designed for supervised feature engineering and appropriate care is taken to prevent data leakage during cross-validation.

5.  **NaN Handling:**
    Many feature engineering operations (e.g., moving averages, lags) will produce `NaN` values at the beginning of the series. Your transformer should handle these, typically by:
    *   Leaving them as is (if downstream models can handle NaNs).
    *   Filling them (e.g., using `ffill`, `bfill`, or a constant). The `BaseFeatureTransformer` provides a `_fillna_methods(df, columns_to_fill)` helper which applies forward fill then backward fill to the specified columns.
    *   Dropping rows with NaNs (usually done after all features are generated, outside the transformer, to ensure consistent data length across features).

6.  **Example Reference:**
    *   See `technical_indicators.py` for examples like `MovingAverageTransformer`, `RSITransformer`.
    *   The `LagFeatureTransformer` defined in the demonstration notebook (`notebooks/01_pipeline_demonstration.ipynb`) also serves as a good example of a custom transformer.

7.  **Import and Use:**
    Once defined, import your custom transformer and include it as a step in an `sklearn.pipeline.Pipeline`:

    ```python
    from sklearn.pipeline import Pipeline
    # from .my_custom_transformers_file import MyCustomTransformer # If in a new file

    # Assuming MyCustomTransformer is defined or imported
    feature_pipeline = Pipeline([
        ('existing_transformer', SomeExistingTransformer(...)),
        ('my_custom_step', MyCustomTransformer(custom_param1='value', custom_param2=5))
    ])
    ```

By following this structure, your custom feature transformers will integrate seamlessly with the rest of the financial pipeline, including scikit-learn's `Pipeline` objects, cross-validation routines, and hyperparameter optimization with Optuna.
