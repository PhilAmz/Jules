# Models Module

## Overview

The `models` module is designed to provide a standardized way to incorporate various machine learning models into the financial pipeline. It achieves this through a system of model wrappers that ensure compatibility with scikit-learn's API (e.g., `fit`, `predict`, `get_params`, `set_params`). This allows any wrapped model to be used seamlessly within scikit-learn `Pipeline` objects and with utilities like `GridSearchCV` or Optuna-based hyperparameter optimization.

Key components:
- `base_model.py`: Contains `BaseModelWrapper`, an abstract base class that all specific model wrappers should inherit from. It defines the common interface expected by the pipeline.
- `wrappers.py`: Contains concrete implementations of wrappers for popular libraries like LightGBM (`LGBMWrapper`), TensorFlow/Keras (`SimpleNNWrapper`), and scikit-learn's own models (`RidgeWrapper`).

## Adding Custom Model Wrappers

If you want to integrate a model that isn't already provided (e.g., from another library like XGBoost, CatBoost, or a custom PyTorch model), you should create a new wrapper class for it.

1.  **Create or Choose a File:** You can add your wrapper to `wrappers.py` or create a new file within the `models` directory (e.g., `my_custom_wrappers.py`).

2.  **Inherit from `BaseModelWrapper`:**
    Your custom model wrapper should inherit from `financial_pipeline.models.base_model.BaseModelWrapper`.

    ```python
    from financial_pipeline.models.base_model import BaseModelWrapper
    # Import the library for the model you are wrapping
    # import some_model_library

    class MyCustomModelWrapper(BaseModelWrapper):
        def __init__(self, param1='default', param2=100, **kwargs):
            super().__init__() # Call the parent's __init__
            self.param1 = param1
            self.param2 = param2
            # Store other relevant parameters passed via kwargs if needed by your model
            self.kwargs = kwargs

            # Initialize your actual model instance here, or defer to fit()
            # self.model_ = some_model_library.Model(param1=self.param1, ...)
            # It's often better to initialize the actual model instance just before fit()
            # after set_params might have been called by hyperparameter tuning tools.
            # The self.model_ attribute is conventionally where the fitted model is stored.
    ```
    Ensure all parameters in `__init__` have default values and are stored as instance attributes with the same names. This is crucial for `get_params` and `set_params` (inherited from `BaseEstimator`) to work correctly.

3.  **Implement `fit(self, X, y, **kwargs)`:**
    This method trains your model.
    *   It should internally instantiate the actual model from the library if not done in `__init__` (this allows `set_params` to work effectively before fitting).
    *   Store the trained model instance in `self.model_`.
    *   The `X` and `y` arguments will typically be pandas DataFrames/Series or NumPy arrays. You can use `self._prepare_input_X(X)` and `self._prepare_input_y(y)` (from `BaseModelWrapper`) to convert pandas inputs to NumPy arrays if your underlying model requires it.
    *   Pass through any relevant `**kwargs` (e.g., `eval_set`, `sample_weight`, `callbacks`) to the underlying model's fit method.

    ```python
    def fit(self, X, y, **kwargs):
        X_prepared = self._prepare_input_X(X) # Example conversion
        y_prepared = self._prepare_input_y(y)   # Example conversion

        # Instantiate the actual model using current parameters
        # This ensures that if set_params was called, the new params are used.
        self.model_ = some_model_library.Model(
            param1=self.param1,
            param2=self.param2,
            **self.kwargs # Pass other stored kwargs
        )

        # Example: handling an eval_set for models that support it
        eval_set_prepared = None
        if 'eval_set' in kwargs and kwargs['eval_set'] is not None:
            eval_X, eval_y = kwargs['eval_set'][0] # Assuming one eval set for simplicity
            eval_set_prepared = [(self._prepare_input_X(eval_X), self._prepare_input_y(eval_y))]

        fit_params = {}
        if eval_set_prepared:
            fit_params['eval_set'] = eval_set_prepared
        # Add other relevant kwargs to fit_params if needed
        # fit_params.update(kwargs.get('specific_fit_args', {}))

        self.model_.fit(X_prepared, y_prepared, **fit_params)
        return self
    ```

4.  **Implement `predict(self, X)`:**
    This method should use the trained model (`self.model_`) to make predictions on new data `X`.
    *   Remember to call `super().predict(X)` first if you want the base class check for `self.model_` being fitted. However, the current `BaseModelWrapper.predict` doesn't raise `NotImplementedError` anymore, it just performs the check. The actual prediction must be done by your model.
    *   Prepare `X` if needed (e.g., `self._prepare_input_X(X)`).

    ```python
    def predict(self, X):
        if self.model_ is None: # Or call super().predict(X) which does this check
            raise AttributeError("Model has not been fitted yet. Call 'fit' first.")

        X_prepared = self._prepare_input_X(X)
        return self.model_.predict(X_prepared)
    ```

5.  **Implement `predict_proba(self, X)` (If Applicable):**
    If your model is a classifier and can output probability estimates, implement this method.
    *   It should return an array-like of shape (n_samples, n_classes).
    *   Follow a similar structure to `predict`.

    ```python
    def predict_proba(self, X):
        if self.model_ is None: # Or call super().predict_proba(X)
            raise AttributeError("Model has not been fitted yet. Call 'fit' first.")
        if not hasattr(self.model_, 'predict_proba'):
            raise AttributeError("Underlying model does not support predict_proba.")

        X_prepared = self._prepare_input_X(X)
        return self.model_.predict_proba(X_prepared)
    ```

6.  **`get_params` and `set_params`:**
    These are usually inherited correctly from `BaseEstimator` (via `BaseModelWrapper`) if you followed the `__init__` conventions. Override `set_params` if special handling is needed when a parameter is changed (e.g., re-initializing parts of the model or clearing cached states). The provided wrappers (`LGBMWrapper`, `SimpleNNWrapper`, `RidgeWrapper`) show examples where `set_params` re-initializes the underlying model to ensure new parameters are picked up before the next `fit` call, which is good practice.

7.  **Example Reference:**
    Study the existing wrappers in `financial_pipeline/models/wrappers.py` (like `LGBMWrapper`, `SimpleNNWrapper`, `RidgeWrapper`) to see concrete implementations.

8.  **Import and Use:**
    After defining your wrapper, import it into `financial_pipeline/models/__init__.py` and add it to the `__all__` list to make it easily accessible. Then, you can instantiate it and use it in the pipeline like any other model wrapper.

    ```python
    # In financial_pipeline/models/__init__.py
    # from .my_custom_wrappers_file import MyCustomModelWrapper
    # __all__ = [..., 'MyCustomModelWrapper']

    # In your pipeline script or notebook
    # from financial_pipeline.models import MyCustomModelWrapper
    # my_model = MyCustomModelWrapper(param1='custom_setting')
    ```

This structure ensures that new models can be readily integrated and are compatible with the broader ecosystem of scikit-learn tools and the financial pipeline's orchestration logic.
