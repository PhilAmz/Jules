# Defines the base class for model wrappers.

from sklearn.base import BaseEstimator, MetaEstimatorMixin
import numpy as np
import pandas as pd

class BaseModelWrapper(BaseEstimator, MetaEstimatorMixin):
    """
    Base class for model wrappers.

    All custom model wrappers should inherit from this class.
    It provides a consistent interface compatible with scikit-learn's pipelines
    and utilities, such as hyperparameter tuning.
    """

    def __init__(self):
        self.model_ = None # Stores the actual fitted model instance

    def fit(self, X, y, **kwargs):
        """
        Fit the model to the training data.

        This method must be implemented by subclasses.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Training target.
            **kwargs: Additional keyword arguments for the underlying model's fit method.

        Returns:
            self: The instance itself.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'fit' method.")

    def predict(self, X):
        """
        Make predictions using the fitted model.

        This method must be implemented by subclasses.

        Args:
            X (pd.DataFrame or np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Predictions.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            AttributeError: If the model has not been fitted yet.
        """
        if self.model_ is None:
            raise AttributeError("Model has not been fitted yet. Call 'fit' first.")
        # Subclasses should implement actual prediction logic.
        # This base method is primarily for the check above.

    def predict_proba(self, X):
        """
        Return probability estimates for classification problems.

        This method should be implemented by subclasses if their underlying
        model supports probability predictions (typically for classifiers).

        Args:
            X (pd.DataFrame or np.ndarray): Features for prediction.

        Returns:
            np.ndarray: Probability estimates of shape (n_samples, n_classes).

        Raises:
            NotImplementedError: If the subclass does not implement this method
                                 or if the model is not a classifier.
            AttributeError: If the model has not been fitted yet.
        """
        if self.model_ is None:
            raise AttributeError("Model has not been fitted yet. Call 'fit' first.")
        # Subclasses should implement actual prediction logic.
        # This base method is primarily for the check above.

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Scikit-learn's BaseEstimator provides a generic get_params which works by
        inspecting __init__ arguments. This can be overridden if more complex
        parameter handling is needed, but usually, it's not required if __init__
        stores all parameters as attributes with the same name.

        Args:
            deep (bool, optional): If True, will return the parameters for this
                                   estimator and contained subobjects that are
                                   estimators. Defaults to True.

        Returns:
            dict: Parameter names mapped to their values.
        """
        # This relies on BaseEstimator's implementation which inspects __init__
        return super().get_params(deep=deep)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Scikit-learn's BaseEstimator provides a generic set_params. This can be
        overridden if parameters need special handling upon being set.

        Args:
            **params: Estimator parameters.

        Returns:
            self: Estimator instance.
        """
        # This relies on BaseEstimator's implementation
        return super().set_params(**params)

    def _prepare_input_X(self, X):
        """
        Helper to convert X to a NumPy array if it's a pandas DataFrame/Series.
        Models usually expect NumPy arrays.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        # Could add more specific type checks or conversions if needed
        # For now, assume it's either pandas or numpy, or directly usable by the model
        print(f"Warning: Input X is of type {type(X)}, not pandas DataFrame/Series or numpy array. Using as is.")
        return X

    def _prepare_input_y(self, y):
        """
        Helper to convert y to a NumPy array if it's a pandas DataFrame/Series.
        Also ensures y is 1-dimensional for typical supervised learning tasks.
        """
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y_np = y.values
        elif isinstance(y, np.ndarray):
            y_np = y
        else:
            print(f"Warning: Input y is of type {type(y)}, not pandas DataFrame/Series or numpy array. Attempting to use as is.")
            y_np = np.array(y) # Attempt conversion

        if y_np.ndim > 1 and y_np.shape[1] == 1: # Handles column vectors
            y_np = y_np.ravel()

        # Add further checks if y_np.ndim > 1 and not a column vector, as that's usually an issue.
        if y_np.ndim > 1:
            print(f"Warning: Target y has {y_np.ndim} dimensions. Ensure this is expected by the model.")

        return y_np


if __name__ == '__main__':
    # This base class itself is not directly runnable for fit/predict
    # But we can demonstrate instantiation and param methods.

    class DummyModel(BaseModelWrapper):
        def __init__(self, param_a=1, param_b='test'):
            super().__init__() # Important to call parent's __init__ if it does something
            self.param_a = param_a
            self.param_b = param_b
            # self.model_ will be set in fit

        def fit(self, X, y, **kwargs):
            print(f"DummyModel fitting with X shape {X.shape if hasattr(X, 'shape') else 'N/A'}, y length {len(y) if hasattr(y, '__len__') else 'N/A'}")
            print(f"Params: param_a={self.param_a}, param_b='{self.param_b}'")
            print(f"Other kwargs: {kwargs}")
            self.model_ = "dummy_fitted_model_instance" # Placeholder for a real model
            # Store information from y to check in predict, for example
            self.y_unique_ = np.unique(y) if y is not None else []
            return self

        def predict(self, X):
            super().predict(X) # Checks if model_ is fitted
            print(f"DummyModel predicting on X shape {X.shape if hasattr(X, 'shape') else 'N/A'}")
            # Example: return first unique value from y seen during fit, repeated for X
            if not hasattr(self, 'y_unique_') or len(self.y_unique_) == 0:
                 return np.zeros(X.shape[0] if hasattr(X, 'shape') else 1) # Default if y was None or empty
            return np.full(X.shape[0] if hasattr(X, 'shape') else 1, self.y_unique_[0])


    print("--- BaseModelWrapper Demonstration (via DummyModel) ---")
    dummy_data_X = np.array([[1, 2], [3, 4], [5, 6]])
    dummy_data_y = np.array([0, 1, 0])

    # DataFrame input
    dummy_df_X = pd.DataFrame(dummy_data_X, columns=['feature1', 'feature2'])
    dummy_series_y = pd.Series(dummy_data_y, name='target')

    # Test instantiation
    print("\n1. Instantiation and get_params:")
    dummy_model = DummyModel(param_a=10, param_b='custom_val')
    print("Initial params:", dummy_model.get_params())

    # Test set_params
    print("\n2. set_params:")
    dummy_model.set_params(param_a=20)
    print("Updated params:", dummy_model.get_params())

    # Test fit and predict (using the dummy implementation)
    print("\n3. Fit and Predict:")
    dummy_model.fit(dummy_df_X, dummy_series_y, sample_weight=np.array([0.5, 1.0, 0.5]))
    predictions = dummy_model.predict(dummy_df_X)
    print("Predictions:", predictions)

    # Test predict_proba (should raise NotImplementedError)
    print("\n4. predict_proba (expect NotImplementedError):")
    try:
        dummy_model.predict_proba(dummy_data_X)
    except NotImplementedError as e:
        print(f"Caught expected error: {e}")
    except AttributeError as e: # If model_ check happens first
        print(f"Caught expected error: {e}")


    # Test calling predict before fit
    print("\n5. Predict before fit (expect AttributeError):")
    fresh_dummy_model = DummyModel()
    try:
        fresh_dummy_model.predict(dummy_data_X)
    except AttributeError as e:
        print(f"Caught expected error: {e}")

    print("\n--- End of BaseModelWrapper Demonstration ---")
