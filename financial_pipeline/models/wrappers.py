import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator # For RidgeWrapper
from sklearn.linear_model import Ridge
from .base_model import BaseModelWrapper

# Try to import LightGBM
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False
    # print("Warning: LightGBM not found. LGBMWrapper will not be usable.")

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers as KerasLayers # Alias for clarity
    _TENSORFLOW_AVAILABLE = True
except ImportError:
    _TENSORFLOW_AVAILABLE = False
    # print("Warning: TensorFlow not found. SimpleNNWrapper will not be usable.")


class LGBMWrapper(BaseModelWrapper):
    """
    Wrapper for LightGBM Regressor or Classifier.
    """
    def __init__(self, objective='regression', metric='rmse', n_estimators=100, learning_rate=0.1,
                 num_leaves=31, max_depth=-1, min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
                 random_state=None, n_jobs=-1, reg_alpha=0.0, reg_lambda=0.0, **kwargs):
        super().__init__()
        if not _LGBM_AVAILABLE:
            raise ImportError("LightGBM library is not installed. Please install it to use LGBMWrapper.")

        self.objective = objective
        self.metric = metric
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.kwargs = kwargs # Store other LightGBM specific params

        self._is_classifier = self.objective in ['binary', 'multiclass', 'softmax'] # Add other classification objectives

        if self._is_classifier:
            self.model_ = lgb.LGBMClassifier(**self.get_params_for_lgbm())
        else:
            self.model_ = lgb.LGBMRegressor(**self.get_params_for_lgbm())

    def get_params_for_lgbm(self, deep=False): # deep is for BaseEstimator compatibility, not used here
        """Helper to get relevant params for LGBM internal model."""
        params = {
            'objective': self.objective,
            'metric': self.metric,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
        }
        params.update(self.kwargs)
        return params

    def fit(self, X, y, eval_set=None, **kwargs):
        """
        Fit the LightGBM model.

        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Training target.
            eval_set (list of tuples, optional): List of (X_eval, y_eval) for early stopping.
                                                Example: [(X_val, y_val)]
            **kwargs: Additional arguments passed to the underlying LGBM model's fit method
                      (e.g., early_stopping_rounds, callbacks).
        """
        X_prepared = self._prepare_input_X(X)
        y_prepared = self._prepare_input_y(y)

        # Re-initialize model here to allow hyperparameter tuning via set_params
        # before fit is called.
        if self._is_classifier:
            self.model_ = lgb.LGBMClassifier(**self.get_params_for_lgbm())
        else:
            self.model_ = lgb.LGBMRegressor(**self.get_params_for_lgbm())

        fit_params = {}
        if eval_set:
            fit_params['eval_set'] = []
            for eval_X, eval_y in eval_set:
                fit_params['eval_set'].append((self._prepare_input_X(eval_X), self._prepare_input_y(eval_y)))

        fit_params.update(kwargs) # Add other fit-specific kwargs like early_stopping_rounds

        self.model_.fit(X_prepared, y_prepared, **fit_params)
        return self

    def predict(self, X):
        super().predict(X) # Checks if model_ is fitted (via self.model_ which is the LGBM model itself)
        X_prepared = self._prepare_input_X(X)
        return self.model_.predict(X_prepared)

    def predict_proba(self, X):
        super().predict_proba(X) # Basic check
        if not self._is_classifier:
            raise AttributeError("predict_proba is only available for classification objectives.")
        if not hasattr(self.model_, 'predict_proba'):
             raise AttributeError("The underlying LGBM model does not have predict_proba method.")
        X_prepared = self._prepare_input_X(X)
        return self.model_.predict_proba(X_prepared)

    # get_params and set_params are inherited from BaseModelWrapper, which uses BaseEstimator's.
    # Need to ensure LGBM constructor params are attributes of the wrapper.
    def set_params(self, **params):
        # Update own attributes
        for key, value in params.items():
            setattr(self, key, value)

        # Re-check if it's a classifier after params update (e.g. objective changed)
        self._is_classifier = self.objective in ['binary', 'multiclass', 'softmax']

        # Re-initialize the underlying LightGBM model with new params
        # This ensures that when fit() is called, the model uses the new params.
        # Scikit-learn's clone mechanism relies on get_params and then set_params on a new instance.
        if self._is_classifier:
            self.model_ = lgb.LGBMClassifier(**self.get_params_for_lgbm())
        else:
            self.model_ = lgb.LGBMRegressor(**self.get_params_for_lgbm())
        return self


class SimpleNNWrapper(BaseModelWrapper):
    """
    Wrapper for a simple Keras Sequential Neural Network.
    """
    def __init__(self, layers_config=None, optimizer='adam', loss='mse', metrics=None,
                 epochs=10, batch_size=32, validation_split=0.1, callbacks=None,
                 input_shape=None, random_seed=42):
        super().__init__()
        if not _TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is not installed. Please install it to use SimpleNNWrapper.")

        self.layers_config = layers_config if layers_config is not None else [{'units': 64, 'activation': 'relu', 'dropout': 0.1}]
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics if metrics is not None else ['mae']
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.callbacks = callbacks if callbacks is not None else []
        self.input_shape = input_shape # Can be inferred in fit if not set
        self.random_seed = random_seed

        # Model and history will be set during fit
        self.model_ = None
        self.history_ = None

        if self.random_seed is not None:
            tf.random.set_seed(self.random_seed)
            # np.random.seed(self.random_seed) # If using numpy for any random operations before tf

    def _build_model(self, current_input_shape):
        """Builds the Keras Sequential model."""
        if current_input_shape is None:
            raise ValueError("input_shape must be defined either at init or inferred during fit.")

        model = keras.Sequential(name="SimpleNNSequential")
        # Input layer can be implicitly defined by the first layer's input_shape,
        # or explicitly using KerasLayers.InputLayer

        first_layer = True
        for layer_conf in self.layers_config:
            units = layer_conf.get('units')
            activation = layer_conf.get('activation', 'relu')
            dropout_rate = layer_conf.get('dropout')

            # For the first Dense layer, specify input_shape
            if first_layer:
                model.add(KerasLayers.Dense(units, activation=activation, input_shape=current_input_shape))
                first_layer = False
            else:
                model.add(KerasLayers.Dense(units, activation=activation))

            if dropout_rate and 0 < dropout_rate < 1:
                model.add(KerasLayers.Dropout(dropout_rate))

        # Add an output layer - for now, assume single output unit.
        # This might need to be more flexible (e.g. based on y shape or loss function)
        # If classification (e.g. binary_crossentropy), output activation might be 'sigmoid'
        # If multiclass, 'softmax'. For regression, often 'linear' (or None).
        output_activation = 'linear'
        num_output_units = 1 # Default for regression or simple binary classification before sigmoid

        if isinstance(self.loss, str):
            if 'binary_crossentropy' in self.loss:
                output_activation = 'sigmoid'
            elif 'categorical_crossentropy' in self.loss: # Requires y to be one-hot
                # num_output_units needs to be known (e.g. from y.shape[1] or a param)
                # This part is tricky without knowing y dimensionality beforehand.
                # For now, this wrapper is simpler. Advanced use would need more params.
                print("Warning: 'categorical_crossentropy' might require 'num_output_units' to be set based on y.")
                output_activation = 'softmax'

        model.add(KerasLayers.Dense(num_output_units, activation=output_activation))

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return model

    def fit(self, X, y, **kwargs):
        X_prepared = self._prepare_input_X(X)
        y_prepared = self._prepare_input_y(y)

        if X_prepared.ndim == 1: # Reshape if 1D feature array
            X_prepared = X_prepared.reshape(-1, 1)

        current_input_shape = self.input_shape
        if current_input_shape is None:
            current_input_shape = (X_prepared.shape[1],) # Infer from X data

        self.model_ = self._build_model(current_input_shape)

        # Handle Keras fit specific kwargs like 'verbose'
        fit_kwargs = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'callbacks': self.callbacks,
            'verbose': kwargs.get('verbose', 1) # Default verbose to 1 if not provided
        }
        fit_kwargs.update({k:v for k,v in kwargs.items() if k not in fit_kwargs})


        self.history_ = self.model_.fit(X_prepared, y_prepared, **fit_kwargs)
        return self

    def predict(self, X):
        super().predict(X) # Checks if model_ is fitted
        X_prepared = self._prepare_input_X(X)
        if X_prepared.ndim == 1:
             X_prepared = X_prepared.reshape(-1, 1)

        preds = self.model_.predict(X_prepared)

        # For binary classification with sigmoid, output is (n_samples, 1). Ravel it.
        if isinstance(self.loss, str) and 'binary_crossentropy' in self.loss and preds.shape[1] == 1:
            return preds.ravel() # Return as (n_samples,) for consistency with sklearn
        return preds

    def predict_proba(self, X):
        super().predict_proba(X) # Basic check
        X_prepared = self._prepare_input_X(X)
        if X_prepared.ndim == 1:
             X_prepared = X_prepared.reshape(-1, 1)

        if isinstance(self.loss, str) and 'binary_crossentropy' in self.loss:
            probs = self.model_.predict(X_prepared) # Output is already probability for sigmoid
            return np.hstack([1 - probs, probs]) # Return as (n_samples, 2) for sklearn compatibility
        elif isinstance(self.loss, str) and 'categorical_crossentropy' in self.loss:
            return self.model_.predict(X_prepared) # Output is (n_samples, n_classes)
        else:
            raise AttributeError("predict_proba is typically for classification losses like 'binary_crossentropy' or 'categorical_crossentropy'.")

    # get_params and set_params will be inherited.
    # Keras models are not easily 're-initialized' by just setting attributes like scikit-learn models.
    # If set_params changes model structure (layers_config, optimizer, loss), _build_model needs to be called again.
    # This usually happens naturally if fit is called after set_params.
    def set_params(self, **params):
        # Default behavior from BaseEstimator
        super().set_params(**params)
        # For Keras, if structural params change, the model needs to be rebuilt.
        # We assume _build_model will be called in fit() using the new params.
        # Clear the existing model to ensure it's rebuilt if fit is called.
        self.model_ = None
        self.history_ = None
        if self.random_seed is not None: # Re-apply seed if it's part of params
            tf.random.set_seed(self.random_seed)
        return self


class RidgeWrapper(BaseModelWrapper):
    """
    Wrapper for scikit-learn's Ridge Regressor.
    """
    def __init__(self, alpha=1.0, fit_intercept=True, solver='auto', random_state=None, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.random_state = random_state
        self.kwargs = kwargs # Store other Ridge specific params

        # Initialize model here, will be re-initialized in fit if params change via set_params
        self.model_ = Ridge(**self.get_params_for_ridge())

    def get_params_for_ridge(self):
        params = {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'solver': self.solver,
            'random_state': self.random_state,
        }
        params.update(self.kwargs)
        return params

    def fit(self, X, y, **kwargs):
        X_prepared = self._prepare_input_X(X)
        y_prepared = self._prepare_input_y(y)

        # Re-initialize with current params (in case set_params was used)
        self.model_ = Ridge(**self.get_params_for_ridge())
        self.model_.fit(X_prepared, y_prepared, **kwargs) # Pass sample_weight etc. if provided
        return self

    def predict(self, X):
        super().predict(X) # Checks if model_ is fitted
        X_prepared = self._prepare_input_X(X)
        return self.model_.predict(X_prepared)

    # predict_proba is not applicable for Ridge regression
    # get_params and set_params are inherited.
    def set_params(self, **params):
        super().set_params(**params)
        # Re-initialize the underlying model with new params for consistency
        self.model_ = Ridge(**self.get_params_for_ridge())
        return self


if __name__ == '__main__':
    from sklearn.datasets import make_regression, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

    # --- Generate Dummy Data ---
    X_reg, y_reg = make_regression(n_samples=100, n_features=10, random_state=42)
    X_clf, y_clf = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    # Convert to DataFrames for more realistic testing of _prepare_input_X
    X_reg_train_df = pd.DataFrame(X_reg_train, columns=[f'f{i}' for i in range(X_reg_train.shape[1])])
    X_reg_test_df = pd.DataFrame(X_reg_test, columns=[f'f{i}' for i in range(X_reg_test.shape[1])])
    y_reg_train_series = pd.Series(y_reg_train, name='target_reg')

    X_clf_train_df = pd.DataFrame(X_clf_train, columns=[f'f{i}' for i in range(X_clf_train.shape[1])])
    X_clf_test_df = pd.DataFrame(X_clf_test, columns=[f'f{i}' for i in range(X_clf_test.shape[1])])
    y_clf_train_series = pd.Series(y_clf_train, name='target_clf')


    if _LGBM_AVAILABLE:
        print("\n--- LGBMWrapper Demonstration ---")
        # Regression
        print("\nLGBM Regression:")
        lgbm_reg = LGBMWrapper(objective='regression', n_estimators=50, random_state=42, metric='l2')
        lgbm_reg.fit(X_reg_train_df, y_reg_train_series, eval_set=[(X_reg_test_df, y_reg_test)], early_stopping_rounds=5, verbose=-1) #verbose for lgbm
        reg_preds = lgbm_reg.predict(X_reg_test_df)
        print(f"LGBM Regression MSE: {mean_squared_error(y_reg_test, reg_preds):.4f}")

        # Classification
        print("\nLGBM Classification:")
        lgbm_clf = LGBMWrapper(objective='binary', n_estimators=50, random_state=42, metric='binary_logloss')
        # Example of using set_params before fit
        lgbm_clf.set_params(learning_rate=0.05, num_leaves=20)
        lgbm_clf.fit(X_clf_train_df, y_clf_train_series, eval_set=[(X_clf_test_df, y_clf_test)], early_stopping_rounds=5, verbose=-1)
        clf_preds = lgbm_clf.predict(X_clf_test_df)
        clf_probas = lgbm_clf.predict_proba(X_clf_test_df)
        print(f"LGBM Classification Accuracy: {accuracy_score(y_clf_test, clf_preds):.4f}")
        print(f"LGBM Classification ROC AUC: {roc_auc_score(y_clf_test, clf_probas[:, 1]):.4f}")
        print("LGBM Params:", lgbm_clf.get_params(deep=True))

    if _TENSORFLOW_AVAILABLE:
        print("\n--- SimpleNNWrapper Demonstration ---")
        # Regression with NN
        print("\nNN Regression:")
        nn_reg_layers = [
            {'units': 32, 'activation': 'relu', 'dropout': 0.05},
            {'units': 16, 'activation': 'relu'}
        ]
        nn_reg = SimpleNNWrapper(layers_config=nn_reg_layers, optimizer='adam', loss='mean_squared_error',
                                 epochs=10, batch_size=8, validation_split=0.1, random_seed=42)
        nn_reg.fit(X_reg_train, y_reg_train, verbose=0) # verbose for keras fit
        nn_reg_preds = nn_reg.predict(X_reg_test)
        print(f"NN Regression MSE: {mean_squared_error(y_reg_test, nn_reg_preds.ravel()):.4f}")

        # Classification with NN
        print("\nNN Classification:")
        nn_clf_layers = [
            {'units': 32, 'activation': 'relu'},
            {'units': 16, 'activation': 'relu', 'dropout': 0.1}
        ]
        # For binary classification, ensure the output layer is appropriate (sigmoid)
        # and loss is binary_crossentropy. The _build_model handles this.
        nn_clf = SimpleNNWrapper(layers_config=nn_clf_layers, optimizer='adam', loss='binary_crossentropy',
                                 metrics=['accuracy'], epochs=10, batch_size=8, validation_split=0.1, random_seed=42)
        nn_clf.fit(X_clf_train, y_clf_train, verbose=0)
        nn_clf_preds_raw = nn_clf.predict(X_clf_test) # Output from sigmoid, continuous
        nn_clf_preds = (nn_clf_preds_raw > 0.5).astype(int)
        nn_clf_probas = nn_clf.predict_proba(X_clf_test) # Should give (n, 2) shape

        print(f"NN Classification Accuracy: {accuracy_score(y_clf_test, nn_clf_preds):.4f}")
        if nn_clf_probas.shape[1] == 2:
             print(f"NN Classification ROC AUC: {roc_auc_score(y_clf_test, nn_clf_probas[:, 1]):.4f}")
        else: # If predict_proba returned raw sigmoid outputs
            print(f"NN Classification ROC AUC (from raw predict): {roc_auc_score(y_clf_test, nn_clf_preds_raw):.4f}")
        print("NN Params:", nn_clf.get_params(deep=True))


    print("\n--- RidgeWrapper Demonstration ---")
    ridge_model = RidgeWrapper(alpha=1.0)
    # Example of using set_params
    ridge_model.set_params(alpha=0.5, solver='svd')
    ridge_model.fit(X_reg_train_df, y_reg_train_series)
    ridge_preds = ridge_model.predict(X_reg_test_df)
    print(f"Ridge Regression MSE: {mean_squared_error(y_reg_test, ridge_preds):.4f}")
    print("Ridge Params:", ridge_model.get_params(deep=True))

    print("\n--- End of Wrappers Demonstration ---")
