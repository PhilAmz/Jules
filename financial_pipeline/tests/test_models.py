# This file will contain tests for the models module.
import unittest
import numpy as np
import pandas as pd

# Add parent directory to sys.path
import sys
import os
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from financial_pipeline.models import LGBMWrapper, SimpleNNWrapper, RidgeWrapper

# Check if TensorFlow and LightGBM are available, skip tests if not.
_LGBM_AVAILABLE = True
try:
    import lightgbm
except ImportError:
    _LGBM_AVAILABLE = False

_TENSORFLOW_AVAILABLE = True
try:
    import tensorflow
except ImportError:
    _TENSORFLOW_AVAILABLE = False


class TestModelWrappers(unittest.TestCase):

    def setUp(self):
        """Set up dummy data for model wrapper tests."""
        self.n_samples = 30 # Slightly more data for NN
        self.n_features = 5

        self.X_reg_df = pd.DataFrame(np.random.rand(self.n_samples, self.n_features),
                                  columns=[f'feat_{i}' for i in range(self.n_features)])
        self.y_reg_series = pd.Series(np.random.rand(self.n_samples), name='target_reg')

        self.X_clf_df = pd.DataFrame(np.random.rand(self.n_samples, self.n_features),
                                   columns=[f'feat_{i}' for i in range(self.n_features)])
        self.y_clf_series = pd.Series(np.random.randint(0, 2, self.n_samples), name='target_clf')

    @unittest.skipIf(not _LGBM_AVAILABLE, "LightGBM not installed, skipping LGBMWrapper tests.")
    def test_lgbm_regressor(self):
        """Test LGBMWrapper for regression."""
        model = LGBMWrapper(objective='regression', n_estimators=10, num_leaves=5, random_state=42, n_jobs=1)
        model.fit(self.X_reg_df, self.y_reg_series)
        predictions = model.predict(self.X_reg_df)
        self.assertEqual(predictions.shape[0], self.n_samples, "Predictions shape mismatch for LGBM regressor.")

    @unittest.skipIf(not _LGBM_AVAILABLE, "LightGBM not installed, skipping LGBMWrapper tests.")
    def test_lgbm_classifier(self):
        """Test LGBMWrapper for classification."""
        model = LGBMWrapper(objective='binary', n_estimators=10, num_leaves=5, random_state=42, n_jobs=1)
        model.fit(self.X_clf_df, self.y_clf_series)

        predictions = model.predict(self.X_clf_df)
        self.assertEqual(predictions.shape[0], self.n_samples, "Predictions shape mismatch for LGBM classifier.")

        proba_predictions = model.predict_proba(self.X_clf_df)
        self.assertEqual(proba_predictions.shape[0], self.n_samples, "Probability predictions shape mismatch for LGBM classifier.")
        self.assertEqual(proba_predictions.shape[1], 2, "Probability predictions should have 2 columns for binary classification.")

    @unittest.skipIf(not _TENSORFLOW_AVAILABLE, "TensorFlow not installed, skipping SimpleNNWrapper tests.")
    def test_simple_nn_regressor(self):
        """Test SimpleNNWrapper for regression."""
        # Minimal config for speed
        nn_layers_config = [{'units': 8, 'activation': 'relu'}]
        model = SimpleNNWrapper(layers_config=nn_layers_config, epochs=2, batch_size=8, random_seed=42)

        model.fit(self.X_reg_df, self.y_reg_series) # verbose is handled by model.fit default or kwargs
        predictions = model.predict(self.X_reg_df)
        self.assertEqual(predictions.shape[0], self.n_samples, "Predictions shape mismatch for NN regressor.")
        # For regression, predict often returns (n_samples, 1), ensure it's raveled if needed by metrics later.
        # The wrapper's predict for binary_crossentropy loss does ravel. For MSE, it might be (n,1) or (n,).
        # Let's check if it's 1D or (N,1)
        self.assertTrue(predictions.ndim == 1 or predictions.shape[1] == 1, "NN Regressor output should be 1D or (N,1)")


    @unittest.skipIf(not _TENSORFLOW_AVAILABLE, "TensorFlow not installed, skipping SimpleNNWrapper tests.")
    def test_simple_nn_classifier(self):
        """Test SimpleNNWrapper for classification."""
        nn_layers_config = [{'units': 8, 'activation': 'relu'}]
        model = SimpleNNWrapper(layers_config=nn_layers_config, loss='binary_crossentropy',
                                epochs=2, batch_size=8, random_seed=42)

        model.fit(self.X_clf_df, self.y_clf_series) # verbose is handled by model.fit default or kwargs

        # predict() for binary_crossentropy in wrapper should return (n_samples,)
        predictions_continuous = model.predict(self.X_clf_df)
        self.assertEqual(predictions_continuous.shape[0], self.n_samples, "Continuous predictions shape mismatch for NN classifier.")
        self.assertEqual(predictions_continuous.ndim, 1, "Continuous predictions for binary NN should be 1D (after ravel).")

        # predict_proba() should return (n_samples, 2)
        proba_predictions = model.predict_proba(self.X_clf_df)
        self.assertEqual(proba_predictions.shape[0], self.n_samples, "Probability predictions shape mismatch for NN classifier.")
        self.assertEqual(proba_predictions.shape[1], 2, "Probability predictions should have 2 columns for NN binary classifier.")


    def test_ridge_wrapper(self):
        """Test RidgeWrapper for regression."""
        model = RidgeWrapper(alpha=1.0, random_state=42)
        model.fit(self.X_reg_df, self.y_reg_series)
        predictions = model.predict(self.X_reg_df)
        self.assertEqual(predictions.shape[0], self.n_samples, "Predictions shape mismatch for Ridge regressor.")

    @unittest.skipIf(not _LGBM_AVAILABLE, "LightGBM not available")
    def test_lgbm_set_params_and_refit(self):
        """Test if set_params followed by fit works for LGBM."""
        model = LGBMWrapper(objective='regression', n_estimators=5, random_state=42, n_jobs=1)
        model.fit(self.X_reg_df, self.y_reg_series) # Initial fit
        initial_preds = model.predict(self.X_reg_df)

        # Change parameters that should affect results more significantly
        model.set_params(n_estimators=10, learning_rate=0.5, num_leaves=2, min_child_samples=5)
        model.fit(self.X_reg_df, self.y_reg_series) # Refit
        new_preds = model.predict(self.X_reg_df)

        self.assertEqual(new_preds.shape[0], self.n_samples)
        # Check that predictions are different. With very different tree structures (due to num_leaves=2),
        # it's highly likely predictions will change, even on small data, unless all splits are trivial.
        if np.allclose(initial_preds, new_preds):
            print(f"Warning: Predictions did not change significantly in test_lgbm_set_params_and_refit. Initial: {initial_preds[:5]}, New: {new_preds[:5]}")
        self.assertFalse(np.allclose(initial_preds, new_preds, atol=1e-5, rtol=1e-5),
                         "Predictions should differ after set_params and refit for LGBM with significant parameter changes.")


if __name__ == '__main__':
    unittest.main()
