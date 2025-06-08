# This file will contain tests for the feature_engineering module.
import unittest
import pandas as pd
import numpy as np

# Add parent directory to sys.path
import sys
import os
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from financial_pipeline.feature_engineering.base import BaseFeatureTransformer
from financial_pipeline.feature_engineering import (
    MovingAverageTransformer,
    RSITransformer,
    MACDTransformer
)

# Define LagFeatureTransformer directly in the test file for simplicity
# (as it was defined in the notebook)
class LagFeatureTransformer(BaseFeatureTransformer):
    """Adds lagged versions of a specified column."""
    def __init__(self, column_to_lag: str, lag_periods: list):
        super().__init__()
        if not isinstance(lag_periods, list) or not all(isinstance(p, int) and p > 0 for p in lag_periods):
            raise ValueError("lag_periods must be a list of positive integers.")
        self.column_to_lag = column_to_lag
        self.lag_periods = lag_periods
        self.new_columns_ = [] # To store names of new columns created

    def fit(self, X: pd.DataFrame, y=None):
        if self.column_to_lag not in X.columns:
            raise ValueError(f"Column '{self.column_to_lag}' not found in input DataFrame.")
        self.new_columns_ = [f'{self.column_to_lag}_lag_{p}' for p in self.lag_periods]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        if self.column_to_lag not in X_transformed.columns:
            # This case might happen if fit was called on a DataFrame with the column,
            # but transform is called on one without it.
            raise ValueError(f"Column '{self.column_to_lag}' not found in DataFrame during transform.")

        for period in self.lag_periods:
            col_name = f'{self.column_to_lag}_lag_{period}'
            X_transformed[col_name] = X_transformed[self.column_to_lag].shift(period)

        # Use the helper from BaseFeatureTransformer for consistent NaN filling
        X_transformed = self._fillna_methods(X_transformed, self.new_columns_)
        return X_transformed


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        """Set up dummy data for feature engineering tests."""
        n_periods = 100
        self.data = pd.DataFrame({
            'Close': np.random.rand(n_periods) * 100 + 50, # Price-like data
            'Open': np.random.rand(n_periods) * 100 + 48,
            'High': np.random.rand(n_periods) * 5 + (np.random.rand(n_periods) * 100 + 50), # High > Close/Open
            'Low': (np.random.rand(n_periods) * 100 + 50) - np.random.rand(n_periods) * 5, # Low < Close/Open
            'Volume': np.random.randint(1000, 10000, size=n_periods)
        }, index=pd.date_range(start='1/1/2020', periods=n_periods))
        # Ensure High is mostly above Close and Low is mostly below Close
        self.data['High'] = self.data[['High', 'Close', 'Open']].max(axis=1) + np.random.rand(n_periods)
        self.data['Low'] = self.data[['Low', 'Close', 'Open']].min(axis=1) - np.random.rand(n_periods)


    def test_moving_average_transformer(self):
        """Test MovingAverageTransformer."""
        window_sizes = [10, 20]
        transformer = MovingAverageTransformer(window_sizes=window_sizes, column='Close')

        # Test fit and transform
        transformed_df = transformer.fit_transform(self.data.copy()) # Use copy to avoid modifying self.data

        self.assertIsInstance(transformed_df, pd.DataFrame)
        for window in window_sizes:
            col_name = f'SMA_Close_{window}'
            self.assertIn(col_name, transformed_df.columns)
            # Check that after initial NaNs (due to window), values are plausible (not all NaN)
            # The _fillna_methods in BaseFeatureTransformer should handle NaNs.
            # So, we expect no NaNs if the input series is long enough.
            if len(self.data) >= window:
                 self.assertFalse(transformed_df[col_name].isnull().all(), f"{col_name} is all NaN.")
                 self.assertEqual(transformed_df[col_name].isnull().sum(), 0, f"{col_name} should have no NaNs after fill.")


    def test_rsi_transformer(self):
        """Test RSITransformer."""
        window = 14
        transformer = RSITransformer(window=window, column='Close')
        transformed_df = transformer.fit_transform(self.data.copy())

        self.assertIsInstance(transformed_df, pd.DataFrame)
        col_name = f'RSI_Close_{window}'
        self.assertIn(col_name, transformed_df.columns)
        if len(self.data) >= window: # RSI needs enough data points
            self.assertFalse(transformed_df[col_name].isnull().all(), f"{col_name} is all NaN.")
            self.assertEqual(transformed_df[col_name].isnull().sum(), 0, f"{col_name} should have no NaNs after fill.")


    def test_macd_transformer(self):
        """Test MACDTransformer."""
        transformer = MACDTransformer(window_slow=26, window_fast=12, window_sign=9, column='Close')
        transformed_df = transformer.fit_transform(self.data.copy())

        self.assertIsInstance(transformed_df, pd.DataFrame)
        expected_cols = ['MACD_line_Close', 'MACD_signal_Close', 'MACD_diff_Close']
        for col_name in expected_cols:
            self.assertIn(col_name, transformed_df.columns)
            # MACD calculations can produce NaNs at the beginning.
            # The _fillna_methods should handle these.
            if len(self.data) >= 26: # Longest window for MACD default
                 self.assertFalse(transformed_df[col_name].isnull().all(), f"{col_name} is all NaN.")
                 self.assertEqual(transformed_df[col_name].isnull().sum(), 0, f"{col_name} should have no NaNs after fill.")

    def test_custom_lag_transformer(self):
        """Test the custom LagFeatureTransformer."""
        lag_periods = [1, 3, 5]
        column_to_lag = 'Volume'
        transformer = LagFeatureTransformer(column_to_lag=column_to_lag, lag_periods=lag_periods)

        transformed_df = transformer.fit_transform(self.data.copy())
        self.assertIsInstance(transformed_df, pd.DataFrame)

        for period in lag_periods:
            col_name = f'{column_to_lag}_lag_{period}'
            self.assertIn(col_name, transformed_df.columns)
            # Check if lagging was done correctly for a few points (excluding NaNs at start)
            # After fillna, the first `period` values will be propagated from the first valid lagged value.
            if len(self.data) > period:
                # The first non-NaN original value is self.data[column_to_lag].iloc[0]
                # After lagging by `period` and then ffill+bfill, the first `period` rows of the lag col
                # should be equal to self.data[column_to_lag].iloc[0].
                # And transformed_df[col_name].iloc[period] should be self.data[column_to_lag].iloc[0]
                pd.testing.assert_series_equal(
                    transformed_df[col_name].iloc[period:],
                    self.data[column_to_lag].shift(period).iloc[period:],
                    check_dtype=False, # joblib dump/load might change int to float if NaNs were ever there
                    check_names=False # Shift changes name
                )
                self.assertEqual(transformed_df[col_name].isnull().sum(), 0, f"{col_name} should have no NaNs after fill.")


if __name__ == '__main__':
    unittest.main()
