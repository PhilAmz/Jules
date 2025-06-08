# This file will contain functions for calculating technical indicators.
import pandas as pd
import ta
from .base import BaseFeatureTransformer

class MovingAverageTransformer(BaseFeatureTransformer):
    """
    Adds Simple Moving Average (SMA) features to the DataFrame.
    """
    def __init__(self, window_sizes: list, column: str = 'Close'):
        """
        Args:
            window_sizes (list of int): List of window sizes for SMA calculation.
            column (str, optional): The column to calculate SMA on. Defaults to 'Close'.
        """
        if not isinstance(window_sizes, list) or not all(isinstance(w, int) and w > 0 for w in window_sizes):
            raise ValueError("window_sizes must be a list of positive integers.")
        if not isinstance(column, str):
            raise ValueError("column must be a string.")

        self.window_sizes = window_sizes
        self.column = column
        self.new_columns_ = []

    def fit(self, X: pd.DataFrame, y=None):
        self.new_columns_ = [f'SMA_{self.column}_{w}' for w in self.window_sizes]
        # Check if the source column exists
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds SMA features to the DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame with a column specified in `self.column`.

        Returns:
            pd.DataFrame: DataFrame with added SMA columns.
        """
        X_transformed = X.copy()
        if self.column not in X_transformed.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame during transform.")

        for window in self.window_sizes:
            col_name = f'SMA_{self.column}_{window}'
            X_transformed[col_name] = ta.trend.SMAIndicator(close=X_transformed[self.column], window=window).sma_indicator()

        # Fill NaNs introduced by indicators
        X_transformed = self._fillna_methods(X_transformed, self.new_columns_)
        return X_transformed

class RSITransformer(BaseFeatureTransformer):
    """
    Adds Relative Strength Index (RSI) feature to the DataFrame.
    """
    def __init__(self, window: int = 14, column: str = 'Close'):
        """
        Args:
            window (int, optional): Window size for RSI calculation. Defaults to 14.
            column (str, optional): The column to calculate RSI on. Defaults to 'Close'.
        """
        if not isinstance(window, int) or window <= 0:
            raise ValueError("window must be a positive integer.")
        if not isinstance(column, str):
            raise ValueError("column must be a string.")

        self.window = window
        self.column = column
        self.new_columns_ = []

    def fit(self, X: pd.DataFrame, y=None):
        self.new_columns_ = [f'RSI_{self.column}_{self.window}']
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds RSI feature to the DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame with a column specified in `self.column`.

        Returns:
            pd.DataFrame: DataFrame with added RSI column.
        """
        X_transformed = X.copy()
        if self.column not in X_transformed.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame during transform.")

        col_name = f'RSI_{self.column}_{self.window}'
        X_transformed[col_name] = ta.momentum.RSIIndicator(close=X_transformed[self.column], window=self.window).rsi()

        X_transformed = self._fillna_methods(X_transformed, self.new_columns_)
        return X_transformed

class MACDTransformer(BaseFeatureTransformer):
    """
    Adds Moving Average Convergence Divergence (MACD) features to the DataFrame.
    """
    def __init__(self, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9, column: str = 'Close'):
        """
        Args:
            window_slow (int, optional): Slow EMA window. Defaults to 26.
            window_fast (int, optional): Fast EMA window. Defaults to 12.
            window_sign (int, optional): Signal line EMA window. Defaults to 9.
            column (str, optional): The column to calculate MACD on. Defaults to 'Close'.
        """
        if not all(isinstance(w, int) and w > 0 for w in [window_slow, window_fast, window_sign]):
            raise ValueError("All window parameters must be positive integers.")
        if not isinstance(column, str):
            raise ValueError("column must be a string.")
        if window_fast >= window_slow:
            raise ValueError("Fast window should be smaller than slow window for MACD.")

        self.window_slow = window_slow
        self.window_fast = window_fast
        self.window_sign = window_sign
        self.column = column
        self.new_columns_ = []

    def fit(self, X: pd.DataFrame, y=None):
        self.new_columns_ = [
            f'MACD_line_{self.column}',
            f'MACD_signal_{self.column}',
            f'MACD_diff_{self.column}'
        ]
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and adds MACD features (line, signal, diff) to the DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame with a column specified in `self.column`.

        Returns:
            pd.DataFrame: DataFrame with added MACD columns.
        """
        X_transformed = X.copy()
        if self.column not in X_transformed.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame during transform.")

        macd_indicator = ta.trend.MACD(
            close=X_transformed[self.column],
            window_slow=self.window_slow,
            window_fast=self.window_fast,
            window_sign=self.window_sign
        )
        X_transformed[f'MACD_line_{self.column}'] = macd_indicator.macd()
        X_transformed[f'MACD_signal_{self.column}'] = macd_indicator.macd_signal()
        X_transformed[f'MACD_diff_{self.column}'] = macd_indicator.macd_diff()

        X_transformed = self._fillna_methods(X_transformed, self.new_columns_)
        return X_transformed

if __name__ == '__main__':
    # Create a sample DataFrame (e.g., from financial data)
    sample_data = {
        'Open': [100, 102, 101, 105, 107, 108, 110, 112, 111, 115, 117, 118, 120, 122, 121, 125, 127, 128, 130, 132],
        'High': [103, 104, 106, 108, 109, 111, 113, 115, 114, 118, 119, 121, 123, 125, 124, 128, 129, 131, 133, 135],
        'Low':  [99, 101, 100, 103, 105, 106, 108, 110, 109, 113, 115, 116, 118, 120, 119, 123, 125, 126, 128, 130],
        'Close':[102, 103, 105, 107, 108, 110, 112, 113, 112, 117, 118, 120, 122, 123, 122, 127, 128, 130, 132, 133],
        'Volume':[1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900]
    }
    dates = pd.date_range(start='2023-01-01', periods=20)
    df = pd.DataFrame(sample_data, index=dates)

    print("Original DataFrame:")
    print(df.head())

    # --- MovingAverageTransformer Example ---
    sma_transformer = MovingAverageTransformer(window_sizes=[5, 10], column='Close')
    df_sma = sma_transformer.fit_transform(df)
    print("\nDataFrame with SMA features:")
    print(df_sma.head(15))
    print("\nNaNs in SMA columns:")
    print(df_sma[['SMA_Close_5', 'SMA_Close_10']].isnull().sum())


    # --- RSITransformer Example ---
    rsi_transformer = RSITransformer(window=14, column='Close')
    df_rsi = rsi_transformer.fit_transform(df) # Use original df for a clean test
    print("\nDataFrame with RSI feature:")
    print(df_rsi[['Close', 'RSI_Close_14']].head(20)) # Print more rows to see RSI values
    print("\nNaNs in RSI column:")
    print(df_rsi['RSI_Close_14'].isnull().sum())


    # --- MACDTransformer Example ---
    macd_transformer = MACDTransformer(window_slow=12, window_fast=5, window_sign=4, column='Close') # Adjusted for smaller dataset
    df_macd = macd_transformer.fit_transform(df) # Use original df for a clean test
    print("\nDataFrame with MACD features:")
    print(df_macd[['Close', 'MACD_line_Close', 'MACD_signal_Close', 'MACD_diff_Close']].head(20))
    print("\nNaNs in MACD columns:")
    print(df_macd[['MACD_line_Close', 'MACD_signal_Close', 'MACD_diff_Close']].isnull().sum())

    # --- Test with a different column (e.g., 'Open') ---
    print("\n--- Testing with 'Open' column ---")
    sma_open_transformer = MovingAverageTransformer(window_sizes=[5], column='Open')
    df_sma_open = sma_open_transformer.fit_transform(df)
    print("\nDataFrame with SMA ('Open') features:")
    print(df_sma_open.head(10))
    print("\nNaNs in SMA_Open_5 column:")
    print(df_sma_open['SMA_Open_5'].isnull().sum())

    # --- Test error handling ---
    try:
        err_transformer = MovingAverageTransformer(window_sizes=[5], column='NonExistentCol')
        err_transformer.fit_transform(df)
    except ValueError as e:
        print(f"\nSuccessfully caught error for non-existent column: {e}")

    try:
        err_transformer_window = MovingAverageTransformer(window_sizes=[-5], column='Close')
    except ValueError as e:
        print(f"\nSuccessfully caught error for invalid window: {e}")

    try:
        err_transformer_macd_window = MACDTransformer(window_fast=10, window_slow=5) # Fast > Slow
    except ValueError as e:
        print(f"\nSuccessfully caught error for invalid MACD windows: {e}")
