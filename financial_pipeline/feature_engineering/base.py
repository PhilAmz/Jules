# Defines the base class for feature transformers.

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class BaseFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for feature engineering transformers.

    All custom feature transformers should inherit from this class.
    It provides a consistent interface with scikit-learn's pipelines.
    """

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit method for the transformer.

        In most feature engineering cases, this method doesn't need to learn
        anything from the data, so it just returns self. However, it can be
        overridden if fitting is required (e.g., learning min/max for scaling).

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series or np.array, optional): Target variable. Defaults to None.

        Returns:
            self: The instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method to add new features.

        This method must be implemented by subclasses to perform the actual
        feature engineering.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: DataFrame with added features.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("Each specific transformer must implement the 'transform' method.")

    def _fillna_methods(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Helper method to apply forward-fill and backward-fill to specified columns.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            columns (list): A list of column names to fill NaN values for.

        Returns:
            pd.DataFrame: The DataFrame with NaN values filled in the specified columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
            raise TypeError("'columns' must be a list of strings.")

        for col in columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
                df[col] = df[col].fillna(method='bfill')
            else:
                # This could be a warning or an error depending on desired strictness
                print(f"Warning: Column '{col}' not found in DataFrame for fillna operation.")
        return df

if __name__ == '__main__':
    # Example Usage (conceptual)
    class MyCustomTransformer(BaseFeatureTransformer):
        def __init__(self, new_col_name='custom_feature'):
            self.new_col_name = new_col_name

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            X_transformed = X.copy()
            if 'Close' in X_transformed.columns:
                X_transformed[self.new_col_name] = X_transformed['Close'] * 2 # Example transformation
                # Suppose this transformation could introduce NaNs
                X_transformed[self.new_col_name].iloc[0] = pd.NA
                X_transformed = self._fillna_methods(X_transformed, [self.new_col_name])
            else:
                print("Warning: 'Close' column not found for MyCustomTransformer.")
            return X_transformed

    # Create a sample DataFrame
    sample_data = {'Close': [10, 20, pd.NA, 40, 50], 'Open': [9, 19, 29, 39, 49]}
    sample_df = pd.DataFrame(sample_data)

    print("Original DataFrame:")
    print(sample_df)

    custom_transformer = MyCustomTransformer()
    transformed_df = custom_transformer.fit_transform(sample_df)

    print("\nTransformed DataFrame:")
    print(transformed_df)
    print("\nCheck NaNs in new column:")
    print(transformed_df[custom_transformer.new_col_name].isnull().sum())

    # Test fillna helper directly
    helper_test_df = pd.DataFrame({'A': [1, pd.NA, 3, pd.NA, 5], 'B': [pd.NA, 2, pd.NA, 4, pd.NA]})
    print("\nHelper Test DataFrame (before):")
    print(helper_test_df)
    filled_helper_df = custom_transformer._fillna_methods(helper_test_df.copy(), ['A', 'B'])
    print("\nHelper Test DataFrame (after):")
    print(filled_helper_df)
