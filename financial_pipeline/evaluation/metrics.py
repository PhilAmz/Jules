# This file will contain functions for calculating evaluation metrics.

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    mean_absolute_percentage_error
)
import numpy as np

# Re-export common sklearn metrics for easy access
# Regression Metrics
mean_squared_error = mean_squared_error
mean_absolute_error = mean_absolute_error
mean_absolute_percentage_error = mean_absolute_percentage_error # MAPE
r2_score = r2_score

# Classification Metrics
accuracy_score = accuracy_score
f1_score = f1_score # Can specify average for multiclass, e.g., f1_score(y_true, y_pred, average='weighted')
precision_score = precision_score
recall_score = recall_score
roc_auc_score = roc_auc_score # For binary or multiclass (ovr/ovo) probabilities
log_loss = log_loss # Cross-entropy loss


# Custom Metrics (or wrappers for more complex calculations)

def sharpe_ratio(y_true_returns, y_pred_signals, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculates the Sharpe Ratio. (Placeholder)

    This is a placeholder function. A proper Sharpe Ratio calculation requires:
    1.  Translating `y_pred_signals` into a portfolio allocation strategy.
    2.  Calculating daily portfolio returns based on `y_true_returns` and the strategy.
    3.  Calculating the annualized mean and standard deviation of these portfolio returns.

    Args:
        y_true_returns (np.ndarray or pd.Series): Actual returns of the asset(s).
        y_pred_signals (np.ndarray or pd.Series): Predicted signals from the model
                                                  (e.g., buy/sell/hold, predicted returns, allocation weights).
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.0.
        periods_per_year (int, optional): Number of trading periods in a year
                                          (e.g., 252 for daily data). Defaults to 252.

    Returns:
        float: Calculated Sharpe Ratio.

    Raises:
        NotImplementedError: As this is a placeholder.
    """
    # Example conceptual steps if it were implemented:
    # 1. Generate portfolio returns based on signals and true_returns
    #    portfolio_returns = ... (this is the complex part)
    # 2. Calculate excess returns over risk-free rate
    #    daily_risk_free_rate = (1 + risk_free_rate)**(1/periods_per_year) - 1
    #    excess_returns = portfolio_returns - daily_risk_free_rate
    # 3. Calculate annualized Sharpe Ratio
    #    mean_excess_return = np.mean(excess_returns)
    #    std_dev_excess_return = np.std(excess_returns)
    #    if std_dev_excess_return == 0:
    #        return np.nan # Or handle as per financial convention (e.g., 0 or very large if mean_excess_return > 0)
    #    annualized_sharpe = (mean_excess_return / std_dev_excess_return) * np.sqrt(periods_per_year)
    #    return annualized_sharpe
    raise NotImplementedError(
        "Sharpe Ratio calculation requires portfolio simulation logic based on model signals."
    )

def adjusted_r2_score(y_true, y_pred, n_features):
    """
    Calculates the Adjusted R-squared score.

    Args:
        y_true (np.ndarray or pd.Series): True target values.
        y_pred (np.ndarray or pd.Series): Predicted target values.
        n_features (int): Number of features used in the model.

    Returns:
        float: Adjusted R-squared score.
    """
    r2 = r2_score(y_true, y_pred)
    n_samples = len(y_true)
    if n_samples - n_features - 1 == 0: # Avoid division by zero
        return r2 # Or np.nan, or handle as appropriate. R2 is often preferred in this case.
    adj_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adj_r2


# Example of a custom metric for finance: Hit Rate (for directional accuracy)
def directional_accuracy(y_true_change, y_pred_change):
    """
    Calculates the directional accuracy of predictions.
    Assumes y_true_change and y_pred_change represent changes (e.g., price differences or returns).
    The prediction is correct if its sign matches the sign of the true change.

    Args:
        y_true_change (np.ndarray or pd.Series): True changes (e.g., y_t - y_{t-1} or log(y_t/y_{t-1})).
        y_pred_change (np.ndarray or pd.Series): Predicted changes.

    Returns:
        float: Directional accuracy (proportion of correct direction predictions).
    """
    if len(y_true_change) != len(y_pred_change):
        raise ValueError("y_true_change and y_pred_change must have the same length.")
    if len(y_true_change) == 0:
        return 0.0 # Or np.nan

    correct_direction = np.sign(y_true_change) == np.sign(y_pred_change)
    # Handle cases where either true or predicted change is zero.
    # If true change is zero, any prediction is arguably "wrong" unless pred is also zero.
    # If pred change is zero, it's wrong unless true change is also zero.
    # A common approach: if sign is same (both +ve or both -ve), it's correct.
    # If one is zero and other is not, it's incorrect direction.
    # If both are zero, it's correct direction (no change predicted, no change happened).
    # np.sign(0) is 0. So (0 == 0) is True.

    return np.mean(correct_direction)


if __name__ == '__main__':
    # Example usage of metrics
    y_true_reg = np.array([3, -0.5, 2, 7])
    y_pred_reg = np.array([2.5, 0.0, 2, 8])
    n_features_reg = 2

    print("--- Regression Metrics Examples ---")
    print(f"MSE: {mean_squared_error(y_true_reg, y_pred_reg):.4f}")
    print(f"MAE: {mean_absolute_error(y_true_reg, y_pred_reg):.4f}")
    print(f"R2 Score: {r2_score(y_true_reg, y_pred_reg):.4f}")
    print(f"Adjusted R2 Score: {adjusted_r2_score(y_true_reg, y_pred_reg, n_features_reg):.4f}")
    print(f"MAPE: {mean_absolute_percentage_error(y_true_reg, y_pred_reg):.4f}")


    y_true_clf = np.array([0, 1, 0, 1, 0, 1])
    y_pred_clf_labels = np.array([0, 1, 1, 1, 0, 0])
    y_pred_clf_probs = np.array([0.1, 0.9, 0.6, 0.8, 0.2, 0.4]) # Probs for class 1

    print("\n--- Classification Metrics Examples ---")
    print(f"Accuracy: {accuracy_score(y_true_clf, y_pred_clf_labels):.4f}")
    print(f"F1 Score (binary): {f1_score(y_true_clf, y_pred_clf_labels):.4f}")
    print(f"Precision: {precision_score(y_true_clf, y_pred_clf_labels):.4f}")
    print(f"Recall: {recall_score(y_true_clf, y_pred_clf_labels):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_true_clf, y_pred_clf_probs):.4f}")
    print(f"Log Loss: {log_loss(y_true_clf, y_pred_clf_probs):.4f}")

    print("\n--- Custom Metrics Examples ---")
    # Directional Accuracy Example
    # Example: stock price changes
    true_price_changes = np.array([1.0, -0.5, 0.2, -0.1, 0.0, 0.3])
    pred_price_changes = np.array([0.8, -0.2, -0.1, 0.05, 0.1, 0.2])
    # Expected: Correct, Correct, Incorrect, Incorrect, Incorrect (0 pred vs 0 true is Correct), Correct
    # Signs:   [ 1, -1,  1, -1,  0,  1]
    # PredS:   [ 1, -1, -1,  1,  1,  1]
    # Match:   [ T,  T,  F,  F,  F,  T] -> 4/6 = 0.6667
    # np.sign(0) == 0. If true is 0, pred must be 0. If pred is 0, true must be 0.
    # My logic: np.sign(true) == np.sign(pred)
    # np.sign([ 1.0, -0.5,  0.2, -0.1,  0.0,  0.3]) -> [ 1, -1,  1, -1,  0,  1]
    # np.sign([ 0.8, -0.2, -0.1, 0.05,  0.1,  0.2]) -> [ 1, -1, -1,  1,  1,  1]
    # Correct:   [ T,  T,  F,  F,  F,  T] -> 4/6
    print(f"Directional Accuracy: {directional_accuracy(true_price_changes, pred_price_changes):.4f}")

    true_zeros = np.array([0.0, 0.0])
    pred_zeros = np.array([0.0, 0.1]) # One match, one not
    print(f"Directional Accuracy (with zeros): {directional_accuracy(true_zeros, pred_zeros):.4f}") # Expected 0.5


    # Sharpe Ratio (placeholder)
    try:
        sharpe_ratio(np.array([]), np.array([]))
    except NotImplementedError as e:
        print(f"Sharpe Ratio Placeholder: {e}")
