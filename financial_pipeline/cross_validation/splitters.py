# This file will contain functions for splitting data for cross-validation.
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
# TimeSeriesSplit can be used as a reference or wrapped/extended if needed.
# For BasicTimeSeriesSplit, we are reimplementing a similar logic.
from sklearn.model_selection import TimeSeriesSplit as SklearnTimeSeriesSplit
from sklearn.utils.validation import indexable, _num_samples

class BasicTimeSeriesSplit(BaseCrossValidator):
    """
    Time Series cross-validator.

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than train indices.

    This implementation is similar to sklearn's TimeSeriesSplit but allows
    for more explicit control or modification if needed in the future.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.

    max_train_size : int, default=None
        Maximum size for a single training set.

    test_size : int, default=None
        Number of samples in each test set. Defaults to n_samples // (n_splits + 1).
        If `gap` is used, `test_size` samples will be selected *after* the gap.

    gap : int, default=0
        Number of samples to exclude between the end of the training set and
        the start of the test set.
    """
    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError("n_splits must be an integer greater than or equal to 2.")
        if max_train_size is not None and (not isinstance(max_train_size, int) or max_train_size <= 0):
            raise ValueError("max_train_size must be a positive integer or None.")
        if test_size is not None and (not isinstance(test_size, int) or test_size <= 0):
            raise ValueError("test_size must be a positive integer or None.")
        if not isinstance(gap, int) or gap < 0:
            raise ValueError("gap must be a non-negative integer.")

        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Logic aims to be similar to sklearn.model_selection.TimeSeriesSplit.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than "
                f"the number of samples={n_samples}."
            )

        # Initial train_end_idx for loop logic, if not using fixed test_size from start
        train_end_idx = -1

        for i in range(self.n_splits):
            current_train_end = 0
            current_test_start = 0
            current_test_s = 0 # current_test_size for this fold

            if self.test_size is None: # Proportional sizing
                fold_sizes = np.full(self.n_splits + 1, n_samples // (self.n_splits + 1), dtype=int)
                fold_sizes[:n_samples % (self.n_splits + 1)] += 1

                if i == 0:
                    current_train_end = fold_sizes[0] - 1
                else:
                    # train_end for split i is sum of sizes of fold_0 to fold_i
                    current_train_end = np.sum(fold_sizes[:i+1]) -1

                current_test_s = fold_sizes[i+1]
                current_test_start = current_train_end + 1 + self.gap

            else: # Fixed test_size
                current_test_s = self.test_size
                # Test sets are anchored to the end of the series, train set expands.
                # test_end_inclusive for split i: (n_samples - 1) - (n_splits - 1 - i) * (test_size_for_calc + gap)
                # where test_size_for_calc is test_size.

                # Let's calculate test_start directly for clarity for fixed size
                # The first test split starts after the initial training period.
                # Size of initial training data: n_samples - (n_splits * test_size) - (n_splits * gap)
                # This is not quite right if gap is only train-test.
                # initial_train_size = n_samples - n_splits * self.test_size - (self.n_splits -1) * self.gap if self.n_splits > 1 else n_samples - self.test_size

                # Corrected logic for fixed test_size, similar to sklearn:
                # Train set expands, test set is of fixed size.
                # test_start for split i: (initial_train_size) + i * (test_size + gap)
                # initial_train_size is n_samples - n_splits * test_size - (n_splits -1) * gap (approx)
                # This is complex to get initial_train_size right.

                # Use the end-anchored logic that was traced correctly:
                # For split i (0-indexed):
                # test_end_idx_inclusive = (n_samples - 1) - (self.n_splits - 1 - i) * (self.test_size + self.gap)
                # This formula assumes gap is between test sets. Let's adjust.
                # If gap is strictly between train and test of *this* split:

                # For split i (0-indexed):
                # The end of the i-th test set (inclusive index)
                test_absolute_end_idx = (n_samples - 1) - (self.n_splits - 1 - i) * self.test_size \
                                      - (self.n_splits - 1 - i) * self.gap
                                      # Each (n_splits-1-i) block of future (test_set+gap) is accounted for

                current_test_start = test_absolute_end_idx - self.test_size + 1
                current_train_end = current_test_start - 1 - self.gap


            if current_test_start >= n_samples: # No more data for a new test set
                break

            # Adjust current_test_size if it overruns n_samples
            if current_test_start + current_test_s > n_samples:
                current_test_s = n_samples - current_test_start

            if current_test_s <= 0: # No samples left for test set
                break

            actual_train_start = 0
            if self.max_train_size is not None and (current_train_end + 1) > self.max_train_size:
                actual_train_start = (current_train_end + 1) - self.max_train_size

            # Ensure train end is not negative
            current_train_end = max(-1, current_train_end)

            train_indices_arr = indices[actual_train_start : current_train_end + 1]
            test_indices_arr = indices[current_test_start : current_test_start + current_test_s]

            if len(test_indices_arr) == 0 : # Should be caught by current_test_s <=0, but safeguard
                break

            # Allow empty train for first split if n_samples is tiny, but not for subsequent ones if test is formable
            if len(train_indices_arr) == 0 and i > 0 and (actual_train_start <= current_train_end) :
                 # This implies an issue if train is empty but should not be.
                 # However, max_train_size could lead to valid empty train if current_train_end < actual_train_start.
                 # The condition (actual_train_start <= current_train_end) means it wasn't due to max_train_size making start > end.
                 # So, if it's empty and wasn't supposed to be, stop.
                 break


            yield train_indices_arr, test_indices_arr

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class WalkForwardSplit(BaseCrossValidator):
    """
    Walk-Forward Cross-Validator for recurrent fine-tuning scenarios.

    Yields (train_indices, val_indices, test_indices) for each step.
    The test set of split `i` becomes part of the training set for split `i+1`.
    Validation data is taken from the end of the current training data.

    Parameters
    ----------
    initial_train_val_size : int
        Minimum total size of the first (train + validation) block.
    n_test_periods : int
        Number of samples/periods to include in each test set.
    n_recurrent_steps : int
        Number of walk-forward steps (splits) to generate.
    train_val_ratio : float, default=0.8
        Ratio of the current (train + validation) block to use for actual training.
        The rest (1 - train_val_ratio) is used for validation.
        E.g., 0.8 means 80% for training, 20% for validation.
    min_val_size : int, default=1
        Minimum number of samples in the validation set. If the ratio calculation
        results in fewer samples, this minimum is used (if possible).
    """
    def __init__(self, initial_train_val_size: int, n_test_periods: int, n_recurrent_steps: int,
                 train_val_ratio: float = 0.8, min_val_size: int = 1):
        if not isinstance(initial_train_val_size, int) or initial_train_val_size <= 0:
            raise ValueError("initial_train_val_size must be a positive integer.")
        if not isinstance(n_test_periods, int) or n_test_periods <= 0:
            raise ValueError("n_test_periods must be a positive integer.")
        if not isinstance(n_recurrent_steps, int) or n_recurrent_steps <= 0:
            raise ValueError("n_recurrent_steps must be a positive integer.")
        if not (isinstance(train_val_ratio, float) and 0 < train_val_ratio < 1):
            raise ValueError("train_val_ratio must be a float between 0 and 1 (exclusive).")
        if not isinstance(min_val_size, int) or min_val_size <=0:
            raise ValueError("min_val_size must be a positive integer.")
        if initial_train_val_size * (1-train_val_ratio) < min_val_size and initial_train_val_size >= min_val_size :
             # This print can be noisy during Optuna, consider logging instead or removing for production.
             # print(f"Warning: Calculated initial validation size ({initial_train_val_size * (1-train_val_ratio):.2f}) "
             #       f"is less than min_val_size ({min_val_size}). Adjusting validation size if possible.")
             pass


        self.initial_train_val_size = initial_train_val_size
        self.n_test_periods = n_test_periods
        self.n_recurrent_steps = n_recurrent_steps
        self.train_val_ratio = train_val_ratio
        self.min_val_size = min_val_size

    def split(self, X, y=None, groups=None):
        """Generate (train, validation, test) indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Always ignored.
        groups : array-like of shape (n_samples,)
            Always ignored.

        Yields
        ------
        train_indices : ndarray
        val_indices : ndarray
        test_indices : ndarray
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        min_total_samples_needed_first_split = self.initial_train_val_size + self.n_test_periods
        if n_samples < min_total_samples_needed_first_split:
            raise ValueError(
                f"Not enough samples ({n_samples}) for the first split. "
                f"Need at least initial_train_val_size ({self.initial_train_val_size}) "
                f"+ n_test_periods ({self.n_test_periods}) = {min_total_samples_needed_first_split}."
            )

        # Calculate max possible steps based on the total samples and expanding window
        # After initial block, each step consumes n_test_periods from remaining data.
        # Remaining data after initial train/val block = n_samples - initial_train_val_size
        # Number of test blocks possible from remaining data = remaining_data // n_test_periods
        max_possible_steps = (n_samples - self.initial_train_val_size) // self.n_test_periods
        # Since the first split is part of n_recurrent_steps, if initial_train_val_size itself is one step.
        # Let's adjust: total samples for all test sets = n_recurrent_steps * n_test_periods
        # Total samples needed = initial_train_val_size + (n_recurrent_steps * n_test_periods)
        # No, this is not right. The train_val block expands.
        # Correct logic for max_possible_steps:
        # First step needs initial_train_val_size + n_test_periods
        # Each subsequent step adds n_test_periods to the total span.
        # Total span = initial_train_val_size + n_recurrent_steps * n_test_periods
        if self.initial_train_val_size + self.n_recurrent_steps * self.n_test_periods > n_samples:
             actual_max_steps = (n_samples - self.initial_train_val_size) // self.n_test_periods
             if actual_max_steps < self.n_recurrent_steps : # Check if it's truly impossible
                raise ValueError(
                    f"n_recurrent_steps ({self.n_recurrent_steps}) is too large for n_samples ({n_samples}), "
                    f"initial_train_val_size ({self.initial_train_val_size}), and n_test_periods ({self.n_test_periods}). "
                    f"Max possible steps for full test sets: {actual_max_steps}."
                )

        current_train_val_end_idx = -1 # Stores the end index of the combined train+val block for current step

        for step in range(self.n_recurrent_steps):
            current_train_val_start_idx = 0 # Expanding window always starts from 0

            if step == 0:
                current_train_val_end_idx = self.initial_train_val_size - 1
            else:
                # The new train_val block ends where the previous test block ended.
                # previous_test_end_idx was current_train_val_end_idx (from prev step) + self.n_test_periods
                current_train_val_end_idx += self.n_test_periods

            current_train_val_size = current_train_val_end_idx + 1

            # Define test set for the current step
            test_start_idx = current_train_val_end_idx + 1
            test_end_idx = test_start_idx + self.n_test_periods - 1

            if test_end_idx >= n_samples:
                # Not enough samples to form the test set for this step. Stop.
                break

            # Split current_train_val_block into actual train and validation
            val_size_float = current_train_val_size * (1.0 - self.train_val_ratio)
            val_size = int(val_size_float) # Standard truncation
            val_size = max(self.min_val_size, val_size) # Ensure min_val_size

            # Ensure val_size does not make train_size zero or negative
            if val_size >= current_train_val_size: # If val_size is all of current block
                val_size = current_train_val_size - 1 # Leave at least one sample for training
                if val_size < 1 and current_train_val_size > 0 : # If current_train_val_size was 1, val_size becomes 0
                     val_size = 0 # No validation possible, all train
                elif val_size < 1 and current_train_val_size ==0: # Should not happen if initial checks are good
                     raise ValueError("Cannot form train/val from zero samples.")


            train_size = current_train_val_size - val_size
            if train_size <= 0: # If train_size became non-positive
                # This can happen if min_val_size pushes val_size to be too large relative to current_train_val_size
                # If train_size is 0, try to make val_size smaller to allow for some training data
                if current_train_val_size > self.min_val_size : # If we can shrink val and still meet min_val_size
                    val_size = current_train_val_size -1 # Maximize training, 1 sample for val (if min_val_size allows)
                    val_size = max(self.min_val_size, val_size) # Re-check min_val_size
                    train_size = current_train_val_size - val_size
                    if train_size <=0: # Still no training data, means current_train_val_size is too small
                         raise ValueError(
                            f"Step {step}: Training size ({train_size}) is not positive after attempting to adjust validation. "
                            f"current_train_val_size={current_train_val_size}, val_size={val_size}. "
                            "Consider increasing initial_train_val_size or reducing min_val_size."
                        )
                else: # current_train_val_size is already <= min_val_size, cannot shrink val further
                    raise ValueError(
                        f"Step {step}: Training size ({train_size}) is not positive. "
                        f"current_train_val_size={current_train_val_size}, val_size={val_size}. "
                        "Consider increasing initial_train_val_size or reducing min_val_size."
                    )


            train_end_idx = current_train_val_start_idx + train_size - 1

            val_start_idx = train_end_idx + 1
            # val_end_idx should align with current_train_val_end_idx
            val_end_idx = current_train_val_end_idx

            # Adjust val_start if val_size was clamped by current_train_val_end_idx logic
            # This ensures val_end_idx is correctly current_train_val_end_idx
            # And train_end_idx is before that by `val_size`.
            actual_val_size = val_end_idx - val_start_idx + 1
            if actual_val_size != val_size : # Should not happen if logic is correct
                # This means train_end_idx + 1 + val_size -1 != current_train_val_end_idx
                # This can occur if initial val_size calculation was too large and got clamped.
                # Re-calculate train_end_idx based on the final val_end_idx and desired val_size
                val_start_idx = val_end_idx - val_size + 1
                train_end_idx = val_start_idx -1
                if train_end_idx < current_train_val_start_idx -1: # -1 because train_end can be -1 for empty train
                     # This implies val_size is too large for the current_train_val_block
                     # Should have been caught by train_size <= 0 check.
                      raise ValueError(f"Step {step}: Logic error led to train_end_idx {train_end_idx} < start {current_train_val_start_idx-1}")


            train_indices_arr = indices[current_train_val_start_idx : train_end_idx + 1]
            val_indices_arr = indices[val_start_idx : val_end_idx + 1]
            test_indices_arr = indices[test_start_idx : test_end_idx + 1]

            # Ensure no empty arrays are yielded if they are not supposed to be
            if not (len(train_indices_arr) > 0 and len(val_indices_arr) > 0 and len(test_indices_arr) > 0):
                 # Check if any are unexpectedly empty
                if len(test_indices_arr) == 0: break # Cannot proceed if test is empty
                if len(val_indices_arr) == 0 and self.min_val_size > 0 : break # Val became empty against min_val_size
                # Train can be empty if initial_train_val_size is very small, but val/test should hold.

            yield train_indices_arr, val_indices_arr, test_indices_arr

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return self.n_recurrent_steps


class TimeBasedGroupShuffleSplit(BaseCrossValidator):
    """
    Time-Based Group Shuffle Split (Conceptual Placeholder).

    This cross-validator aims to split data by groups (e.g., assets, series)
    while respecting temporal order, either within groups or between groups
    for train/test splits.

    The core idea is to ensure that training data for a group or set of groups
    does not inadvertently include future information from test groups,
    especially if groups have overlapping time ranges or if global time order
    is paramount.

    Parameters
    ----------
    n_splits : int
        Number of splits.
    test_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the test split. If int, represents the
        absolute number of groups in the test split.
    train_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the proportion
        of groups to include in the train split. If int, represents the
        absolute number of groups in the train split. If None, it is set to
        1.0 - test_size.
    group_column : str
        The name of the column in X (if X is a DataFrame) that contains
        the group identifiers.
    date_column : str
        The name of the column in X (if X is a DataFrame) that contains
        date/time information, used for ensuring temporal consistency.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the groups before splitting.
        Pass an int for reproducible output across multiple function calls.

    Challenges and Considerations:
    -----------------------------
    - Defining "chronologically after": Does this mean all data points in
      test groups must be after all data points in train groups? Or just that
      the *start* of test group data is after the *end* of train group data?
      Or on a per-group basis if groups are persistent across splits?
    - Handling groups with different start/end times: A group might appear late
      in the dataset. How does this affect its assignment to train/test?
    - Lookahead bias: The primary goal is to prevent lookahead bias. If groups
      are, for example, different financial assets, training on asset A up to
      time T and testing on asset B from time T-k to T+m might be problematic
      if asset B's behavior at T-k is influenced by information also available
      to asset A at T.
    - API: How to best integrate with pandas DataFrames and ensure easy use.
    """
    def __init__(self, n_splits: int = 5, test_size=0.2, train_size=None,
                 group_column: str = 'group_id', date_column: str = 'date', random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.group_column = group_column
        self.date_column = date_column
        self.random_state = random_state # For shuffling groups

    def split(self, X, y=None, groups=None):
        """Generate (train, test) indices based on time-aware group splitting.

        Parameters
        ----------
        X : pd.DataFrame typically, with group_column and date_column.
        y : array-like, optional
        groups : array-like, optional (might be redundant if group_column used)

        Yields
        ------
        train_indices : ndarray
        test_indices : ndarray
        """
        # X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        # if self.group_column not in X_df.columns:
        #     raise ValueError(f"Group column '{self.group_column}' not found in X.")
        # if self.date_column not in X_df.columns:
        #     raise ValueError(f"Date column '{self.date_column}' not found in X.")

        # unique_groups = X_df[self.group_column].unique()
        # # Potentially sort groups by their first appearance date?
        # # Shuffle groups using self.random_state

        # # Loop n_splits times:
        # #   - Select train_groups and test_groups
        # #   - Get all indices for train_groups, all for test_groups
        # #   - Apply temporal constraints:
        # #     Example: max_date_in_train_groups < min_date_in_test_groups (if strict separation)
        # #     Or, for each group in test_groups, its data must be after its own data in train_groups
        # #     if groups can be in both train and test over time (unlikely for pure group split).

        raise NotImplementedError(
            "TimeBasedGroupShuffleSplit is a conceptual placeholder. "
            "Implementing it correctly requires careful consideration of: \n"
            "1. How groups are defined and if they persist over time. \n"
            "2. The exact temporal relationship desired between train/test sets (e.g., no overlap, test strictly after train globally, or per group). \n"
            "3. Handling of groups with different start/end dates. \n"
            "This functionality is complex and application-dependent."
        )

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


if __name__ == '__main__':
    # --- BasicTimeSeriesSplit Example ---
    print("--- BasicTimeSeriesSplit Example ---")
    X_basic_main = np.arange(20).reshape(10, 2) # 10 samples
    y_basic_main = np.arange(10)

    print("Input X shape:", X_basic_main.shape)

    # Example 1: Default settings
    print("\nExample 1: Default settings (n_splits=5)")
    bts_1 = BasicTimeSeriesSplit(n_splits=5)
    for i_main, (train_idx_main, test_idx_main) in enumerate(bts_1.split(X_basic_main)):
        print(f"Split {i_main+1}: Train={train_idx_main}, Test={test_idx_main}")
    print(f"Total splits: {bts_1.get_n_splits()}")

    # Example Test for BasicTimeSeriesSplit: n_samples=10, n_splits=3, test_size=2, gap=0
    print("\nExample BasicTimeSeriesSplit: n_samples=10, n_splits=3, test_size=2, gap=0")
    X_10 = np.arange(20).reshape(10,2)
    bts_10_3_2_0 = BasicTimeSeriesSplit(n_splits=3, test_size=2, gap=0)
    for i_main, (train_idx_main, test_idx_main) in enumerate(bts_10_3_2_0.split(X_10)):
        print(f"Split {i_main+1}: Train={train_idx_main}, Test={test_idx_main}")
    # Expected from trace:
    # Split 1: Train=[0 1 2 3], Test=[4 5]
    # Split 2: Train=[0 1 2 3 4 5], Test=[6 7]
    # Split 3: Train=[0 1 2 3 4 5 6 7], Test=[8 9]

    # --- WalkForwardSplit Example ---
    print("\n--- WalkForwardSplit Example ---")
    X_walk_main = np.arange(100).reshape(50, 2) # 50 samples
    y_walk_main = np.arange(50)
    print("Input X shape for WalkForwardSplit:", X_walk_main.shape)

    print("\nExample W1: initial_train_val_size=20, n_test_periods=5, n_recurrent_steps=3, train_val_ratio=0.75")
    wfs_1 = WalkForwardSplit(initial_train_val_size=20, n_test_periods=5, n_recurrent_steps=3, train_val_ratio=0.75, min_val_size=3)
    for i_main, (train_idx_main, val_idx_main, test_idx_main) in enumerate(wfs_1.split(X_walk_main)):
        print(f"Split {i_main+1}: Train len={len(train_idx_main)}, Val len={len(val_idx_main)}, Test len={len(test_idx_main)}")
        # print(f"  Train: {train_idx_main}\n  Val: {val_idx_main}\n  Test: {test_idx_main}")


    # --- TimeBasedGroupShuffleSplit Example ---
    print("\n--- TimeBasedGroupShuffleSplit Example ---")
    tbgss = TimeBasedGroupShuffleSplit()
    try:
        for _ in tbgss.split(X_walk_main):
            pass
    except NotImplementedError as e:
        print(f"Correctly caught: {e}")
    except ValueError as e:
        print(f"Caught ValueError (expected if X is not DataFrame with specific columns): {e}")
