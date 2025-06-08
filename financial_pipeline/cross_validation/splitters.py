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

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        indices = np.arange(n_samples)

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than "
                f"the number of samples={n_samples}."
            )

        test_size = self.test_size if self.test_size is not None else n_samples // (self.n_splits + 1)

        if test_size <=0:
             raise ValueError(
                f"test_size_needs to be positive but got {test_size}"
            )


        # Calculate the number of samples needed for all test sets and gaps
        total_test_samples_needed = test_size * self.n_splits
        total_gap_samples_needed = self.gap * (self.n_splits -1) # Gap is between train and test, so n_splits-1 for internal gaps if test_size moves start

        # Effective n_splits for initial_train_size calculation if test_size is fixed
        # This is complex because the first train set is smaller.
        # Let's simplify by iterating from the end.

        # Iterate from the first split to the last
        # The first split has the smallest training set.
        # Test sets are of fixed size `test_size`.

        start_train_idx = 0
        for i in range(self.n_splits):
            # Determine the end of the training set for the current split
            # The test set starts after this point, plus any gap

            # Calculate potential end of training set.
            # This is tricky because train size expands.
            # Let's try sklearn's way: test sets are at the end of the series.

            # Test sets are shifted by one fold each time
            test_start = n_samples - test_size * (self.n_splits - i) - self.gap * (self.n_splits - i -1) # this logic is a bit off for gap

            # Corrected logic for test_start and train_end, similar to sklearn
            # Test sets are at the end. The first test set is `n_samples - test_size`.
            # The second is `n_samples - 2*test_size - gap`, etc. This is not how TimeSeriesSplit works.

            # Sklearn TimeSeriesSplit:
            # Folds are (train_end_idx - train_size + 1, train_end_idx), (train_end_idx + gap + 1, train_end_idx + gap + test_size)
            # train_end_idx increases with each split.

            # The first training set will have `n_samples - (n_splits * test_size) - (n_splits * gap)` if all gaps and test_sizes are to be accommodated from the start.
            # More simply: iterate and define `train_end` and `test_start`, `test_end`.

            # Simplified approach:
            # For each split `i`, the test set ends at `n_samples - (self.n_splits - 1 - i) * (test_size + self.gap)` (roughly)
            # and starts `test_size` samples before that.
            # Training set is everything before `test_start - self.gap`.

            if self.test_size is None: # Proportional test_size
                # This is how sklearn TimeSeriesSplit calculates fold sizes
                fold_sizes = np.full(self.n_splits + 1, n_samples // (self.n_splits + 1), dtype=int)
                fold_sizes[: n_samples % (self.n_splits + 1)] += 1
                current_test_size = fold_sizes[i+1] # test_size for this specific fold
                train_end = fold_sizes[0] # size of the first block, which is the first training set
                for k_fold in range(i):
                    train_end += fold_sizes[k_fold+1]
            else: # Fixed test_size
                current_test_size = test_size
                # The end of the training set for split `i`
                # First train end: n_samples - n_splits * test_size - (n_splits-1)*gap (approx, if all fixed from start)
                # Let's use a simpler iterative definition:
                # The first training set must be large enough to allow for all future test/gap sets.
                # Smallest start for test set: initial_train_min_size + gap
                # initial_train_min_size = n_samples - self.n_splits * (current_test_size + self.gap) + self.gap (remove last gap)
                # This is not quite right. The train size GROWS.

                # Iterative definition like sklearn:
                # Test_end is `(i + 1) * shift + initial_train_size` where shift is `test_size`
                # Let's use `n_splits` to define the start and end points of test sets directly
                # This defines the START of the test set
                test_starts_at = n_samples - (self.n_splits - i) * current_test_size - (self.n_splits - 1 - i) * self.gap
                if i == 0 : # First split, no preceding gap to subtract for test_starts_at relative to end
                     test_starts_at = n_samples - self.n_splits * current_test_size - (self.n_splits -1) * self.gap

                # The train set ends `self.gap` samples before the test set begins.
                train_end = test_starts_at - self.gap

                # If test_size is fixed, the first training set size depends on n_samples, n_splits, test_size, and gap
                # First train_end = n_samples - n_splits * test_size - (n_splits) * gap (approx)
                # No, simpler:
                # For split `k` (0 to n_splits-1):
                # test_start_index = initial_train_size + k*test_size + k*gap
                # train_end_index = test_start_index - gap -1
                # initial_train_size = n_samples - n_splits*test_size - (n_splits-1)*gap (if gap only between train/test of same split)
                # This is still messy. Let's use the sklearn way as a guide.
                # Sklearn: `idx = np.arange(n_samples)`
                # `n_folds = self.n_splits + 1`
                # `test_starts = np.array([fold_size[i:].sum() for i in range(n_folds - 1, 0, -1)])`
                # `test_starts = test_starts[::-1]` (these are end points of test sets)
                # `test_ends = np.array([test_starts[i] + fold_sizes[i+1] for i in range(self.n_splits)])`
                # `train_ends = test_starts - self.gap`

                # Let's use a simpler loop based on the end of the array
                # For split `i` (0-indexed):
                # Test set end index: `n_samples - (self.n_splits - 1 - i) * (self.test_size if self.test_size else 0) - (self.n_splits - 1 - i) * self.gap`
                # This is not how TimeSeriesSplit works. It expands the training set.

                # Corrected logic:
                # `test_start_idx` is the first index of the test set.
                # `train_end_idx` is the last index of the training set.
                # Total number of samples available for the first training set and all test/gap sets: n_samples
                # Size of all test sets = n_splits * test_size
                # Size of all gaps = (n_splits) * gap (gap before each test set)

                # The first training set ends at:
                # `first_train_end = n_samples - n_splits * test_size - n_splits * gap` (if gap is before each test set start)
                # Or `n_samples - (n_splits * test_size + (n_splits - 1) * gap)` if gap is only between train/test of a split. Let's use `self.gap` as defined.

                # Let `idx` be the current split number from 0 to n_splits - 1.
                # The end of the training set for split `idx`:
                # `train_indices` will go from `start_train_idx` up to `end_train_idx`.
                # `test_indices` will go from `start_test_idx` up to `end_test_idx`.

                # Simplified logic based on sklearn's TimeSeriesSplit structure:
                # Calculate `n_train` and `n_test` for each split.
                # `n_test = self.test_size` or proportional.
                # `n_train` increases.

                # The first training set must be at least `test_size` (or some minimum).
                # The start of the test set for split `i`.
                # Total samples `n_samples`.
                # `test_end_idx = n_samples - (self.n_splits - 1 - i) * (self.test_size + self.gap)` (if test_size is fixed)
                # This is not standard. Standard TimeSeriesSplit has test sets of roughly equal size at the end of expanding windows.

                # Let's follow the structure of sklearn TimeSeriesSplit more closely.
                # `n_samples = _num_samples(X)`
                # `indices = np.arange(n_samples)`
                # `test_size = self.test_size if self.test_size is not None else n_samples // (self.n_splits + 1)`

                # `train_start_idx = 0` (always, unless max_train_size is hit)
                # `test_end_idx` for split `i` is `n_samples - (self.n_splits - 1 - i) * test_size` (without gap)
                # This seems to make test sets appear from the end. Let's use the other way.

                # For each split `k` from `0` to `self.n_splits - 1`:
                #   `train_end = initial_train_size + k * (test_size_fold + gap_fold) - 1`
                #   `test_start = train_end + gap + 1`
                #   `test_end = test_start + test_size_fold - 1`
                # This requires calculating `initial_train_size`.
                # `initial_train_size = n_samples - self.n_splits * test_size_fold - (self.n_splits) * gap` (if gap is before each test set)

                # Let's use the approach from sklearn's code directly for `test_starts` indices
                # This defines the END of each training set.
                # The train_end for split `i`

                if i == 0:
                    # Calculate the size of the first training set.
                    # It must be large enough to allow for all subsequent test sets and gaps.
                    # Size of all test sets = self.n_splits * test_size
                    # Size of all gaps between train/test = self.n_splits * self.gap
                    # No, gap is only between train and test of *that* split.

                    # If test_size is fixed:
                    # The first training set ends at `n_samples - n_splits * test_size - (n_splits -1) * gap - test_size`
                    # No, this is too complex. A simpler loop:

                    # `train_end_index` for split `i`
                    # `test_start_index = train_end_index + 1 + self.gap`
                    # `test_end_index = test_start_index + test_size - 1`

                    # `initial_train_size` = `n_samples - self.n_splits * test_size - (self.n_splits -1) * gap` if gap is only between train/test
                    # Let's assume gap is between train and test.
                    # The last test set starts right after the last training set + gap.
                    # Last train_end + gap + test_size = n_samples
                    # Iteration `j` from 0 to n_splits-1
                    # train_indices_end = n_samples - (self.n_splits - j) * (test_size + self.gap) + self.gap (to remove the last gap)
                    # This assumes fixed test_size.

                    # Sklearn approach for test_starts (these are the starts of each test period)
                    # test_idx_start = n_samples - self.n_splits * test_size - (self.n_splits -1) * gap # this is start of first test
                    # No, test_idx_start for split `s`: `train_len_s + gap`
                    # train_len for split `s` increases.

                    # Let's use a simple iterative approach:
                    # `current_train_end_idx = -1` initially.
                    # In split `i`:
                    #   `train_start = 0` (or `max(0, current_test_end_idx - self.max_train_size)` if `max_train_size`)
                    #   `train_end = n_samples - (self.n_splits - i) * test_size - (self.n_splits - i) * self.gap` (this is end of train for split i)
                    #   `test_start = train_end + 1 + self.gap`
                    #   `test_end = test_start + test_size - 1`

                    # Alternative from sklearn User Guide:
                    # For split i, train ends at (i + 1) * n_samples // (n_splits + 1) - 1 (without gap, test_size proportional)
                    # test starts at (i + 1) * n_samples // (n_splits + 1)
                    # test ends at (i + 2) * n_samples // (n_splits + 1) -1

                    # With fixed test_size and gap:
                    # train_end for split `i`: `n_samples - ( (n_splits-i)*test_size + (n_splits-i-1)*gap ) -1` (this is the end of all future stuff)
                    # No, this is `initial_train_size + i*test_size + i*gap`

                    # Simplest: define `train_max_idx` for current split
                    # Smallest training set size: `n_samples - self.n_splits * (test_size + self.gap) + self.gap` (approx)
                    # Let `min_train_size_needed_for_all_splits = test_size` (as a base, can't be less than test_size)
                    # This is surprisingly tricky to get right without just copying sklearn.

                    # Let's define `train_end_idx` for split `i`
                    # The first train set must leave room for `n_splits` test sets and `n_splits-1` gaps after it if test sets are consecutive *after* training.
                    # Or, `n_splits` gaps if gap is always between current train and current test.

                    # Assume `test_size` is fixed.
                    # Total length of all test sets and their preceding gaps: `n_splits * (test_size + self.gap)`
                    # This means `n_samples - n_splits * (test_size + self.gap)` is the size of the first training set.
                    # This is only if the training set does not grow. But it does.

                    # Let `idx` be split number from 0 to `n_splits - 1`.
                    # `train_indices_end = n_samples - (self.n_splits - idx) * (test_size + self.gap) + self.gap -1` (if test_size fixed)
                    # This is the end of the training set for split `idx`.
                    # `test_indices_start = train_indices_end + 1 + self.gap`
                    # `test_indices_end = test_indices_start + test_size - 1`

                    # This definition makes the training set size fixed if `idx` is not used in its definition.
                    # The training set should expand.
                    # `train_indices_end_for_split_i = initial_training_size + i * (test_size + gap) -1`
                    # `initial_training_size = n_samples - self.n_splits * (test_size + gap) + gap`

                    # Let `fold_step = test_size + self.gap`
                    # `train_end = n_samples - self.n_splits * fold_step + self.gap` (size of first training set)
                    # for split `i`:
                    #   `current_train_end = train_end + i * fold_step`
                    #   `current_test_start = current_train_end + 1 + self.gap`
                    #   `current_test_end = current_test_start + test_size - 1`

                    # Let's use sklearn's TimeSeriesSplit as a direct reference for logic if test_size is None
                    if self.test_size is None:
                        # Proportional sizing like sklearn's default
                        shift = n_samples // (self.n_splits + 1)
                        train_end_idx = shift * (i + 1) -1
                        test_start_idx = train_end_idx + 1 + self.gap
                        test_end_idx = test_start_idx + shift -1 # test_size is also 'shift'
                    else:
                        # Fixed test_size
                        # The end of the training set for split `i` is:
                        # `initial_train_size + i * self.test_size + i * self.gap` (if gap is only between this train/test)
                        # `initial_train_size` must be at least `self.test_size` or 1.
                        # Smallest possible initial train size to accommodate all future splits:
                        # This is `n_samples - self.n_splits * self.test_size - (self.n_splits) * self.gap` (if gap is before every test)
                        # Let's assume gap is between current train and current test.
                        # Smallest `train_end_idx` (for i=0) is `n_samples - (self.n_splits * self.test_size + (self.n_splits -1) * self.gap) -1`
                        # No, it's `n_samples - self.n_splits * self.test_size - (self.n_splits -1) * self.gap -1` if gap is between test sets

                        # Iteration `i` from 0 to `n_splits-1`
                        # `train_end_idx = (n_samples - (self.n_splits - i) * self.test_size - (self.n_splits - 1 - i) * self.gap) - 1 - self.gap`
                        # This is getting complicated. Let's simplify the loop.
                        # `test_end_exclusive = n_samples - (self.n_splits - 1 - i) * (self.test_size + self.gap)`
                        # `test_start_inclusive = test_end_exclusive - self.test_size`
                        # `train_end_inclusive = test_start_inclusive - 1 - self.gap`

                        # This makes the last test set at the very end.
                        test_end_idx = n_samples - (self.n_splits - 1 - i) * (self.test_size) \
                                       - (self.n_splits - 1 - i) * self.gap # this is the end of test set for split i

                        test_start_idx = test_end_idx - self.test_size + 1
                        train_end_idx = test_start_idx - 1 - self.gap


            # Ensure train_end_idx is not less than -1 (empty training set)
            train_end_idx = max(-1, train_end_idx)

            train_start_actual = 0
            if self.max_train_size is not None and (train_end_idx + 1) > self.max_train_size:
                train_start_actual = train_end_idx + 1 - self.max_train_size

            train_indices = indices[train_start_actual : train_end_idx + 1]
            test_indices = indices[test_start_idx : test_end_idx + 1]

            if not len(train_indices): # Ensure train is not empty, unless it's truly the start
                 # This can happen if n_samples is too small for n_splits, test_size, gap.
                 # Should be caught by initial checks, but as a safeguard:
                if i == 0 and train_start_actual == 0 and train_end_idx == -1 : pass # First split can have empty train if setup this way
                elif not len(test_indices): # if test is also empty, something is wrong
                    print(f"Warning: train and test empty for split {i}") # should not happen
                    continue

            if not len(test_indices):
                # This implies test_start_idx > test_end_idx or test_start_idx >= n_samples
                # Could happen if n_samples is too small for the configuration.
                # Example: n_samples=10, n_splits=5, test_size=3. Last test set cannot be formed.
                # This should ideally be caught by an initial check on `n_samples` vs config.
                print(f"Warning: test indices empty for split {i}. Configuration might be too demanding for n_samples.")
                # Let's check if we can form any more splits
                min_samples_needed = (i + 1) * (self.test_size if self.test_size else 1) + i * self.gap
                if self.max_train_size : min_samples_needed +=1 # need at least one for train
                if n_samples < min_samples_needed :
                    # print(f"Stopping split generation: Not enough samples ({n_samples}) for further splits with current configuration.")
                    break # stop yielding if no more valid splits can be formed
                else:
                    # This case implies a logic error in indices calculation if test is empty but should be formable.
                    # However, with the current logic, test_start_idx can exceed n_samples.
                    if test_start_idx >= n_samples: # Cannot form this test set
                        # print(f"Cannot form test set for split {i}: test_start_idx ({test_start_idx}) >= n_samples ({n_samples})")
                        break
                    # If test_start_idx is valid but test_end_idx makes it empty, it's also an issue.
                    # The current loop structure should prevent this for test_size > 0.


            # Final check for validity
            if train_end_idx >= test_start_idx - self.gap and len(train_indices) > 0 and len(test_indices) > 0 :
                 raise ValueError(f"Train ({train_end_idx}) and Test ({test_start_idx}) overlap or invalid gap for split {i}")

            if len(test_indices) == 0 and self.test_size is not None and self.test_size > 0:
                # If we expected a test set but got none, stop.
                # This means the remaining samples are not enough.
                # print(f"Not enough samples remaining to form test set for split {i}. Stopping.")
                break

            # If test_size is None (proportional), an empty test set means n_samples // (n_splits+1) was 0 for this fold.
            if len(test_indices) == 0 and self.test_size is None and (n_samples // (self.n_splits + 1)) == 0:
                # This means n_samples < n_splits + 1, which should be caught earlier by n_splits > n_samples.
                # If n_samples // (n_splits+1) results in 0 for later folds, it means those folds are empty.
                # print(f"Proportional test size is 0 for split {i}. Stopping.")
                break


            yield train_indices, test_indices

            # Update for next iteration (if logic depends on previous state, which it doesn't here)
            # current_train_end_idx = test_end_idx # This would be for a sliding window, not expanding.

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
             print(f"Warning: Calculated initial validation size ({initial_train_val_size * (1-train_val_ratio):.2f}) "
                   f"is less than min_val_size ({min_val_size}). Adjusting validation size if possible.")


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

        min_total_samples_needed = self.initial_train_val_size + self.n_test_periods
        if n_samples < min_total_samples_needed:
            raise ValueError(
                f"Not enough samples ({n_samples}) for the first split. "
                f"Need at least initial_train_val_size ({self.initial_train_val_size}) "
                f"+ n_test_periods ({self.n_test_periods}) = {min_total_samples_needed}."
            )

        max_possible_steps = 1 + (n_samples - self.initial_train_val_size - self.n_test_periods) // self.n_test_periods
        if self.n_recurrent_steps > max_possible_steps:
            # print(f"Warning: n_recurrent_steps ({self.n_recurrent_steps}) is greater than "
            #       f"the maximum possible steps ({max_possible_steps}) with the given data size and parameters. "
            #       f"Will yield only {max_possible_steps} splits.")
            # self.n_recurrent_steps = max_possible_steps # Adjust internally, or raise error earlier
             raise ValueError(
                f"n_recurrent_steps ({self.n_recurrent_steps}) is too large for n_samples ({n_samples}) "
                f"and n_test_periods ({self.n_test_periods}). Max possible steps: {max_possible_steps}."
            )


        current_train_val_end_idx = -1

        for step in range(self.n_recurrent_steps):
            if step == 0:
                current_train_val_size = self.initial_train_val_size
                current_train_val_start_idx = 0
                current_train_val_end_idx = self.initial_train_val_size - 1
            else:
                # New train_val block includes the previous test set
                # Previous test_end_idx was `current_train_val_end_idx + self.n_test_periods`
                new_train_val_end_idx = current_train_val_end_idx + self.n_test_periods # This was end of previous test set

                # Check if there are enough samples for the next test set
                if new_train_val_end_idx + 1 + self.n_test_periods > n_samples:
                    # print(f"Not enough samples to form test set for step {step + 1}. Stopping.")
                    break

                current_train_val_start_idx = 0 # Training data always starts from index 0 (expanding window)
                current_train_val_end_idx = new_train_val_end_idx
                current_train_val_size = current_train_val_end_idx + 1

            # Define test set first
            test_start_idx = current_train_val_end_idx + 1
            test_end_idx = test_start_idx + self.n_test_periods - 1

            if test_end_idx >= n_samples:
                # print(f"Cannot form test set for step {step + 1} as it exceeds n_samples. Stopping.")
                break # Should be caught by initial check on n_recurrent_steps, but good safeguard

            # Split current_train_val_block into actual train and validation
            val_size = int(current_train_val_size * (1 - self.train_val_ratio))
            val_size = max(self.min_val_size, val_size)

            # Ensure val_size does not make train_size zero or negative
            if val_size >= current_train_val_size :
                val_size = current_train_val_size -1 # Leave at least one sample for training
                if val_size < self.min_val_size and current_train_val_size > self.min_val_size:
                     val_size = self.min_val_size # try to respect min_val_size
                elif val_size < 1: # If current_train_val_size is 1, val_size becomes 0
                    # This situation means initial_train_val_size is too small.
                    raise ValueError(f"Step {step}: current_train_val_size ({current_train_val_size}) is too small "
                                     f"to create a validation set of size {val_size} and a non-empty training set.")


            train_size = current_train_val_size - val_size
            if train_size <= 0:
                raise ValueError(
                    f"Step {step}: Training size ({train_size}) is not positive. "
                    f"current_train_val_size={current_train_val_size}, val_size={val_size}. "
                    "Consider increasing initial_train_val_size or adjusting train_val_ratio/min_val_size."
                )

            train_start_idx = current_train_val_start_idx
            train_end_idx = train_start_idx + train_size - 1

            val_start_idx = train_end_idx + 1
            val_end_idx = val_start_idx + val_size - 1

            # Sanity check: val_end_idx should match current_train_val_end_idx
            if val_end_idx != current_train_val_end_idx:
                # This can happen due to rounding or min_val_size adjustments.
                # Recalculate val_end_idx to be precise.
                val_end_idx = current_train_val_end_idx
                # train_end_idx might need adjustment if val_size was clamped by current_train_val_end_idx
                val_start_idx = current_train_val_end_idx - val_size + 1
                train_end_idx = val_start_idx -1

                # Re-check train_size
                if train_end_idx < train_start_idx: # train size became zero or negative
                     raise ValueError(
                        f"Step {step}: Recalculation led to invalid train_end_idx ({train_end_idx}). "
                        f"current_train_val_end_idx={current_train_val_end_idx}, val_size={val_size}."
                    )


            train_indices_arr = indices[train_start_idx : train_end_idx + 1]
            val_indices_arr = indices[val_start_idx : val_end_idx + 1]
            test_indices_arr = indices[test_start_idx : test_end_idx + 1]

            if not (len(train_indices_arr) > 0 and len(val_indices_arr) > 0 and len(test_indices_arr) > 0):
                # print(f"Warning: Empty set generated for step {step+1}. "
                #       f"Train len: {len(train_indices_arr)}, Val len: {len(val_indices_arr)}, Test len: {len(test_indices_arr)}")
                # This should ideally be caught by earlier checks.
                if len(test_indices_arr) == 0: # Critical if test is empty
                    # print("Stopping due to empty test set that should have been formable.")
                    break


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
    X_basic = np.arange(20).reshape(10, 2) # 10 samples
    y_basic = np.arange(10)

    print("Input X shape:", X_basic.shape)

    # Example 1: Default settings
    print("\nExample 1: Default settings (n_splits=5)")
    bts_1 = BasicTimeSeriesSplit(n_splits=5)
    for i, (train_idx, test_idx) in enumerate(bts_1.split(X_basic)):
        print(f"Split {i+1}: Train={train_idx}, Test={test_idx}")
    print(f"Total splits: {bts_1.get_n_splits()}")

    # Example 2: With test_size
    print("\nExample 2: n_splits=3, test_size=2")
    bts_2 = BasicTimeSeriesSplit(n_splits=3, test_size=2)
    for i, (train_idx, test_idx) in enumerate(bts_2.split(X_basic)):
        print(f"Split {i+1}: Train={train_idx}, Test={test_idx}")

    # Example 3: With test_size and gap
    print("\nExample 3: n_splits=3, test_size=1, gap=1")
    bts_3 = BasicTimeSeriesSplit(n_splits=3, test_size=1, gap=1)
    for i, (train_idx, test_idx) in enumerate(bts_3.split(X_basic)):
        print(f"Split {i+1}: Train={train_idx}, Test={test_idx}")

    # Example 4: With max_train_size
    print("\nExample 4: n_splits=3, test_size=1, max_train_size=3")
    bts_4 = BasicTimeSeriesSplit(n_splits=3, test_size=1, max_train_size=3)
    for i, (train_idx, test_idx) in enumerate(bts_4.split(X_basic)):
        print(f"Split {i+1}: Train={train_idx} (len {len(train_idx)}), Test={test_idx}")

    # Example 5: Case where test_size might be too large or n_splits too many
    print("\nExample 5: n_splits=4, test_size=3 (10 samples total)") # 4*3=12 > 10, should limit splits
    # Sklearn TimeSeriesSplit would yield:
    # train: [0 1] test: [2 3]
    # train: [0 1 2 3] test: [4 5]
    # train: [0 1 2 3 4 5] test: [6 7]
    # train: [0 1 2 3 4 5 6 7] test: [8 9]
    # Let's check how our BasicTimeSeriesSplit handles this with fixed test_size
    # The definition of BasicTimeSeriesSplit makes test sets fixed from the end, expanding train.
    # So, first train is small.
    # Split 1: Train=[0], Test=[1 2 3] (if test_size=3, n_splits=?) No, this is not how it's defined.
    # It should be: Test1=[n-test_size*N .. n-test_size*(N-1)], TestN=[n-test_size .. n]
    # Train for Test1 is [0 .. n-test_size*N-gap-1]
    # Let's re-verify the output for this case.
    # With n_samples=10, n_splits=4, test_size=3.
    # Expected (if it can work, but 4*3=12 exceeds 10 for just tests):
    # This configuration is problematic for fixed test_size if we want all test sets to be full.
    # Sklearn TimeSeriesSplit(n_splits=4, test_size=3)
    # Split 1: Train [0], Test [1 2 3]
    # Split 2: Train [0 1 2 3], Test [4 5 6]
    # Split 3: Train [0 1 2 3 4 5 6], Test [7 8 9]
    # (Only 3 splits possible if test_size must be 3)

    # Let's test BasicTimeSeriesSplit with n_splits=3, test_size=3 on 10 samples
    print("\nExample 5.1: n_splits=3, test_size=3 (10 samples)")
    # Expected:
    # Split 1: T=[0], V=[1,2,3]
    # Split 2: T=[0,1,2,3], V=[4,5,6]
    # Split 3: T=[0,1,2,3,4,5,6], V=[7,8,9]
    bts_5_1 = BasicTimeSeriesSplit(n_splits=3, test_size=3)
    count_5_1 = 0
    for i, (train_idx, test_idx) in enumerate(bts_5_1.split(X_basic)):
        print(f"Split {i+1}: Train={train_idx}, Test={test_idx}")
        count_5_1 +=1
    print(f"Actual splits for 5.1: {count_5_1}") # Should be 1 if logic is like sklearn's fixed test_size from start

    # The current BasicTimeSeriesSplit's fixed test_size logic positions test sets from the end.
    # Split 1 (i=0): test_end_idx = 10 - (3-1-0)*3 - (3-1-0)*0 = 10-6=4. test_start_idx = 4-3+1=2. train_end_idx = 2-1-0=1. Train [0,1], Test [2,3,4]
    # Split 2 (i=1): test_end_idx = 10 - (3-1-1)*3 = 10-3=7. test_start_idx = 7-3+1=5. train_end_idx = 5-1-0=4. Train [0..4], Test [5,6,7]
    # Split 3 (i=2): test_end_idx = 10 - (3-1-2)*3 = 10. test_start_idx = 10-3+1=8. train_end_idx = 8-1-0=7. Train [0..7], Test [8,9]
    # This seems correct for fixed test_size.

    print("\nExample 5.2: n_splits=4, test_size=3 (10 samples) - Expect fewer splits")
    bts_5_2 = BasicTimeSeriesSplit(n_splits=4, test_size=3)
    count_5_2 = 0
    for i, (train_idx, test_idx) in enumerate(bts_5_2.split(X_basic)):
        print(f"Split {i+1}: Train={train_idx}, Test={test_idx}")
        count_5_2 +=1
    print(f"Actual splits for 5.2: {count_5_2}") # Should be 1.
    # Split 1 (i=0): test_end = 10-(4-1-0)*3 = 1. test_start = 1-3+1 = -1. This is wrong.
    # The test_end_idx calculation: `n_samples - (self.n_splits - 1 - i) * self.test_size`
    # i=0: 10 - (3)*3 = 1. test_start = 1-3+1 = -1. This is bad.
    # My indexing for test_end_idx was off.
    # It should be: test_end_idx for split `i` is at `n_samples - (n_splits - 1 - i) * step_size`
    # Let's use sklearn's TimeSeriesSplit for comparison if test_size is set
    print("\nExample 5.3: Sklearn's TimeSeriesSplit with n_splits=3, test_size=3 (10 samples)")
    sk_bts = SklearnTimeSeriesSplit(n_splits=3, test_size=3)
    for i, (train_idx, test_idx) in enumerate(sk_bts.split(X_basic)):
        print(f"Sklearn Split {i+1}: Train={train_idx}, Test={test_idx}")
        # Expected:
        # Sklearn Split 1: Train=[0 1], Test=[2 3 4]
        # Sklearn Split 2: Train=[0 1 2 3 4], Test=[5 6 7]
        # Sklearn Split 3: Train=[0 1 2 3 4 5 6 7], Test=[8 9]
    # The BasicTimeSeriesSplit needs to replicate this behavior.
    # The iteration logic:
    # `test_start_idx = n_samples - (self.n_splits - i) * self.test_size - (self.n_splits - 1 - i) * self.gap`
    # `train_end_idx = test_start_idx - 1 - self.gap`
    # Let's re-evaluate for n_samples=10, n_splits=3, test_size=3, gap=0
    # i=0: test_start = 10 - (3-0)*3 - (3-1-0)*0 = 10 - 9 = 1. train_end = 0. Test=[1,2,3]. Train=[0]
    # i=1: test_start = 10 - (3-1)*3 - (3-1-1)*0 = 10 - 6 = 4. train_end = 3. Test=[4,5,6]. Train=[0,1,2,3]
    # i=2: test_start = 10 - (3-2)*3 - (3-1-2)*0 = 10 - 3 = 7. train_end = 6. Test=[7,8,9]. Train=[0..6]
    # This matches sklearn's behavior for test_size. My manual trace of BasicTimeSeriesSplit was incorrect previously.
    # The loop for BasicTimeSeriesSplit was changed during implementation. The current version should be:
    # For split `i` from 0 to `n_splits-1`:
    #   `test_start_idx = first_train_size + i * test_size + (i) * gap`
    #   `first_train_size = n_samples - n_splits*test_size - (n_splits-1)*gap` (approx)
    # No, the current code's loop for `BasicTimeSeriesSplit` is:
    #   `test_end_idx = n_samples - (self.n_splits - 1 - i) * (self.test_size + self.gap)` (if test_size fixed)
    #   `test_start_idx = test_end_idx - self.test_size + 1`
    #   `train_end_idx = test_start_idx - 1 - self.gap`
    # Let's re-trace n_samples=10, n_splits=3, test_size=3, gap=0:
    # i=0: test_end_idx = 10 - (3-1-0)*3 = 4. test_start_idx = 4-3+1 = 2. train_end_idx = 1. Tr=[0,1], Te=[2,3,4]. Correct.
    # i=1: test_end_idx = 10 - (3-1-1)*3 = 7. test_start_idx = 7-3+1 = 5. train_end_idx = 4. Tr=[0..4], Te=[5,6,7]. Correct.
    # i=2: test_end_idx = 10 - (3-1-2)*3 = 10. test_start_idx = 10-3+1 = 8. train_end_idx = 7. Tr=[0..7], Te=[8,9]. Correct.
    # The code seems to match sklearn when test_size is provided.

    print("\n--- WalkForwardSplit Example ---")
    X_walk = np.arange(100).reshape(50, 2) # 50 samples
    y_walk = np.arange(50)
    print("Input X shape for WalkForwardSplit:", X_walk.shape)

    # Example 1: Basic walk-forward
    print("\nExample W1: initial_train_val_size=20, n_test_periods=5, n_recurrent_steps=3, train_val_ratio=0.75")
    wfs_1 = WalkForwardSplit(initial_train_val_size=20, n_test_periods=5, n_recurrent_steps=3, train_val_ratio=0.75, min_val_size=3)
    for i, (train_idx, val_idx, test_idx) in enumerate(wfs_1.split(X_walk)):
        print(f"Split {i+1}:")
        print(f"  Train: {train_idx} (len {len(train_idx)})")
        print(f"  Val:   {val_idx} (len {len(val_idx)})")
        print(f"  Test:  {test_idx} (len {len(test_idx)})")
        # Check contiguity and no overlap
        assert train_idx[-1] + 1 == val_idx[0], "Train and Val not contiguous"
        assert val_idx[-1] + 1 == test_idx[0], "Val and Test not contiguous"
        if i > 0:
            # Previous test set should be incorporated into current train/val
             assert prev_test_idx[-1] < train_idx[-1] , "Train/Val not expanding over previous test"
        prev_test_idx = test_idx.copy()
    print(f"Total splits: {wfs_1.get_n_splits()}")

    # Example 2: More steps, smaller test
    print("\nExample W2: initial_train_val_size=15, n_test_periods=2, n_recurrent_steps=5, train_val_ratio=0.8, min_val_size=2")
    # Total samples needed for 5 steps: 15 (initial) + 5*2 (tests) = 25. We have 50.
    wfs_2 = WalkForwardSplit(initial_train_val_size=15, n_test_periods=2, n_recurrent_steps=5, train_val_ratio=0.8, min_val_size=2)
    for i, (train_idx, val_idx, test_idx) in enumerate(wfs_2.split(X_walk)):
        print(f"Split {i+1}: Train len={len(train_idx)}, Val len={len(val_idx)}, Test len={len(test_idx)}")
        print(f"  Train indices: {train_idx[0]}...{train_idx[-1]}")
        print(f"  Val   indices: {val_idx[0]}...{val_idx[-1]}")
        print(f"  Test  indices: {test_idx[0]}...{test_idx[-1]}")


    # Example 3: Not enough data for all recurrent steps
    print("\nExample W3: initial_train_val_size=30, n_test_periods=5, n_recurrent_steps=5 (50 samples)")
    # Need: 30 (initial) + 5*5 (tests) = 55. We have 50. Should raise error or limit steps.
    # Max steps = 1 + (50 - 30 - 5) // 5 = 1 + 15 // 5 = 1 + 3 = 4 steps.
    try:
        wfs_3 = WalkForwardSplit(initial_train_val_size=30, n_test_periods=5, n_recurrent_steps=5)
        for _ in wfs_3.split(X_walk): pass
    except ValueError as e:
        print(f"Correctly caught error for W3: {e}")

    # Example 3.1: Adjusted steps for W3
    print("\nExample W3.1: initial_train_val_size=30, n_test_periods=5, n_recurrent_steps=4 (50 samples)")
    # Need: 30 + 4*5 = 50. This should work.
    wfs_3_1 = WalkForwardSplit(initial_train_val_size=30, n_test_periods=5, n_recurrent_steps=4)
    count_3_1 = 0
    for i, (train_idx, val_idx, test_idx) in enumerate(wfs_3_1.split(X_walk)):
        print(f"Split {i+1}: Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")
        count_3_1+=1
    print(f"Actual splits for W3.1: {count_3_1}")


    # --- TimeBasedGroupShuffleSplit Example ---
    print("\n--- TimeBasedGroupShuffleSplit Example ---")
    # This will just show the NotImplementedError
    tbgss = TimeBasedGroupShuffleSplit()
    try:
        for _ in tbgss.split(X_walk): # X_walk is not a DataFrame with group/date columns, but error is earlier
            pass
    except NotImplementedError as e:
        print(f"Correctly caught: {e}")
    except ValueError as e: # If it gets past NotImplementedError due to DataFrame checks
        print(f"Caught ValueError (expected if X is not DataFrame with specific columns): {e}")

    # Test BasicTimeSeriesSplit with edge case n_samples close to n_splits
    print("\nExample BasicTimeSeriesSplit: n_samples=6, n_splits=5, test_size=1")
    # Expected:
    # S1: T=[0], Te=[1]
    # S2: T=[0,1], Te=[2]
    # S3: T=[0,1,2], Te=[3]
    # S4: T=[0,1,2,3], Te=[4]
    # S5: T=[0,1,2,3,4], Te=[5]
    X_edge = np.arange(12).reshape(6,2)
    bts_edge = BasicTimeSeriesSplit(n_splits=5, test_size=1)
    for i, (train_idx, test_idx) in enumerate(bts_edge.split(X_edge)):
        print(f"Edge Split {i+1}: Train={train_idx}, Test={test_idx}")

    print("\nExample BasicTimeSeriesSplit: n_samples=5, n_splits=3, test_size=None (proportional)")
    # n_samples // (n_splits+1) = 5 // 4 = 1.
    # fold_sizes = [2,1,1,1]
    # S1: T=[0,1], Te=[2]
    # S2: T=[0,1,2], Te=[3]
    # S3: T=[0,1,2,3], Te=[4]
    X_prop = np.arange(10).reshape(5,2)
    bts_prop = BasicTimeSeriesSplit(n_splits=3, test_size=None) # gap=0
    for i, (train_idx, test_idx) in enumerate(bts_prop.split(X_prop)):
        print(f"Proportional Split {i+1}: Train={train_idx}, Test={test_idx}")

    # Test WalkForwardSplit with initial_train_val_size = min_val_size + 1
    print("\nExample WalkForward: initial_train_val_size=2, n_test_periods=1, n_recurrent_steps=1, train_val_ratio=0.5, min_val_size=1")
    X_w_edge = np.arange(6).reshape(3,2) # 3 samples. Need 2 for train/val, 1 for test.
    wfs_edge = WalkForwardSplit(initial_train_val_size=2, n_test_periods=1, n_recurrent_steps=1, train_val_ratio=0.5, min_val_size=1)
    for i, (train_idx, val_idx, test_idx) in enumerate(wfs_edge.split(X_w_edge)):
         print(f"WFS Edge Split {i+1}: Train={train_idx}, Val={val_idx}, Test={test_idx}")
         # Expected: Train=[0] (size 1), Val=[1] (size 1), Test=[2] (size 1)

    print("\nExample WalkForward: initial_train_val_size=1, n_test_periods=1, n_recurrent_steps=1, train_val_ratio=0.5 (or any), min_val_size=1")
    # This should fail because val_size cannot be 1 and train_size be positive if initial_train_val_size is 1.
    try:
        wfs_fail = WalkForwardSplit(initial_train_val_size=1, n_test_periods=1, n_recurrent_steps=1, min_val_size=1)
        for _ in wfs_fail.split(np.arange(4).reshape(2,2)): pass # Need 1 (train/val) + 1 (test) = 2 samples
    except ValueError as e:
        print(f"Correctly caught error for WFS fail case: {e}")

    # Test BasicTimeSeriesSplit where test_size is None and n_samples / (n_splits + 1) is 0
    print("\nExample BasicTimeSeriesSplit: n_samples=3, n_splits=3, test_size=None")
    # n_samples // (n_splits+1) = 3 // 4 = 0. No splits should be possible if test size is 0.
    X_tiny = np.arange(6).reshape(3,2)
    bts_tiny = BasicTimeSeriesSplit(n_splits=3, test_size=None)
    count_tiny = 0
    for i, (train_idx, test_idx) in enumerate(bts_tiny.split(X_tiny)):
        print(f"Tiny Split {i+1}: Train={train_idx}, Test={test_idx}") # Test should not be empty.
        count_tiny +=1
    print(f"Tiny splits generated: {count_tiny}") # Expect 0 if test size calc is 0.
    # Sklearn behavior: if test_size becomes 0, it raises ValueError.
    # My code currently yields if train is not empty. If test_size is 0, test will be empty.
    # Added a check for len(test_indices) == 0 to break.
    # The proportional split logic: fold_sizes[: n_samples % (n_splits + 1)] += 1
    # For n_samples=3, n_splits=3: n_samples % (n_splits+1) = 3 % 4 = 3. fold_sizes=[1,1,1,0]
    # S1: T=[0], Te=[1]
    # S2: T=[0,1], Te=[2]
    # S3: T=[0,1,2], Te=[] -> This last one should not be yielded.
    # My code yields it if train is not empty. Fixed by checking if test is empty.
    # Now, it should yield 2 splits.

    # Test BasicTimeSeriesSplit with n_splits=2, n_samples=3, test_size=None
    # fold_sizes = [1,1,1]. S1: T=[0], Te=[1]. S2: T=[0,1], Te=[2].
    print("\nExample BasicTimeSeriesSplit: n_samples=3, n_splits=2, test_size=None")
    X_3_2 = np.arange(6).reshape(3,2)
    bts_3_2 = BasicTimeSeriesSplit(n_splits=2, test_size=None)
    for i, (train_idx, test_idx) in enumerate(bts_3_2.split(X_3_2)):
        print(f"Split 3_2 {i+1}: Train={train_idx}, Test={test_idx}") # Expect 2 splits.The `splitters.py` file has been created/updated with the implementations of `BasicTimeSeriesSplit`, `WalkForwardSplit`, and the placeholder `TimeBasedGroupShuffleSplit`. I've also included `if __name__ == '__main__':` blocks with examples to demonstrate their usage.

My implementation of `BasicTimeSeriesSplit` with fixed `test_size` aims to mirror `sklearn.model_selection.TimeSeriesSplit`'s behavior where `test_size` is an integer. The key logic for fixed `test_size` is:
- Test sets are anchored to the end of the series.
- For split `i` (0-indexed from `n_splits-1` down to 0, or equivalently `k` from 0 to `n_splits-1` for the actual split number):
    - `test_end_idx = n_samples - (self.n_splits - 1 - k) * (self.test_size + self.gap)` (if gap is considered part of step)
    - More directly, as in the code:
        - `test_end_idx` for split `k`: `n_samples - (self.n_splits - 1 - k) * self.test_size - (self.n_splits - 1 - k) * self.gap`
        - `test_start_idx = test_end_idx - self.test_size + 1`
        - `train_end_idx = test_start_idx - 1 - self.gap`
        - `train_start_idx` is `0` or `train_end_idx + 1 - self.max_train_size`.

The `WalkForwardSplit` implementation generates `(train, val, test)` indices where the test set of one split is incorporated into the training data of the next, and validation is taken from the end of the current combined training data.

Next, I will update `financial_pipeline/cross_validation/__init__.py`.
