# This file will contain tests for the cross_validation module.
import unittest
import numpy as np
import pandas as pd

# Add parent directory to sys.path
import sys
import os
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from financial_pipeline.cross_validation import BasicTimeSeriesSplit, WalkForwardSplit

class TestCrossValidation(unittest.TestCase):

    def setUp(self):
        """Set up dummy data for cross-validation tests."""
        self.n_samples = 30 # Increased for more realistic splits
        self.X = pd.DataFrame({'feature1': np.arange(self.n_samples),
                               'feature2': np.arange(self.n_samples, self.n_samples * 2)})
        self.y = pd.Series(np.random.rand(self.n_samples))

    def test_basic_time_series_split(self):
        """Test BasicTimeSeriesSplit."""
        n_splits = 4
        test_size = 5 # Ensure n_samples is large enough: 30 samples, 4 splits * 5 test_size = 20 for tests. Min train = 30 - 4*5 = 10.

        splitter = BasicTimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=1)

        all_train_indices = []
        all_test_indices = []
        count = 0
        for train_idx, test_idx in splitter.split(self.X, self.y):
            count += 1
            self.assertIsInstance(train_idx, np.ndarray)
            self.assertIsInstance(test_idx, np.ndarray)
            self.assertTrue(len(train_idx) > 0, "Train set should not be empty.")
            self.assertTrue(len(test_idx) > 0, "Test set should not be empty.")
            self.assertEqual(len(test_idx), test_size, "Test set size is incorrect.")

            # Test chronological order: max train index < min test index - gap
            self.assertTrue(np.max(train_idx) < np.min(test_idx) - splitter.gap,
                            "Train set must come before test set, respecting the gap.")

            all_train_indices.append(train_idx)
            all_test_indices.append(test_idx)

        self.assertEqual(count, n_splits, f"Expected {n_splits} splits, but got {count}.")

        # Assert no overlap between any two test sets
        for i in range(len(all_test_indices)):
            for j in range(i + 1, len(all_test_indices)):
                self.assertTrue(len(np.intersect1d(all_test_indices[i], all_test_indices[j])) == 0,
                                "Test sets from different folds should not overlap.")

        # Assert that test sets are ordered chronologically
        for i in range(len(all_test_indices) - 1):
            self.assertTrue(np.max(all_test_indices[i]) < np.min(all_test_indices[i+1]),
                            "Test sets should be chronologically ordered.")


    def test_walk_forward_split(self):
        """Test WalkForwardSplit."""
        initial_train_val_size = 15
        n_test_periods = 3
        n_recurrent_steps = 4 # Total samples needed: 15 + 4*3 = 27. self.n_samples = 30.
        train_val_ratio = 0.7 # Approx 10 for train, 5 for val initially.
        min_val_size = 2

        splitter = WalkForwardSplit(
            initial_train_val_size=initial_train_val_size,
            n_test_periods=n_test_periods,
            n_recurrent_steps=n_recurrent_steps,
            train_val_ratio=train_val_ratio,
            min_val_size=min_val_size
        )

        count = 0
        previous_test_indices_end = -1

        for train_idx, val_idx, test_idx in splitter.split(self.X, self.y):
            count += 1
            self.assertIsInstance(train_idx, np.ndarray)
            self.assertIsInstance(val_idx, np.ndarray)
            self.assertIsInstance(test_idx, np.ndarray)

            self.assertTrue(len(train_idx) > 0, "Train set should not be empty.")
            self.assertTrue(len(val_idx) >= min_val_size, f"Val set size should be at least {min_val_size}.")
            self.assertEqual(len(test_idx), n_test_periods, "Test set size is incorrect.")

            # Check contiguity and order
            self.assertTrue(np.max(train_idx) < np.min(val_idx), "Train must precede validation.")
            self.assertTrue(np.max(val_idx) < np.min(test_idx), "Validation must precede test.")

            # Check if current train/val starts after previous test (or from 0 for expanding window)
            if count > 1:
                # The start of the current training data should be index 0 (expanding window)
                self.assertEqual(np.min(train_idx), 0, "Training window should be expanding from start.")
                # The end of the previous test set should be part of the current train+val block
                self.assertTrue(previous_test_indices_end <= np.max(val_idx),
                                "Previous test data not fully incorporated or ordering issue.")

            previous_test_indices_end = np.max(test_idx)


        self.assertEqual(count, n_recurrent_steps, f"Expected {n_recurrent_steps} splits, but got {count}.")

        # Check validation set size calculation (approximate due to int conversion and min_val_size)
        # For the first split:
        expected_val_size_first_split = max(min_val_size, int(initial_train_val_size * (1 - train_val_ratio)))
        first_split_indices = next(splitter.split(self.X,self.y)) # Get first split again
        self.assertEqual(len(first_split_indices[1]), expected_val_size_first_split,
                         f"Validation set size for the first split is not as expected. Got {len(first_split_indices[1])}, expected {expected_val_size_first_split}")


if __name__ == '__main__':
    unittest.main()
