import unittest
import os
import pathlib
import logging # For checking logger setup
import tempfile # For creating temporary files/dirs securely

# Add parent directory to sys.path
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from financial_pipeline.utils.io_utils import save_object, load_object
from financial_pipeline.utils.logging_utils import setup_basic_logger

class TestUtils(unittest.TestCase):

    def test_save_load_object(self):
        """Test saving and loading an object using joblib via utils."""
        sample_dict = {'a': 1, 'b': [1, 2, 3], 'c': {'nested': 'string'}}

        # Use tempfile for robust temporary file creation
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = pathlib.Path(tmpdir) / "test_object.joblib"

            save_object(sample_dict, filepath)
            self.assertTrue(filepath.exists(), "Saved object file should exist.")

            loaded_dict = load_object(filepath)
            self.assertEqual(sample_dict, loaded_dict, "Loaded object does not match original.")

            # Test loading a non-existent file
            with self.assertRaises(FileNotFoundError):
                load_object(pathlib.Path(tmpdir) / "non_existent.joblib")

    def test_setup_basic_logger_console_only(self):
        """Test setting up a basic logger for console output."""
        logger_name = 'TestConsoleLogger'
        logger = setup_basic_logger(logger_name=logger_name, log_level=logging.DEBUG)

        self.assertIsInstance(logger, logging.Logger, "Should return a Logger instance.")
        self.assertEqual(logger.name, logger_name, "Logger name mismatch.")
        self.assertEqual(logger.level, logging.DEBUG, "Logger level mismatch.")

        # Check for StreamHandler (console)
        self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in logger.handlers),
                        "Logger should have a StreamHandler.")

        # Ensure no FileHandler if file_path is None
        self.assertFalse(any(isinstance(h, logging.FileHandler) for h in logger.handlers),
                         "Logger should not have a FileHandler when file_path is None.")

        # Test if it logs messages (visual check or capture stdout if needed for full automation)
        # For now, just call log methods to ensure no errors.
        logger.debug("Test debug message from console logger.")
        logger.info("Test info message from console logger.")


    def test_setup_basic_logger_with_file(self):
        """Test setting up a logger with file output."""
        logger_name = 'TestFileLogger'

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file_path = pathlib.Path(tmpdir) / "test_app.log"

            logger = setup_basic_logger(logger_name=logger_name,
                                        log_level=logging.INFO,
                                        file_path=str(log_file_path)) # Ensure str for older Pythons if needed

            self.assertTrue(any(isinstance(h, logging.FileHandler) for h in logger.handlers),
                            "Logger should have a FileHandler.")

            # Log some messages
            test_info_msg = "Test info message for file logger."
            test_error_msg = "Test error message for file logger."
            logger.info(test_info_msg)
            logger.error(test_error_msg)

            # Check if log file was created and contains messages
            self.assertTrue(log_file_path.exists(), "Log file should be created.")
            with open(log_file_path, 'r') as f:
                log_content = f.read()
            self.assertIn(test_info_msg, log_content, "Info message not found in log file.")
            self.assertIn(test_error_msg, log_content, "Error message not found in log file.")
            self.assertIn(logger_name, log_content, "Logger name not found in log file content.")

    def test_logger_handler_clearing(self):
        """Test that existing handlers are cleared if specified."""
        logger_name = "TestHandlerClearing"
        logger = setup_basic_logger(logger_name=logger_name, log_level=logging.INFO, clear_existing_handlers=False)
        initial_handler_count = len(logger.handlers)

        # Add another handler manually (or by calling setup_basic_logger without clearing)
        logger.addHandler(logging.StreamHandler()) # Add a dummy handler
        self.assertTrue(len(logger.handlers) > initial_handler_count)

        # Now setup again with clear_existing_handlers=True (default)
        logger_cleared = setup_basic_logger(logger_name=logger_name, log_level=logging.INFO)
        # The number of handlers should be back to what setup_basic_logger adds (1 or 2)
        # For console only, it's 1.
        expected_handlers_after_clear = 1 # Assuming console only in this call
        self.assertEqual(len(logger_cleared.handlers), expected_handlers_after_clear,
                         "Handlers were not cleared and re-added as expected.")


if __name__ == '__main__':
    unittest.main()
