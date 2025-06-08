import joblib
from pathlib import Path
import logging

# Use a logger for this module
# It's good practice to get a logger specific to the module rather than using root.
# If setup_basic_logger was called for 'financial_pipeline' or root, this logger will inherit that config,
# or it can be configured independently.
logger = logging.getLogger(__name__) # Gets 'financial_pipeline.utils.io_utils' if part of package

def save_object(obj: object, filepath: str or Path, compress: int = 3) -> None:
    """
    Saves an object to a file using joblib.

    Args:
        obj (object): The object to save.
        filepath (str or Path): The path to the file where the object will be saved.
        compress (int, optional): Compression level (0-9). Defaults to 3.

    Raises:
        IOError: If there's an error during file writing.
        Exception: For other joblib related errors.
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, filepath, compress=compress)
        logger.info(f"Object successfully saved to: {filepath}")
    except IOError as e:
        logger.error(f"IOError saving object to {filepath}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to save object to {filepath}: {e}", exc_info=True)
        raise


def load_object(filepath: str or Path) -> object:
    """
    Loads an object from a file using joblib.

    Args:
        filepath (str or Path): The path to the file from which to load the object.

    Returns:
        object: The loaded object.

    Raises:
        FileNotFoundError: If the filepath does not exist.
        IOError: If there's an error during file reading.
        Exception: For other joblib related errors.
    """
    try:
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"File not found for loading: {filepath}")
            raise FileNotFoundError(f"No such file or directory: '{filepath}'")

        loaded_obj = joblib.load(filepath)
        logger.info(f"Object successfully loaded from: {filepath}")
        return loaded_obj
    except FileNotFoundError: # Already handled, but good to be explicit
        raise
    except IOError as e:
        logger.error(f"IOError loading object from {filepath}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to load object from {filepath}: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # To make this example runnable and see log messages, setup a basic logger first.
    # This assumes logging_utils.py is in the same directory or financial_pipeline is in PYTHONPATH
    try:
        from .logging_utils import setup_basic_logger
        # Setup a logger that will also cover the logger used within io_utils functions
        # if they use getLogger(__name__) which is a child of 'financial_pipeline.utils'
        # For standalone run, getLogger(__name__) is '__main__', so we configure it.
        # Alternatively, configure the root logger, or the specific 'financial_pipeline.utils.io_utils' logger.
        if __name__ == '__main__': # If script is run directly
             # Configure a logger for this specific script's execution scope
            script_logger = setup_basic_logger(logger_name='io_utils_example', log_level=logging.DEBUG)
        else: # If imported
            # This setup might be redundant if a higher-level setup (e.g. in main app) is done.
            # For library modules, it's often better not to configure logging directly
            # but let the application configure it. However, for __main__ examples, it's useful.
            pass


    except ImportError:
        # Fallback if running this script standalone and logging_utils is not found in expected path
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        script_logger = logging.getLogger(__name__) # Use the module's own logger
        script_logger.warning("Could not import setup_basic_logger. Using basicConfig for logging in example.")


    script_logger.info("--- I/O Utilities Demonstration ---")

    # Example 1: Saving and loading a dictionary
    sample_dict = {'name': 'Test Object', 'version': 1.0, 'data': [1, 2, 3, 4, 5]}
    dict_filepath = Path("temp_test_dict.joblib")

    script_logger.info(f"Attempting to save dictionary to {dict_filepath}...")
    try:
        save_object(sample_dict, dict_filepath)
        loaded_dict = load_object(dict_filepath)
        script_logger.info(f"Loaded dictionary: {loaded_dict}")

        if sample_dict == loaded_dict:
            script_logger.info("Dictionary save/load test: SUCCESS (content matches)")
        else:
            script_logger.error("Dictionary save/load test: FAILED (content mismatch)")

    except Exception as e:
        script_logger.error(f"Error in dictionary save/load test: {e}", exc_info=True)
    finally:
        if dict_filepath.exists():
            dict_filepath.unlink() # Clean up
            script_logger.debug(f"Cleaned up {dict_filepath}")

    # Example 2: Saving and loading a scikit-learn model (conceptual)
    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver='liblinear') # liblinear is simple, few warnings
        # Fit with some dummy data
        dummy_X = [[0,0],[1,1],[0,1],[1,0]]
        dummy_y = [0,1,1,0]
        model.fit(dummy_X, dummy_y)

        model_filepath = Path("temp_test_model.joblib")
        script_logger.info(f"Attempting to save sklearn model to {model_filepath}...")
        save_object(model, model_filepath)
        loaded_model = load_object(model_filepath)
        script_logger.info(f"Loaded sklearn model: {type(loaded_model)}")

        # Basic check if model seems loaded (not a thorough test)
        if hasattr(loaded_model, 'predict'):
            script_logger.info("Sklearn model save/load test: SUCCESS (model seems loadable)")
        else:
            script_logger.error("Sklearn model save/load test: FAILED (model not loaded correctly)")

    except ImportError:
        script_logger.warning("Scikit-learn not installed. Skipping model save/load example.")
    except Exception as e:
        script_logger.error(f"Error in sklearn model save/load test: {e}", exc_info=True)
    finally:
        if 'model_filepath' in locals() and model_filepath.exists(): # locals() to check if defined
            model_filepath.unlink() # Clean up
            script_logger.debug(f"Cleaned up {model_filepath}")

    # Example 3: Testing non-existent file load
    script_logger.info("\nAttempting to load a non-existent file (expect FileNotFoundError)...")
    try:
        load_object("non_existent_file.joblib")
    except FileNotFoundError as e:
        script_logger.info(f"Correctly caught expected error: {e}")
    except Exception as e:
        script_logger.error(f"Unexpected error when loading non-existent file: {e}", exc_info=True)

    script_logger.info("--- I/O Utilities Demonstration Finished ---")
