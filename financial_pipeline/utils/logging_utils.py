# This file will contain functions for configuring and using logging.
import logging
import sys # For StreamHandler to output to stdout

def setup_basic_logger(logger_name='financial_pipeline', log_level=logging.INFO, file_path=None, clear_existing_handlers=True):
    """
    Sets up a basic logger with console and optional file output.

    Args:
        logger_name (str, optional): Name of the logger to configure.
                                     Defaults to 'financial_pipeline'.
        log_level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
                                   Defaults to logging.INFO.
        file_path (str, optional): Path to a log file. If provided, logs will also
                                   be written to this file. Defaults to None.
        clear_existing_handlers (bool, optional): Whether to remove existing handlers
                                                  from the logger before adding new ones.
                                                  Defaults to True.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    if clear_existing_handlers:
        # Remove existing handlers to prevent duplicate logs if called multiple times
        if logger.hasHandlers():
            logger.handlers.clear()

    # Define a basic formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # Add StreamHandler for console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level) # Set level for handler as well
    logger.addHandler(console_handler)

    # Add FileHandler if file_path is provided
    if file_path:
        try:
            file_handler = logging.FileHandler(file_path, mode='a') # Append mode
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level) # Set level for handler
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to set up file handler at {file_path}: {e}", exc_info=True)
            # Continue with console logging even if file logging fails

    # Prevent logger from propagating messages to the root logger if it's not the root logger itself
    # and we've set up specific handlers. This is often desired to avoid duplicate outputs
    # if the root logger also has handlers.
    if logger_name != logging.getLogger().name: # Check if it's not the root logger
         logger.propagate = False


    return logger

if __name__ == '__main__':
    # Demonstrate usage

    # 1. Basic console logger
    logger1 = setup_basic_logger(logger_name='AppLogger1', log_level=logging.DEBUG)
    logger1.debug("This is a debug message for AppLogger1 (console only).")
    logger1.info("This is an info message for AppLogger1.")
    logger1.warning("This is a warning for AppLogger1.")

    # 2. Logger with file output
    log_file = "app_demonstration.log"
    logger2 = setup_basic_logger(logger_name='AppLogger2', log_level=logging.INFO, file_path=log_file)
    logger2.info("This is an info message for AppLogger2 (console and file).")
    logger2.error("This is an error message for AppLogger2, also in the file.")

    print(f"Check '{log_file}' for AppLogger2's file output.")

    # 3. Demonstrate getting the same logger instance and handler clearing
    logger2_again = setup_basic_logger(logger_name='AppLogger2', log_level=logging.DEBUG, clear_existing_handlers=True)
    logger2_again.debug("This is a new debug message for AppLogger2 after re-setup. Old handlers should be gone.")
    logger2_again.info("Another info for AppLogger2. If clear_existing_handlers worked, this won't be duplicated in console from old setup.")

    # 4. Root logger configuration (if logger_name is the root logger's name)
    # root_logger_name = logging.getLogger().name # This is usually 'root'
    # root_logger_configured = setup_basic_logger(logger_name=root_logger_name, log_level=logging.WARNING)
    # logging.warning("This is a warning from the root logger, configured by setup_basic_logger.")
    # Other modules using logging.info() directly might not show if root level is WARNING.

    # 5. Using a different logger after configuring one
    another_logger = logging.getLogger("AnotherModule")
    # If 'AnotherModule' doesn't have handlers and propagate=True (default),
    # its messages would go to the root logger's handlers.
    # If AppLogger1 was setup with propagate=False, its messages won't go to root.
    # Our setup_basic_logger sets propagate=False for non-root loggers.

    # To test propagation, let's get the root logger and add a handler to it.
    # If we don't call setup_basic_logger for the root, it might have default handlers or none.
    # Python's default level for root logger is WARNING.

    logging.info("This INFO message from root logger might not show by default.") # Default level is WARNING
    logging.warning("This WARNING message from root logger should show by default (if no other config changed it).")

    logger1.info("Info from logger1 again (should not propagate to root if setup correctly).")

    # Clean up the log file created by the demo
    try:
        import os
        if os.path.exists(log_file):
            os.remove(log_file)
            print(f"Cleaned up demo log file: {log_file}")
    except Exception as e:
        print(f"Error cleaning up demo log file: {e}")
