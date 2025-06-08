# Generic End-to-End Financial ML Pipeline

## Overview

This project provides a flexible, scikit-learn compatible pipeline designed for developing and evaluating machine learning models for financial time series forecasting and analysis. It aims to offer a structured approach from data ingestion and feature engineering through to model training, cross-validation, optimization, and evaluation.

## Key Features

*   **Data Ingestion:** Easily fetch historical financial data (stocks, crypto, etc.) using `yfinance`.
*   **Customizable Feature Engineering:**
    *   Includes common technical indicators (SMA, RSI, MACD).
    *   Supports easy integration of custom feature transformers by inheriting from a base class.
*   **Time-Series Cross-Validation:**
    *   `BasicTimeSeriesSplit`: Standard time series splitting with expanding or rolling windows.
    *   `WalkForwardSplit`: Implements a walk-forward validation strategy suitable for recurrent model fine-tuning, yielding train, validation, and test sets for each period.
*   **Flexible Model Wrappers:**
    *   Support for various models including LightGBM, simple Keras Neural Networks, and Ridge regression.
    *   Easily extensible to wrap other scikit-learn compatible or custom models.
*   **TensorBoard Integration:** Utilities for logging metrics and losses during training and cross-validation, viewable in TensorBoard.
*   **Hyperparameter Optimization:** Built-in support for hyperparameter tuning using Optuna, allowing for optimization of feature engineering steps and model parameters.
*   **Modular and Extensible Design:** Components are organized into logical modules, making it straightforward to customize or extend any part of the pipeline.
*   **Utilities:** Includes helpers for logging, saving/loading Python objects (e.g., trained models, Optuna studies).

## Project Structure

The project is organized into the following main directories:

-   `financial_pipeline/`: Root package directory.
    -   `data_ingestion/`: Scripts for fetching financial data.
    -   `feature_engineering/`: Modules for creating and transforming features. Contains base classes and example transformers. (See `feature_engineering/README.md`)
    -   `cross_validation/`: Custom time-series cross-validation splitters.
    -   `models/`: Wrappers for various ML models and a base class for custom model integration. (See `models/README.md`)
    -   `evaluation/`: Tools for model evaluation, including metrics and plotting utilities (e.g., TensorBoard logging, reliability plots).
    -   `optimization/`: Hyperparameter optimization using Optuna.
    -   `utils/`: General utility functions (logging, I/O).
    -   `notebooks/`: Jupyter notebooks for demonstrating pipeline usage and experimentation.
    -   `tests/`: Unit tests for various components of the pipeline.
    -   `main.py`: Example script orchestrating an end-to-end pipeline run (data -> features -> model -> CV -> evaluation).
-   `logs/`: Default directory for TensorBoard logs and other log files. (This directory will be created when runs are executed).
-   `requirements.txt`: Python dependencies for the project.
-   `README.md`: This file.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/financial_pipeline.git # Replace with actual URL
    cd financial_pipeline
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    ```
    -   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
    -   On Windows:
        ```bash
        venv\\Scripts\\activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start / Running the Example Notebook

The easiest way to see the pipeline in action is to run the example Jupyter Notebook:

1.  **Install Jupyter (if not already installed in your venv):**
    ```bash
    pip install jupyterlab notebook
    ```

2.  **Run the demonstration notebook:**
    Navigate to the `notebooks/` directory and start Jupyter:
    ```bash
    cd notebooks
    jupyter notebook 01_pipeline_demonstration.ipynb
    # Or use jupyter lab:
    # jupyter lab 01_pipeline_demonstration.ipynb
    ```
    This notebook (`01_pipeline_demonstration.ipynb`) provides a comprehensive walkthrough of the pipeline's capabilities, from data fetching to model optimization.

## Running Tests

To ensure all components are working correctly, run the unit tests:

```bash
# From the root directory of the project (where requirements.txt is)
pytest financial_pipeline/tests/
```
Or, for more verbose output:
```bash
pytest -v --capture=no financial_pipeline/tests/
```

## Using TensorBoard

The pipeline supports logging metrics to TensorBoard, especially during cross-validation and hyperparameter optimization.

1.  **Log Generation:** When you run scripts like `main.py` (if configured) or `optimizer.py`, logs will typically be saved into subdirectories within the `logs/` directory (e.g., `logs/notebook_cv_run/`, `logs/optuna_financial_study/`).

2.  **Viewing Logs:**
    Start TensorBoard by pointing it to the main logs directory:
    ```bash
    # From the root directory of the project
    tensorboard --logdir logs/
    ```
    Then open your web browser to the URL provided by TensorBoard (usually `http://localhost:6006`).

## Extensibility

*   **Custom Features:** See `financial_pipeline/feature_engineering/README.md` for guidance on adding new feature transformers.
*   **Custom Models:** See `financial_pipeline/models/README.md` for how to integrate new model types.
*   **Custom CV Strategies:** Implement classes inheriting from `sklearn.model_selection.BaseCrossValidator`.
*   **Custom Metrics:** Add functions to `financial_pipeline/evaluation/metrics.py`.

## Future Work (Optional Ideas)

*   Integration of more advanced models (e.g., attention-based NNs, Prophet).
*   Advanced feature selection techniques.
*   Portfolio construction and backtesting modules.
*   Cloud integration for distributed training and data storage.
*   Enhanced plotting and reporting capabilities.
*   More sophisticated hyperparameter optimization strategies.
*   Support for live trading (with appropriate risk management).
