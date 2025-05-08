# AIRCHECK ML Pipeline

## Overview

The AIRCHECK ML Pipeline is a modular machine learning pipeline for model training, evaluation, and selection. It supports multiple machine learning models, automatic hyperparameter tuning, and model fusion.

## Key Features

* Supports multiple ML models (Random Forest, Logistic Regression, SVM, Gradient Boosting, etc.)
* Automatic model selection using Bayesian optimization (optional)
* Cross-validation and final model training
* Automated model selection and score fusion
* Detailed model evaluation with radar plots
* Configurable through a single YAML configuration file

## Project Structure

* **`runner.py`**: Main script to run the entire pipeline.
* **`config.yaml`**: Configuration file that defines data paths, model settings, and evaluation criteria.
* **`config_utils.py`**: Utilities for reading, validating, and writing the configuration file.
* **`data_utils.py`**: Functions for data loading, processing, and balancing.
* **`model_utils.py`**: Functions for model initialization, training, cross-validation, and hyperparameter optimization.
* **`eval_utils.py`**: Functions for model evaluation and metric calculation.
* **`fusion_utils.py`**: Functions for model selection, fusion, and radar plot generation.

## How to Use

### 1. Configure Your Settings

* Edit the `config.yaml` file to specify:

  * Training and test data paths (`train_data`, `test_data`)
  * Desired columns (`desired_columns`)
  * Model names (`desired_models`)
  * Whether to enable hyperparameter tuning (`hyperparameters_tuning`)
  * Fusion settings (`Fusion`) and number of top models (`num_top_models`)

### 2. Run the Pipeline

```bash
python runner.py
```

### 3. Check the Results

* Results are saved in the `Results` directory.
* Includes:

  * Model folders with saved models.
  * A `results.csv` file with training and evaluation metrics.
  * A radar plot image (`RadarChart_TopModels.png`) showing the performance of top models.

## How It Works

* The pipeline reads your configuration and trains models using specified settings.
* Cross-validation is performed to ensure model robustness.
* Top-performing models are selected based on evaluation metrics.
* Selected models are fused to create an ensemble model.

## Customization

* Add or remove models in `config.yaml` using the `desired_models` list.
* Customize model hyperparameters using the `hyperparameters` section.
* Adjust the number of cross-validation folds with `Nfold`.
* Modify fusion settings (`Fusion`, `num_top_models`) to control the ensemble.

