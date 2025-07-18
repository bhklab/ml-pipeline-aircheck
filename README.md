# ML Pipeline for Drug Discovery

A comprehensive machine learning pipeline for drug discovery and virtual screening, featuring model training, selection, fusion, and deployment capabilities.

## 🚀 Features

- **End-to-end ML Pipeline**: Complete workflow from data preparation to model deployment
- **Model Training & Selection**: Support for multiple ML algorithms with automated model selection
- **Model Fusion**: Advanced ensemble methods for improved performance
- **Virtual Screening**: High-throughput screening of chemical compounds
- **MLflow Integration**: Experiment tracking and model versioning
- **Cloud Storage**: Automatic artifact upload to Google Cloud Storage
- **Conformal Prediction**: Uncertainty quantification for predictions
- **Chemistry Filters**: Built-in molecular property filters

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Git

### Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_mapie_conformal.txt #install this to use conformal
   ```

## 🚀 Quick Start

### Basic Usage

```bash
# Run with default configuration
python -m src

# Run with custom configuration
python -m src --config path/to/your/config.yaml

# Run with verbose logging
python -m src --verbose
```

## 🔄 Pipeline Overview

The ML pipeline consists of 10 main steps:

### 1. **Configuration Loading**

- Loads YAML configuration file
- Sets up run folder and logging

### 2. **Data Preparation**

- Creates balanced datasets
- Handles data preprocessing
- Feature engineering

### 3. **Model Training**

- Trains multiple ML models
- Supports various algorithms (RF, SVM, XGBoost, Neural Networks)
- Cross-validation and hyperparameter tuning

### 4. **Model Testing**

- Evaluates trained models on test sets
- Generates performance metrics
- Creates evaluation reports

### 5. **Model Selection**

- Selects best performing models
- Supports multiple evaluation criteria
- Generates radar charts for visualization

### 6. **Model Fusion**

- Combines predictions from multiple models
- Ensemble methods for improved performance
- Optional step based on configuration

### 7. **Virtual Screening**

- Screens large compound libraries
- Applies chemistry filters
- Generates screening reports

### 8. **MLflow Logging**

- Logs experiments and models
- Tracks metrics and parameters
- Model versioning and registry

### 9. **Artifact Upload**

- Uploads results to Google Cloud Storage
- Organizes artifacts for easy access

### 10. **Cleanup**

- Removes run_name files
- Configurable cleanup options (default True, if you want to keep run_file set `cleanup_after_run` to `True` )

## ⚙️ Configuration

### Main Configuration File (`config_loader.yaml`)

```yaml
# Data Configuration

protein_name: "WDR91" # Target Name
is_train: True # Running train phase (False:for no, True: for yes)
is_test: True # Running test phase (False:for no, True: for yes)
is_screen: True # Running screen phase (False:for no, True: for yes)

train_data:
  - ./data/TrainFiles/company1.parquet

test_data:
  - ./data/TestFiles/sampled_data_test_1.parquet

desired_columns:
  - ECFP4 # Correct format: [ECFP4], and [ECFP4, ECFP6, ...] if multuple columns
label_column_train: LABEL
label_column_test: LABEL
nrows_train: None # integer or None
nrows_test: None
feature_fusion_method: None # options: None, All, Pairwise
balance_flag: False # Creating blanced train sets (True/False)
balance_ratios: # balance_ratios: [1, 2, 4, 8]
  - 1 # Ratio of positive to negative samples in the balanced dataset

# Model Configuration
desired_models:
  - lgbm

hyperparameters_tuning: False #  (N:for no, Y: for yes)
tf_models:
  - tf_ff
  - tf_cnn1D
# Specifying hyperparameters
hyperparameters:
  tf_ff:
  input_shape: 2048
  hidden_units:
    - 128
    - 64
  learning_rate: 0.0005

# Training Configuration
Nfold: 2

# Conformal Prediction
conformal_prediction: False # Running conformal prediction (N:for no, Y: for yes)
conformal_test_size: 0.3
conformal_confidence_level: 0.95

# Model selection

trainfile_for_modelselection: [] # If empty, the top model by evaluation columns and result on the evaluation set is selected. Example: trainfile_for_modelselection: WDR91_SGC.parquet
evaluationfile_for_modelselection: [] # If empty, the top model by evaluation columns is selected. Exmple: evaluationfile_for_modelselection: evaluation.parquet
evaluation_column:
  - Test_HitsAt200
  - Test_HitsAt500
crossvalidation_column:
  - CV_HitsAt200
  - CV_Precision
  - CV_Recall
  - CV_Accuracy
  - CV_PlatePPV

# Model Fusion
Fusion: True # Running model fusion (N:for no, Y: for yes)
num_top_models: 2

# Cloud Storage
bucket_name: "your-gcs-bucket"
prefix_name: "mlflow-artifacts"

# Cleanup
cleanup_after_run: true
```

### Environment Variables (`.env`)

## 📖 Usage

### Command Line Interface

```bash
# Basic usage
python -m src

# With custom config
python -m src --config configs/custom_config.yaml

# With verbose logging
python -m src --verbose

# Help
python -m src --help
```

## 📁 Project Structure

---

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## 📊 Monitoring & Logging

### MLflow UI

```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Access UI at http://127.0.0.1:5000/
```

## 🔧 Customization

### Adding New Models

1. **Create model class** in `src/models/`
2. **Add to configuration** in `config_loader.yaml`
3. **Update training pipeline** in `train_model.py`

### Custom Evaluation Metrics

1. **Add metric function** to `src/utils/eval_utils11.py`
2. **Update configuration** to include new metric
3. **Modify selection criteria** as needed

### Custom Screening Filters

1. **Add filter function** to `src/screening/clustering.py`
2. **Update configuration** to enable new filter
3. **Test with sample data**

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**

   ```yaml
   # Reduce batch size in config
   nrows_train: 10000
   nrows_test: 5000
   ```

2. **MLflow Connection Issues**
   ```bash
   # Check MLflow server status
   mlflow server --help
   ```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MLflow](https://mlflow.org/) for experiment tracking
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [RDKit](https://www.rdkit.org/) for cheminformatics
- [Google Cloud](https://cloud.google.com/) for storage solutions

---

**Happy Drug Discovery! 🧬💊**
