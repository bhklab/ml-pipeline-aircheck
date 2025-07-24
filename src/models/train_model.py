
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

from src.data.data_reader import DataLoader
from src.data.make_dataset import DataProcessor
from src.utils.config_utils import MLPipelineConfig

from src.models.eval_model import ModelEvaluator
from src.models.nn_model import SimpleFNNClassifier, ConfigurableCNN1DClassifier

warnings.filterwarnings("ignore")


class ModelTrainer:
    """A utility class for training and evaluating machine learning models."""

    def get_model(self, model_name: str, best_params: dict | None = None) -> object:
        """Initialize a machine learning model with optional parameters.

        Args:
            model_name: Name of the model to initialize (e.g., 'rf', 'lr', 'svc').
            best_params: Optional dictionary of model parameters. If None, uses default parameters.

        Returns:
            Initialized model instance.

        Raises:
            ValueError: If the model_name is unsupported.
        """
        if not isinstance(model_name, str):
            raise ValueError("model_name must be a string")
        if best_params is not None and not isinstance(best_params, dict):
            raise ValueError("best_params must be a dictionary or None")

        best_params = best_params or {}
        model_name = model_name.lower()

        model_configs = {
            'rf': (RandomForestClassifier, {}),
            'lr': (LogisticRegression, {}),
            'sgd': (SGDClassifier, {'loss': 'log_loss'}),
            'svc': (SVC, {'probability': True}),
            'nb': (GaussianNB, {}),
            'dt': (DecisionTreeClassifier, {}),
            'knn': (KNeighborsClassifier, {}),
            'gb': (GradientBoostingClassifier, {}),
            'ada': (AdaBoostClassifier, {}),
            'bag': (BaggingClassifier, {}),
            'mlp': (MLPClassifier, {}),
            'lgbm': (LGBMClassifier, {}),
            'catboost': (CatBoostClassifier, {'silent': True}),
            'tf_ff': (SimpleFNNClassifier, {'epochs': 50, 'batch_size': 32}),
            'tf_cnn1d': (ConfigurableCNN1DClassifier, {'conv_layers': [(32, 3), (64, 3)], 'ff_layers': [64, 32], 'epochs': 100}),
        }

        if model_name not in model_configs:
            raise ValueError(f"Unsupported model: {model_name}")

        model_class, default_params = model_configs[model_name]
        params = {**default_params, **best_params}
        try:
            return model_class(**params)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize {model_name} with parameters {params}: {str(e)}")

    def train_model(self, model: object, X_train: np.ndarray, y_train: np.ndarray) -> object:
        """Train a machine learning model on the provided data.

        Args:
            model: Model instance to train.
            X_train: Training feature array.
            y_train: Training label array.

        Returns:
            Trained model instance.

        Raises:
            ValueError: If input arrays are invalid or training fails.
        """
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("X_train and y_train must be numpy arrays")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must have the same number of samples")
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("Input arrays cannot be empty")

        try:

            model.fit(X_train, y_train)
            return model
        except Exception as e:
            raise ValueError(f"Failed to train model: {str(e)}")

    # @staticmethod
    def cross_validate_and_save_models(
        self,
        X_train_array: np.ndarray,
        Y_train_array: np.ndarray,
        model_name: str,
        model_subfolder: str | Path,
        Nfold: int,
        best_params: dict | None = None,
        tf_dnn: bool = False
    ) -> dict:
        """Perform cross-validation, save fold models, and compute average metrics.

        Args:
            X_train_array: Training feature array.
            Y_train_array: Training label array.
            model_name: Name of the model to train.
            model_subfolder: Folder path to save fold models.
            Nfold: Number of cross-validation folds.
            get_model: Function to initialize a model instance.
            train_model: Function to train a model.
            evaluate_model: Function to evaluate a model.
            best_params: Optional dictionary of model parameters.

        Returns:
            Dictionary of average evaluation metrics across folds.

        Raises:
            ValueError: If inputs are invalid or cross-validation fails.
            FileNotFoundError: If model_subfolder cannot be created.
        """
        if not isinstance(X_train_array, np.ndarray) or not isinstance(Y_train_array, np.ndarray):
            raise ValueError(
                "X_train_array and Y_train_array must be numpy arrays")
        if X_train_array.shape[0] != Y_train_array.shape[0]:
            raise ValueError(
                "X_train_array and Y_train_array must have the same number of samples")
        if not isinstance(Nfold, int) or Nfold < 2:
            raise ValueError("Nfold must be an integer >= 2")
        if not isinstance(model_subfolder, (str, Path)):
            raise ValueError("model_subfolder must be a string or Path")

        model_subfolder = Path(model_subfolder)
        os.makedirs(model_subfolder, exist_ok=True)

        try:
            skf = StratifiedKFold(
                n_splits=Nfold, shuffle=True, random_state=42)
            fold_metrics = []

            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_train_array, Y_train_array)):
                X_train_fold, X_test_fold = X_train_array[train_idx], X_train_array[test_idx]
                y_train_fold, y_test_fold = Y_train_array[train_idx], Y_train_array[test_idx]

                model_fold = self.get_model(model_name, best_params)
                model_fold = self.train_model(
                    model_fold, X_train_fold, y_train_fold)

                fold_model_path = os.path.join(
                    model_subfolder, f"model_fold{fold_idx + 1}")

                if tf_dnn:

                    os.makedirs(os.path.dirname(
                        fold_model_path), exist_ok=True)
                    model_fold.save_model(fold_model_path)
                else:

                    fold_model_path = f"{fold_model_path}/model.pkl"
                    os.makedirs(os.path.dirname(
                        fold_model_path), exist_ok=True)
                    with open(fold_model_path, 'wb') as f:
                        pickle.dump(model_fold, f)
                metrics = ModelEvaluator.evaluate_model(
                    model_fold,  X_test_fold, y_test_fold)

                fold_metrics.append(metrics)
            avg_metrics = {metric: np.mean(
                [fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0]}
            return avg_metrics
        except Exception as e:
            raise ValueError(f"Failed to perform cross-validation: {str(e)}")

    def train_and_save_final_model(
        self,
        config: dict,
        X_train_array: np.ndarray,
        Y_train_array: np.ndarray,
        model_name: str,
        model_subfolder: str | Path,
        best_params: dict | None = None
    ) -> None:
        """Train a final model and save it to the specified folder.

        Args:
            X_train_array: Training feature array.
            Y_train_array: Training label array.
            model_name: Name of the model to train.
            model_subfolder: Folder path to save the model.
            get_model: Function to initialize a model instance.
            train_model: Function to train a model.
            best_params: Optional dictionary of model parameters.

        Returns:
            None

        Raises:
            ValueError: If inputs are invalid or training fails.
            FileNotFoundError: If model_subfolder cannot be created.
        """
        if not isinstance(X_train_array, np.ndarray) or not isinstance(Y_train_array, np.ndarray):
            raise ValueError(
                "X_train_array and Y_train_array must be numpy arrays")
        if X_train_array.shape[0] != Y_train_array.shape[0]:
            raise ValueError(
                "X_train_array and Y_train_array must have the same number of samples")
        if not isinstance(model_subfolder, (str, Path)):
            raise ValueError("model_subfolder must be a string or Path")
        tf_models = config.get("tf_models")
        tf_dnn = True if model_name in tf_models else False

        # model_subfolder = Path(model_subfolder)
        # os.makedirs(model_subfolder, exist_ok=True)
        model_path = os.path.join(model_subfolder, "model")

        try:
            model = self.get_model(model_name, best_params)
            model = self.train_model(model, X_train_array, Y_train_array)
            if tf_dnn:
                model.save_model(model_path)
            else:
                model_path = model_path + ".pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
        except Exception as e:
            raise ValueError(f"Failed to train and save final model: {str(e)}")

    def bayesian_hyperparameter_search(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        cv: int = 5,
        n_iter: int = 32,
        random_state: int = 42
    ) -> dict:
        """Perform Bayesian hyperparameter optimization for a specified model.

        Args:
            model_name: Name of the model to optimize.
            X_train: Training feature array.
            y_train: Training label array.
            cv: Number of cross-validation folds.
            n_iter: Number of parameter settings to sample.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary of best hyperparameters.

        Raises:
            ValueError: If inputs are invalid, model is unsupported, or search fails.
        """
        if not isinstance(X_train, np.ndarray) or not isinstance(y_train, np.ndarray):
            raise ValueError("X_train and y_train must be numpy arrays")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must have the same number of samples")
        if not isinstance(cv, int) or cv < 2:
            raise ValueError("cv must be an integer >= 2")
        if not isinstance(n_iter, int) or n_iter < 1:
            raise ValueError("n_iter must be a positive integer")

        param_spaces = {
            'rf': {'n_estimators': (100, 200), 'max_depth': (5, 15)},
            'lr': {'C': (0.01, 10.0, 'log-uniform'), 'penalty': ['l2']},
            'ridge': {'alpha': (0.01, 10.0, 'log-uniform')},
            'sgd': {'alpha': (1e-5, 1e-2, 'log-uniform'), 'penalty': ['l2', 'l1']},
            'perceptron': {'penalty': ['l2', 'l1', None], 'alpha': (1e-5, 1e-2, 'log-uniform')},
            'svc': {'C': (0.1, 10.0, 'log-uniform'), 'kernel': ['linear', 'rbf']},
            'nb': {'var_smoothing': (1e-11, 1e-8, 'log-uniform')},
            'dt': {'max_depth': (3, 10), 'min_samples_split': (2, 10)},
            'knn': {'n_neighbors': (3, 15), 'weights': ['uniform', 'distance']},
            'gb': {'n_estimators': (100, 200), 'learning_rate': (0.01, 0.3, 'log-uniform'), 'max_depth': (3, 7)},
            'ada': {'n_estimators': (50, 150), 'learning_rate': (0.5, 1.5, 'uniform')},
            'bag': {'n_estimators': (10, 50), 'max_samples': (0.5, 1.0, 'uniform')},
            'mlp': {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'activation': ['relu', 'tanh'], 'alpha': (1e-5, 1e-2, 'log-uniform')},
            'lgbm': {},
            'catboost': {},
            'tf_ff': {},
            'tf_cnn1D': {},

        }

        model_name = model_name.lower()
        if model_name not in param_spaces:
            raise ValueError(f"Unsupported model for tuning: {model_name}")
        if not param_spaces[model_name]:
            print(
                f"Bayesian hyperparameter search is not available for {model_name} in this pipeline.")
            return {}

        try:
            model = self.get_model(model_name)
            search = BayesSearchCV(
                estimator=model,
                search_spaces=param_spaces[model_name],
                n_iter=n_iter,
                cv=cv,
                random_state=random_state
            )
            search.fit(X_train, y_train)
            return search.best_params_
        except Exception as e:
            raise ValueError(
                f"Failed to perform hyperparameter search for {model_name}: {str(e)}")

    def train_pipeline(
        self,
        RunFolderName: str,
        config: dict | None = None,
        train_paths: list[str] = None,
    ) -> None:
        """Execute a training pipeline for multiple models and datasets.

        Args:
            Train: Flag to enable training ('y' or 'n').
            train_paths: List of paths to training parquet files.
            column_names: List of feature column names.
            label_column_train: Name of the label column.
            nrows_train: Number of rows to load from training data, or None for all rows.
            model_names: List of model names to train.
            hyperparameters_tuning: Flag to enable hyperparameter tuning ('y' or 'n').
            RunFolderName: Folder to save models and results.
            feature_fusion_method: Feature fusion method ('all', 'pairwise', or None).
            load_data: Function to load data.
            fuse_columns: Function to fuse columns.
            get_model: Function to initialize a model.
            train_model: Function to train a model.
            cross_validate_and_save_models: Function for cross-validation.
            train_and_save_final_model: Function to train and save final model.
            bayesian_hyperparameter_search: Function for hyperparameter search.
            write_results_csv: Function to write results to CSV.
            hyperparameters: Optional dictionary of model hyperparameters.
            Nfold: Number of cross-validation folds.
            config: Optional configuration dictionary to override parameters.

        Returns:
            None

        Raises:
            ValueError: If inputs are invalid or pipeline execution fails.
            FileNotFoundError: If any training file does not exist.
        """
        if config:
            is_train = config.get('is_train')
            # train_paths = config.get('train_data')
            column_names = config.get('desired_columns')
            label_column_train = config.get('label_column_train', "LABEL")
            nrows_train = config.get('nrows_train')
            model_names = config.get('desired_models')
            hyperparameters_tuning = config.get(
                'hyperparameters_tuning', False)
            hyperparameters = config.get('hyperparameters')
            Nfold = config.get('Nfold')
            feature_fusion_method = config.get('feature_fusion_method')
            tf_models = config.get('tf_models')
            tf_dnn = True if model_names in tf_models else False
            tf_dnn = any(elem in model_names for elem in tf_models)

        utils_class = MLPipelineConfig()

        if not isinstance(is_train, bool) or is_train is None:
            raise ValueError("Train must be true or false")
        if not train_paths or not isinstance(train_paths, list):
            raise ValueError("train_paths must be a non-empty list")
        if not column_names or not isinstance(column_names, list):
            raise ValueError("column_names must be a non-empty list")
        if not isinstance(label_column_train, str):
            raise ValueError("label_column_train must be a string")
        if nrows_train is not None and (not isinstance(nrows_train, int) or nrows_train <= 0):
            raise ValueError("nrows_train must be a positive integer or None")
        if not model_names or not isinstance(model_names, list):
            raise ValueError("model_names must be a non-empty list")
        if not isinstance(hyperparameters_tuning, bool) or hyperparameters_tuning is None:
            raise ValueError("hyperparameters_tuning must be True or False")
        if not isinstance(RunFolderName, (str, Path)):
            raise ValueError("RunFolderName must be a string or Path")
        if feature_fusion_method and feature_fusion_method.lower() not in ['all', 'pairwise', 'none']:
            raise ValueError(
                "feature_fusion_method must be 'all', 'pairwise', 'none', or None")
        if not isinstance(Nfold, int) or Nfold < 2:
            raise ValueError("Nfold must be an integer >= 2")

        if not is_train:
            print(
                "Training is disabled in the configuration. Skipping training pipeline.")
            return
        RunFolderName = Path(RunFolderName)
        os.makedirs(RunFolderName, exist_ok=True)

        try:
            for train_path in train_paths:
                train_path = Path(train_path)

                if not train_path.exists():
                    raise FileNotFoundError(
                        f"Training file not found: {train_path}")

                train_filename = train_path.stem
                total_columns = column_names

                if feature_fusion_method and feature_fusion_method.lower() != "none":
                    X_train, Y_train = DataLoader.load_data(
                        train_path, column_names, label_column_train, nrows_train)

                    if Y_train.empty:
                        raise ValueError(
                            f"No data found in the label column '{label_column_train}' of the training file: {train_path}")
                    if not isinstance(Y_train, pd.DataFrame):
                        raise ValueError(
                            f"Y_train must be a pandas DataFrame, got {type(Y_train)}")
                    Y_train_array = np.stack(Y_train.iloc[:, 0])
                    X_train, fused_column_names = DataProcessor.fuse_columns(
                        X_train, column_names, feature_fusion_method)
                    total_columns = fused_column_names
                else:
                    fused_column_names = None

                for model_name in model_names:
                    for column_name in total_columns:
                        if not fused_column_names:
                            X_train, Y_train = DataLoader.load_data(
                                train_path, [column_name], label_column_train, nrows_train)

                            if Y_train.empty:
                                raise ValueError(
                                    f"No data found in the label column '{label_column_train}' of the training file: {train_path}")
                            if not isinstance(Y_train, pd.DataFrame):
                                raise ValueError(
                                    f"Y_train must be a pandas DataFrame, got {type(Y_train)}")
                            Y_train_array = np.stack(Y_train.iloc[:, 0])

                        X_train_array = np.stack(X_train[column_name])
                        model_subfolder = RunFolderName / \
                            f"{train_filename}_{model_name}_{column_name}"
                        os.makedirs(model_subfolder, exist_ok=True)

                        best_params = (
                            self.bayesian_hyperparameter_search(
                                model_name, X_train_array, Y_train_array)
                            if hyperparameters_tuning
                            else hyperparameters.get(model_name, {})
                        )

                        avg_metrics = self.cross_validate_and_save_models(
                            X_train_array=X_train_array,
                            Y_train_array=Y_train_array,
                            model_name=model_name,
                            model_subfolder=model_subfolder,
                            Nfold=Nfold,
                            best_params=best_params,
                            tf_dnn=tf_dnn,
                        )

                        self.train_and_save_final_model(
                            config,
                            X_train_array,
                            Y_train_array,
                            model_name,
                            model_subfolder,
                            best_params
                        )
                        experiment_results = config.copy()
                        experiment_results["train_path"] = train_path
                        experiment_results["TrainFileName"] = train_filename
                        experiment_results["ModelType"] = model_name
                        experiment_results["ColumnName"] = column_names
                        experiment_results["ModelPath"] = model_subfolder
                        experiment_results["UsedHyperParameters"] = best_params

                        for key, value in avg_metrics.items():
                            experiment_results[f"CV_{key}"] = value
                        utils_class.write_results_csv(
                            experiment_results, RunFolderName)
        except Exception as e:
            raise ValueError(f"Failed to execute training pipeline: {str(e)}")
