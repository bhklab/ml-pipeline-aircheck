from turtle import mode
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    balanced_accuracy_score,
    average_precision_score,
)
from rdkit import DataStructs
from rdkit.SimDivFilters import rdSimDivPickers
from sklearn.model_selection import train_test_split
from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import classification_coverage_score
import ast
from tqdm import tqdm
from src.data.data_reader import DataLoader
from src.data.make_dataset import DataProcessor
# from src.utils.model_utils111 import ModelTrainer
# from src.models.train_model import ModelTrainer
from src.models.nn_model import SimpleFNNClassifier, ConfigurableCNN1DClassifier
from tensorflow.keras.models import load_model

class ModelEvaluator:
    """A utility class for evaluating machine learning models and computing performance metrics."""

    @staticmethod
    def evaluate_model(model: object, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate a model on test data and compute performance metrics.

        Args:
            model: Trained model instance with predict and predict_proba methods.
            X_test: Test feature array.
            y_test: Test label array.

        Returns:
            Dictionary of evaluation metrics.

        Raises:
            ValueError: If inputs are invalid or evaluation fails.
        """
        # if not hasattr(model, 'predict') or not hasattr(model, 'predict_proba'):
        #     raise ValueError("Model must have predict and predict_proba methods")
        if not isinstance(X_test, np.ndarray) or not isinstance(y_test, np.ndarray):
            raise ValueError("X_test and y_test must be numpy arrays")
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError("X_test and y_test must have the same number of samples")
        if X_test.size == 0 or y_test.size == 0:
            raise ValueError("Input arrays cannot be empty")

        try:
       
            if isinstance(model, SimpleFNNClassifier) or isinstance(model, ConfigurableCNN1DClassifier):
             
            # if model_path.endswith('.h5'):
                y_proba = model.predict(X_test).flatten()
                y_pred = (y_proba > 0.5).astype(int)
            elif isinstance(model, str) and model.endswith(".h5"):

                model = load_model(model)
                y_proba = model.predict(X_test).flatten() 
                y_pred = (y_proba > 0.5).astype(int) 

            elif isinstance(model, str) and model.endswith(".pkl"):
                with open(model, 'rb') as f:
                    model = pickle.load(f)
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:, 1]
            else:
    
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
            return ModelEvaluator.calculate_metrics(X_test, y_test, y_pred, y_proba)
        except Exception as e:
            raise ValueError(f"Failed to evaluate model: {str(e)}")

    @staticmethod
    def calculate_metrics(X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
        """Calculate evaluation metrics for model predictions.

        Args:
            X_test: Test feature array.
            y_test: True labels.
            y_pred: Predicted labels.
            y_proba: Predicted probabilities for the positive class.

        Returns:
            Dictionary containing various evaluation metrics.

        Raises:
            ValueError: If inputs are invalid or metric calculation fails.
        """
        if not all(isinstance(arr, np.ndarray) for arr in [X_test, y_test, y_pred, y_proba]):
            raise ValueError("All inputs must be numpy arrays")
        if not (X_test.shape[0] == y_test.shape[0] == y_pred.shape[0] == y_proba.shape[0]):
            raise ValueError("All input arrays must have the same number of samples")

        try:
            y_test_array = np.array(y_test)
            y_pred_array = np.array(y_pred)
            y_proba_array = np.array(y_proba)

            ppv = precision_score(y_test_array, y_pred_array, zero_division=0)
            p_ppv = ModelEvaluator.plate_ppv(y_test_array, y_pred_array, top_n=128)
            # dp_ppv = ModelEvaluator.diverse_plate_ppv(y_test_array, y_pred_array, clusters=ModelEvaluator.cluster_leader_from_array(X_test).tolist())
            dp_ppv = -1  # Disabled due to performance concerns for large datasets

            hits_200, prec_200 = ModelEvaluator.hits_and_precision_at_k(y_test_array, y_pred_array, y_proba_array, k=200)
            hits_500, prec_500 = ModelEvaluator.hits_and_precision_at_k(y_test_array, y_pred_array, y_proba_array, k=500)
            total_hits = int(np.sum((y_test_array == 1) & (y_pred_array == 1)))

            return {
                "Accuracy": accuracy_score(y_test_array, y_pred_array),
                "Precision": ppv,
                "Recall": recall_score(y_test_array, y_pred_array, zero_division=0),
                "F1Score": f1_score(y_test_array, y_pred_array, zero_division=0),
                "AUC-ROC": roc_auc_score(y_test_array, y_proba_array),
                "AUC-PR": average_precision_score(y_test_array, y_proba_array),
                "MCC": matthews_corrcoef(y_test_array, y_pred_array),
                "Cohen Kappa": cohen_kappa_score(y_test_array, y_pred_array),
                "balanced_accuracy": balanced_accuracy_score(y_test_array, y_pred_array),
                "PlatePPV": p_ppv,
                "DivPlatePPV": dp_ppv,
                "HitsAt200": hits_200,
                "PrecisionAt200": prec_200,
                "HitsAt500": hits_500,
                "PrecisionAt500": prec_500,
                "TotalHits": total_hits
            }
        except Exception as e:
            raise ValueError(f"Failed to calculate metrics: {str(e)}")

    @staticmethod
    def hits_and_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray, k: int) -> tuple[int, float]:
        """Calculate hits and precision at top k predictions.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_scores: Predicted probabilities/scores.
            k: Number of top predictions to consider.

        Returns:
            Tuple of (hits, precision_at_k), where hits is the number of true positives in top k,
            and precision_at_k is the precision among predicted positives in top k.

        Raises:
            ValueError: If inputs are invalid or calculation fails.
        """
        if not all(isinstance(arr, np.ndarray) for arr in [y_true, y_pred, y_scores]):
            raise ValueError("All inputs must be numpy arrays")
        if not (y_true.shape[0] == y_pred.shape[0] == y_scores.shape[0]):
            raise ValueError("All input arrays must have the same number of samples")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")

        try:
            k = min(k, len(y_scores))  # Prevent index overflow
            top_k_idx = np.argsort(y_scores)[::-1][:k]
            top_k_true = y_true[top_k_idx]
            top_k_pred = y_pred[top_k_idx]
            hits = int(np.sum((top_k_true == 1) & (top_k_pred == 1)))
            predicted_positives = np.sum(top_k_pred == 1)
            precision_at_k = hits / predicted_positives if predicted_positives > 0 else 0.0
            return hits, precision_at_k
        except Exception as e:
            raise ValueError(f"Failed to calculate hits and precision at k={k}: {str(e)}")

    @staticmethod
    def plate_ppv(y: np.ndarray, y_pred: np.ndarray, top_n: int = 128) -> float:
        """Calculate Plate Positive Predictive Value (PPV) for top-n predictions.

        Args:
            y: True labels.
            y_pred: Predicted labels.
            top_n: Number of top predictions to consider (default: 128, typical for plate-based assays).

        Returns:
            PPV for the top-n predicted positives.

        Raises:
            ValueError: If inputs are invalid or calculation fails.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise ValueError("y and y_pred must be numpy arrays")
        if y.shape[0] != y_pred.shape[0]:
            raise ValueError("y and y_pred must have the same number of samples")
        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer")

        try:
            y = np.atleast_1d(y)
            y_pred = np.atleast_1d(y_pred)
            top_indices = y_pred.argsort()[::-1][:top_n]
            selected = np.vstack((y, y_pred)).T[top_indices]
            selected = selected[selected[:, 1] > 0.5]
            return np.sum(selected[:, 0]) / len(selected) if len(selected) > 0 else 0.0
        except Exception as e:
            raise ValueError(f"Failed to calculate Plate PPV: {str(e)}")

    @staticmethod
    def diverse_plate_ppv(y: np.ndarray, y_pred: np.ndarray, clusters: list, top_n_per_group: int = 15) -> float:
        """Calculate Diverse Plate PPV across clusters.

        Args:
            y: True labels.
            y_pred: Predicted labels.
            clusters: List of cluster IDs for each sample.
            top_n_per_group: Number of top predictions per cluster (default: 15).

        Returns:
            Mean PPV across clusters for top-n predictions per group.

        Raises:
            ValueError: If inputs are invalid or calculation fails.
        """
        if not isinstance(y, np.ndarray) or not isinstance(y_pred, np.ndarray):
            raise ValueError("y and y_pred must be numpy arrays")
        if not isinstance(clusters, list):
            raise ValueError("clusters must be a list")
        if y.shape[0] != y_pred.shape[0] or y.shape[0] != len(clusters):
            raise ValueError("y, y_pred, and clusters must have the same length")
        if not isinstance(top_n_per_group, int) or top_n_per_group < 1:
            raise ValueError("top_n_per_group must be a positive integer")

        try:
            df = pd.DataFrame({"pred": y_pred, "real": y, "CLUSTER_ID": clusters})
            df_groups = df.groupby("CLUSTER_ID")
            ppv_values = []

            for _, idx in df_groups.groups.items():
                group_data = df.iloc[idx].copy()
                if sum(group_data["pred"] > 0.5) == 0:
                    continue
                group_data = group_data[group_data["pred"] > 0.5]
                top_n = np.vstack((group_data["real"].to_numpy(), group_data["pred"].to_numpy())).T
                top_n = top_n[top_n[:, 1].argsort()[::-1]][:top_n_per_group]
                ppv = np.sum(top_n[:, 0]) / len(top_n) if len(top_n) > 0 else 0.0
                ppv_values.append(ppv)

            return np.mean(ppv_values) if ppv_values else 0.0
        except Exception as e:
            raise ValueError(f"Failed to calculate Diverse Plate PPV: {str(e)}")

    @staticmethod
    def cluster_leader_from_array(X: np.ndarray, thresh: float = 0.65) -> np.ndarray:
        """Generate cluster IDs for an array using Tanimoto similarity-based leader clustering.

        Args:
            X: Feature array (numeric).
            thresh: Tanimoto similarity threshold for clustering (default: 0.65).

        Returns:
            Array of cluster IDs.

        Raises:
            ValueError: If inputs are invalid or clustering fails.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array")
        if not isinstance(thresh, float) or not 0 <= thresh <= 1:
            raise ValueError("thresh must be a float between 0 and 1")
        if X.size == 0:
            raise ValueError("Input array X cannot be empty")

        try:
            X = np.array(X, dtype=float)
            fps = [DataStructs.CreateFromBitString(
                "".join(["1" if val > 0 else "0" for val in row])) for row in X]
            lp = rdSimDivPickers.LeaderPicker()
            centroids = lp.LazyBitVectorPick(fps, len(fps), thresh)
            centroid_fps = [fps[i] for i in centroids]
            cluster_ids = [np.argmax(DataStructs.BulkTanimotoSimilarity(fp, centroid_fps)) for fp in fps]
            return np.array(cluster_ids, dtype=int)
        except Exception as e:
            raise ValueError(f"Failed to perform leader clustering: {str(e)}")

    @staticmethod
    def compute_conformal_prediction(
        row: pd.Series,
        nrows_train: int | None,
        feature_fusion_method: str | None,
        X_test_array: np.ndarray,
        Y_test_array: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        from src.models.train_model import ModelTrainer
        """Compute conformal prediction metrics for a model.

        Args:
            get_model: Function to initialize a model instance.
            train_model: Function to train a model.
            load_data: Function to load training data.
            fuse_columns: Function to fuse columns.
            row: DataFrame row containing model metadata (ModelType, ColumnName, etc.).
            nrows_train: Number of rows to load from training data, or None for all rows.
            feature_fusion_method: Feature fusion method ('all', 'pairwise', or None).
            X_test_array: Test feature array.
            Y_test_array: Test label array.

        Returns:
            Tuple of (coverage_score, confidence_score, prediction_sets).

        Raises:
            ValueError: If inputs are invalid or conformal prediction fails.
        """
        if not isinstance(row, pd.Series):
            raise ValueError("row must be a pandas Series")
        if not all(isinstance(arr, np.ndarray) for arr in [X_test_array, Y_test_array]):
            raise ValueError("X_test_array and Y_test_array must be numpy arrays")
        if X_test_array.shape[0] != Y_test_array.shape[0]:
            raise ValueError("X_test_array and Y_test_array must have the same number of samples")
        if nrows_train is not None and (not isinstance(nrows_train, int) or nrows_train <= 0):
            raise ValueError("nrows_train must be a positive integer or None")
        if feature_fusion_method and feature_fusion_method.lower() not in ['all', 'pairwise', 'none']:
            raise ValueError("feature_fusion_method must be 'all', 'pairwise', 'none', or None")

        try:
            model_name = row["ModelType"]
            column_names = [row["ColumnName"]]
            label_column_train = row["label_column_train"]
            conformal_test_size = row["confromal_test_size"]
            conformal_confidence_level = row["confromal_confidence_level"]
            train_path = row["train_path"]
            model_parameters = ast.literal_eval(row["UsedHyperParameters"]) if isinstance(row["UsedHyperParameters"], str) else row["UsedHyperParameters"]

            if not isinstance(model_parameters, dict):
                raise ValueError("UsedHyperParameters must be a dictionary or valid string representation")

            X_train, Y_train = DataLoader.load_data(train_path, column_names, label_column_train, nrows_train)
            Y_train_array = np.stack(Y_train.iloc[:, 0])
            X_train, _ = DataProcessor.fuse_columns(X_train, column_names, feature_fusion_method)
            X_train_array = np.stack(X_train[column_names[0]])

            X_train_model, X_calib, y_train_model, y_calib = train_test_split(
                X_train_array, Y_train_array, test_size=conformal_test_size, random_state=42
            )

            base_model = ModelTrainer.get_model(model_name, model_parameters)
            base_model = ModelTrainer.train_model(base_model, X_train_model, y_train_model)

            mapie = SplitConformalClassifier(
                estimator=base_model,
                confidence_level=conformal_confidence_level,
                prefit=True
            )
            mapie.conformalize(X_calib, y_calib)
            y_pred, y_pred_set = mapie.predict_set(X_test_array)

            coverage_score = classification_coverage_score(Y_test_array, y_pred_set)
            avg_set_size = y_pred_set.sum(axis=1).mean()
            confidence_score = 2 - avg_set_size  # Assumes binary classification (K=2)

            return coverage_score, confidence_score, y_pred_set
        except Exception as e:
            raise ValueError(f"Failed to compute conformal prediction: {str(e)}")

    @staticmethod
    def test_pipeline(
        RunFolderName: str | Path,
        config: dict | None = None
    ) -> None:
        """Execute a test pipeline for evaluating models on test data.

        Args:
            Test: Flag to enable testing ('y' or 'n').
            test_paths: List of paths to test parquet files.
            column_names: List of feature column names.
            label_column_test: Name of the label column in test data.
            nrows_test: Number of rows to load from test data, or None for all rows.
            nrows_train: Number of rows to load from training data for conformal prediction, or None.
            feature_fusion_method: Feature fusion method ('all', 'pairwise', or None).
            conformal_prediction: Flag to enable conformal prediction ('y' or 'n').
            conformal_test_size: Fraction of training data for calibration in conformal prediction.
            conformal_confidence_level: Confidence level for conformal prediction.
            RunFolderName: Folder containing results.csv and saved models.
            load_data: Function to load data.
            fuse_columns: Function to fuse columns.
            evaluate_model: Function to evaluate a model.
            get_model: Function to initialize a model.
            train_model: Function to train a model.
            config: Optional configuration dictionary to override parameters.

        Returns:
            None

        Raises:
            ValueError: If inputs are invalid or pipeline execution fails.
            FileNotFoundError: If results.csv or model files are missing.
        """
        if config:
            is_test = config.get('is_test')
            test_paths = config.get('test_data')
            column_names = config.get('desired_columns')
            label_column_test = config.get('label_column_test')
            nrows_test = config.get('nrows_test')
            nrows_train = config.get('nrows_train')
            feature_fusion_method = config.get('feature_fusion_method')
            conformal_prediction = config.get('conformal_prediction')
            conformal_test_size = config.get('confromal_test_size')
            confromal_confidence_level = config.get('confromal_confidence_level')
            

        if not isinstance(is_test, bool) or is_test is None:
            raise ValueError("Test must be a boolean value")
        if not test_paths or not isinstance(test_paths, list):
            raise ValueError("test_paths must be a non-empty list")
        if not column_names or not isinstance(column_names, list):
            raise ValueError("column_names must be a non-empty list")
        if not isinstance(label_column_test, str):
            raise ValueError("label_column_test must be a string")
        if nrows_test is not None and (not isinstance(nrows_test, int) or nrows_test <= 0):
            raise ValueError("nrows_test must be a positive integer or None")
        if nrows_train is not None and (not isinstance(nrows_train, int) or nrows_train <= 0):
            raise ValueError("nrows_train must be a positive integer or None")
        if feature_fusion_method and feature_fusion_method.lower() not in ['all', 'pairwise', 'none']:
            raise ValueError("feature_fusion_method must be 'all', 'pairwise', 'none', or None")
        if not isinstance(conformal_prediction, bool) or conformal_prediction is None:
            raise ValueError("conformal_prediction must be a boolean value")
        if not isinstance(conformal_test_size, float) or not 0 < conformal_test_size < 1:
            raise ValueError("conformal_test_size must be a float between 0 and 1")
        if not isinstance(confromal_confidence_level, float) or not 0 < confromal_confidence_level < 1:
            raise ValueError("conformal_confidence_level must be a float between 0 and 1")
        if not isinstance(RunFolderName, (str, Path)):
            raise ValueError("RunFolderName must be a string or Path")

        if not is_test:
            print("Test pipeline skipped: Test flag is 'n' or 'N'.")
            return

        RunFolderName = Path(RunFolderName)
        results_path = RunFolderName / "results.csv"

        if not results_path.exists():
            raise FileNotFoundError("results.csv not found. Ensure the training phase was run.")

        try:
            df = pd.read_csv(results_path)
            updated_rows = []

            for test_path in test_paths:
                test_path = Path(test_path)
                if not test_path.exists():
                    raise FileNotFoundError(f"Test file not found: {test_path}")

                if feature_fusion_method and feature_fusion_method.lower() != "none":
                    X_test, Y_test = DataLoader.load_data(test_path, column_names, label_column_test, nrows_test)
                    #Y_test_array = np.stack(Y_test.iloc[:, 0])
                    X_test, fused_column_names = DataProcessor.fuse_columns(X_test, column_names, feature_fusion_method)
                else:
                    fused_column_names = None

                for rowcount, row in df.iterrows():
                    # model_path = Path(row["ModelPath"]) / "model.pkl" if os.path.isdir(row["ModelPath"]) else Path(row["ModelPath"])
                    model_path = row["ModelPath"]
                    model_name = row["ModelType"]
                    tf_models = config.get("tf_models")
                    tf_dnn = True if model_name in tf_models else False
                    column_name = row["ColumnName"]
                    if os.path.isdir(model_path):
                        if tf_dnn:
                            model_path = os.path.join(model_path, "model.h5")
                        else:
                            model_path = os.path.join(model_path, "model.pkl")

                    if not fused_column_names:
                        X_test, Y_test = DataLoader.load_data(test_path, [column_name], label_column_test, nrows_test)
                        #Y_test_array = np.stack(Y_test.iloc[:, 0])

                    if column_name not in X_test.columns:
                        print(f"Column '{column_name}' not in test file: {test_path}. Skipping.")
                        continue

                    # if not model_path.exists():
                    #     raise FileNotFoundError(f"Model file not found at: {model_path}")

                    # with open(model_path, 'rb') as f:
                    #     model = pickle.load(f)
                    
                    Y_test_array = np.stack(Y_test.iloc[:, 0])
                    X_test_array = np.stack(X_test[column_name])
                    test_metrics = ModelEvaluator.evaluate_model(model_path, X_test_array, Y_test_array)

                    row_result = row.copy()
                    if conformal_prediction:
                        coverage_score, confidence_score, _ = ModelEvaluator.compute_conformal_prediction(
                            row,
                            nrows_train, feature_fusion_method, X_test_array, Y_test_array
                        )
                        row_result["confromal_coverage_score"] = coverage_score
                        row_result["confromal_confidence_score"] = confidence_score

                    for key, value in test_metrics.items():
                        row_result[f"Test_{key}"] = value

                    row_result["TestFile"] = test_path.name
                    updated_rows.append(row_result)

            df_updated = pd.DataFrame(updated_rows)
            df_updated.to_csv(results_path, index=False)
        except Exception as e:
            raise ValueError(f"Failed to execute test pipeline: {str(e)}")