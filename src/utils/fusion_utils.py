import pandas as pd
import os
import warnings
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from src.models.eval_model import ModelEvaluator
from src.data.data_reader import DataLoader
from src.data.make_dataset import DataProcessor

from tensorflow.keras.models import load_model


class ModelMetricsVisualizer:
    """Handles visualization of model metrics."""

    @staticmethod
    def plot_radar_chart(df: pd.DataFrame,
                         metric_columns: List[str],
                         save_path: str = "") -> None:
        """
        Plots a radar chart for model metrics.

        Args:
            df: DataFrame containing model evaluation metrics
            metric_columns: List of metric column names to plot
            save_path: Path to save the radar chart image
        """
        # Calculate mean values for each metric
        metric_values = df[metric_columns].mean().values

        # Make the values a complete loop for radar plot
        values = list(metric_values) + [metric_values[0]]

        # Radar chart setup
        angles = np.linspace(
            0, 2 * np.pi, len(metric_columns), endpoint=False).tolist()
        angles += angles[:1]  # Closing the loop for the radar chart

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='blue', alpha=0.2)
        ax.plot(angles, values, color='blue', linewidth=2)

        # Add labels
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_columns, fontsize=12)
        ax.set_title("Model Metrics", fontsize=16, fontweight='bold')

        # Save the plot if save_path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"Radar chart saved at {save_path}")

        plt.show()


class ModelSelector:
    """Handles model selection and evaluation."""

    def __init__(self, config: Dict[str, Any], run_folder_name: str):
        """
        Initialize ModelSelector with configuration and run folder.

        Args:
            config: Configuration dictionary
            run_folder_name: Name of the run folder
        """
        self.config = config
        self.run_folder_name = Path(run_folder_name)
        self.results_csv_path = self.run_folder_name / "results.csv"
        self.output_txt_path = self.run_folder_name / "BestModels.txt"
        self.best_models_folder = self.run_folder_name / "BestModels"

        # Extract configuration parameters
        self.trainfile_for_modelselection = config.get(
            "trainfile_for_modelselection")
        self.evaluationfile_for_modelselection = config.get(
            "evaluationfile_for_modelselection")
        self.evaluation_column = config.get("evaluation_column")
        self.crossvalidation_column = config.get("crossvalidation_column")
        self.num_top_models = config.get("num_top_models", 1)
        self.fusion = config.get("Fusion", False)
        self.tf_models = config.get("tf_models", [])

        # Adjust num_top_models based on fusion setting
        if not self.fusion:
            print("Fusion is disabled. Setting num_top_models to 1.")
            self.num_top_models = 1

    def _load_and_filter_data(self) -> pd.DataFrame:
        """Load and filter the results data."""
        if not self.results_csv_path.exists():
            raise FileNotFoundError(
                f"Results file not found: {self.results_csv_path}")

        df = pd.read_csv(self.results_csv_path)

        # Drop duplicates excluding Date and Time columns (if they exist)
        drop_cols = [col for col in df.columns if col not in ['Date', 'Time']]
        df = df.drop_duplicates(subset=drop_cols)

        # Filter by TrainFile if specified
        if self.trainfile_for_modelselection:
            df = df[df['TrainFile'] == self.trainfile_for_modelselection]

        # Filter by TestFile if specified
        if self.evaluationfile_for_modelselection:
            df = df[df['TestFile'] == self.evaluationfile_for_modelselection]

        return df

    def _sort_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort models based on evaluation criteria."""
        # Sort by specified metric columns in descending order
        if self.evaluation_column and all(col in df.columns for col in self.evaluation_column):
            df_sorted = df.sort_values(by=self.evaluation_column, ascending=[
                                       False] * len(self.evaluation_column))
        elif self.crossvalidation_column and all(col in df.columns for col in self.crossvalidation_column):
            print('Sorting based on cross-validation metrics')
            df_sorted = df.sort_values(by=self.crossvalidation_column, ascending=[
                                       False] * len(self.crossvalidation_column))
        else:
            df_sorted = df  # Keep df as is without sorting

        return df_sorted

    def _get_unique_models(self, df_sorted: pd.DataFrame) -> pd.DataFrame:
        """Get unique models and handle insufficient model count."""
        # Drop duplicates based on ModelPath
        df_sorted_unique = df_sorted.drop_duplicates(subset=['ModelPath'])

        # Save the sorted unique DataFrame for further analysis
        sorted_unique_csv_path = self.run_folder_name / "SortedUniqueResults.csv"
        df_sorted_unique.to_csv(sorted_unique_csv_path, index=False)

        # Check available distinct models and warn if fewer than num_top_models
        available_models = len(df_sorted_unique)
        if available_models < self.num_top_models:
            warnings.warn(
                f"Only {available_models} distinct models available. Returning all.")
            self.num_top_models = available_models

        return df_sorted_unique

    def _save_best_models(self, best_models: pd.DataFrame) -> None:
        """Save best models to the BestModels folder."""
        # Create BestModels folder
        self.best_models_folder.mkdir(exist_ok=True)

        for idx, row in best_models.iterrows():
            path = Path(row['ModelPath'])
            model_name = path.name
            model_type = row.get("ModelType", "")

            # Determine if it's a TensorFlow model
            tf_dnn = model_type in self.tf_models

            # Determine model file extension
            model_file = path / ("model.h5" if tf_dnn else "model.pkl")

            if model_file.exists():
                # Create the new name with the folder name prefix
                extension = ".h5" if tf_dnn else ".pkl"
                new_model_name = f"{model_name}_model{extension}"
                model_save_path = self.best_models_folder / new_model_name

                # Copy the model file with the new name
                shutil.copy(str(model_file), str(model_save_path))
                print(f"Saved: {model_save_path}")
            else:
                print(f"Model file not found in {path}. Skipping.")

    def _generate_radar_chart(self, best_models: pd.DataFrame) -> None:
        """Generate radar chart for top models."""
        radar_columns = ['CV_F1Score', 'CV_Precision', 'CV_Recall',
                         'CV_Accuracy', 'CV_PlatePPV', 'CV_DivPlatePPV']

        if all(col in best_models.columns for col in radar_columns):
            # Calculate mean values for the specified evaluation columns
            radar_data = best_models[radar_columns].mean().to_dict()

            # Remove "Test_" prefix from the metric names for the radar plot
            radar_data_clean = {col.replace(
                "Test_", ""): value for col, value in radar_data.items()}
            radar_df = pd.DataFrame([radar_data_clean])

            radar_chart_path = self.run_folder_name / "RadarChart_TopModels.png"
            ModelMetricsVisualizer.plot_radar_chart(
                radar_df,
                list(radar_data_clean.keys()),
                save_path=str(radar_chart_path)
            )

    def select_best_models(self) -> pd.DataFrame:
        """
        Select the best models based on evaluation criteria.

        Returns:
            DataFrame containing the best models
        """
        if not self.evaluation_column:
            raise ValueError(
                "You must provide evaluation_column in the configuration")

        # Load and filter data
        df = self._load_and_filter_data()

        # Sort models
        df_sorted = self._sort_models(df)

        # Get unique models
        df_sorted_unique = self._get_unique_models(df_sorted)

        # Take top k distinct models
        best_models = df_sorted_unique.head(self.num_top_models)
        print("BEST MODELS:", best_models["ModelPath"].tolist())

        # Save best models results
        best_models_results_path = self.run_folder_name / "BestModelsResults.csv"
        best_models.to_csv(best_models_results_path, index=False)

        # Write model paths to output txt file
        with open(self.output_txt_path, 'w') as f:
            for path in best_models['ModelPath']:
                f.write(f"{path}\n")

        # Save best models to BestModels folder
        self._save_best_models(best_models)

        # Generate radar chart
        self._generate_radar_chart(best_models)

        print(
            f"Top {self.num_top_models} distinct model paths written to {self.output_txt_path}")

        return best_models


class ModelFusionPipeline:
    """Handles model fusion pipeline."""

    VALID_EXTENSIONS = [".pkl", ".h5", ".pt"]

    def __init__(self, config: Dict[str, Any], run_folder_name: str):
        """
        Initialize ModelFusionPipeline with configuration and run folder.

        Args:
            config: Configuration dictionary
            run_folder_name: Name of the run folder
        """
        self.config = config
        self.run_folder_name = Path(run_folder_name)
        self.best_models_folder = self.run_folder_name / "BestModels"

        # Extract configuration parameters
        self.test_paths = config['test_data']
        self.column_names = config['desired_columns']
        self.label_column_test = config['label_column_test']
        self.nrows_test = config.get('nrows_test')
        self.feature_fusion_method = config.get('feature_fusion_method')

    def _get_model_files(self) -> List[Path]:
        """Get all model files from the BestModels folder."""
        if not self.best_models_folder.exists():
            raise ValueError("BestModels folder not found.")

        model_files = [
            self.best_models_folder / f
            for f in os.listdir(self.best_models_folder)
            if any(f.endswith(ext) for ext in self.VALID_EXTENSIONS)
        ]

        if not model_files:
            raise ValueError("No models found in BestModels folder.")

        return model_files

    def _load_model(self, model_file: Path) -> Any:
        """Load a model based on its file extension."""
        ext = model_file.suffix.lower()

        if ext == ".pkl":
            with open(model_file, 'rb') as f:
                return pickle.load(f)
        elif ext == ".h5":
            return load_model(str(model_file))
        else:
            raise ValueError(f"Unsupported model format: {ext}")

    def _predict_with_model(self, model: Any, X_test_array: np.ndarray, model_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with a model."""
        ext = model_file.suffix.lower()

        if ext == ".pkl":
            y_pred = model.predict(X_test_array)
            y_proba = model.predict_proba(X_test_array)[:, 1]
        elif ext == ".h5":
            y_proba = model.predict(X_test_array).flatten()
            y_pred = (y_proba > 0.5).astype(int)
        else:
            raise ValueError(f"Unsupported model format: {ext}")

        return y_pred, y_proba

    def _process_test_file(self, test_path: str, model_files: List[Path]) -> Dict[str, Any]:
        """Process a single test file with all models."""
        # Load test data
        X_test, Y_test = DataLoader.load_data(
            test_path, self.column_names, self.label_column_test, self.nrows_test
        )
        Y_test_array = np.stack(Y_test.iloc[:, 0])

        # Feature Fusion (if specified)
        X_test, fused_column_name = DataProcessor.fuse_columns(
            X_test, self.column_names, self.feature_fusion_method
        )

        # Store predictions for fusion
        y_preds = []
        y_probas = []

        # Run each model
        for model_file in model_files:
            model_name = model_file.stem.replace("_model", "")
            # Assuming column name is the last part
            column_name = model_name.split("_")[-1]
            X_test_array = np.stack(X_test[column_name])

            # Load and predict with model
            model = self._load_model(model_file)
            y_pred, y_proba = self._predict_with_model(
                model, X_test_array, model_file)

            y_preds.append(y_pred)
            y_probas.append(y_proba)

        # Average predictions and probabilities across all models
        y_pred_fusion = np.mean(y_preds, axis=0).round().astype(int)
        y_proba_fusion = np.mean(y_probas, axis=0)

        # Calculate metrics on the fusion results
        metrics = ModelEvaluator.calculate_metrics(
            X_test_array, Y_test_array, y_pred_fusion, y_proba_fusion)

        # Prepare results for this test file
        fusion_result = {"TestFile": Path(test_path).name}
        fusion_result.update(metrics)

        return fusion_result

    def run_fusion_pipeline(self) -> pd.DataFrame:
        """
        Run the model fusion pipeline.

        Returns:
            DataFrame containing fusion results
        """
        model_files = self._get_model_files()
        fusion_results = []

        for test_path in self.test_paths:
            fusion_result = self._process_test_file(test_path, model_files)
            fusion_results.append(fusion_result)

        # Save the fusion results in a CSV file
        fusion_results_df = pd.DataFrame(fusion_results)
        fusion_results_csv_path = self.run_folder_name / "FusionResults.csv"
        fusion_results_df.to_csv(fusion_results_csv_path, index=False)

        print(f"Model fusion results saved to {fusion_results_csv_path}")
        return fusion_results_df


class ModelPipeline:
    """Main pipeline class that orchestrates model selection and fusion."""

    def __init__(self, config: Dict[str, Any], run_folder_name: str):
        """
        Initialize ModelPipeline with configuration and run folder.

        Args:
            config: Configuration dictionary
            run_folder_name: Name of the run folder
        """
        self.config = config
        self.run_folder_name = run_folder_name
        self.model_selector = ModelSelector(config, run_folder_name)
        self.fusion_pipeline = ModelFusionPipeline(config, run_folder_name)

    def run_model_selection(self) -> pd.DataFrame:
        """Run model selection process."""
        return self.model_selector.select_best_models()

    def run_model_fusion(self) -> pd.DataFrame:
        """Run model fusion process."""
        return self.fusion_pipeline.run_fusion_pipeline()

    def run_full_pipeline(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Run the complete pipeline (selection + fusion if enabled).

        Returns:
            Tuple of (best_models_df, fusion_results_df or None)
        """
        # Run model selection
        best_models = self.run_model_selection()

        # Run fusion if enabled
        fusion_results = None
        if self.config.get("Fusion", False):
            fusion_results = self.run_model_fusion()

        return best_models, fusion_results


# Convenience functions for backward compatibility
def select_best_models(config: Dict[str, Any], run_folder_name: str) -> pd.DataFrame:
    """
    Convenience function for selecting best models.

    Args:
        config: Configuration dictionary
        run_folder_name: Name of the run folder

    Returns:
        DataFrame containing the best models
    """
    pipeline = ModelPipeline(config, run_folder_name)
    return pipeline.run_model_selection()


def fusion_pipeline(config: Dict[str, Any], run_folder_name: str) -> pd.DataFrame:
    """
    Convenience function for running fusion pipeline.

    Args:
        config: Configuration dictionary
        run_folder_name: Name of the run folder

    Returns:
        DataFrame containing fusion results
    """
    pipeline = ModelPipeline(config, run_folder_name)
    return pipeline.run_model_fusion()


def plot_model_metrics_radar(df: pd.DataFrame,
                             metric_columns: List[str],
                             save_path: str = "") -> None:
    """
    Convenience function for plotting radar chart.

    Args:
        df: DataFrame containing model evaluation metrics
        metric_columns: List of metric column names to plot
        save_path: Path to save the radar chart image
    """
    ModelMetricsVisualizer.plot_radar_chart(df, metric_columns, save_path)
