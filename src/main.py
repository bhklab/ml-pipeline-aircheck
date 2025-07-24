"""
ML Pipeline Main Entry Point

This module provides the main entry point for running the complete ML pipeline
for drug discovery and virtual screening.

Date: 2025
License: MIT
"""


import logging
import argparse
import os
import shutil
from typing import Any
import sys
import time
import datetime

from src.data.make_dataset import DataProcessor
from src.models.train_model import ModelTrainer
from src.models.eval_model import ModelEvaluator
from src.screen.clustering import PipelineConfig, ScreeningPipeline
from src.utils.config_utils import MLPipelineConfig
from src.mlflow_logging.mlflow_serve import MLflowModelRegistration
from src.utils.upload_to_gcs import upload_folder_to_gcs
from src.utils.fusion_utils import ModelSelector, ModelFusionPipeline
from src.visualization.visualize import plot_function
# ==========================================================================
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()

service_account_path = 'service_account.json'
# service_account_path = '../app/service_account.json'

# Check if the file exists before setting the environment variable
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    logging.info("Service account credentials set.")
else:
    logging.info("Service account file not found. Skipping credential setup.")


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class MLPipelineRunner:
    """Main ML Pipeline Runner class."""

    def __init__(self, config_path: str):
        """
        Initialize the ML Pipeline Runner.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config: dict[str, Any] | None = None
        self.run_folder: str | None = None
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('pipeline.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)

    def _load_config(self) -> None:
        """Load configuration from file."""
        self.logger.info("Step 1: Reading config file")
        mlconfig = MLPipelineConfig()
        self.config, self.run_folder = mlconfig.read_config(self.config_path)
        self.logger.info(f"Run folder: {self.run_folder}")

    def _prepare_data(self) -> list:
        """Prepare and balance datasets."""
        self.logger.info("Step 2: Data preparation")
        train_paths = DataProcessor.create_balanced_datasets(self.config)
        self.logger.info(f"Created {len(train_paths)} balanced datasets")
        return train_paths

    def _train_models(self, train_paths: list) -> None:
        """Train ML models."""
        self.logger.info("Step 3: Training pipeline")
        trainer = ModelTrainer()
        trainer.train_pipeline(self.run_folder, self.config, train_paths)
        self.logger.info("Training completed successfully")

    def _test_models(self) -> None:
        """Test trained models."""
        self.logger.info("Step 4: Testing pipeline")
        ModelEvaluator.test_pipeline(self.run_folder, self.config)
        self.logger.info("Testing completed successfully")

    def _select_best_models(self) -> None:
        """Select best performing models."""
        self.logger.info("Step 5: Model selection")
        best_model = ModelSelector(
            config=self.config, run_folder_name=self.run_folder)
        best_model.select_best_models()
        self.logger.info("Model selection completed")

    def _run_model_fusion(self) -> None:
        """Run model fusion if enabled."""
        self.logger.info("Step 6: Model fusion")
        fusion = ModelFusionPipeline(
            config=self.config, run_folder_name=self.run_folder)
        fusion.run_fusion_pipeline()
        self.logger.info("Model fusion completed")

    def _run_virtual_screening(self) -> None:
        """Run virtual screening pipeline."""
        self.logger.info("Step 7: Virtual screening")
        pipeline_config = PipelineConfig(
            is_screen=self.config.get("is_screen", True),
            screen_data=self.config.get("screen_data"),
            smiles_column=self.config.get("smiles_column", "SMILES"),
            desired_columns=self.config.get("desired_columns", []),
            feature_fusion_method=self.config.get("feature_fusion_method"),
            conformal_prediction=self.config.get("conformal_prediction"),
            conformal_test_size=self.config.get("conformal_test_size"),
            conformal_confidence_level=self.config.get(
                "conformal_confidence_level"),
            nrows_train=self.config.get("nrows_train"),
            is_chemistry_filters=self.config.get("is_chemistry_filters"),
        )
        screening = ScreeningPipeline(pipeline_config, self.run_folder)
        screening.run_pipeline()
        self.logger.info("Virtual screening completed")

    def _plot_result(self) -> None:
        plot_function(self.run_folder)

    def _log_with_mlflow(self) -> None:
        """Log results using MLflow."""
        self.logger.info("Step 8: Logging results with MLflow")
        best_model_file = os.path.join(
            self.run_folder, "BestModelsResults.csv")
        print("best model file---------", best_model_file)

        if not os.path.exists(best_model_file):
            self.logger.warning(
                f"Best models file not found: {best_model_file}")
            return

        mlflow_logger = MLflowModelRegistration()
        experiment_name = self.config.get("experiment_name", "DEL-Experiment")
        mlflow_logger.process_csv(best_model_file, experiment_name)
        self.logger.info("MLflow logging completed")

    def _upload_artifacts(self) -> None:
        """Upload artifacts to GCS bucket."""
        self.logger.info("Step 9: Uploading artifacts to GCS")
        bucket_name = self.config.get("bucket_name")
        prefix_name = self.config.get("prefix_name", "mlflow-artifacts")

        if bucket_name:
            upload_folder_to_gcs(self.run_folder, bucket_name, prefix_name)
            self.logger.info("Artifacts uploaded to GCS successfully")
        else:
            self.logger.warning(
                "No bucket_name specified, skipping GCS upload")

    def _cleanup(self) -> None:
        """Clean up temporary files."""
        self.logger.info("Step 10: Cleaning up temporary run folder")
        """Removes the run folder and its contents."""
        if os.path.exists(self.run_folder):
            try:
                shutil.rmtree(self.run_folder)
                print(f"Successfully removed run folder: {self.run_folder}")
            except Exception as e:
                print(f"Error removing folder {self.run_folder}: {e}")
        else:
            print(f"No folder found at: {self.run_folder}")

        self.logger.info("Cleanup completed")

    def run_pipeline(self) -> None:
        """Run the complete ML pipeline."""
        try:
            # Load configuration
            self._load_config()

            # Data preparation
            train_paths = self._prepare_data()

            # Model training
            self._train_models(train_paths)

            # Model testing
            self._test_models()

            # Model selection
            self._select_best_models()

            # Model fusion (if enabled)
            if self.config.get("Fusion", False):
                self._run_model_fusion()
            else:
                self.logger.info(
                    "Step 6: Model fusion - SKIPPED (disabled in config)")

            # Virtual screening (if enabled)
            if self.config.get("is_screen", False):
                self._run_virtual_screening()
            else:
                self.logger.info(
                    "Step 7: Virtual screening - SKIPPED (disabled in config)")

            # MLflow logging
            self._log_with_mlflow()

            # Upload artifacts
            # self._upload_artifacts()

            self._plot_result()

            # Cleanup
            if self.config.get("cleanup_after_run", True):
                self._cleanup()
            else:
                self.logger.info(
                    "Step 10: Cleanup - SKIPPED (disabled in config)")

            self.logger.info("Pipeline completed successfully!")

        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            raise


def main():
    """Main entry point for the ML pipeline."""
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Run ML Pipeline for Drug Discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="src/configs/config_loader.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Run the pipeline
    try:
        runner = MLPipelineRunner(args.config)
        runner.run_pipeline()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\nTotal runtime: {elapsed_minutes:.2f} minutes")


def cleanup_run_folder(folder_path: str):
    """Removes the run folder and its contents."""
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print(f"Successfully removed run folder: {folder_path}")
        except Exception as e:
            print(f"Error removing folder {folder_path}: {e}")
    else:
        print(f"No folder found at: {folder_path}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\nTotal runtime: {elapsed_minutes:.2f} minutes")
