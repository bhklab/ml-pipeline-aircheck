
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import pickle
import joblib
import tensorflow as tf
from pathlib import Path
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import numpy as np
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()


service_account_path = 'service_account.json'


# Check if the file exists before setting the environment variable
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    logging.info("Service account credentials set.")
else:
    logging.info("Service account file not found. Skipping credential setup.")


class MLflowModelRegistration:
    """
    A class to handle model registration in MLflow 3 from CSV metadata
    """
    
    def __init__(self, mlflow_tracking_uri: str = mlflow_uri):
        """
        Initialize MLflow client
        
        Args:
            mlflow_tracking_uri: URI for MLflow tracking server
        """
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        
    def load_model_from_path(self, model_path: str, model_type: str):
        """
        Load model from file path based on extension
        
        Args:
            model_path: Path to model file
            model_type: Type of model (for context)
            
        Returns:
            Loaded model object or None if failed
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return None
                
            file_extension = Path(model_path).suffix.lower()
            
            if file_extension == '.pkl':
                # Try different pickle loading methods
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                except:
                    # Try joblib if pickle fails
                    model = joblib.load(model_path)
                logger.info(f"Successfully loaded pickle model from {model_path}")
                return model
                
            elif file_extension == '.h5':
                # Load Keras/TensorFlow model
                model = tf.keras.models.load_model(model_path)
                logger.info(f"Successfully loaded H5 model from {model_path}")
                return model
                
            else:
                logger.warning(f"Unsupported file extension: {file_extension}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None
    
    def prepare_metrics(self, row: pd.Series) -> Dict[str, float]:
        """
        Extract and prepare metrics from CSV row
        
        Args:
            row: Pandas Series containing model metadata
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Define metric columns
        metric_columns = [
            'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1Score',
            'Test_AUC-ROC', 'Test_AUC-PR', 'Test_MCC', 'Test_Cohen Kappa',
            'Test_balanced_accuracy', 'Test_PlatePPV', 'Test_DivPlatePPV',
            'Test_HitsAt200', 'Test_PrecisionAt200', 'Test_HitsAt500',
            'Test_PrecisionAt500', 'Test_TotalHits', 'CV_HitsAt200', 'CV_HitsAt500', 'CV_PrecisionAt200', 'CV_PrecisionAt500', 'CV_TotalHits', 'CV_F1Score', 'CV_Precision', 'CV_Recall', 'CV_Accuracy', 'CV_PlatePPV',
        ]

        
        for col in metric_columns:
            if col in row.index and pd.notna(row[col]):
                try:
                    # Clean metric name for MLflow
                    # metric_name = col.replace('Test_', '').replace(' ', '_').lower()
                    metric_name = col
                    metrics[metric_name] = float(row[col])
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {col} to float: {row[col]}")
        

                    
        return metrics
    
    def prepare_params(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extract and prepare parameters from CSV row
        
        Args:
            row: Pandas Series containing model metadata
            
        Returns:
            Dictionary of parameters
        """
        params = {}
        
        # Basic parameters
        param_columns = [
            'ModelType', 'ColumnName', 'UsedHyperParameters', 
            'TrainFileName', 'TestFile'
        ]
        
        for col in param_columns:
            if col in row.index and pd.notna(row[col]):
                params[col.lower()] = str(row[col])
                
        # Add date and time
        if 'Date' in row.index and pd.notna(row['Date']):
            params['training_date'] = str(row['Date'])
        if 'Time' in row.index and pd.notna(row['Time']):
            params['training_time'] = str(row['Time'])
            
        return params
    
    def prepare_tags(self, row: pd.Series) -> Dict[str, str]:
        """
        Prepare tags for model registration
        
        Args:
            row: Pandas Series containing model metadata
            
        Returns:
            Dictionary of tags
        """
        tags = {
            'source': 'csv_import',
            'model_type': str(row.get('ModelType', 'unknown')),
            'training_date': str(row.get('Date', 'unknown')),
            'data_source': str(row.get('TrainFileName', 'unknown'))
        }
        
        return tags


    def register_model1(self, row: pd.Series, experiment_name: str = "CSV_Model_Import", model_path: str = "", run_name: str | None = None) -> bool:
        """
        Register a single model with MLflow using nested runs
        
        Args:
            row: Pandas Series containing model metadata
            experiment_name: Name of MLflow experiment
            model_path: Path to the model file

        Returns:
            Boolean indicating success
        """
        try:

            # Set or create experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            logger.info(f"Experiment: {experiment}")
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
                
            mlflow.set_experiment(experiment_name)
            logger.info(f"Experiment ID: {experiment_id}")
            
            if not model_path or pd.isna(model_path):
                logger.error(f"No model path found in row: {row.name}")
                return False
                
            # Check if file exists and has supported extension
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
                
            file_extension = Path(model_path).suffix.lower()
            if file_extension not in ['.pkl', '.h5']:
                logger.info(f"Skipping unsupported file type: {file_extension}")
                return False
            
            # Prepare MLflow logging data
            metrics = self.prepare_metrics(row)
            params = self.prepare_params(row)
            tags = self.prepare_tags(row)

            
            # Create run name
            parent_run_name = f"{row.get('ModelType', 'model')}_{Path(model_path).stem}"
            parent_run_name = parent_run_name.replace(' ', '_').replace('.', '_')
            
            # Start parent run for this model
            logger.info(f"Starting MLflow run for model: {row.get('ModelType', 'unknown')}")
            
            with mlflow.start_run(run_name=parent_run_name) as parent_run:
                
                # Log common parameters, metrics, and tags at parent level
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
                
                # Log common metadata
                mlflow.log_param("original_model_path", model_path)
                mlflow.log_param("file_extension", file_extension)
                mlflow.log_param("model_name", run_name)
                
                parent_run_id = parent_run.info.run_id
                
                # Handle different model types with nested runs
                if file_extension == '.pkl':
                    # Nested run for PKL model
                    with mlflow.start_run(run_name=f"{run_name}_sklearn", nested=True) as nested_run:
                        try:
                            with open(model_path, 'rb') as files:
                                clf = pickle.load(files)

                            # Log model-specific parameters
                            mlflow.log_param("model_framework", "sklearn")
                            mlflow.log_param("model_type", str(type(clf).__name__))
                            
                            # Try to get model parameters if available
                            if hasattr(clf, 'get_params'):
                                model_params = clf.get_params()
                                for param_name, param_value in model_params.items():
                                    if param_value is not None:
                                        mlflow.log_param(f"model_{param_name}", str(param_value))
                            
                            # Log the model
                            mlflow.sklearn.log_model(clf, "model",registered_model_name=None)
                            
                            nested_run_id = nested_run.info.run_id
                            
                        except Exception as e:
                            print(f"Error logging sklearn model: {e}")
                            mlflow.log_param("error_message", str(e))

                elif file_extension == '.h5':
                    # Nested run for H5 model
                    with mlflow.start_run(run_name=f"{run_name}_keras", nested=True) as nested_run:
                        try:
                            import tensorflow as tf
                            
                            # Load the model
                            model = tf.keras.models.load_model(model_path)
                            
                            # Log model-specific parameters
                            mlflow.log_param("model_framework", "keras")
                            mlflow.log_param("model_type", "keras_sequential" if isinstance(model, tf.keras.Sequential) else "keras_functional")
                            
                            # Log model architecture info
                            mlflow.log_param("total_params", model.count_params())
                            mlflow.log_param("trainable_params", sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
                            mlflow.log_param("non_trainable_params", sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]))
                            mlflow.log_param("num_layers", len(model.layers))
                            
                            # Log optimizer info if available
                            if hasattr(model, 'optimizer') and model.optimizer is not None:
                                mlflow.log_param("optimizer", model.optimizer.__class__.__name__)
                                if hasattr(model.optimizer, 'learning_rate'):
                                    mlflow.log_param("learning_rate", float(model.optimizer.learning_rate))
                            
                            # Log loss function if available
                            if hasattr(model, 'loss') and model.loss is not None:
                                mlflow.log_param("loss_function", str(model.loss))
                            
                            # Log the model
                            mlflow.keras.log_model(model, "model")
                            
                            nested_run_id = nested_run.info.run_id
                            print(f"Nested run ID (keras): {nested_run_id}")
                            
                        except Exception as e:
                            print(f"Error logging keras model: {e}")
                            # Fallback to artifact logging
                            mlflow.log_artifact(model_path, "model")
                            mlflow.log_param("error_message", str(e))
                
                logger.info(f"Successfully registered model: {run_name}, Parent Run ID: {parent_run_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error registering model from row {row.name}: {str(e)}")
            return False
        

    def process_csv1(self, csv_path: str, experiment_name: str = "CSV_Model_Import") -> Dict[str, int]:
        """
        Process entire CSV file and register all models
        
        Args:
            csv_path: Path to CSV file
            experiment_name: Name of MLflow experiment
            
        Returns:
            Dictionary with success/failure counts
        """
        try:
            # Load CSV
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(df)} rows")
            
            # Validate required columns
            required_columns = ['ModelPath']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Process each row
            results = {'success': 0, 'failed': 0, 'skipped': 0}
            df.replace("", np.nan, inplace=True)
            df = df.dropna(how='all')
        
    
            for index, row in df.iterrows():
                logger.info(f"Processing row {index + 1}/{len(df)}")
                
                model_path = "run_test/BestModels/"

                # Loop through everything inside model_path
                for entry in os.listdir(model_path):
                    entry_path = os.path.join(model_path, entry)

                    if os.path.isfile(entry_path) and (entry.endswith(".pkl") or entry.endswith(".h5")):
                        target_files = entry_path
                    else:
                        logger.warning(f"No files found in model path: {model_path}")
                        results['skipped'] += 1
                        raise FileNotFoundError(f"File not found or unsupported format: {entry_path}")

                run_name = entry.split(".")[0]
            
            
                if self.register_model1(row, experiment_name, model_path=target_files, run_name = run_name):
                 results['success'] += 1
                else:
                    results['failed'] += 1

            logger.info(f"Processing complete. Results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            raise

    # mlflow server --host 127.0.0.1   