import yaml
import pandas as pd
import os
from datetime import datetime
import warnings
from typing import Any


class MLPipelineConfig:
    """
    A class to manage ML pipeline configuration validation, loading, and result writing.
    """
    
    def __init__(self, config_path: str|None = None):
        """
        Initialize the ML Pipeline Configuration Manager.
        
        Args:
            config_path (str, optional): Path to the configuration file
        """
        self.config_path = config_path
        self.config = None
        self.run_folder_name = None
        self.default_values = {
            'run_name': 'Results',
            'protein_name': 'protein_name',
            'is_train': False,
            'is_test': False,
            'train_data': [],
            'test_data': [],
            'desired_columns': ['ECFP4'],
            'label_column_train': ['LABEL'],
            'label_column_test': ['LABEL'],
            'nrows_train': None,
            'nrows_test': None,
            'feature_fusion_method': 'None',
            'balance_flag': False,
            'balance_ratios': [1],
            'desired_models': ['rf'],
            'hyperparameters_tuning': False,
            'hyperparameters': {},
            'Nfold': 2,
            'trainfile_for_modelselection': [],
            'evaluationfile_for_modelselection': [],
            'evaluation_column': ['Test_F1 Score', 'Test_Precision', 'Test_Recall', 'Test_Accuracy', 'Test_PlatePPV', 'Test_DivPlatePPV'],
            'crossvalidation_column': ['CV_F1 Score', 'CV_Precision', 'CV_Recall', 'CV_Accuracy', 'CV_PlatePPV', 'CV_DivPlatePPV'],
            'Fusion': False,
            'num_top_models': 5,
            'conformal_prediction': False,
            'confromal_test_size': 0.3,
            'confromal_confidence_level': 0.95,
            'smiles_column': 'SMILES',
            'chemistry_filters': True
        }
        self.supported_models = [
            'rf', 'lr', 'svc', 'nb', 'dt', 'knn', 'gb', 'ada', 
            'bag', 'mlp', 'lgbm', 'catboost', 'tf_ff', 'tf_cnn1D'
        ]
        self.valid_fusion_methods = ['None', 'All', 'Pairwise']
    
    def load_config(self, config_path: str|None = None) -> dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str, optional): Path to configuration file
            
        Returns:
            Dict[str, Any]: Loaded configuration dictionary
        """
        if config_path:
            self.config_path = config_path
        
        if not self.config_path:
            raise ValueError("No configuration path provided")
        
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        return config.get("ml-pipeline", {})
    
    def _set_default_values(self, config: dict[str, Any]) -> None:
        """
        Set default values for missing configuration keys.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary to update
        """
        for key, default_value in self.default_values.items():
            if key not in config:
                config[key] = default_value
                warnings.warn(f"Config key '{key}' missing. Default value '{default_value}' has been set.")
    
    def _validate_required_fields(self, config: dict[str, Any]) -> list[str]:
        """
        Validate required configuration fields.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Validate boolean fields
        if not config.get('is_train'):
            errors.append("The 'is_train' field is required and must be set to 'True'.")
        
        if config.get('is_test') is None:
            errors.append("The 'is_test' field is required and must be set to True or False.")
        
        if config.get('chemistry_filters') is None:
            errors.append("The 'chemistry_filters' field is required and must be set to True or False.")
        
        if config.get('conformal_prediction') is None:
            errors.append("The 'conformal_prediction' field is required and must be set to True or False.")
        
        if config.get('Fusion') is None:
            errors.append("The 'Fusion' field is required and must be set to True or False.")
        
        return errors
    
    def _validate_data_paths(self, config: dict[str, Any]) -> list[str]:
        """
        Validate data file paths.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Validate train data paths
        if not isinstance(config.get('train_data'), list):
            errors.append("train_data must be a list of paths.")
        else:
            for path in config['train_data']:
                if not os.path.exists(path):
                    errors.append(f"Train data path not found: {path}")
        
        # Validate test data paths
        if not isinstance(config.get('test_data'), list):
            errors.append("test_data must be a list of paths.")
        else:
            for path in config['test_data']:
                if not os.path.exists(path):
                    errors.append(f"Test data path not found: {path}")
        
        return errors
    
    def _validate_numeric_fields(self, config: dict[str, Any]) -> list[str]:
        """
        Validate numeric configuration fields.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Validate conformal prediction parameters
        for key in ['confromal_test_size', 'confromal_confidence_level']:
            value = config.get(key)
            if not isinstance(value, (float, int)) or not (0 < value < 1):
                errors.append(f"'{key}' must be a float between 0 and 1. Got: {value}")
        
        # Validate cross-validation folds
        if not isinstance(config.get('Nfold'), int) or config['Nfold'] < 2:
            config['Nfold'] = 2
        
        # Validate number of top models
        if not isinstance(config.get('num_top_models'), int) or config['num_top_models'] <= 0:
            config['num_top_models'] = 5
        
        return errors
    
    def _validate_models(self, config: dict[str, Any]) -> list[str]:
        """
        Validate model configuration.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        if not isinstance(config.get('desired_models'), list):
            errors.append("desired_models must be a list.")
        else:
            for model in config['desired_models']:
                if model not in self.supported_models:
                    errors.append(f"Unsupported model: {model}")
        
        return errors
    
    def _validate_data_types(self, config: dict[str, Any]) -> list[str]:
        """
        Validate data types and set defaults where appropriate.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            
        Returns:
            List[str]: List of validation errors
        """
        errors = []
        
        # Validate label columns
        if not isinstance(config.get('label_column_train'), str):
            errors.append("label_column_train must be a string.")
        if not isinstance(config.get('label_column_test'), str):
            errors.append("label_column_test must be a string.")
        
        # Validate and set defaults for various fields
        if not isinstance(config.get('desired_columns'), list):
            config['desired_columns'] = []
        
        if not isinstance(config.get('hyperparameters'), dict):
            config['hyperparameters'] = {}
        
        if not isinstance(config.get('balance_flag'), bool):
            config['balance_flag'] = False
        
        if not isinstance(config.get('balance_ratios'), list):
            config['balance_ratios'] = [1, 2]
        
        # Handle nrows fields
        for field in ['nrows_train', 'nrows_test']:
            if not isinstance(config.get(field), (int, str)):
                config[field] = None
            elif isinstance(config.get(field), str) and config[field].lower() == 'none':
                config[field] = None
        
        # Validate feature fusion method
        if config.get('feature_fusion_method') not in self.valid_fusion_methods:
            config['feature_fusion_method'] = 'None'
        
        # Validate list fields
        list_fields = ['trainfile_for_modelselection', 'evaluationfile_for_modelselection']
        for field in list_fields:
            if not isinstance(config.get(field), list):
                config[field] = []
        
        # Validate evaluation columns
        if not isinstance(config.get('evaluation_column'), list):
            config['evaluation_column'] = self.default_values['evaluation_column']
        
        if not isinstance(config.get('crossvalidation_column'), list):
            config['crossvalidation_column'] = self.default_values['crossvalidation_column']
        
        return errors
    
    def validate_config(self, config_path: str | None = None) -> dict[str, Any]:
        """
        Validate the configuration file and return the validated configuration.
        
        Args:
            config_path (str, optional): Path to configuration file
            
        Returns:
            Dict[str, Any]: Validated configuration dictionary
            
        Raises:
            ValueError: If configuration validation fails
        """
        config = self.load_config(config_path)
        
        # Set default values
        self._set_default_values(config)
        
        # Collect all validation errors
        errors = []
        errors.extend(self._validate_required_fields(config))
        errors.extend(self._validate_data_paths(config))
        errors.extend(self._validate_numeric_fields(config))
        errors.extend(self._validate_models(config))
        errors.extend(self._validate_data_types(config))
        
        # Raise error if any validation failed
        if errors:
            raise ValueError("Configuration Error:\n" + "\n".join(errors))
        

        self.config = config
        return config
    
    def read_config(self, config_path: str | None = None) -> tuple[dict[str, Any], str]:
        """
        Read and validate configuration, create run folder, and save config.
        
        Args:
            config_path (str, optional): Path to configuration file
            
        Returns:
            tuple: (config dictionary, run folder name)
        """

        config = self.validate_config(config_path)
 
        
        # Create run folder
        run_folder_name = os.path.join(config['run_name'])
        os.makedirs(run_folder_name, exist_ok=True)

        
        # Save config to run folder
        config_path_in_folder = os.path.join(run_folder_name, "config.yaml")
        with open(config_path_in_folder, 'w') as f:
            yaml.dump(config, f)
        
        self.run_folder_name = run_folder_name
        return config, run_folder_name
    
    def write_results_csv(self, experiment_results: dict[str, Any], 
                         run_folder_name: str | None = None) -> None:
        """
        Write experiment results to CSV file.
        
        Args:
            experiment_results (Dict[str, Any]): Dictionary of experiment results
            run_folder_name (str, optional): Path to run folder
        """
        if run_folder_name is None:
            run_folder_name = self.run_folder_name
        
        if run_folder_name is None:
            raise ValueError("No run folder specified. Call read_config() first or provide run_folder_name.")
        
        # Convert all values to strings, handling lists properly
        results_row = {
            k: ','.join(map(str, v)) if isinstance(v, list) else str(v) 
            for k, v in experiment_results.items()
        }
        
        results_csv_path = os.path.join(run_folder_name, "results.csv")
        
        # Add Date and Time
        now = datetime.now()
        results_row["Date"] = now.date().isoformat()
        results_row["Time"] = now.strftime("%H:%M:%S")
        
        # Create new file if needed
        if not os.path.exists(results_csv_path):
            df = pd.DataFrame(columns=list(results_row.keys()))
            df.to_csv(results_csv_path, index=False)
        
        # Append row
        df_existing = pd.read_csv(results_csv_path)
        df_existing = pd.concat([df_existing, pd.DataFrame([results_row])], ignore_index=True)
        df_existing.to_csv(results_csv_path, index=False)
    
    def get_config(self) -> dict[str, Any] | None:
        """
        Get the current configuration.
        
        Returns:
            Dict[str, Any]: Current configuration dictionary
        """
        return self.config
    
    def get_run_folder(self) -> str | None :
        """
        Get the current run folder path.
        
        Returns:
            str: Current run folder path
        """
        return self.run_folder_name

