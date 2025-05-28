import yaml
import pandas as pd
import os
from datetime import datetime
import warnings
#==============================================================================
def read_config(config_name):
    config = validate_config(config_name)
    
    '''with open(config_name, 'r') as file:
        config = yaml.safe_load(file)'''
        

    RunFolderName = os.path.join("Results")
    os.makedirs(RunFolderName, exist_ok=True)


    # Save config to this path
    config_path_in_folder = os.path.join(RunFolderName, config_name)
    with open(config_path_in_folder, 'w') as f:
        yaml.dump(config, f)

    return config, RunFolderName
#==============================================================================
def write_results_csv(experiment_results, RunFolderName):
    # Convert all values to strings, handling lists properly
    results_row = {
        k: ','.join(map(str, v)) if isinstance(v, list) else str(v) 
        for k, v in experiment_results.items()
    }
    
    results_csv_path = os.path.join(RunFolderName, "results.csv")
    
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

#==============================================================================
# Function to validate the configuration file
def validate_config(config_path):
    errors = []
    #-------------------------------
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file) 
    #-------------------------------
    # Default values for missing keys
    default_values = {
        'protein_name': 'protein_name',
        'Train': 'N',
        'Test': 'N',
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
        'hyperparameters_tuning': 'N',
        'hyperparameters': {},
        'Nfold': 2,
        'trainfile_for_modelselection': [],
        'evaluationfile_for_modelselection': [],
        'evaluation_column': ['Test_F1 Score', 'Test_Precision', 'Test_Recall', 'Test_Accuracy', 'Test_PlatePPV', 'Test_DivPlatePPV'],
        'crossvalidation_column': ['CV_F1 Score', 'CV_Precision', 'CV_Recall', 'CV_Accuracy', 'CV_PlatePPV', 'CV_DivPlatePPV'],
        'Fusion': 'N',
        'num_top_models': 5,
        'conformal_prediction': 'N',
        'confromal_test_size': 0.3,
        'confromal_confidence_level': 0.95
    }
    
    # Set default values and warn user
    for key, default_value in default_values.items():
        if key not in config:
            config[key] = default_value
            warnings.warn(f"Config key '{key}' missing. Default value '{default_value}' has been set.")

    #--------------------------------

    # Validate Train and Test (Required)
    if config.get('Train') not in ['Y', 'N', 'n', 'y']:
        errors.append("The 'Train' field is required in the configuration file and must be set to 'Y/y' (yes) or 'N/n' (no).")

    if config.get('Test') not in ['Y', 'N', 'n', 'y']:
        errors.append("The 'Test' field is required in the configuration file and must be set to 'Y/y' (yes) or 'N/n' (no).")

    if config.get('conformal_prediction') not in ['Y', 'N', 'n', 'y']:
        errors.append("The 'conformal_prediction' field is required in the configuration file and must be set to 'Y/y' (yes) or 'N/n' (no).")
        
    # Validate Train and Test Data Paths (Required)
    if not isinstance(config.get('train_data'), list):
        errors.append("train_data must be a list of paths.")
    else:
        for path in config['train_data']:
            if not os.path.exists(path):
                errors.append(f"Train data path not found: {path}")

    if not isinstance(config.get('test_data'), list):
        errors.append("test_data must be a list of paths.")
    else:
        for path in config['test_data']:
            if not os.path.exists(path):
                errors.append(f"Test data path not found: {path}")

    # Validate Desired Columns (Default: Empty List)
    if not isinstance(config.get('desired_columns'), list):
        config['desired_columns'] = []  # Default value

    # Validate Label Columns (Required)
    if not isinstance(config.get('label_column_train'), list):
        errors.append("label_column_train must be a list.")
    if not isinstance(config.get('label_column_test'), list):
        errors.append("label_column_test must be a list.")

    # Validate Number of Rows for Train and Test (Default: None)
    if not isinstance(config.get('nrows_train'), (int, str)):
        config['nrows_train'] = None
    elif isinstance(config.get('nrows_train'), str) and config['nrows_train'].lower() == 'none':
        config['nrows_train'] = None

    if not isinstance(config.get('nrows_test'), (int, str)):
        config['nrows_test'] = None
    elif isinstance(config.get('nrows_test'), str) and config['nrows_test'].lower() == 'none':
        config['nrows_test'] = None

    # Validate Feature Fusion Method (Default: None)
    valid_fusion_methods = ['None', 'All', 'Pairwise']
    if config.get('feature_fusion_method') not in valid_fusion_methods:
        config['feature_fusion_method'] = 'None'

    # Validate Balance Settings (Defaults)
    if not isinstance(config.get('balance_flag'), bool):
        config['balance_flag'] = False

    if not isinstance(config.get('balance_ratios'), list):
        config['balance_ratios'] = [1, 2]

    # Validate Model Names (Required)
    supported_models = ['rf', 'lr', 'ridge', 'sgd', 'perceptron', 'svc', 'nb', 
                        'dt', 'knn', 'gb', 'ada', 'bag', 'mlp']
    
    if not isinstance(config.get('desired_models'), list):
        errors.append("desired_models must be a list.")
    else:
        for model in config['desired_models']:
            if model not in supported_models:
                errors.append(f"Unsupported model: {model}")

    # Validate Hyperparameters (Default: Empty Dict)
    if not isinstance(config.get('hyperparameters'), dict):
        config['hyperparameters'] = {}

    # Validate Hyperparameter Tuning (Default: 'N')
    if config.get('hyperparameters_tuning') not in ['Y', 'N']:
        config['hyperparameters_tuning'] = 'N'

    # Validate Cross-Validation (Default: 2)
    if not isinstance(config.get('Nfold'), int) or config['Nfold'] < 2:
        config['Nfold'] = 2

    # Validate Model Selection (Defaults)
    if not isinstance(config.get('trainfile_for_modelselection'), list):
        config['trainfile_for_modelselection'] = []

    if not isinstance(config.get('evaluationfile_for_modelselection'), list):
        config['evaluationfile_for_modelselection'] = []

    if not isinstance(config.get('evaluation_column'), list):
        config['evaluation_column'] = ['Test_F1 Score', 'Test_Precision', 'Test_Recall', 'Test_Accuracy', 'Test_PlatePPV', 'Test_DivPlatePPV']
        
    if not isinstance(config.get('crossvalidation_column'), list):
        config['crossvalidation_column'] = ['CV_F1 Score', 'CV_Precision', 'CV_Recall', 'CV_Accuracy', 'CV_PlatePPV', 'CV_DivPlatePPV']

    # Validate Fusion Settings (Defaults)
    if config.get('Fusion') not in ['Y', 'N', 'n', 'y']:
        errors.append("The 'Fusion' field is required in the configuration file and must be set to 'Y/y' (yes) or 'N/n' (no).")

    if not isinstance(config.get('num_top_models'), int) or config['num_top_models'] <= 0:
        config['num_top_models'] = 5

    # Output errors or return validated config
    if errors:
        raise ValueError("Configuration Error:\n" + "\n".join(errors))

    print("Configuration is valid and complete.")
    return config
#==============================================================================

