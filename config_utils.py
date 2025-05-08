import yaml
import pandas as pd
import os
from datetime import datetime
#==============================================================================
def read_config(config_name):
    validate_config(config_name)
    
    with open(config_name, 'r') as file:
        config = yaml.safe_load(file)
        

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
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    errors = []

    # Validate Train and Test
    if config.get('Train') not in ['Y', 'N']:
        errors.append("Train must be 'Y' or 'N'.")

    if config.get('Test') not in ['Y', 'N']:
        errors.append("Test must be 'Y' or 'N'.")

    # Validate Train and Test Data Paths
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

    # Validate desired columns
    if not isinstance(config.get('desired_columns'), list):
        errors.append("desired_columns must be a list of column names.")

    # Validate Model Names
    supported_models = ['rf', 'lr', 'ridge', 'sgd', 'perceptron', 'svc', 'nb', 'dt', 'knn', 'gb', 'ada', 'bag', 'mlp']
    if not isinstance(config.get('desired_models'), list):
        errors.append("desired_models must be a list.")
    else:
        for model in config['desired_models']:
            if model not in supported_models:
                errors.append(f"Unsupported model: {model}")

    # Validate Hyperparameters
    if not isinstance(config.get('hyperparameters'), dict):
        errors.append("hyperparameters must be a dictionary.")

    # Validate Cross-Validation
    if not isinstance(config.get('Nfold'), int) or config['Nfold'] < 2:
        errors.append("Nfold must be an integer greater than 1.")

    # Set default values if missing
    if 'Fusion' not in config:
        config['Fusion'] = 'N'

    if 'num_top_models' not in config:
        config['num_top_models'] = 5

    # Output errors or return validated config
    if errors:
        raise ValueError("Configuration Error:\n" + "\n".join(errors))

    print("Configuration is valid.")
    return config
