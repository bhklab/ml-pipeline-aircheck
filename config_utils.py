import yaml
import pandas as pd
import os
from datetime import datetime

def read_config(config_name):
    with open(config_name, 'r') as file:
        config = yaml.safe_load(file)
        
    RunName = config['RunName']
    RunFolderName = os.path.join("Results", RunName)
    os.makedirs(RunFolderName, exist_ok=True) 


    # Save config to this path
    config_path_in_folder = os.path.join(RunFolderName, config_name)
    with open(config_path_in_folder, 'w') as f:
        yaml.dump(config, f)

    return config, RunFolderName

def write_results_csv(experiment_results, RunFolderName):
    # Prepare row with config values
    results_row = {k: ','.join(v) if isinstance(v, list) else v for k, v in experiment_results.items()}
    
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


'''def CheckConfigFile:
    # Check if n,y is correct for fusion
    # Checking all config entries and raise error if not correct
    # label file should be just one string and check if it exists
    # if n_rows is not None, check if it is integer and in range and convert to number from string'
    # Checking the correct input format and raising error if not!!!!!!!!!!!!!!!!!Label should be one string, the other one should be a list
    # label and columns should be as a list in 
    # Column and label should exist in the dataset
    '''
