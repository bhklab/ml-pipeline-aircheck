#==========================================================================
# Import necessary modules and utility functions from other scripts
from data_utils import load_data, fuse_columns
from data_utils import create_balanced_datasets
from model_utils import get_model, train_model
from model_utils import cross_validate_and_save_models, train_and_save_final_model
from model_utils import bayesian_hyperparameter_search
from model_utils import train_pipeline
from eval_utils import evaluate_model, test_pipeline
from config_utils import read_config, write_results_csv
from fusion_utils import select_best_models 
from fusion_utils import fusion_pipeline
#==========================================================================


#==========================================================================
"""
Main function to execute the AIRCHECK ML pipeline.
Loads configuration, initiates training, testing, model selection, and fusion.
"""
def run_pipeline(config_name="config.yaml"):
    
    # Load and validate configuration file
    config, RunFolderName = read_config(config_name)  
    

    # Prepare training paths based on balance settings
    train_paths = create_balanced_datasets(config)


    # Prepare training arguments for the train pipeline function
    train_pipeline(config, RunFolderName, load_data, fuse_columns, get_model, train_model,
               cross_validate_and_save_models, train_and_save_final_model,
               bayesian_hyperparameter_search, write_results_csv)
    

    # Prepare testing arguments for the test pipeline function
    test_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model, get_model, train_model)


    # Prepare model selection arguments
    select_best_models(config, RunFolderName)


    # Prepare score fusion arguments for the fusion pipeline function
    fusion_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model)
#=========================================================================='''


# Example usage: Execute the pipeline if the script is run directly
if __name__ == "__main__":
    run_pipeline()
