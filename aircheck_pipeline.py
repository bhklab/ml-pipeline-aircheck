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
from screening_utils import screening_pipeline
from log_results import log_results_to_mlflow
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

    # Execute the training pipeline
    train_pipeline(config, RunFolderName, load_data, fuse_columns, get_model, train_model,
               cross_validate_and_save_models, train_and_save_final_model,
               bayesian_hyperparameter_search, write_results_csv)  

    # Execute the testing pipeline
    test_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model, get_model, train_model)

    # Execute model selection to find the best models
    select_best_models(config, RunFolderName)

    # Execute the fusion pipeline to combine the top models
    fusion_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model)
    
    # Virtual Screening: Calculating screening data probability, applying chemistry filters, and clustering results
    screening_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model, get_model, train_model)
    
    # Loggingparameters, metric, artifacts and models using mlflow
    log_results_to_mlflow (RunFolderName)
#=========================================================================='''


# Example usage: Execute the pipeline if the script is run directly
if __name__ == "__main__":
    run_pipeline()
