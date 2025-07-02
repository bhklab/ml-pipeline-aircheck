#==========================================================================
# Import necessary modules and utility functions from other scripts
from pdb import run
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
import time
from plot_results import plot_function
import datetime
#==========================================================================


#==========================================================================
"""
Main function to execute the AIRCHECK ML pipeline.
Loads configuration, initiates training, testing, model selection, and fusion.
"""

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")



def run_pipeline(config_name="config.yaml"):
    
    # Load and validate configuration file
    print("Step 1: Reading config file")
    config, RunFolderName = read_config(config_name)  

    # Prepare training paths based on balance settings
    print("Step 2: Data preperation")
    train_paths = create_balanced_datasets(config)

    # Execute the training pipeline
    print("Step 3: Train pipeline")
    train_pipeline(config, RunFolderName, load_data, fuse_columns, get_model, train_model,
               cross_validate_and_save_models, train_and_save_final_model,
               bayesian_hyperparameter_search, write_results_csv)

    # Execute the testing pipeline
    print("Step 4: Test pipeline")
    test_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model, get_model, train_model)

    # Execute model selection to find the best models
    print("Step 5: Model selection")
    select_best_models(config, RunFolderName)

    # Execute the fusion pipeline to combine the top models
    print("Step 6: Model fusion")
    fusion_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model)
    
    # Virtual Screening: Calculating screening data probability, applying chemistry filters, and clustering results
    print("Step 7: Virtual screening")
    screening_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model, get_model, train_model)
    
    print("Step 8: Logging the results using mlflow")
    # Loggingparameters, metric, artifacts and models using mlflow
    run_name = f"{config.get('protein_name')}_{timestamp}"
    log_results_to_mlflow (RunFolderName, config.get("experiment_name", "Default_Experiment"), run_name)
    
    print("Step 9: Loplotting some results")
    plot_function(RunFolderName)
#=========================================================================='''


# Example usage: Execute the pipeline if the script is run directly
if __name__ == "__main__":
    start_time = time.time()
    run_pipeline()
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nTotal runtime: {elapsed_minutes:.2f} minutes")
