#==========================================================================
# Import necessary modules and utility functions from other scripts
from pdb import run
from data_utils import load_data, fuse_columns
from data_utils import create_balanced_datasets
from model_utils import get_model, train_model
from model_utils import cross_validate_and_save_models, train_and_save_final_model
from model_utils import create_cv_folds, plot_cv_folds_umap
from model_utils import bayesian_hyperparameter_search
from model_utils import train_pipeline
from eval_utils import evaluate_model, test_pipeline, plot_hit_curves, plot_best_models_hit_curves, plot_cv_hit_curves, plot_test_pvalue_enrichment_curves
from config_utils import read_config, write_results_csv
from fusion_utils import select_best_models
from fusion_utils import fusion_pipeline, plot_fusion_hit_curves
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
    config['train_data'] = train_paths
    print(train_paths)

    # Build CV folds and persist them under {RunFolderName}/CVFolds/.
    # This step is heavy (especially for 'cluster' method); comment it out to reuse
    # the folds saved on disk from a previous run.
    print("Step 2.5: Create CV folds")
    create_cv_folds(config, RunFolderName)

    # Per-fold UMAP plots (Jaccard) so train/test positives can be inspected visually.
    # Heavy step — comment out to skip.
    print("Step 2.6: Plot CV folds UMAP")
    plot_cv_folds_umap(config, RunFolderName)

    # Execute the training pipeline
    print("Step 3: Train pipeline")
    train_pipeline(config, RunFolderName, load_data, fuse_columns, get_model, train_model,
               cross_validate_and_save_models, train_and_save_final_model,
               bayesian_hyperparameter_search, write_results_csv)

    # CV hit-curve plots (per-fold + mean over folds) for every CV prediction set
    # under {RunFolderName}/CVResults/. Comment out to skip.
    print("Step 3.5: CV hit curve plots")
    plot_cv_hit_curves(config, RunFolderName)

    # Execute the testing pipeline
    print("Step 4: Test pipeline")
    test_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model, get_model, train_model)

    # Cumulative-hits plots (model vs random baseline) for every *_predictions_sorted.parquet
    # under {RunFolderName}/Predictions/. Comment out to skip.
    print("Step 4.5: Hit curve plots")
    plot_hit_curves(config, RunFolderName)

    # Hypergeometric p-value and enrichment curves vs. top-n for the same
    # test predictions parquets (no random baseline). Test set only.
    print("Step 4.6: Test p-value and enrichment plots")
    plot_test_pvalue_enrichment_curves(config, RunFolderName)

    # Execute model selection to find the best models
    print("Step 5: Model selection")
    select_best_models(config, RunFolderName)

    # Overlay hit curves of all selected best models, one figure per test file.
    print("Step 5.5: Best-models hit curves")
    plot_best_models_hit_curves(config, RunFolderName)

    # Execute the fusion pipeline to combine the top models
    print("Step 6: Model fusion")
    fusion_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model)

    # Hit@K curves for fusion outputs (mean & max overlaid per test file).
    print("Step 6.5: Fusion hit curves")
    plot_fusion_hit_curves(config, RunFolderName)
    
    # Virtual Screening: Calculating screening data probability, applying chemistry filters, and clustering results
    #print("Step 7: Virtual screening")
    #screening_pipeline(config, RunFolderName, load_data, fuse_columns, evaluate_model, get_model, train_model)
    
    #print("Step 8: Logging the results using mlflow")
    # Loggingparameters, metric, artifacts and models using mlflow
    # run_name = f"{config.get('protein_name')}_{timestamp}"
    # log_results_to_mlflow (RunFolderName, config.get("experiment_name", "Default_Experiment"), run_name)
    
    #print("Step 9: Plotting some results")
    # plot_function(RunFolderName)
#=========================================================================='''


# Example usage: Execute the pipeline if the script is run directly
if __name__ == "__main__":
    start_time = time.time()
    run_pipeline()
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"\nTotal runtime: {elapsed_minutes:.2f} minutes")
