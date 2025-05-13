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
def run_pipeline(config_name="config.yaml"):
    """
    Main function to execute the AIRCHECK ML pipeline.
    Loads configuration, initiates training, testing, model selection, and fusion.
    """
    
    #==============================================================================
    # Load and validate configuration file
    config, RunFolderName = read_config(config_name)  
    
    # Extract necessary parameters from the configuration file
    protein_name = config['protein_name']
    train_paths = config['train_data']
    test_paths = config['test_data']
    model_names = config['desired_models']
    column_names = config['desired_columns']
    nrows_train = config['nrows_train']
    nrows_test = config['nrows_test']  
    label_column_train = config['label_column_train']
    label_column_test = config['label_column_test']
    feature_fusion_method = config['feature_fusion_method']
    Nfold = config['Nfold']
    Train = config['Train']
    Test = config['Test']
    hyperparameters_tuning = config['hyperparameters_tuning']
    hyperparameters = config.get('hyperparameters', {})
    trainfile_for_modelselection = config['trainfile_for_modelselection']
    evaluationfile_for_modelselection = config['evaluationfile_for_modelselection']
    evaluation_column = config['evaluation_column']
    crossvalidation_column = config['crossvalidation_column']
    Fusion = config['Fusion']
    num_top_models = config['num_top_models']
    balance_flag = config['balance_flag']
    balance_ratios = config['balance_ratios']
    #==============================================================================

    # Prepare training paths based on balance settings
    kwargs = {
        "train_paths": train_paths,
        "label_column": label_column_train,
        "balance_ratios": balance_ratios,
        "balance_flag": balance_flag
    }   
    train_paths = create_balanced_datasets(**kwargs)

    #==============================================================================
    # Prepare training arguments for the train pipeline function
    train_args = {
        "Train": Train,
        "RunFolderName": RunFolderName,
        "train_paths": train_paths,
        "column_names": column_names,
        "label_column_train": label_column_train,
        "nrows_train": nrows_train,
        "model_names": model_names,
        "hyperparameters_tuning": hyperparameters_tuning,
        "hyperparameters": hyperparameters,
        "Nfold": Nfold,
        "feature_fusion_method": feature_fusion_method,
        "load_data": load_data,
        "fuse_columns": fuse_columns,
        "get_model": get_model,
        "train_model": train_model,
        "cross_validate_and_save_models": cross_validate_and_save_models,
        "train_and_save_final_model": train_and_save_final_model,
        "bayesian_hyperparameter_search": bayesian_hyperparameter_search,
        "write_results_csv": write_results_csv,
        "config": config
    }
    # Execute the training pipeline
    train_pipeline(**train_args)
    
    #==============================================================================
    # Prepare testing arguments for the test pipeline function
    test_args = {
        'Test': Test,
        'RunFolderName': RunFolderName,
        'test_paths': test_paths,
        'column_names': column_names,
        'label_column_test': label_column_test,
        'nrows_test': nrows_test,
        'load_data': load_data,
        'fuse_columns': fuse_columns,
        'evaluate_model': evaluate_model,
        'feature_fusion_method': feature_fusion_method
    }   
    # Execute the testing pipeline
    test_pipeline(**test_args)
    #==============================================================================

    # Prepare model selection arguments
    model_selection_args = {
        'RunFolderName': RunFolderName,
        'trainfile_for_modelselection': trainfile_for_modelselection,
        'evaluationfile_for_modelselection': evaluationfile_for_modelselection,   
        'evaluation_column': evaluation_column,
        'crossvalidation_column': crossvalidation_column,       
        'num_top_models': num_top_models,
        'Fusion': Fusion                      
    }
    # Execute model selection to find the best models
    select_best_models(**model_selection_args)
    #==============================================================================

    # Prepare score fusion arguments for the fusion pipeline function
    fusion_args = {
        'RunFolderName': RunFolderName,
        'test_paths': test_paths,
        'column_names': column_names,
        'label_column_test': label_column_test,
        'nrows_test': nrows_test,
        'load_data': load_data,
        'fuse_columns': fuse_columns,
        'evaluate_model': evaluate_model,
        'feature_fusion_method': feature_fusion_method
    }
    # Execute the fusion pipeline to combine the top models
    fusion_pipeline(**fusion_args)
#==========================================================================


# Example usage: Execute the pipeline if the script is run directly
if __name__ == "__main__":
    run_pipeline()
