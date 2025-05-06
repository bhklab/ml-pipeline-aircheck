#==========================================================================
from data_utils import load_data, fuse_columns
from model_utils import get_model, train_model
from model_utils import cross_validate_and_save_models, train_and_save_final_model
from model_utils import bayesian_hyperparameter_search
from model_utils import train_pipeline
from eval_utils import evaluate_model, test_pipeline
from config_utils import read_config, write_results_csv
from fusion_utils import select_best_models 
#==========================================================================





#==========================================================================
def run_pipeline(config_name="config.yaml"):
    
    #==============================================================================
    # Load and check config
    config, RunFolderName = read_config(config_name)  
    train_path = config['train_data']
    test_path = config['test_data']
    model_names = config['desired_models']
    column_names = config['desired_columns']
    nrows_train = config['nrows_train']
    nrows_test = config['nrows_test']  
    label_column_train = config['label_column_train']
    label_column_test = config['label_column_test']
    FeatureFusionAll = config['FeatureFusionAll']
    Nfold = config['Nfold']
    Train = config['Train']
    Test = config['Test']
    hyperparameters_tuning = config['hyperparameters_tuning']
    hyperparameters = config.get('hyperparameters', {})
    evaluation_file_name = config['evaluation_file_name']
    evaluation_column = config['evaluation_column']
    Fusion = config['Fusion']
    num_top_models = config['num_top_models']
    #==============================================================================
    # Extrac feature and dataprocessing steps
    # fuse n features   
    # Feature manipulation, pca, ...
    # Maybe combing train data
    # Maybe adding differnt ratio of negatives and adding to train data and the train_path    
    # Cleaning train data with similarity check
    #==============================================================================
    train_args = {
        "Train": Train,
        "RunFolderName": RunFolderName,
        "train_path": train_path,
        "column_names": column_names,
        "label_column_train": label_column_train,
        "nrows_train": nrows_train,
        "model_names": model_names,
        "hyperparameters_tuning": hyperparameters_tuning,
        "hyperparameters": hyperparameters,
        "Nfold": Nfold,
        "FeatureFusionAll": FeatureFusionAll,
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
    train_pipeline(**train_args)
    
    #==============================================================================
    # Test
    test_args = {
        'Test': Test,
        'RunFolderName': RunFolderName,
        'test_paths': test_path,
        'column_names': column_names,
        'label_column_test': label_column_test,
        'nrows_test': nrows_test,
        'load_data': load_data,
        'fuse_columns': fuse_columns,
        'evaluate_model': evaluate_model,
        'FeatureFusionAll': FeatureFusionAll
    }   
    test_pipeline(**test_args)
    #==============================================================================
    # Model Selection
    model_selection_args = {
        'RunFolderName': RunFolderName,
        'evaluation_file_name': evaluation_file_name,   
        'evaluation_column': evaluation_column,        
        'num_top_models': num_top_models,
        'Fusion': Fusion                      
    }
    select_best_models(**model_selection_args)
    #==============================================================================
    #==============================================================================
    # Clustering
    # Evaluate after clustering
    
    # Applying Chemistry based filters
    # Evaluate after applying these filters
    
    # Criteria for selecting final model
    
    # Selecting best model for staging
    # for this we need to combine all result.csv files, query for the experiment we want to decide
    # we also need a very precise criteria, here maybe thinking about statistical significanse and ...
#==========================================================================


# Example usage
if __name__ == "__main__":
    run_pipeline()


