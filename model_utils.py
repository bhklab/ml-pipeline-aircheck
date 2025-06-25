from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import pickle
import os
import numpy as np
from skopt import BayesSearchCV
from eval_utils import evaluate_model
import warnings
warnings.filterwarnings("ignore")
#==============================================================================
#==============================================================================
def get_model(model_name, best_params={}):
    if model_name == 'rf':
        return RandomForestClassifier(**best_params) if best_params else RandomForestClassifier()
    elif model_name == 'lr':
        return LogisticRegression(**best_params) if best_params else LogisticRegression()
    elif model_name == 'sgd':
        best_params['loss'] = 'log_loss'
        return SGDClassifier(**best_params) if best_params else SGDClassifier(loss = 'log_loss')
    elif model_name == 'svc':
        best_params['probability'] = True
        return SVC(**best_params) if best_params else SVC(probability=True)
    elif model_name == 'nb':
        return GaussianNB(**best_params) if best_params else GaussianNB()
    elif model_name == 'dt':
        return DecisionTreeClassifier(**best_params) if best_params else DecisionTreeClassifier()
    elif model_name == 'knn':
        return KNeighborsClassifier(**best_params) if best_params else KNeighborsClassifier()
    elif model_name == 'gb':
        return GradientBoostingClassifier(**best_params) if best_params else GradientBoostingClassifier()
    elif model_name == 'ada':
        return AdaBoostClassifier(**best_params) if best_params else AdaBoostClassifier()
    elif model_name == 'bag':
        return BaggingClassifier(**best_params) if best_params else BaggingClassifier()
    elif model_name == 'mlp':
        return MLPClassifier(**best_params) if best_params else MLPClassifier()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
#==============================================================================
#==============================================================================
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
#==============================================================================
#==============================================================================
def train_pipeline(config,
                   RunFolderName,
                   load_data,
                   fuse_columns,
                   get_model,
                   train_model,
                   cross_validate_and_save_models,
                   train_and_save_final_model,
                   bayesian_hyperparameter_search,
                   write_results_csv):
    
    Train = config['Train']
    train_paths = config['train_data']
    column_names = config['desired_columns']
    label_column_train = config['label_column_train']
    nrows_train = config['nrows_train']
    model_names = config['desired_models']
    hyperparameters_tuning = config['hyperparameters_tuning']
    hyperparameters = config.get('hyperparameters', {})
    Nfold = config['Nfold']
    feature_fusion_method = config['feature_fusion_method']

    if Train.lower() != 'y':
        return

    for train_path in train_paths:
        # Extract the filename without extension for naming
        train_filename = os.path.basename(train_path).split('.')[0]

        # === Load Training Data ===
        X_train, Y_train = load_data(train_path, column_names, label_column_train, nrows_train)
        Y_train_array = np.stack(Y_train.iloc[:, 0])

        # === Feature Fusion (optional) ===
        X_train, fused_column_name = fuse_columns(X_train, column_names , feature_fusion_method)

        for model_name_i in model_names:
            for column_names_j in fused_column_name:
                X_train_array = np.stack(X_train[column_names_j])
                
                # Model subfolder includes train filename
                model_subfolder = os.path.join(RunFolderName, f"{train_filename}_{model_name_i}_{column_names_j}")
                os.makedirs(model_subfolder, exist_ok=True)

                # === Hyperparameter Tuning ===
                if hyperparameters_tuning.lower() == 'y':
                    best_params = bayesian_hyperparameter_search(model_name_i, X_train_array, Y_train_array)
                else:
                    best_params = hyperparameters.get(model_name_i, {})

                # === Cross Validation ===
                avg_metrics = cross_validate_and_save_models(
                    X_train_array=X_train_array,
                    Y_train_array=Y_train_array,
                    model_name=model_name_i,
                    model_subfolder=model_subfolder,
                    Nfold=Nfold,
                    get_model=get_model,
                    train_model=train_model,
                    evaluate_model=evaluate_model,
                    best_params=best_params
                )

                # === Final Model Training ===
                train_and_save_final_model(
                    X_train_array,
                    Y_train_array,
                    model_name_i,
                    model_subfolder,
                    get_model,
                    train_model,
                    best_params
                )

                # === Save Results ===
                experiment_results = config.copy()
                experiment_results["train_path"] = train_path
                experiment_results["TrainFileName"] = train_filename
                experiment_results["ModelType"] = model_name_i
                experiment_results["ColumnName"] = column_names_j
                experiment_results["ModelPath"] = model_subfolder
                experiment_results["UsedHyperParameters"] = best_params

                for key, value in avg_metrics.items():
                    experiment_results[f"CV_{key}"] = value

                write_results_csv(experiment_results, RunFolderName)

#==============================================================================
#==============================================================================
def cross_validate_and_save_models(X_train_array, Y_train_array, model_name, model_subfolder, Nfold, get_model, train_model, evaluate_model, best_params):
    """
    Perform cross-validation, save fold models, and return average metrics.

    Parameters:
    - X_train_array: np.ndarray
    - Y_train_array: np.ndarray
    - model_name: str — name of the model to initialize
    - model_subfolder: str — folder path to save fold models
    - Nfold: int — number of cross-validation folds
    - get_model: function to get model instance
    - train_model: function to train a model
    - evaluate_model: function to evaluate a model

    Returns:
    - avg_metrics: dict — average evaluation metrics across folds
    """

    skf = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=42)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_train_array, Y_train_array)):
        CrossVal_data_train, CrossVal_data_test = X_train_array[train_idx], X_train_array[test_idx]
        CrossVal_label_train, CrossVal_label_test = Y_train_array[train_idx], Y_train_array[test_idx]

        model_fold = get_model(model_name , best_params)
        model_fold = train_model(model_fold, CrossVal_data_train, CrossVal_label_train)

        metrics = evaluate_model(model_fold, CrossVal_data_test, CrossVal_label_test)
        fold_metrics.append(metrics)

        # Save model for this fold
        fold_model_path = os.path.join(model_subfolder, f"model_fold{fold_idx + 1}.pkl")
        with open(fold_model_path, 'wb') as f:
            pickle.dump(model_fold, f)

    # Average metrics across folds
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0]}
    return avg_metrics
#==============================================================================
#==============================================================================
def train_and_save_final_model(X_train_array, Y_train_array, model_name, model_subfolder, get_model, train_model, best_params):
    """
    Trains the final model and saves it to the specified folder.

    Parameters:
    - X_train_array: np.ndarray — training features
    - Y_train_array: np.ndarray — training labels
    - model_name: str — name of the model to initialize
    - model_subfolder: str — folder path to save the model
    - get_model: function to get model instance
    - train_model: function to train a model

    Returns:
    - None
    """
    model = get_model(model_name , best_params)
    model = train_model(model, X_train_array, Y_train_array)

    model_path = os.path.join(model_subfolder, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
#==============================================================================
#==============================================================================
def bayesian_hyperparameter_search(model_name, X_train, y_train, cv=5, n_iter=32, random_state=42):
    if model_name == 'rf':
        model = RandomForestClassifier()
        param_space = {
            'n_estimators': (100, 200),
            'max_depth': (5, 15),
        }
    elif model_name == 'lr':
        model = LogisticRegression()
        param_space = {
            'C': (0.01, 10.0, 'log-uniform'),
            'penalty': ['l2'],
        }
    elif model_name == 'ridge':
        model = RidgeClassifier()
        param_space = {
            'alpha': (0.01, 10.0, 'log-uniform'),
        }
    elif model_name == 'sgd':
        model = SGDClassifier()
        param_space = {
            'alpha': (1e-5, 1e-2, 'log-uniform'),
            'penalty': ['l2', 'l1'],
        }
    elif model_name == 'perceptron':
        model = Perceptron()
        param_space = {
            'penalty': ['l2', 'l1', None],
            'alpha': (1e-5, 1e-2, 'log-uniform'),
        }
    elif model_name == 'svc':
        model = SVC()
        param_space = {
            'C': (0.1, 10.0, 'log-uniform'),
            'kernel': ['linear', 'rbf'],
        }
    elif model_name == 'nb':
        model = GaussianNB()
        param_space = {
            'var_smoothing': (1e-11, 1e-8, 'log-uniform'),
        }
    elif model_name == 'dt':
        model = DecisionTreeClassifier()
        param_space = {
            'max_depth': (3, 10),
            'min_samples_split': (2, 10),
        }
    elif model_name == 'knn':
        model = KNeighborsClassifier()
        param_space = {
            'n_neighbors': (3, 15),
            'weights': ['uniform', 'distance'],
        }
    elif model_name == 'gb':
        model = GradientBoostingClassifier()
        param_space = {
            'n_estimators': (100, 200),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'max_depth': (3, 7),
        }
    elif model_name == 'ada':
        model = AdaBoostClassifier()
        param_space = {
            'n_estimators': (50, 150),
            'learning_rate': (0.5, 1.5, 'uniform'),
        }
    elif model_name == 'bag':
        model = BaggingClassifier()
        param_space = {
            'n_estimators': (10, 50),
            'max_samples': (0.5, 1.0, 'uniform'),
        }
    elif model_name == 'mlp':
        model = MLPClassifier(max_iter=300)
        param_space = {
            'hidden_layer_sizes': [(50,), (100,), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': (1e-5, 1e-2, 'log-uniform'),
        }
    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")
    
    search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=cv,
        random_state=random_state
    )
    
    search.fit(X_train, y_train)
    return search.best_params_
#==============================================================================
#==============================================================================

