# -*- coding: utf-8 -*-
"""
1- navigate to the root directory of your ML project

2- open a terminal and run:
     mlflow ui
     
     
3- Then go to your browser and open:
     http://127.0.0.1:5000
"""

import mlflow
import pandas as pd
import os
import joblib

import logging
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

load_dotenv()
# Suppress only MLflow model logger warnings
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)

service_account_path = 'service_account.json'
# service_account_path = '../app/service_account.json'

# Check if the file exists before setting the environment variable
if os.path.exists(service_account_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
    logging.info("Service account credentials set.")
else:
    logging.info("Service account file not found. Skipping credential setup.")

mlflow_uri = mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', '').strip()
mlflow.set_tracking_uri(mlflow_uri)
print(f"MLflow tracking URI set to: {mlflow_uri}")


'''metric_cols = ['CV_Accuracy', 'CV_Precision', 'CV_Recall', 'CV_F1Score', 'CV_AUC-ROC', 'CV_AUC-PR', 'CV_MCC', 'CV_Cohen Kappa',
                'CV_balanced_accuracy', 'CV_PlatePPV', 'CV_DivPlatePPV', 'Test_Accuracy', 'Test_Precision',
                'Test_Recall', 'Test_F1Score', 'Test_AUC-ROC', 'Test_AUC-PR', 'Test_MCC', 'Test_Cohen Kappa',
                'Test_balanced_accuracy','Test_PlatePPV', 'Test_DivPlatePPV', 'Test_Hits@200', 'Test_Hits@500', 'Test_Precision@200',
                'Test_Precision@500', 'CV_Hits@200', 'CV_Hits@500', 'CV_Precision@200', 'CV_Precision@500']'''


def log_results_to_mlflow(RunFolderName, experiment_name ,run_name):
    print(f"Logging results to MLflow for run: {run_name} in experiment: {experiment_name}")
    print(f"MLflow tracking URI set to: {mlflow_uri}")
    param_cols = ['nrows_train', 'nrows_test', 'Nfold', 'confromal_test_size', 'confromal_confidence_level', 'balance_ratios',
                  'run_name', 'protein_name', 'TrainFileName', 'TestFile', 'feature_fusion_method',
                  'ModelType', 'UsedHyperParameters', 'conformal_prediction', 'ColumnName', 'Date', 'Time']
    
    metric_cols = ['CV_Accuracy', 'CV_Precision', 'CV_Recall', 'CV_F1Score', 'CV_AUC-ROC', 'CV_AUC-PR', 'CV_MCC', 'CV_Cohen Kappa',
                   'CV_balanced_accuracy', 'CV_PlatePPV', 'CV_DivPlatePPV', 'Test_Accuracy', 'Test_Precision',
                   'Test_Recall', 'Test_F1Score', 'Test_AUC-ROC', 'Test_AUC-PR', 'Test_MCC', 'Test_Cohen Kappa',
                   'Test_balanced_accuracy','Test_PlatePPV', 'Test_DivPlatePPV', 'Test_HitsAt200', 'Test_HitsAt500', 'Test_PrecisionAt200',
                   'Test_PrecisionAt500', 'CV_HitsAt200', 'CV_HitsAt500', 'CV_PrecisionAt200', 'CV_PrecisionAt500']
    
    artifact_cols = ['run_name', 'protein_name', 'TrainFileName', 'TestFile']
    model_cols = ['ModelPath']
    
    results_path = os.path.join(RunFolderName, "results.csv")
    results_df = pd.read_csv(results_path)
    
    mlflow.set_experiment(experiment_name)
    
    for i, row in results_df.iterrows():
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            for col in param_cols:
                try:
                    mlflow.log_param(col, row[col])
                except Exception as e:
                    print(f"Could not log parameter {col}: {e}")
            
            # Log metrics
            for col in metric_cols:
                value = row[col]
                try:
                    if pd.notnull(value):
                        mlflow.log_metric(col, float(value))
                except Exception as e:
                    print(f"Could not log metric {col}: {value} ({type(value)}). Error: {e}")
            
            # Log models - FIXED VERSION
            # exit()
            for col in model_cols:
                model_path = row[col]
                '''model_file = os.path.join(model_path, "model.pkl")
                if os.path.exists(model_file):
                    try:
                        # Option 1: Log as artifact 
                        mlflow.log_artifact(model_file, "model")
                        print(f"Successfully logged model as artifact from {model_file}")
                        
                        # Option 2: If you want to try sklearn logging, uncomment below:
                        # model_obj = joblib.load(model_file)
                        # mlflow.sklearn.log_model(model_obj, "model")
                        
                    except Exception as e:
                        print(f"Could not log model from {model_path}. Error: {e}")
                        # Fallback: just log the model file as an artifact
                        try:
                            mlflow.log_artifact(model_file, "model_backup")
                        except Exception as e2:
                            print(f"Fallback artifact logging also failed: {e2}")
                else:
                    print(f"Warning: model file {model_file} does not exist.")'''
                model_file = None
                for ext in [".pkl", ".h5"]:
                    candidate = os.path.join(model_path, f"model{ext}")
                    if os.path.exists(candidate):
                        model_file = candidate
                        break
            
                if model_file:
                    try:
                        if model_file.endswith(".pkl"):
                            model_obj = joblib.load(model_file)
                            mlflow.sklearn.log_model(model_obj, artifact_path="model")
                            print(f"Logged sklearn model from {model_file}")
            
                        elif model_file.endswith(".h5"):
                            model_obj = load_model(model_file)
                            mlflow.keras.log_model(model_obj, artifact_path="model")
                            print(f"Logged Keras model from {model_file}")
            
                    except Exception as e:
                        print(f"Could not log model from {model_file}. Error: {e}")
                        try:
                            mlflow.log_artifact(model_file, artifact_path="model_backup")
                            print(f"Logged model file as artifact (backup): {model_file}")
                        except Exception as e2:
                            print(f"Backup artifact logging failed: {e2}")
                else:
                    print(f"Warning: No model file (.pkl or .h5) found in {model_path}")
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    