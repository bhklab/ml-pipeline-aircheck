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
# Suppress only MLflow model logger warnings
logging.getLogger("mlflow.models.model").setLevel(logging.ERROR)


def log_results_to_mlflow(RunFolderName):
    
    param_cols = ['nrows_train', 'nrows_test', 'Nfold', 'confromal_test_size', 'confromal_confidence_level', 'balance_ratios',
                  'run_name', 'protein_name', 'TrainFileName', 'TestFile', 'feature_fusion_method',
                  'ModelType', 'UsedHyperParameters', 'conformal_prediction', 'ColumnName', 'Date', 'Time']
    
    metric_cols = ['CV_Accuracy', 'CV_Precision', 'CV_Recall', 'CV_F1 Score', 'CV_AUC-ROC', 'CV_AUC-PR', 'CV_MCC', 'CV_Cohen Kappa',
                   'CV_balanced_accuracy', 'CV_PlatePPV', 'CV_DivPlatePPV', 'Test_Accuracy', 'Test_Precision',
                   'Test_Recall', 'Test_F1 Score', 'Test_AUC-ROC', 'Test_AUC-PR', 'Test_MCC', 'Test_Cohen Kappa',
                   'Test_balanced_accuracy','Test_PlatePPV', 'Test_DivPlatePPV']
    
    artifact_cols = ['run_name', 'protein_name', 'TrainFileName', 'TestFile']
                     #, 'feature_fusion_method', 'ModelType', 'conformal_prediction', 'ColumnName', 'Date', 'Time']
    
    #'UsedHyperParameters', 
    model_cols = ['ModelPath']
    
    results_path = os.path.join(RunFolderName, "results.csv")
    results_df = pd.read_csv(results_path)
    
    #mlflow.set_experiment("aircheck-experiment") 
    for i, row in results_df.iterrows():
        with mlflow.start_run():
            # Log parameters
            for col in param_cols:
                mlflow.log_param(col, row[col])

            # Log metrics
            for col in metric_cols:
                mlflow.log_metric(col, row[col])

 
            # Log models
            for col in model_cols:
                model_path = row[col]
                model_file = os.path.join(model_path, "model.pkl")
                if os.path.exists(model_file):
                    model_obj = joblib.load(model_file)
                    mlflow.sklearn.log_model(model_obj)
                else:
                    print(f"Warning: model file {model_path} does not exist.")
        #mlflow.end_run()
