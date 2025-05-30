import pandas as pd
import pickle
import os
import numpy as np
import rdkit
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit import DataStructs   
from rdkit.DataStructs  import BulkTanimotoSimilarity
from tqdm import tqdm 
import ast
from sklearn.model_selection import train_test_split
from mapie.classification import SplitConformalClassifier
from mapie.metrics.classification import classification_coverage_score
# top 100, 2000 metrics?? Number of hits?
# Threshold??????


#==============================================================================
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    classification_report,
    balanced_accuracy_score,
    average_precision_score,
)
#==============================================================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(X_test,y_test,  y_pred, y_proba)
    return metrics
#==============================================================================
def calculate_metrics(X_test,y_test,  y_pred, y_proba):  
    ppv = precision_score(y_test, y_pred)
    p_ppv = plate_ppv(y_test, y_pred, top_n=128) # Why this number!!! edit later !!!!!!!!!!
    clusters = cluster_leader_from_array (X_test)
    dp_ppv = diverse_plate_ppv(y_test, y_pred, clusters=clusters.tolist())

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
        "AUC-PR": average_precision_score(y_test, y_proba),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "PlatePPV": p_ppv,
        "DivPlatePPV": dp_ppv
    }
    return metrics
#==============================================================================
# PPV (Positive Predictive Value)
#                  True Positives
# PPV = ---------------------------------
#       (True Positives + False Positives)
def plate_ppv(y, y_pred, top_n: int = 128):
    y_pred = np.atleast_1d(y_pred)
    y = np.atleast_1d(y)
    _tmp = np.vstack((y, y_pred)).T[y_pred.argsort()[::-1]][:top_n, :]
    _tmp = _tmp[np.where(_tmp[:, 1] > 0.5)[0]].copy()
    return np.sum(_tmp[:, 0]) / len(_tmp)


def diverse_plate_ppv(y, y_pred, clusters: list, top_n_per_group: int = 15):
    df = pd.DataFrame({"pred": y_pred, "real": y, "CLUSTER_ID": clusters})
    df_groups = df.groupby("CLUSTER_ID")

    _vals = []
    for group, idx in df_groups.groups.items():
        _tmp = df.iloc[idx].copy()
        if sum(df.iloc[idx]["pred"] > 0.5) == 0:
            continue
        _tmp = _tmp[_tmp["pred"] > 0.5].copy()
        _tmp = np.vstack((_tmp["real"].to_numpy(), _tmp["pred"].to_numpy())).T[_tmp["pred"].to_numpy().argsort()[::-1]][:top_n_per_group, :]
        _val = np.sum(_tmp[:, 0]) / len(_tmp)
        _vals.append(_val)

    return np.mean(_vals)

#==============================================================================
def cluster_leader_from_array(X, thresh: float = 0.65):
    """
    Generate a cluster ID map for an already featurized array such that each cluster centroid has a Tanimoto similarity
    below the passed threshold.
    
    Parameters:
    - X (np.ndarray): Array of numeric features.
    - thresh (float): Tanimoto similarity threshold for clustering.
    
    Returns:
    - np.ndarray[int]: An array of cluster IDs.
    """
    # Ensure values are numeric
    X = np.array(X, dtype=float)  # Convert to numeric (float)
    # Convert each row to a binary fingerprint
    _fps = [DataStructs.CreateFromBitString(
        "".join(["1" if __ > 0 else "0" for __ in _])) for _ in X]
    # Leader clustering (RDKit)
    lp = rdSimDivPickers.LeaderPicker()
    _centroids = lp.LazyBitVectorPick(_fps, len(_fps), thresh)
    _centroid_fps = [_fps[i] for i in _centroids]
    # Assign each fingerprint to the closest centroid
    _cluster_ids = [
        np.argmax(DataStructs.BulkTanimotoSimilarity(_fp, _centroid_fps))
        for _fp in _fps
    ]
    return np.array(_cluster_ids)
#==============================================================================
def test_pipeline(config,
                  RunFolderName,
                  load_data,
                  fuse_columns,
                  evaluate_model,
                  get_model,
                  train_model):
    
    Test = config['Test']
    test_paths = config['test_data']
    column_names = config['desired_columns']
    label_column_test = config['label_column_test']
    nrows_test = config['nrows_test']
    nrows_train = config['nrows_train']
    feature_fusion_method = config['feature_fusion_method']
    conformal_prediction = config['conformal_prediction']
    confromal_test_size = config['confromal_test_size']
    confromal_confidence_level = config['confromal_confidence_level']
    
    if Test.lower() != 'y':
        return

    results_path = os.path.join(RunFolderName, "results.csv")
    
    # Check if results.csv exists
    if not os.path.exists(results_path):
        print("Error: 'results.csv' not found. Please ensure you have run the training phase before testing.")
        return

    df = pd.read_csv(results_path)

    updated_rows = []

    for test_path in test_paths:
        # Load test data
        X_test, Y_test = load_data(test_path, column_names, label_column_test, nrows_test)
        Y_test_array = np.stack(Y_test.iloc[:, 0])

        # === Feature Fusion  ===
        X_test, fused_column_name = fuse_columns(X_test, column_names, feature_fusion_method)

        for _, row in df.iterrows():
            model_path = row["ModelPath"]
            if os.path.isdir(model_path):
                model_path = os.path.join(model_path, "model.pkl")

            column_name = row["ColumnName"]

            try:
                X_test_array = np.stack(X_test[column_name])
            except KeyError:
                print(f"Column '{column_name}' not in test file: {test_path}. Skipping.")
                continue
            
            # Check if the model file exists
            if not os.path.exists(model_path):
                print(f"Error: Model file not found at '{model_path}'. Please ensure the model was saved correctly during training.")
                raise SystemExit("Terminating: Model file missing.")

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            test_metrics = evaluate_model(model, X_test_array, Y_test_array)
            
            
            if conformal_prediction.lower() == 'y':               
                [confromal_coverage_score, confromal_confidence_score] = compute_conformal_prediction(get_model, train_model, load_data,fuse_columns,row,nrows_train,feature_fusion_method, X_test_array, Y_test_array)
                row["confromal_coverage_score"] = confromal_coverage_score
                row["onfromal_confidence_score"] = confromal_confidence_score

            row_result = row.copy()
            for key, value in test_metrics.items():
                row_result[f"Test_{key}"] = value

            row_result["TestFile"] = os.path.basename(test_path)
            updated_rows.append(row_result)

    df_updated = pd.DataFrame(updated_rows)
    df_updated.to_csv(results_path, index=False)
#------------------------------------------------------------------------------


def compute_conformal_prediction(get_model, train_model, load_data, fuse_columns, row, nrows_train, feature_fusion_method, X_test_array, Y_test_array):
    #----
    model_name = row ["ModelType"]
    column_names = row["ColumnName"]
    label_column_train = row["label_column_train"]
    confromal_test_size = row["confromal_test_size"]
    confromal_confidence_level= row['confromal_confidence_level']
    train_path = row["train_path"]
    model_parameters = row["UsedHyperParameters"]    
    model_parameters = ast.literal_eval(model_parameters)
    #-----
    X_train, Y_train = load_data(train_path, [column_names], [label_column_train], nrows_train)
    Y_train_array = np.stack(Y_train.iloc[:, 0])
    X_train, fused_column_name = fuse_columns(X_train, column_names , feature_fusion_method)
    X_train_array = np.stack(X_train[column_names])
    #-----
    
    #----------------------Conformal prediction ---------------------------
    # Split train data into training and calibration (conformalization) sets
    X_train_model, X_calib, y_train_model, y_calib = train_test_split(
        X_train_array,Y_train_array, test_size= confromal_test_size, random_state=42
    )
    
    base_model = get_model(model_name , model_parameters)
    base_model.fit(X_train_model, y_train_model)
    
    # Wrap with MAPIE
    mapie = SplitConformalClassifier(
        estimator=base_model,
        confidence_level= confromal_confidence_level,
        prefit=True
    )

    # Use calibration data to compute nonconformity scores and set the prediction threshold
    # Run conformal calibration
    mapie.conformalize(X_calib, y_calib)
    
    # Predict sets on test data
    y_pred, y_pred_set = mapie.predict_set(X_test_array)

    # If we want to select a model based on conformal prediction:
    # We choose the one that balances high coverage and small prediction sets    
    # Calculate how often true labels are in the prediction sets
    confromal_coverage_score = classification_coverage_score(Y_test_array, y_pred_set)
    #print(f"Effective coverage: {coverage_score[0]:.3f}")
    
    # Calculate average number of labels in prediction sets
    # 1 ≤ avg_set_size ≤ K (K total classes)
    # 1 → perfect confidence
    # K → maximum uncertainty
    avg_set_size = y_pred_set.sum(axis=1).mean()
    
    # confidence_score = 1 - (avg_set_size - 1) / (K - 1)
    # 1: Fully confident, 0: Fully uncertain
    confromal_confidence_score = 2 - avg_set_size
    
    #confromal_coverage_score=1
    #confromal_confidence_score=1
    
    return confromal_coverage_score, confromal_confidence_score























