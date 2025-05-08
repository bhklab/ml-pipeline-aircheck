import pandas as pd
import pickle
import os
import numpy as np
import rdkit
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit import DataStructs   
from rdkit.DataStructs  import BulkTanimotoSimilarity
from tqdm import tqdm 
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
    average_precision_score
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
def test_pipeline(
    Test,
    RunFolderName,
    test_paths,
    column_names,
    label_column_test,
    nrows_test,
    load_data,
    fuse_columns,
    evaluate_model,
    feature_fusion_method):
    
    if Test.lower() != 'y':
        return

    results_path = os.path.join(RunFolderName, "results.csv")
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

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            test_metrics = evaluate_model(model, X_test_array, Y_test_array)

            row_result = row.copy()
            for key, value in test_metrics.items():
                row_result[f"Test_{key}"] = value

            row_result["TestFile"] = os.path.basename(test_path)
            updated_rows.append(row_result)

    df_updated = pd.DataFrame(updated_rows)
    df_updated.to_csv(results_path, index=False)


#------------------------------------------------------------------------------
