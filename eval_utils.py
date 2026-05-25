import pandas as pd
import pickle
import os
import textwrap
import numpy as np
import rdkit
from rdkit.SimDivFilters import rdSimDivPickers
from rdkit import DataStructs   
from rdkit.DataStructs  import BulkTanimotoSimilarity
from tqdm import tqdm 
import ast
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model



#-------------------------------------
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
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
#-------------------------------------
def evaluate_model(model_path, X_test, y_test, area_hits_K=500,
                   regression_threshold='median', regression_target=None,
                   binary_threshold=0.5):
    """Evaluate a saved model on (X_test, y_test).

    Classifier path (model has predict_proba):
      y_proba = model.predict_proba(X)[:, 1]   # probability of positive class
      y_pred  = (y_proba > binary_threshold).astype(int)   # default 0.5

    Regressor path (no predict_proba):
      y_proba = model.predict(X)               # continuous score, used for RANKING
      y_pred  = (y_proba > threshold).astype(int)
        - threshold = median(y_proba) when regression_threshold == 'median'
        - threshold = float(regression_threshold) otherwise
      regression_target (continuous true values) enables Test_RMSE / Test_MAE / Test_R2.

    y_test stays the binary LABEL in both paths so ranking metrics
    (HitsAt*, AreaHitsAtK, NDCG, BEDROC, ...) are comparable across model types.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'. Please ensure the model was saved correctly during training.")
        raise SystemExit("Terminating: Model file missing.")

    # Row-chunked predict: avoids LightGBM/sklearn/Keras internally allocating a
    # 3.5+ GiB float32 cast of the whole input matrix at once. We slice X_test by
    # rows, run predict on each slice, concatenate.
    chunk_size = 20000

    def _predict_chunks(predict_fn, two_d_out=False):
        n = X_test.shape[0]
        outs = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            outs.append(predict_fn(X_test[start:end]))
        if not outs:
            return np.empty((0, 2) if two_d_out else (0,), dtype=np.float32)
        if two_d_out:
            return np.concatenate(outs, axis=0)
        return np.concatenate([np.asarray(o).flatten() for o in outs], axis=0)

    if model_path.endswith('.h5'):
        model = load_model(model_path)
        y_proba = _predict_chunks(lambda x: model.predict(x).flatten())
        try:
            bt = float(binary_threshold)
        except (TypeError, ValueError):
            bt = 0.5
        y_pred = (y_proba > bt).astype(int)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        if hasattr(model, 'predict_proba'):
            # Classifier — compute y_pred ourselves from y_proba so binary_threshold
            # actually takes effect (model.predict() would silently use 0.5).
            proba_2d = _predict_chunks(lambda x: model.predict_proba(x), two_d_out=True)
            y_proba = proba_2d[:, 1]
            try:
                bt = float(binary_threshold)
            except (TypeError, ValueError):
                bt = 0.5
            y_pred = (y_proba > bt).astype(int)
        else:
            # Regressor: predict() returns a continuous score; binarize for y_pred.
            y_proba = _predict_chunks(lambda x: np.asarray(model.predict(x)).flatten())
            if isinstance(regression_threshold, str) and regression_threshold.lower() == 'median':
                threshold = float(np.median(y_proba))
            else:
                try:
                    threshold = float(regression_threshold)
                except (TypeError, ValueError):
                    threshold = float(np.median(y_proba))
            y_pred = (y_proba > threshold).astype(int)

    metrics = calculate_metrics(
        X_test, y_test, y_pred, y_proba,
        area_hits_K=area_hits_K,
        regression_target=regression_target,
    )
    return metrics, y_pred, y_proba
#-------------------------------------



#-------------------------------------
def NormPrecision_at_k(y_true, y_pred, y_scores, k, random_seed=42, return_per_group=False):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_scores = np.asarray(y_scores)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    P = len(pos_idx)
    G = P // k  # number of full groups

    # If not enough positives for one group, fall back to normal precision@k
    if G == 0:
        _, prec = hits_and_precision_at_k(y_true, y_pred, y_scores, k)
        if return_per_group:
            return prec, [prec]
        return prec

    # Truncate positives to a multiple of k
    P_trunc = G * k

    # Shuffle positives reproducibly
    rng = np.random.default_rng(random_seed)
    pos_idx_shuffled = pos_idx.copy()
    rng.shuffle(pos_idx_shuffled)
    pos_idx_used = pos_idx_shuffled[:P_trunc]

    per_group_precisions = []

    # Build groups: each group = k positives + all negatives
    for g in range(G):
        group_pos = pos_idx_used[g * k : (g + 1) * k]
        group_idx = np.concatenate([neg_idx, group_pos])

        yt = y_true[group_idx]
        yp = y_pred[group_idx]
        ys = y_scores[group_idx]

        _, prec = hits_and_precision_at_k(yt, yp, ys, k)
        per_group_precisions.append(prec)

    mean_norm_precision = float(np.mean(per_group_precisions))

    if return_per_group:
        return mean_norm_precision, per_group_precisions
    return mean_norm_precision
#-------------------------------------



#-------------------------------------
def hits_and_precision_at_k(y_true, y_pred, y_scores, k):
    
    k = min(k, len(y_scores))  # Prevent index overflow

    top_k_idx = np.argsort(y_scores)[::-1][:k]
    top_k_true = np.array(y_true)[top_k_idx]
    top_k_pred = np.array(y_pred)[top_k_idx]
    
    hits = np.sum((top_k_true == 1) & (top_k_pred == 1))
    #predicted_positives = np.sum(top_k_pred == 1)

    #precision_at_k = hits / predicted_positives if predicted_positives > 0 else 0.0
    precision_at_k = hits / k
    print('hits:',hits,precision_at_k)
    return int(hits), precision_at_k
#-------------------------------------


# -------------------------------------------------------------------
# Area / weighted metrics over the top-K of a probability ranking.
# Used as the primary model-selection sort keys in evaluation_column /
# crossvalidation_column. K is plumbed through evaluate_model from
# config['area_hits_K'].
# -------------------------------------------------------------------

def _ideal_area_hits_at_k(P, K):
    """Maximum possible value of Σ_{k=1..K} hits@k (perfect ranker)."""
    P = int(P); K = int(K)
    if P >= K:
        return K * (K + 1) // 2
    return P * (P + 1) // 2 + P * (K - P)


def area_hits_at_k(y_true, y_score, K):
    """Raw cumulative-hits area: Σ_{k=1..K} hits@k. Higher = better.

    Earlier hits contribute more (they're counted at every k from their rank
    to K), and total volume still matters (more hits = more terms).
    """
    K = int(K)
    if K <= 0 or len(y_true) == 0:
        return 0
    K = min(K, len(y_true))
    order = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true)[order][:K].astype(int)
    return int(np.cumsum(y_sorted).sum())


def area_hits_at_k_norm(y_true, y_score, K):
    """Normalized cumulative-hits area in [0, 1] (raw / ideal)."""
    P = int(np.sum(np.asarray(y_true) == 1))
    ideal = _ideal_area_hits_at_k(P, K)
    if ideal == 0:
        return 0.0
    return float(area_hits_at_k(y_true, y_score, K) / ideal)


def log_weighted_hits_at_k(y_true, y_score, K):
    """Log-discounted cumulative area: Σ_{k=1..K} hits@k / log2(k+1).

    Gives more weight to early ranks than plain area while still rewarding
    total volume. Raw value (not normalized).
    """
    K = int(K)
    if K <= 0 or len(y_true) == 0:
        return 0.0
    K = min(K, len(y_true))
    order = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true)[order][:K].astype(int)
    cum = np.cumsum(y_sorted)
    weights = 1.0 / np.log2(np.arange(1, K + 1) + 1)
    return float(np.sum(cum * weights))


def ndcg_at_k(y_true, y_score, K):
    """Standard NDCG@K with binary relevance, in [0, 1]."""
    K = int(K)
    if K <= 0 or len(y_true) == 0:
        return 0.0
    K = min(K, len(y_true))
    y_arr = np.asarray(y_true).astype(int)
    P = int(np.sum(y_arr == 1))
    if P == 0:
        return 0.0
    order = np.argsort(y_score)[::-1]
    y_sorted = y_arr[order][:K]
    discounts = 1.0 / np.log2(np.arange(1, K + 1) + 1)
    dcg = float(np.sum(y_sorted * discounts))
    ideal_K = min(P, K)
    ideal_dcg = float(np.sum(discounts[:ideal_K]))
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def bedroc_at_k(y_true, y_score, K, alpha=20.0):
    """BEDROC-style early-rank-weighted score restricted to top-K, in [0, 1].

    For every hit at rank r in top-K: contribution = exp(-alpha * r / K).
    Score = sum_of_contributions / ideal_sum (all P hits at the very top).
    Aggressive front-loading: rank 1 is worth ~3x rank 50 (alpha=20, K=500).
    """
    K = int(K)
    if K <= 0 or len(y_true) == 0:
        return 0.0
    K = min(K, len(y_true))
    order = np.argsort(y_score)[::-1][:K]
    y_top = np.asarray(y_true)[order].astype(int)
    P = int(np.sum(y_top == 1))
    if P == 0:
        return 0.0
    hit_ranks = np.where(y_top == 1)[0] + 1  # 1-indexed
    score = float(np.sum(np.exp(-alpha * hit_ranks / K)))
    ideal = float(np.sum(np.exp(-alpha * np.arange(1, P + 1) / K)))
    if ideal == 0:
        return 0.0
    return score / ideal


def calculate_metrics(X_test, y_test, y_pred, y_proba, area_hits_K=500, regression_target=None):
    ppv = precision_score(y_test, y_pred, zero_division=0)
    p_ppv = plate_ppv(y_test, y_pred, top_n=128)
    
    # Commented this metric as it is so slow for big datasets!!
    # clusters = cluster_leader_from_array(X_test)
    # dp_ppv = diverse_plate_ppv(y_test, y_pred, clusters=clusters.tolist())
    dp_ppv = -1

    y_test_array = np.array(y_test)
    y_pred_array = np.array(y_pred)
    y_proba_array = np.array(y_proba)

    hits_5, prec_5 = hits_and_precision_at_k(y_test_array, y_pred_array, y_proba_array, 5)
    hits_10, prec_10 = hits_and_precision_at_k(y_test_array, y_pred_array, y_proba_array, 10)
    hits_50, prec_50 = hits_and_precision_at_k(y_test_array, y_pred_array, y_proba_array, 50)
    hits_100, prec_100 = hits_and_precision_at_k(y_test_array, y_pred_array, y_proba_array, 100)
    hits_200, prec_200 = hits_and_precision_at_k(y_test_array, y_pred_array, y_proba_array, 200)
    hits_500, prec_500 = hits_and_precision_at_k(y_test_array, y_pred_array, y_proba_array, 500)
    total_hits = int(np.sum((y_test_array == 1) & (y_pred_array == 1)))
    
    norm_prec_50 = NormPrecision_at_k(y_test_array, y_pred_array, y_proba_array, 50)
    norm_prec_100 = NormPrecision_at_k(y_test_array, y_pred_array, y_proba_array, 100)
    norm_prec_200 = NormPrecision_at_k(y_test_array, y_pred_array, y_proba_array, 200)
    norm_prec_500 = NormPrecision_at_k(y_test_array, y_pred_array, y_proba_array, 500)

    # Area / weighted metrics over the top-K of the probability ranking.
    K = int(area_hits_K)
    area_hits_K_val = area_hits_at_k(y_test_array, y_proba_array, K)
    area_hits_K_norm_val = area_hits_at_k_norm(y_test_array, y_proba_array, K)
    log_weighted_K = log_weighted_hits_at_k(y_test_array, y_proba_array, K)
    ndcg_K = ndcg_at_k(y_test_array, y_proba_array, K)
    bedroc_K = bedroc_at_k(y_test_array, y_proba_array, K, alpha=20.0)

    metrics = {
        "Accuracy": accuracy_score(y_test_array, y_pred_array),
        "Precision": ppv,
        "Recall": recall_score(y_test_array, y_pred_array, zero_division=0),
        "F1Score": f1_score(y_test_array, y_pred_array, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test_array, y_proba_array) if len(set(y_test_array)) > 1 else None,
        "AUC-PR": average_precision_score(y_test_array, y_proba_array) if len(set(y_test_array)) > 1 else None,
        "MCC": matthews_corrcoef(y_test_array, y_pred_array),
        "Cohen Kappa": cohen_kappa_score(y_test_array, y_pred_array),
        "balanced_accuracy": balanced_accuracy_score(y_test_array, y_pred_array),
        "PlatePPV": p_ppv,
        "DivPlatePPV": dp_ppv,
        "HitsAt5": hits_5,
        "HitsAt10": hits_10,
        "HitsAt50": hits_50,
        "PrecisionAt50": prec_50,
        "HitsAt100": hits_100,
        "PrecisionAt100": prec_100,
        "HitsAt200": hits_200,
        "PrecisionAt200": prec_200,
        "HitsAt500": hits_500,
        "PrecisionAt500": prec_500,
        "TotalHits": total_hits,
        "norm_prec_50": norm_prec_50,
        "norm_prec_100": norm_prec_100,
        "norm_prec_200": norm_prec_200,
        "norm_prec_500": norm_prec_500,
        # Area / weighted metrics — column names embed K so cross-K runs are distinguishable.
        f"AreaHitsAt{K}": area_hits_K_val,
        f"AreaHitsAt{K}_norm": area_hits_K_norm_val,
        f"LogWeightedHitsAt{K}": log_weighted_K,
        f"NDCG_at_{K}": ndcg_K,
        f"BEDROC_alpha20_at{K}": bedroc_K,
    }

    # Regression-only metrics. Populated when caller (regressor path in
    # evaluate_model) supplies the continuous true targets via regression_target.
    # Classifier rows leave these as None so the column exists but is empty.
    if regression_target is not None:
        try:
            y_true_cont = np.asarray(regression_target, dtype=float)
            y_pred_cont = np.asarray(y_proba, dtype=float)  # raw regressor score
            metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_true_cont, y_pred_cont)))
            metrics["MAE"] = float(mean_absolute_error(y_true_cont, y_pred_cont))
            metrics["R2"] = float(r2_score(y_true_cont, y_pred_cont))
        except Exception as exc:
            print(f"Warning: regression metrics failed ({exc}); leaving as None.")
            metrics["RMSE"] = None
            metrics["MAE"] = None
            metrics["R2"] = None
    else:
        metrics["RMSE"] = None
        metrics["MAE"] = None
        metrics["R2"] = None

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
        print("The test pipeline doesn’t run because it’s not requested. The test flag in the config file is set to 'N' or 'n'.")
        return

    results_path = os.path.join(RunFolderName, "results.csv")
    
    # Check if results.csv exists
    if not os.path.exists(results_path):
        print("Error: 'results.csv' not found. Please ensure you have run the training phase before testing.")
        return

    df = pd.read_csv(results_path)

    updated_rows = []

    # Per-test prediction parquets are written here so the run folder stays clean.
    predictions_dir = os.path.join(RunFolderName, "Predictions")
    os.makedirs(predictions_dir, exist_ok=True)


    


    for test_path in test_paths:
        print(test_path)
        if config['feature_fusion_method']!="None":
            X_test, Y_test = load_data(test_path, column_names, label_column_test, nrows_test)
            #Y_test_array = np.stack(Y_test.iloc[:, 0])
            #X_test_array = np.stack(X_test[column_name])
            # === Feature Fusion  ===
            X_test, fused_column_name = fuse_columns(X_test, column_names, feature_fusion_method)

        for rowcount, row in df.iterrows():
            # print("testing models, row:", rowcount)
            model_path = row["ModelPath"]
            model_name = row ["ModelType"]
            tf_models = config['tf_models']
            regressor_models = config.get('regressor_models', [])
            tf_dnn = True if model_name in tf_models else False
            is_regressor = model_name in regressor_models

            if os.path.isdir(model_path):
                if tf_dnn:
                    model_path = os.path.join(model_path, "model.h5")
                else:
                    model_path = os.path.join(model_path, "model.pkl")

            column_name = row["ColumnName"]

            if config['feature_fusion_method']=="None":
                # Load test data
                X_test, Y_test = load_data(test_path, [column_name], label_column_test, nrows_test)
            Y_test_array = np.stack(Y_test.iloc[:, 0])
            # Chunked uint8 stack — avoids the 3.5+ GiB float32 allocation that
            # crashes on memory-tight machines for big test files. Safe for binary
            # fingerprints and counts <= 255; see data_utils.stack_to_uint8.
            from data_utils import stack_to_uint8
            X_test_array = stack_to_uint8(X_test[column_name])
            # Only drop X_test if we loaded it fresh in THIS iteration (no-fusion
            # path). When feature fusion is on, X_test is loaded once outside the
            # inner loop and reused across rows — deleting it here would break
            # iteration 2+ with UnboundLocalError.
            if config['feature_fusion_method']=="None":
                del X_test

            # For regressor rows: also load the continuous target so RMSE/MAE/R^2
            # are reported alongside the binary ranking metrics.
            regression_target = None
            if is_regressor:
                regression_column_test = config.get('regression_column_test', []) or []
                if regression_column_test:
                    _, Y_test_cont_df = load_data(
                        test_path, [column_name], regression_column_test, nrows_test,
                    )
                    regression_target = np.stack(Y_test_cont_df.iloc[:, 0]).astype(float)
                    # Treat missing continuous targets as 0 (same convention as train side).
                    # RMSE/MAE/R^2 will treat any NaN-target row as if its true score was 0.
                    regression_target = np.nan_to_num(regression_target, nan=0.0)

            test_metrics, y_pred, y_prob  = evaluate_model(
                model_path, X_test_array, Y_test_array,
                area_hits_K=int(config.get('area_hits_K', 500)),
                regression_threshold=config.get('regression_threshold', 'median'),
                regression_target=regression_target,
                binary_threshold=config.get('binary_threshold', 0.5),
            )
            
            
            #---------------------------
            #--- New in 2026.03.04
            # load SMILES and label
            # read SMILES and LABEL
            #label_column_test
            label_column_name= row["label_column_test"]
            smiles_column_name = row["smiles_column"]
            #df_meta = pd.read_parquet(test_path, columns=["SMILES", "LABEL"])
            #df_meta = pd.read_parquet(test_path, columns=["smiles", "label"])
            df_meta = pd.read_parquet(test_path, columns=[smiles_column_name, label_column_name])
            # Mirror nrows_test so df_meta length matches y_pred/y_prob length;
            # otherwise nrows_test < #parquet_rows produces a length mismatch on assignment.
            if isinstance(nrows_test, int) and nrows_test > 0:
                df_meta = df_meta.head(nrows_test)

            # add predictions
            df_meta["y_pred"] = y_pred
            df_meta["y_prob"] = y_prob
            
            trainname = row["TrainFileName"]
    
            # output paths (under {RunFolderName}/Predictions/)
            base_name = os.path.splitext(os.path.basename(test_path))[0]
            out_path = os.path.join(predictions_dir, f"{base_name}_{trainname}_{model_name}_{column_name}_predictions.parquet")
            sorted_path = os.path.join(predictions_dir, f"{base_name}_{trainname}_{model_name}_{column_name}_predictions_sorted.parquet")

            
            # save file
            df_meta.to_parquet(out_path, index=False)
            # save sorted file
            df_meta.sort_values("y_prob", ascending=False).to_parquet(sorted_path, index=False)
            #--- New in 2026.03.04
            #---------------------------
            

            
            if conformal_prediction.lower() == 'y' and not tf_dnn:   
                [confromal_coverage_score, confromal_confidence_score, y_pred_set] = compute_conformal_prediction(get_model, train_model, load_data,fuse_columns,row,nrows_train,feature_fusion_method, X_test_array, Y_test_array)
                row["confromal_coverage_score"] = confromal_coverage_score
                row["confromal_confidence_score"] = confromal_confidence_score
            else:
                row["confromal_coverage_score"] = None
                row["confromal_confidence_score"] = None

            row_result = row.copy()
            for key, value in test_metrics.items():
                row_result[f"Test_{key}"] = value

            row_result["TestFile"] = os.path.basename(test_path)
            updated_rows.append(row_result)

    df_updated = pd.DataFrame(updated_rows)
    df_updated.to_csv(results_path, index=False)

    # Compact summary CSV with only the headline test-time columns.
    K = int(config.get('area_hits_K', 500))
    selected_columns = [
        "TrainFileName", "TestFile", "ColumnName", "ModelType",
        "Test_HitsAt50", "Test_HitsAt100", "Test_HitsAt200", "Test_HitsAt500",
        f"Test_AreaHitsAt{K}", f"Test_AreaHitsAt{K}_norm",
        f"Test_LogWeightedHitsAt{K}", f"Test_NDCG_at_{K}",
    ]
    keep = [c for c in selected_columns if c in df_updated.columns]
    if keep:
        summary_path = os.path.join(RunFolderName, "results_selectedcolumns.csv")
        df_updated[keep].to_csv(summary_path, index=False)
        print(f"Saved selected-columns summary: {summary_path}")
#------------------------------------------------------------------------------


def plot_test_pvalue_enrichment_curves(config, RunFolderName):
    """For every test predictions parquet under {RunFolderName}/Predictions/,
    save two figures next to it:
      - *_pvalue_curve.png:  hypergeometric p-value vs. top-n rank.
      - *_enrichment_curve.png:  fold enrichment vs. top-n rank.

    Notation here is intentionally distinct from the hit-curve plots
    (which use K to mean the rank cutoff) because the hypergeometric
    formulation reuses K for the population positives:
        N_total = total molecules in the test file.
        K_pos   = total positives (LABEL=1) in the test file.
        n_rank  = number of top-ranked predictions considered (x-axis).
        k_hits  = positives observed within the top n_rank predictions.
        p-value = P(X >= k_hits | N_total, K_pos, n_rank) = hypergeom.sf(k_hits - 1, N_total, K_pos, n_rank).
        enrichment = (k_hits / n_rank) / (K_pos / N_total).
    No random baseline is drawn (per request).
    """
    import matplotlib.pyplot as plt
    import glob
    from scipy.stats import hypergeom

    # Top-n cap on the x-axis (only the top of the ranking matters here).
    N_RANK_MAX = 1000

    predictions_dir = os.path.join(RunFolderName, "Predictions")
    if not os.path.isdir(predictions_dir):
        print(f"No Predictions folder at {predictions_dir}; skipping p-value/enrichment plots.")
        return

    sorted_files = sorted(glob.glob(os.path.join(predictions_dir, "*_predictions_sorted.parquet")))
    if not sorted_files:
        print(f"No *_predictions_sorted.parquet found in {predictions_dir}; skipping p-value/enrichment plots.")
        return

    for sorted_file in sorted_files:
        df = pd.read_parquet(sorted_file)
        label_col = next((c for c in ('LABEL', 'label') if c in df.columns), None)
        if label_col is None:
            print(f"No label column in {sorted_file}; skipping.")
            continue

        y_sorted = df[label_col].astype(int).values
        N_total = len(y_sorted)
        K_pos = int(y_sorted.sum())
        if K_pos == 0 or N_total == 0:
            print(f"No positives in {sorted_file}; skipping p-value/enrichment.")
            continue

        n_rank = np.arange(1, N_total + 1)
        k_hits = np.cumsum(y_sorted)

        p_values = hypergeom.sf(k_hits - 1, N_total, K_pos, n_rank)
        # Floor to a tiny positive value so log-scale plotting doesn't blow up.
        p_values = np.clip(p_values, 1e-300, 1.0)
        enrichment = (k_hits / n_rank) / (K_pos / N_total)

        title_base = os.path.splitext(os.path.basename(sorted_file))[0]
        x_max = min(N_RANK_MAX, N_total)

        # Restrict autoscale to the visible top-n window so the long flat
        # tail past x_max doesn't skew the y-axis range.
        vis_slice = slice(0, x_max)
        enr_vis = enrichment[vis_slice]

        # ---- p-value plot — plot -log10(p) so significance grows upward ----
        # Tiny p-values (e.g. 1e-50) become large positive numbers (50),
        # which is far more readable than a near-zero line on a raw p-value axis.
        neg_log_p = -np.log10(p_values)
        neg_log_p_vis = neg_log_p[vis_slice]
        plt.figure(figsize=(8, 6))
        plt.plot(n_rank, neg_log_p, color='darkblue', linewidth=2, label='-log10(p-value)')
        plt.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=1,
                    label='p = 0.05  (-log10 = 1.30)')
        plt.xlabel('Top-n')
        plt.ylabel('-log10(p-value)   (higher = more significant)')
        plt.title(textwrap.fill(f"p-value vs Top-n - {title_base}", width=60), fontsize=10)
        plt.xlim(0, x_max)
        ymin = min(float(np.nanmin(neg_log_p_vis)), -float(np.log10(0.05)))
        ymax = max(float(np.nanmax(neg_log_p_vis)), -float(np.log10(0.05)))
        pad = max((ymax - ymin) * 0.05, 0.5)
        plt.ylim(ymin - pad, ymax + pad)
        plt.legend()
        plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(predictions_dir, f"{title_base}_pvalue_curve.png")
        plt.savefig(out_png, dpi=200)
        plt.close()

        # ---- enrichment plot ----------------------------------------------
        plt.figure(figsize=(8, 6))
        plt.plot(n_rank, enrichment, color='darkgreen', linewidth=2, label='Enrichment')
        plt.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='No enrichment (=1)')
        plt.xlabel('Top-n')
        plt.ylabel('Enrichment  (k/n) / (K/N)')
        plt.title(textwrap.fill(f"Enrichment vs Top-n - {title_base}", width=60), fontsize=10)
        plt.xlim(0, x_max)
        # Autoscale around the visible data + the y=1 reference, with 10% padding,
        # clamped at 0 so the lower bound never goes negative.
        e_min = min(float(np.nanmin(enr_vis)), 1.0)
        e_max = max(float(np.nanmax(enr_vis)), 1.0)
        pad = max((e_max - e_min) * 0.10, 0.05)
        plt.ylim(max(0.0, e_min - pad), e_max + pad)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(predictions_dir, f"{title_base}_enrichment_curve.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
#------------------------------------------------------------------------------


def plot_cv_hit_curves(config, RunFolderName):
    """Plot cumulative-hits curves for CV predictions under
    {RunFolderName}/CVResults/.

    For each (train, model, column) triplet that has a *_mean_predictions_sorted.parquet,
    overlay each fold's curve plus the mean curve and the random baseline on a
    single figure named {train}_{model}_{column}_cv_hit_curve.png.
    """
    import matplotlib.pyplot as plt
    import glob
    import re

    # Hard-coded axis caps — CV plots zoom in to top-500 ranks.
    Y_MAX = 100
    X_MAX = 500
    N_RANDOM_RUNS = 10

    cv_dir = os.path.join(RunFolderName, "CVResults")
    if not os.path.isdir(cv_dir):
        print(f"No CVResults folder at {cv_dir}; skipping CV hit plots.")
        return

    mean_files = sorted(glob.glob(os.path.join(cv_dir, "*_mean_predictions_sorted.parquet")))
    if not mean_files:
        print(f"No *_mean_predictions_sorted.parquet found in {cv_dir}; skipping CV hit plots.")
        return

    fold_pat = re.compile(r"_fold(\d+)_predictions_sorted\.parquet$")

    for mean_file in mean_files:
        fname = os.path.basename(mean_file)
        prefix = fname[: -len("_mean_predictions_sorted.parquet")]

        df_mean = pd.read_parquet(mean_file)
        label_col = next((c for c in ('LABEL', 'label') if c in df_mean.columns), None)
        if label_col is None:
            print(f"No label column in {mean_file}; skipping.")
            continue

        y_mean = df_mean[label_col].astype(int).values
        n_total_mean = len(y_mean)
        n_pos_mean = int(y_mean.sum())
        if n_pos_mean == 0:
            print(f"No positives in {mean_file}; skipping.")
            continue

        plt.figure(figsize=(9, 6))

        # Random fan + theoretical baseline (sized to the mean curve length).
        ranks_mean = np.arange(1, n_total_mean + 1)
        if N_RANDOM_RUNS > 0:
            rng = np.random.default_rng(42)
            sim_labels = np.zeros(n_total_mean, dtype=np.int8)
            sim_labels[:n_pos_mean] = 1
            for i in range(N_RANDOM_RUNS):
                sim_curve = np.cumsum(rng.permutation(sim_labels))
                kw = dict(color='lightgray', linewidth=0.8, alpha=0.3)
                if i == 0:
                    kw['label'] = f'Random simulated ({N_RANDOM_RUNS} runs)'
                plt.plot(ranks_mean, sim_curve, **kw)
        plt.plot(ranks_mean, ranks_mean * (n_pos_mean / n_total_mean),
                 color='gray', linestyle='--', linewidth=1.5, label='Random expected')

        # Per-fold curves (faint coloured lines).
        fold_files = sorted(
            glob.glob(os.path.join(cv_dir, f"{prefix}_fold*_predictions_sorted.parquet")),
            key=lambda p: int(fold_pat.search(os.path.basename(p)).group(1))
                if fold_pat.search(os.path.basename(p)) else 0,
        )
        cmap = plt.get_cmap('tab10')
        for i, fold_file in enumerate(fold_files):
            df_fold = pd.read_parquet(fold_file)
            fcol = next((c for c in ('LABEL', 'label') if c in df_fold.columns), None)
            if fcol is None:
                continue
            yf = df_fold[fcol].astype(int).values
            if int(yf.sum()) == 0:
                continue
            ranks = np.arange(1, len(yf) + 1)
            cum = np.cumsum(yf)
            m = fold_pat.search(os.path.basename(fold_file))
            fold_num = m.group(1) if m else str(i + 1)
            plt.plot(ranks, cum, color=cmap(i % cmap.N), linewidth=1.2,
                     alpha=0.7, label=f'Fold {fold_num}')

        # Mean-across-folds curve (bold, dark).
        plt.plot(ranks_mean, np.cumsum(y_mean), color='darkblue', linewidth=2.2,
                 label='Mean over folds')

        plt.xlabel('K')
        plt.ylabel('Hit@K')
        plt.title(textwrap.fill(f"CV hit curves - {prefix}", width=60), fontsize=10)
        plt.xlim(0, X_MAX)
        plt.ylim(0, Y_MAX)
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(cv_dir, f"{prefix}_cv_hit_curve.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved: {out_png}")
#------------------------------------------------------------------------------


def plot_best_models_hit_curves(config, RunFolderName):
    """For each test file, overlay the hit curves of all selected best models on
    a single plot (plus the random baseline). Uses BestModelsResults.csv (now in
    the BestModels folder) to find which models were selected, then locates their
    *_predictions_sorted.parquet in {RunFolderName}/Predictions/. Saves one PNG
    per test file into {RunFolderName}/BestModels/.
    """
    import matplotlib.pyplot as plt

    # Hard-coded axis caps — keep in sync with plot_hit_curves.
    Y_MAX = 100
    X_MAX = 1000
    # Number of simulated random-permutation curves to overlay (faint grey fan
    # behind the theoretical 'Random expected' line). 0 = theoretical only.
    N_RANDOM_RUNS = 10

    best_models_folder = os.path.join(RunFolderName, "BestModels")
    best_models_csv = os.path.join(best_models_folder, "BestModelsResults.csv")
    if not os.path.exists(best_models_csv):
        print(f"No BestModelsResults.csv at {best_models_csv}; skipping best-models hit plots.")
        return

    predictions_dir = os.path.join(RunFolderName, "Predictions")
    if not os.path.isdir(predictions_dir):
        print(f"No Predictions folder at {predictions_dir}; skipping best-models hit plots.")
        return

    best_df = pd.read_csv(best_models_csv)

    for test_file, group in best_df.groupby('TestFile'):
        base_name = os.path.splitext(test_file)[0]

        plt.figure(figsize=(9, 6))
        random_drawn = False
        plotted_any = False

        for _, row in group.iterrows():
            trainname = row['TrainFileName']
            model_type = row['ModelType']
            column_name = row['ColumnName']

            sorted_file = os.path.join(
                predictions_dir,
                f"{base_name}_{trainname}_{model_type}_{column_name}_predictions_sorted.parquet",
            )
            if not os.path.exists(sorted_file):
                print(f"Predictions file not found: {sorted_file}")
                continue

            df = pd.read_parquet(sorted_file)
            label_col = next((c for c in ('LABEL', 'label') if c in df.columns), None)
            if label_col is None:
                continue
            y = df[label_col].astype(int).values
            n_total = len(y)
            n_pos = int(y.sum())
            if n_pos == 0:
                continue

            ranks = np.arange(1, n_total + 1)
            cum_hits = np.cumsum(y)

            if not random_drawn:
                # Simulated random runs (drawn first so they sit behind everything).
                if N_RANDOM_RUNS > 0:
                    rng = np.random.default_rng(42)
                    sim_labels = np.zeros(n_total, dtype=np.int8)
                    sim_labels[:n_pos] = 1
                    for i in range(N_RANDOM_RUNS):
                        sim_curve = np.cumsum(rng.permutation(sim_labels))
                        kw = dict(color='lightgray', linewidth=0.8, alpha=0.3)
                        if i == 0:
                            kw['label'] = f'Random simulated ({N_RANDOM_RUNS} runs)'
                        plt.plot(ranks, sim_curve, **kw)
                random_curve = ranks * (n_pos / n_total)
                plt.plot(ranks, random_curve, color='gray', linestyle='--', linewidth=1.5, label='Random expected')
                random_drawn = True

            plt.plot(ranks, cum_hits, linewidth=2, label=f"{model_type}_{column_name}")
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel('K')
        plt.ylabel('Hit@K')
        title_text = f"Best-models hit curves - {base_name}"
        plt.title(textwrap.fill(title_text, width=60), fontsize=10)
        plt.xlim(0, X_MAX)  # hard-coded cap (keep in sync with plot_hit_curves)
        plt.ylim(0, Y_MAX)  # hard-coded cap (keep in sync with plot_hit_curves)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(best_models_folder, f"{base_name}_best_models_hit_curve.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved: {out_png}")
#------------------------------------------------------------------------------


def plot_hit_curves(config, RunFolderName):
    """Plot cumulative-hits curves for every sorted predictions parquet under
    {RunFolderName}/Predictions/. Uses only the *_predictions_sorted.parquet
    files (the unsorted twins encode the same ranking once sorted).

    Each plot shows:
      - Model curve: cumulative number of true positives in the top-K predictions.
      - Random baseline: expected hits if predictions were ranked randomly.
    """
    import matplotlib.pyplot as plt
    import glob

    # Hard-coded axis caps. Y_MAX keeps the visual scale consistent across runs;
    # X_MAX zooms into the top of the ranking (only the top-1000 predictions matter
    # for hit-rate inspection). Bump either if your dataset needs more.
    Y_MAX = 100
    X_MAX = 1000
    # Number of simulated random-permutation curves to overlay (drawn as a faint
    # grey fan behind the theoretical 'Random expected' line). 0 = theoretical only.
    # Bump for visual variability sense; rendering slows when N_RANDOM_RUNS is large.
    N_RANDOM_RUNS = 10

    predictions_dir = os.path.join(RunFolderName, "Predictions")
    if not os.path.isdir(predictions_dir):
        print(f"No Predictions folder at {predictions_dir}; skipping hit plots.")
        return

    sorted_files = sorted(glob.glob(os.path.join(predictions_dir, "*_predictions_sorted.parquet")))
    if not sorted_files:
        print(f"No *_predictions_sorted.parquet found in {predictions_dir}; skipping hit plots.")
        return

    for sorted_file in sorted_files:
        df = pd.read_parquet(sorted_file)
        label_col = next((c for c in ('LABEL', 'label') if c in df.columns), None)
        if label_col is None:
            print(f"No label column in {sorted_file}; skipping.")
            continue

        y = df[label_col].astype(int).values
        n_total = len(y)
        n_pos = int(y.sum())
        if n_pos == 0:
            print(f"No positives in {sorted_file}; skipping.")
            continue

        ranks = np.arange(1, n_total + 1)
        cum_hits = np.cumsum(y)
        random_curve = ranks * (n_pos / n_total)

        plt.figure(figsize=(8, 6))
        # Simulated random runs (drawn first so they sit behind everything).
        if N_RANDOM_RUNS > 0:
            rng = np.random.default_rng(42)
            sim_labels = np.zeros(n_total, dtype=np.int8)
            sim_labels[:n_pos] = 1
            for i in range(N_RANDOM_RUNS):
                sim_curve = np.cumsum(rng.permutation(sim_labels))
                kw = dict(color='lightgray', linewidth=0.8, alpha=0.3)
                if i == 0:
                    kw['label'] = f'Random simulated ({N_RANDOM_RUNS} runs)'
                plt.plot(ranks, sim_curve, **kw)
        plt.plot(ranks, random_curve, color='gray', linestyle='--', linewidth=1.5, label='Random expected')
        plt.plot(ranks, cum_hits, color='darkblue', linewidth=2, label='Model')
        plt.xlabel('K')
        plt.ylabel('Hit@K')
        title_text = os.path.splitext(os.path.basename(sorted_file))[0]
        plt.title(textwrap.fill(title_text, width=60), fontsize=10)
        plt.xlim(0, X_MAX)  # hard-coded cap (see X_MAX above)
        plt.ylim(0, Y_MAX)  # hard-coded cap (see Y_MAX above)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(
            predictions_dir,
            os.path.splitext(os.path.basename(sorted_file))[0] + "_hit_curve.png",
        )
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved: {out_png}")
#------------------------------------------------------------------------------


def compute_conformal_prediction(get_model, train_model, load_data, fuse_columns, row, nrows_train, feature_fusion_method, X_test_array, Y_test_array):
    #---
    from mapie.classification import SplitConformalClassifier
    from mapie.metrics.classification import classification_coverage_score
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
    
    return confromal_coverage_score, confromal_confidence_score, y_pred_set























