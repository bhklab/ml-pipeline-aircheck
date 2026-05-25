

import pandas as pd
import os
import re
import warnings
import shutil
import textwrap
import glob

import pickle
import numpy as np
import matplotlib.pyplot as plt
from eval_utils import calculate_metrics, plate_ppv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import load_model
#==============================================================================
# Function to plot radar chart for model metrics
def plot_model_metrics_radar(df, metric_columns, save_path="", title="Model Metrics"):
    """
    Plots a radar chart for model metrics.

    Parameters:
    - df (pd.DataFrame): DataFrame containing model evaluation metrics.
    - metric_columns (list): List of metric column names to plot.
    - save_path (str, optional): Path to save the radar chart image.
    - title (str, optional): Plot title.

    Assumes all metrics are in [0, 1] (e.g. F1, Precision, Recall, Accuracy,
    PlatePPV, balanced_accuracy). Mixing different scales (like Hit@K counts)
    on one radar makes the polygon shape meaningless.
    """
    # Calculate mean values for each metric
    metric_values = df[metric_columns].mean().values

    # Make the values a complete loop for radar plot
    values = list(metric_values) + [metric_values[0]]

    # Radar chart setup
    angles = np.linspace(0, 2 * np.pi, len(metric_columns), endpoint=False).tolist()
    angles += angles[:1]  # Closing the loop for the radar chart

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.2)
    ax.plot(angles, values, color='blue', linewidth=2)

    # Radial axis: fixed [0, 1] scale with visible tick labels.
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color='gray')
    ax.set_rlabel_position(135)  # rotate the radial labels off the metric axes

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_columns, fontsize=11)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Radar chart saved at {save_path}")

    plt.close(fig)
#-----------------------------------------------------------------------------

def _precision_at_k_metrics_from_parquet(predictions_path, k_values=(10, 50, 100, 200, 500)):
    """Compute Precision@K for each K from a predictions parquet.

    Precision@K = (# true positives in the top-K predictions ranked by y_prob) / K.
    The parquet must have a label column (LABEL/label) and a probability column
    (y_prob/y_proba). Returns a dict keyed 'PrecisionAt{K}' or None if columns missing.
    """
    df = pd.read_parquet(predictions_path)
    label_col = next((c for c in ('LABEL', 'label') if c in df.columns), None)
    prob_col = next((c for c in ('y_prob', 'y_proba') if c in df.columns), None)
    if label_col is None or prob_col is None:
        return None
    y_true = df[label_col].astype(int).values
    y_proba = df[prob_col].astype(float).values
    order = np.argsort(y_proba)[::-1]
    y_sorted = y_true[order]
    out = {}
    for k in k_values:
        kk = min(k, len(y_sorted))
        out[f'PrecisionAt{k}'] = float(y_sorted[:kk].sum()) / kk if kk > 0 else 0.0
    return out


def _radar_metrics_from_parquet(predictions_path, threshold):
    """Recompute the 6 radar metrics from a predictions parquet at the given threshold.

    The parquet must contain a label column (LABEL/label) and a probability column
    (y_prob/y_proba). Returns a dict of metrics or None if columns are missing.
    """
    df = pd.read_parquet(predictions_path)
    label_col = next((c for c in ('LABEL', 'label') if c in df.columns), None)
    prob_col = next((c for c in ('y_prob', 'y_proba') if c in df.columns), None)
    if label_col is None or prob_col is None:
        return None
    y_true = df[label_col].astype(int).values
    y_proba = df[prob_col].astype(float).values
    y_pred = (y_proba > threshold).astype(int)
    return {
        'F1Score': f1_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'Accuracy': accuracy_score(y_true, y_pred),
        'PlatePPV': plate_ppv(y_true, y_pred, top_n=128),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }


def select_best_models(config, RunFolderName):
    trainfile_for_modelselection = config['trainfile_for_modelselection']
    evaluationfile_for_modelselection = config['evaluationfile_for_modelselection']
    evaluation_column = config['evaluation_column']
    crossvalidation_column = config['crossvalidation_column']
    modelselection_basis = config.get('modelselection_basis', 'cv').lower()
    num_top_models = config['num_top_models']
    Fusion = config['Fusion']
    tf_models = config['tf_models']
    
    
    if Fusion.lower() != 'y':
        num_top_models = 1

    results_csv_path = os.path.join(RunFolderName, "results.csv")
    output_txt_path = os.path.join(RunFolderName, "BestModels.txt")

    if evaluation_column is None:
        raise ValueError("You must provide a list of sort_columns")

    df = pd.read_csv(results_csv_path)

    # Drop duplicates excluding Date and Time columns (if they exist)
    drop_cols = [col for col in df.columns if col not in ['Date', 'Time']]
    df = df.drop_duplicates(subset=drop_cols)

    # Filter by TestFile if specified
    if trainfile_for_modelselection:
        df = df[df['TrainFile'] == trainfile_for_modelselection]
        
    # Filter by TestFile if specified
    if evaluationfile_for_modelselection:
        df = df[df['TestFile'] == evaluationfile_for_modelselection]

    # Sort by either evaluation_column (test-based) or crossvalidation_column (cv-based)
    # depending on modelselection_basis. Missing columns are warned about and skipped;
    # if none of the chosen columns exist, the original order is kept.
    if modelselection_basis == 'test':
        chosen_columns = evaluation_column or []
        basis_label = 'test evaluation'
    elif modelselection_basis == 'cv':
        chosen_columns = crossvalidation_column or []
        basis_label = 'cross validation'
    else:
        raise ValueError(f"modelselection_basis must be 'cv' or 'test', got: {modelselection_basis}")

    cols_present = [col for col in chosen_columns if col in df.columns]
    cols_missing = [col for col in chosen_columns if col not in df.columns]

    if cols_present:
        print(f'based on {basis_label}')
        if cols_missing:
            warnings.warn(f"Missing {basis_label} columns ignored: {cols_missing}")
        df_sorted = df.sort_values(by=cols_present, ascending=[False] * len(cols_present))
    else:
        warnings.warn(f"No {basis_label} columns found in results; keeping original order.")
        df_sorted = df  # Keep df as is without sorting

    # Drop duplicates based on ModelPath
    df_sorted_unique = df_sorted.drop_duplicates(subset=['ModelPath'])
    
    # Save the sorted unique DataFrame for further analysis
    sorted_unique_csv_path = os.path.join(RunFolderName, "SortedUniqueResults.csv")
    df_sorted_unique.to_csv(sorted_unique_csv_path, index=False)
    
    # Check available distinct models and warn if fewer than num_top_models
    available_models = len(df_sorted_unique)
    if available_models < num_top_models:
        warnings.warn(f"Only {available_models} distinct models available. Returning all.")
        num_top_models = available_models

    # Take top k distinct models
    best_models = df_sorted_unique.head(num_top_models)

    # All best-model artifacts (results CSV, model files, hit-curve plots) live here.
    best_models_folder = os.path.join(RunFolderName, "BestModels")
    os.makedirs(best_models_folder, exist_ok=True)

    BestModelsResults_csv_path = os.path.join(best_models_folder, "BestModelsResults.csv")
    best_models.to_csv(BestModelsResults_csv_path, index=False)


    # Write model paths to output txt file
    with open(output_txt_path, 'w') as f:
        for path in best_models['ModelPath']:
            f.write(f"{path}\n")
    for path in best_models['ModelPath']:
        model_name = os.path.basename(path)
        
        modelname = best_models['ModelType'].iloc[0]
        tf_dnn = True if modelname in tf_models else False

        if tf_dnn:
            model_file = os.path.join(path, "model.h5") 
        else:
            model_file = os.path.join(path, "model.pkl") 
        
        if os.path.exists(model_file):
            # Create the new name with the folder name prefix
            if tf_dnn:
                new_model_name = f"{model_name}_model.h5"
            else:
                new_model_name = f"{model_name}_model.pkl"
                
            
            model_save_path = os.path.join(best_models_folder, new_model_name)
            
            # Copy the model file with the new name
            shutil.copy(model_file, model_save_path)
            print(f"Saved: {model_save_path}")
        else:
            print(f"Model file not found in {path}. Skipping.")


    print(f"Top {num_top_models} distinct model paths written to {output_txt_path}")
    
    # Both CV and Test radar plots go into a dedicated RadarPlots/ folder.
    radar_plots_dir = os.path.join(RunFolderName, "RadarPlots")
    os.makedirs(radar_plots_dir, exist_ok=True)

    cv_radar_threshold = float(config.get('cv_radar_threshold', 0.5))
    test_radar_threshold = float(config.get('test_radar_threshold', 0.5))

    cv_results_dir = os.path.join(RunFolderName, "CVResults")
    predictions_dir = os.path.join(RunFolderName, "Predictions")

    # ---- CV radar — recompute metrics from CVResults mean parquets ---------
    cv_metric_rows = []
    for _, row in best_models.iterrows():
        train_fn = row.get('TrainFileName')
        model_type = row.get('ModelType')
        column_name = row.get('ColumnName')
        if pd.isna(train_fn) or pd.isna(model_type) or pd.isna(column_name):
            continue
        cv_pred_file = os.path.join(
            cv_results_dir,
            f"{train_fn}_{model_type}_{column_name}_mean_predictions_sorted.parquet",
        )
        if not os.path.exists(cv_pred_file):
            print(f"CV mean predictions not found for radar: {cv_pred_file}")
            continue
        m = _radar_metrics_from_parquet(cv_pred_file, cv_radar_threshold)
        if m is not None:
            cv_metric_rows.append(m)

    if cv_metric_rows:
        cv_df = pd.DataFrame(cv_metric_rows)
        plot_model_metrics_radar(
            cv_df, list(cv_df.columns),
            save_path=os.path.join(radar_plots_dir, "RadarChart_TopModels_CV.png"),
            title=f"CV Metrics - Top Models (threshold={cv_radar_threshold})",
        )
    else:
        print("No CV mean predictions available; CV radar plot skipped.")

    # ---- CV Precision@K radar (K = 10, 50, 100, 200, 500) -----------------
    cv_pk_rows = []
    for _, row in best_models.iterrows():
        train_fn = row.get('TrainFileName')
        model_type = row.get('ModelType')
        column_name = row.get('ColumnName')
        if pd.isna(train_fn) or pd.isna(model_type) or pd.isna(column_name):
            continue
        cv_pred_file = os.path.join(
            cv_results_dir,
            f"{train_fn}_{model_type}_{column_name}_mean_predictions_sorted.parquet",
        )
        if not os.path.exists(cv_pred_file):
            continue
        m = _precision_at_k_metrics_from_parquet(cv_pred_file)
        if m is not None:
            cv_pk_rows.append(m)

    if cv_pk_rows:
        cv_pk_df = pd.DataFrame(cv_pk_rows)
        plot_model_metrics_radar(
            cv_pk_df, list(cv_pk_df.columns),
            save_path=os.path.join(radar_plots_dir, "RadarChart_TopModels_CV_PrecisionAtK.png"),
            title="CV Precision@K - Top Models",
        )
    else:
        print("No CV mean predictions available; CV Precision@K radar skipped.")

    # ---- Test radar — recompute metrics from Predictions parquets ----------
    test_metric_rows = []
    for _, row in best_models.iterrows():
        test_file = row.get('TestFile')
        train_fn = row.get('TrainFileName')
        model_type = row.get('ModelType')
        column_name = row.get('ColumnName')
        if pd.isna(test_file) or pd.isna(train_fn) or pd.isna(model_type) or pd.isna(column_name):
            continue
        base_test = os.path.splitext(test_file)[0]
        test_pred_file = os.path.join(
            predictions_dir,
            f"{base_test}_{train_fn}_{model_type}_{column_name}_predictions_sorted.parquet",
        )
        if not os.path.exists(test_pred_file):
            print(f"Test predictions not found for radar: {test_pred_file}")
            continue
        m = _radar_metrics_from_parquet(test_pred_file, test_radar_threshold)
        if m is not None:
            test_metric_rows.append(m)

    if test_metric_rows:
        test_df = pd.DataFrame(test_metric_rows)
        plot_model_metrics_radar(
            test_df, list(test_df.columns),
            save_path=os.path.join(radar_plots_dir, "RadarChart_TopModels_Test.png"),
            title=f"Test Metrics - Top Models (threshold={test_radar_threshold})",
        )
    else:
        print("No test predictions available; Test radar plot skipped.")

    # ---- Test Precision@K radar (K = 10, 50, 100, 200, 500) ---------------
    test_pk_rows = []
    for _, row in best_models.iterrows():
        test_file = row.get('TestFile')
        train_fn = row.get('TrainFileName')
        model_type = row.get('ModelType')
        column_name = row.get('ColumnName')
        if pd.isna(test_file) or pd.isna(train_fn) or pd.isna(model_type) or pd.isna(column_name):
            continue
        base_test = os.path.splitext(test_file)[0]
        test_pred_file = os.path.join(
            predictions_dir,
            f"{base_test}_{train_fn}_{model_type}_{column_name}_predictions_sorted.parquet",
        )
        if not os.path.exists(test_pred_file):
            continue
        m = _precision_at_k_metrics_from_parquet(test_pred_file)
        if m is not None:
            test_pk_rows.append(m)

    if test_pk_rows:
        test_pk_df = pd.DataFrame(test_pk_rows)
        plot_model_metrics_radar(
            test_pk_df, list(test_pk_df.columns),
            save_path=os.path.join(radar_plots_dir, "RadarChart_TopModels_Test_PrecisionAtK.png"),
            title="Test Precision@K - Top Models",
        )
    else:
        print("No test predictions available; Test Precision@K radar skipped.")
#==============================================================================





#==============================================================================
def _hit_at_k_from_parquet(path, k=200):
    """Hit@K read straight from a per-fold predictions parquet (no re-sorting needed
    if the file already contains all rows for the fold; we re-rank on y_prob here)."""
    df = pd.read_parquet(path)
    label_col = next((c for c in ('LABEL', 'label') if c in df.columns), None)
    prob_col = next((c for c in ('y_prob', 'y_proba') if c in df.columns), None)
    if label_col is None or prob_col is None:
        return None
    y = df[label_col].astype(int).values
    p = df[prob_col].astype(float).values
    if len(y) == 0:
        return 0
    kk = min(k, len(y))
    top = np.argsort(p)[::-1][:kk]
    return int(y[top].sum())


def _train_weighted_fusion_meta(config, RunFolderName, best_models_df, tf_models):
    """Train a logistic-regression fusion meta-learner on a single chosen CV fold.

    Strategy:
      1. Restrict best models to the dominant TrainFileName (cv_splits indices are
         only consistent within one train file; mixed train files are rare).
      2. For each fold N, average Hit@200 across the best models on their
         CVResults/{train}_{model}_{col}_fold{N}_predictions.parquet files.
      3. Pick fold N* with the highest mean Hit@200.
      4. Inner-join those per-fold parquets on SMILES → (n_test_in_fold, K) matrix.
      5. Fit LogisticRegression(class_weight='balanced').

    Returns a dict with the fitted meta + ordered model column ids, or None on failure.
    The meta + the inputs needed to reproduce it are persisted under BestModels/.
    """
    cv_dir = os.path.join(RunFolderName, "CVResults")
    if not os.path.isdir(cv_dir):
        warnings.warn("CVResults folder not found; weighted fusion skipped.")
        return None

    if best_models_df.empty:
        warnings.warn("No best models available; weighted fusion skipped.")
        return None

    # Step 1 — dominant TrainFileName.
    train_counts = best_models_df['TrainFileName'].value_counts()
    dominant_train = train_counts.idxmax()
    if len(train_counts) > 1:
        warnings.warn(
            f"Best models span multiple TrainFileNames {list(train_counts.index)}. "
            f"Weighted fusion uses only models trained on '{dominant_train}'."
        )
    bm = best_models_df[best_models_df['TrainFileName'] == dominant_train].copy()
    if bm.empty:
        return None

    def _fold_path(row, fold_n):
        return os.path.join(
            cv_dir,
            f"{row['TrainFileName']}_{row['ModelType']}_{row['ColumnName']}_fold{fold_n}_predictions.parquet",
        )

    # Step 2 — discover available fold ids from the first model's files.
    first_row = bm.iloc[0]
    pat_str = f"{first_row['TrainFileName']}_{first_row['ModelType']}_{first_row['ColumnName']}_fold*_predictions.parquet"
    fold_files = glob.glob(os.path.join(cv_dir, pat_str))
    fold_pat = re.compile(r"_fold(\d+)_predictions\.parquet$")
    fold_ids = sorted({
        int(fold_pat.search(os.path.basename(p)).group(1))
        for p in fold_files if fold_pat.search(os.path.basename(p))
    })
    if not fold_ids:
        warnings.warn(f"No per-fold CV parquets found for the dominant train file; weighted fusion skipped.")
        return None

    # Pick fold by mean Hit@200 across best models.
    best_fold = None
    best_score = -1.0
    fold_scores = {}
    for fn in fold_ids:
        scores = []
        for _, row in bm.iterrows():
            p = _fold_path(row, fn)
            if os.path.exists(p):
                hits = _hit_at_k_from_parquet(p, 200)
                if hits is not None:
                    scores.append(hits)
        if not scores:
            continue
        mean_h = float(np.mean(scores))
        fold_scores[fn] = mean_h
        if mean_h > best_score:
            best_score = mean_h
            best_fold = fn
    if best_fold is None:
        warnings.warn("Could not score any fold; weighted fusion skipped.")
        return None

    # Step 4 — build joined matrix on the chosen fold.
    smiles_cfg = config.get('smiles_column', 'SMILES')
    merged = None
    model_cols = []
    for _, row in bm.iterrows():
        col_id = f"{row['ModelType']}_{row['ColumnName']}"
        path = _fold_path(row, best_fold)
        if not os.path.exists(path):
            warnings.warn(f"Missing per-fold parquet for {col_id}: {path}; skipping.")
            continue
        df = pd.read_parquet(path)
        prob_col = next((c for c in ('y_prob', 'y_proba') if c in df.columns), None)
        smiles_col = next((c for c in (smiles_cfg, 'SMILES', 'smiles') if c in df.columns), None)
        label_col = next((c for c in ('LABEL', 'label') if c in df.columns), None)
        if prob_col is None or smiles_col is None or label_col is None:
            warnings.warn(f"Required columns missing in {path}; skipping.")
            continue
        sub = df[[smiles_col, label_col, prob_col]].rename(
            columns={smiles_col: 'SMILES', label_col: 'LABEL', prob_col: col_id}
        )
        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub.drop(columns=['LABEL']), on='SMILES', how='inner')
        model_cols.append(col_id)

    if merged is None or not model_cols:
        warnings.warn("Weighted fusion: no usable per-fold data; skipped.")
        return None
    if int(merged['LABEL'].sum()) == 0:
        warnings.warn(f"Weighted fusion: chosen fold {best_fold} has 0 positives after join; skipped.")
        return None

    X_meta = merged[model_cols].values.astype(float)
    y_meta = merged['LABEL'].astype(int).values

    fusion_C = float(config.get('fusion_weighted_C', 1.0))
    meta = LogisticRegression(class_weight='balanced', C=fusion_C, max_iter=500)
    meta.fit(X_meta, y_meta)

    # Persist meta + a human-readable weights CSV.
    bm_folder = os.path.join(RunFolderName, "BestModels")
    os.makedirs(bm_folder, exist_ok=True)
    payload = {
        'meta': meta,
        'model_cols': model_cols,
        'best_fold': best_fold,
        'best_fold_mean_hit200': best_score,
        'all_fold_mean_hit200': fold_scores,
        'train_filename': dominant_train,
        'n_train_rows': int(len(merged)),
        'n_train_pos': int(y_meta.sum()),
        'C': fusion_C,
    }
    with open(os.path.join(bm_folder, "fusion_weights.pkl"), 'wb') as f:
        pickle.dump(payload, f)
    pd.DataFrame({
        'model_col': model_cols + ['__intercept__'],
        'coefficient': list(meta.coef_[0]) + [float(meta.intercept_[0])],
    }).to_csv(os.path.join(bm_folder, "fusion_weights.csv"), index=False)
    pd.DataFrame([{
        'best_fold': best_fold,
        'best_fold_mean_hit200': best_score,
        'train_filename': dominant_train,
        'n_train_rows': len(merged),
        'n_train_pos': int(y_meta.sum()),
        'C': fusion_C,
        'model_cols': ";".join(model_cols),
        'fold_scores': str(fold_scores),
    }]).to_csv(os.path.join(bm_folder, "fusion_weights_summary.csv"), index=False)
    print(f"Weighted fusion: fold {best_fold} chosen (mean Hit@200 = {best_score:.2f}); "
          f"trained on {len(merged)} rows ({int(y_meta.sum())} positives) over {len(model_cols)} models.")
    return payload


def fusion_pipeline(config,
                    RunFolderName,
                    load_data,
                    fuse_columns,
                    evaluate_model):

    test_paths = config['test_data']
    column_names = config['desired_columns']
    label_column_test = config['label_column_test']
    nrows_test = config['nrows_test']
    feature_fusion_method = config['feature_fusion_method']
    tf_models = config['tf_models']

    # Path to Best Models
    best_models_folder = os.path.join(RunFolderName, "BestModels")
    valid_extensions = [".pkl", ".h5", ".pt"]  # Add .pt if you support PyTorch later
    # Skip the weighted-fusion meta-learner pickle (and any other helper artifacts
    # whose name doesn't end with the conventional '_model.{pkl,h5}' suffix).
    excluded_basenames = {"fusion_weights.pkl"}
    model_files = [
        os.path.join(best_models_folder, f)
        for f in os.listdir(best_models_folder)
        if any(f.endswith(ext) for ext in valid_extensions) and f not in excluded_basenames
    ]

    if not model_files:
        raise ValueError("No models found in BestModels folder.")

    # All fusion outputs (parquets + fusion.csv + plots) go here.
    fusion_dir = os.path.join(RunFolderName, "Fusion")
    os.makedirs(fusion_dir, exist_ok=True)

    # Map each model file in BestModels to its ColumnName via BestModelsResults.csv,
    # which is more robust than parsing the file name (column names can contain '-').
    best_models_csv = os.path.join(best_models_folder, "BestModelsResults.csv")
    best_models_df = pd.read_csv(best_models_csv)
    model_file_to_column = {}
    model_file_to_id = {}  # used to align with weighted-fusion meta-model column ids
    for _, row in best_models_df.iterrows():
        folder_name = os.path.basename(row['ModelPath'])
        is_tf = row['ModelType'] in tf_models
        ext = ".h5" if is_tf else ".pkl"
        fname = f"{folder_name}_model{ext}"
        model_file_to_column[fname] = row['ColumnName']
        model_file_to_id[fname] = f"{row['ModelType']}_{row['ColumnName']}"

    # Resolve fusion methods (list-valued, lowercased). Defaults preserve old behavior.
    fusion_methods = config.get('fusion_method', ['mean', 'max'])
    if isinstance(fusion_methods, str):
        fusion_methods = [fusion_methods]
    fusion_methods = [m.strip().lower() for m in fusion_methods if m]
    if not fusion_methods:
        fusion_methods = ['mean', 'max']

    # Train weighted-fusion meta-learner once on the best CV fold (if requested).
    weighted_meta = None
    if 'weighted' in fusion_methods:
        weighted_meta = _train_weighted_fusion_meta(config, RunFolderName, best_models_df, tf_models)
        if weighted_meta is None:
            warnings.warn("Weighted fusion requested but meta training failed; weighted fusion disabled.")

    fusion_results = []

    for test_path in test_paths:
        print(test_path)
        # Store predictions for fusion
        y_preds = []
        y_probas = []
        # Per-model id -> y_proba (so weighted fusion can align with meta column order).
        y_probas_by_id = {}
        # Run each model
        for model_file in model_files:
            print(model_file)
            ext = os.path.splitext(model_file)[-1].lower()
            model_basename = os.path.basename(model_file)
            if model_basename not in model_file_to_column:
                raise ValueError(f"Cannot find ColumnName for {model_basename} in {best_models_csv}")
            column_name = model_file_to_column[model_basename]
            # A fused column name uses '-' to join base descriptors (see data_utils.fuse_columns).
            base_cols = column_name.split("-")

            print ("read test start")
            #-----------------------------------------------------
            #---- read test file here ---------------------------
            # Load test data — read all base columns at once
            X_test, Y_test = load_data(test_path, base_cols, label_column_test, nrows_test)
            Y_test_array = np.stack(Y_test.iloc[:, 0])
            # Chunked uint8 stack — same OOM avoidance as in eval_utils.test_pipeline.
            from data_utils import stack_to_uint8
            if len(base_cols) == 1:
                X_test_array = stack_to_uint8(X_test[base_cols[0]])
            else:
                X_test_array = np.concatenate(
                    [stack_to_uint8(X_test[c]) for c in base_cols], axis=1
                )
            # Try to get SMILES column from file, otherwise from X_test if present
            if str(test_path).lower().endswith(".parquet"):
                smiles_df = pd.read_parquet(test_path, columns=["SMILES"])
            elif str(test_path).lower().endswith(".csv"):
                smiles_df = pd.read_csv(test_path, usecols=["SMILES"], nrows=nrows_test)
            else:
                raise ValueError(f"Unsupported test file format for SMILES extraction: {test_path}")
            smiles_series = smiles_df["SMILES"].reset_index(drop=True)
            if len(smiles_series) != len(X_test):
                smiles_series = smiles_series.iloc[:len(X_test)].reset_index(drop=True)
            #---- end read test file ----------------------------
            #-----------------------------------------------------
            print ("read test end")


            # Row-chunked predict (same OOM avoidance as eval_utils.evaluate_model).
            chunk_size = 20000
            def _predict_chunks_local(predict_fn, two_d_out=False):
                n = X_test_array.shape[0]
                outs = []
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    outs.append(predict_fn(X_test_array[start:end]))
                if not outs:
                    return np.empty((0, 2) if two_d_out else (0,), dtype=np.float32)
                if two_d_out:
                    return np.concatenate(outs, axis=0)
                return np.concatenate([np.asarray(o).flatten() for o in outs], axis=0)

            if ext == ".pkl":
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)

                if hasattr(model, 'predict_proba'):
                    # Classifier — use predicted positive-class probability.
                    proba_2d = _predict_chunks_local(lambda x: model.predict_proba(x), two_d_out=True)
                    y_proba = proba_2d[:, 1]
                    try:
                        bt = float(config.get('binary_threshold', 0.5))
                    except (TypeError, ValueError):
                        bt = 0.5
                    y_pred = (y_proba > bt).astype(int)
                else:
                    # Regressor — predict() returns a continuous score; binarize
                    # with regression_threshold (median or fixed cutoff).
                    y_proba = _predict_chunks_local(lambda x: np.asarray(model.predict(x)).flatten())
                    rt = config.get('regression_threshold', 'median')
                    if isinstance(rt, str) and rt.lower() == 'median':
                        threshold = float(np.median(y_proba))
                    else:
                        try:
                            threshold = float(rt)
                        except (TypeError, ValueError):
                            threshold = float(np.median(y_proba))
                    y_pred = (y_proba > threshold).astype(int)

            elif ext == ".h5":
                model = load_model(model_file)
                y_proba = _predict_chunks_local(lambda x: model.predict(x).flatten())
                try:
                    bt = float(config.get('binary_threshold', 0.5))
                except (TypeError, ValueError):
                    bt = 0.5
                y_pred = (y_proba > bt).astype(int)

            else:
                raise ValueError(f"Unsupported model format: {ext}")

            y_preds.append(y_pred)
            y_probas.append(y_proba)
            y_probas_by_id[model_file_to_id[model_basename]] = y_proba

        test_base = os.path.splitext(os.path.basename(test_path))[0]
        fusion_result = {"TestFile": os.path.basename(test_path)}

        # =========================
        # Mean fusion
        # =========================
        if 'mean' in fusion_methods:
            y_pred_fusion_mean = np.mean(y_preds, axis=0).round().astype(int)
            y_proba_fusion_mean = np.mean(y_probas, axis=0)
            metrics_mean = calculate_metrics(
                X_test_array, Y_test_array, y_pred_fusion_mean, y_proba_fusion_mean,
                area_hits_K=int(config.get('area_hits_K', 500)),
            )

            mean_df = pd.DataFrame({
                "SMILES": smiles_series,
                "LABEL": Y_test_array,
                "y_pred": y_pred_fusion_mean,
                "y_proba": y_proba_fusion_mean,
            })
            mean_df.to_parquet(os.path.join(fusion_dir, f"fusion_mean_{test_base}.parquet"), index=False)
            mean_df.sort_values("y_proba", ascending=False).to_parquet(
                os.path.join(fusion_dir, f"fusion_mean_sorted_{test_base}.parquet"), index=False,
            )
            fusion_result.update({f"mean_{k}": v for k, v in metrics_mean.items()})

        # =========================
        # Max fusion
        # =========================
        if 'max' in fusion_methods:
            y_proba_fusion_max = np.max(y_probas, axis=0)
            y_pred_fusion_max = (y_proba_fusion_max > 0.5).astype(int)
            metrics_max = calculate_metrics(
                X_test_array, Y_test_array, y_pred_fusion_max, y_proba_fusion_max,
                area_hits_K=int(config.get('area_hits_K', 500)),
            )

            max_df = pd.DataFrame({
                "SMILES": smiles_series,
                "LABEL": Y_test_array,
                "y_pred": y_pred_fusion_max,
                "y_proba": y_proba_fusion_max,
            })
            max_df.to_parquet(os.path.join(fusion_dir, f"fusion_max_{test_base}.parquet"), index=False)
            max_df.sort_values("y_proba", ascending=False).to_parquet(
                os.path.join(fusion_dir, f"fusion_max_sorted_{test_base}.parquet"), index=False,
            )
            fusion_result.update({f"max_{k}": v for k, v in metrics_max.items()})

        # =========================
        # Weighted fusion (logistic-regression meta-learner trained on best CV fold)
        # =========================
        if 'weighted' in fusion_methods and weighted_meta is not None:
            cols = weighted_meta['model_cols']
            missing = [c for c in cols if c not in y_probas_by_id]
            if missing:
                warnings.warn(
                    f"Weighted fusion: missing test predictions for {missing} on {test_base}; skipping."
                )
            else:
                X_meta_test = np.column_stack([y_probas_by_id[c] for c in cols]).astype(float)
                y_proba_w = weighted_meta['meta'].predict_proba(X_meta_test)[:, 1]
                y_pred_w = (y_proba_w > 0.5).astype(int)
                metrics_w = calculate_metrics(
                    X_test_array, Y_test_array, y_pred_w, y_proba_w,
                    area_hits_K=int(config.get('area_hits_K', 500)),
                )

                weighted_df = pd.DataFrame({
                    "SMILES": smiles_series,
                    "LABEL": Y_test_array,
                    "y_pred": y_pred_w,
                    "y_proba": y_proba_w,
                })
                weighted_df.to_parquet(os.path.join(fusion_dir, f"fusion_weighted_{test_base}.parquet"), index=False)
                weighted_df.sort_values("y_proba", ascending=False).to_parquet(
                    os.path.join(fusion_dir, f"fusion_weighted_sorted_{test_base}.parquet"), index=False,
                )
                fusion_result.update({f"weighted_{k}": v for k, v in metrics_w.items()})

        fusion_results.append(fusion_result)

    # Save the fusion results in a separate CSV file
    fusion_results_df = pd.DataFrame(fusion_results)
    fusion_results_csv_path = os.path.join(fusion_dir, "fusion.csv")
    print(fusion_results_csv_path)
    fusion_results_df.to_csv(fusion_results_csv_path, index=False)

    print(f"Fusion results saved to {fusion_results_csv_path}")


def plot_fusion_hit_curves(config, RunFolderName):
    """Hit@K curves for the fusion outputs in {RunFolderName}/Fusion/.

    For each test file, overlays the mean and max fusion curves on a single plot
    along with the random baseline (and optional simulated random fan).
    """
    # Hard-coded axis caps — keep in sync with the other hit-curve plots.
    Y_MAX = 100
    X_MAX = 1000
    # 0 = theoretical baseline only; bump to overlay simulated random permutations.
    N_RANDOM_RUNS = 0

    fusion_dir = os.path.join(RunFolderName, "Fusion")
    if not os.path.isdir(fusion_dir):
        print(f"No Fusion folder at {fusion_dir}; skipping fusion hit plots.")
        return

    # Collect all (method, sorted-parquet) pairs by union of any fusion_*_sorted_*.parquet.
    test_to_files = {}
    for prefix, label in (("fusion_mean_sorted_", "Mean"),
                          ("fusion_max_sorted_", "Max"),
                          ("fusion_weighted_sorted_", "Weighted")):
        for f in glob.glob(os.path.join(fusion_dir, f"{prefix}*.parquet")):
            base = os.path.basename(f)
            test_base = base[len(prefix):-len(".parquet")]
            test_to_files.setdefault(test_base, []).append((label, f))
    if not test_to_files:
        print(f"No fusion_*_sorted_*.parquet found in {fusion_dir}; skipping fusion hit plots.")
        return

    for test_base, files_to_plot in test_to_files.items():
        # Order curves consistently: Mean, Max, Weighted.
        order = {"Mean": 0, "Max": 1, "Weighted": 2}
        files_to_plot = sorted(files_to_plot, key=lambda x: order.get(x[0], 99))

        plt.figure(figsize=(9, 6))
        random_drawn = False
        plotted_any = False

        for mode_label, path in files_to_plot:
            df = pd.read_parquet(path)
            label_col = next((c for c in ('LABEL', 'label') if c in df.columns), None)
            if label_col is None:
                print(f"No label column in {path}; skipping.")
                continue
            y = df[label_col].astype(int).values
            n_total = len(y)
            n_pos = int(y.sum())
            if n_pos == 0:
                continue

            ranks = np.arange(1, n_total + 1)
            cum_hits = np.cumsum(y)

            if not random_drawn:
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
                plt.plot(ranks, random_curve, color='gray', linestyle='--',
                         linewidth=1.5, label='Random expected')
                random_drawn = True

            plt.plot(ranks, cum_hits, linewidth=2, label=f"Fusion ({mode_label})")
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel('K')
        plt.ylabel('Hit@K')
        title_text = f"Fusion hit curves - {test_base}"
        plt.title(textwrap.fill(title_text, width=60), fontsize=10)
        plt.xlim(0, X_MAX)
        plt.ylim(0, Y_MAX)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_png = os.path.join(fusion_dir, f"fusion_hit_curve_{test_base}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved: {out_png}")
#==============================================================================







