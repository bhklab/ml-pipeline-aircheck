"""
ModelGuidedScreening.py — standalone model-guided screening loop.

Bypasses the rest of the pipeline (no CV, no model selection, no fusion, no feature
fusion). Uses a single LGBM classifier with the hyperparameters defined under
config['hyperparameters']['lgbm'] in config.yaml, on the first train file, the
first test file, and the first fingerprint listed in desired_columns.

Loop:
  Step 0: Train on train_data, test on full test_data; save metrics.
  Step 1: From step-0 top-200 predictions, find rows with LABEL=1 (true positives).
          Move those rows from the test set into the training set's positives.
          Retrain. Test on the shrunken test set; save metrics.
  Step 2: Repeat the same move/retrain/test on top-200 of step-1 predictions.

Outputs (under {RunFolderName}/GuidedScreening/):
  models/model_step{0,1,2}.pkl
  predictions/step{0,1,2}_predictions.parquet           (un-sorted)
  predictions/step{0,1,2}_predictions_sorted.parquet    (sorted by y_prob desc)
  results_guided_screening.csv                          (one row per step)

Run:
  python ModelGuidedScreening.py
"""
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from data_utils import load_data
from eval_utils import calculate_metrics
from config_utils import read_config


# How many true positives to move from test → train per guidance step.
TOP_N = 500
# How many guidance rounds to run AFTER the initial step. Total rows in the CSV = N_GUIDANCE_STEPS + 1.
N_GUIDANCE_STEPS = 2


def _train_lgbm(X, y, params):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(**params)
    model.fit(X, y)
    return model


def _predict(model, X):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)
    return y_pred, y_proba


def _read_smiles_aligned(path, smiles_col, nrows):
    """Read SMILES in the same row order load_data uses (head(nrows), no dropna)."""
    df = pd.read_parquet(path, columns=[smiles_col])
    if isinstance(nrows, int) and nrows > 0:
        df = df.head(nrows)
    return df[smiles_col].values


def run_guided_screening(config_name="config.yaml",
                         top_n=TOP_N,
                         n_guidance_steps=N_GUIDANCE_STEPS):
    config, RunFolderName = read_config(config_name)

    out_dir = os.path.join(RunFolderName, "GuidedScreening")
    models_dir = os.path.join(out_dir, "models")
    preds_dir = os.path.join(out_dir, "predictions")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    train_path = config['train_data'][0]
    test_path = config['test_data'][0]
    column_name = config['desired_columns'][0]
    label_col_train = config['label_column_train']
    label_col_test = config['label_column_test']
    nrows_train = config.get('nrows_train', None)
    nrows_test = config.get('nrows_test', None)
    smiles_col = config.get('smiles_column', 'SMILES')
    area_hits_K = int(config.get('area_hits_K', 500))

    lgbm_params = (config.get('hyperparameters') or {}).get('lgbm', {})

    print(f"[ModelGuidedScreening] Train file : {train_path}")
    print(f"[ModelGuidedScreening] Test  file : {test_path}")
    print(f"[ModelGuidedScreening] Column     : {column_name}")
    print(f"[ModelGuidedScreening] LGBM params: {lgbm_params}")
    print(f"[ModelGuidedScreening] top_n      : {top_n}")
    print(f"[ModelGuidedScreening] guidance   : {n_guidance_steps} extra step(s) after step 0")

    # --- Initial train + test arrays ---
    X_train_df, Y_train_df = load_data(train_path, [column_name], label_col_train, nrows_train)
    train_X = np.stack(X_train_df[column_name])
    train_y = np.stack(Y_train_df.iloc[:, 0]).astype(int)
    train_smiles = _read_smiles_aligned(train_path, smiles_col, nrows_train)

    X_test_df, Y_test_df = load_data(test_path, [column_name], label_col_test, nrows_test)
    test_X = np.stack(X_test_df[column_name])
    test_y = np.stack(Y_test_df.iloc[:, 0]).astype(int)
    test_smiles = _read_smiles_aligned(test_path, smiles_col, nrows_test)

    # Mutable references that get updated each guidance round. We rebind the
    # cur_* names instead of copying — the originals aren't used after this
    # point, and a .copy() of test_X (~3.5 GiB on 460k×2048 float32) would
    # double-allocate and OOM on memory-tight machines. Disk files are not
    # touched; this only affects in-memory layout.
    cur_train_X = train_X
    cur_train_y = train_y
    cur_train_smiles = train_smiles
    cur_test_X = test_X
    cur_test_y = test_y
    cur_test_smiles = test_smiles
    del train_X, train_y, train_smiles, test_X, test_y, test_smiles

    train_filename = os.path.basename(train_path).split('.')[0]
    test_filename = os.path.basename(test_path).split('.')[0]

    rows = []
    for step in range(n_guidance_steps + 1):  # 0, 1, 2, ...
        print(f"\n[ModelGuidedScreening] === Step {step} | "
              f"train={len(cur_train_y)} ({int(cur_train_y.sum())} pos) | "
              f"test={len(cur_test_y)} ({int(cur_test_y.sum())} pos) ===")

        # Train on current pool.
        model = _train_lgbm(cur_train_X, cur_train_y, lgbm_params)
        model_path = os.path.join(models_dir, f"model_step{step}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Predict on current test pool.
        y_pred, y_proba = _predict(model, cur_test_X)

        # Reuse the same metric set the rest of the pipeline emits as Test_*.
        metrics = calculate_metrics(
            cur_test_X, cur_test_y, y_pred, y_proba,
            area_hits_K=area_hits_K, regression_target=None,
        )

        # Save predictions (unsorted + sorted by y_prob desc).
        pred_df = pd.DataFrame({
            smiles_col: cur_test_smiles,
            'LABEL': cur_test_y,
            'y_pred': y_pred,
            'y_prob': y_proba,
        })
        pred_df.to_parquet(os.path.join(preds_dir, f"step{step}_predictions.parquet"), index=False)
        pred_df.sort_values('y_prob', ascending=False).to_parquet(
            os.path.join(preds_dir, f"step{step}_predictions_sorted.parquet"), index=False,
        )

        # Build the result row (Test_* prefix matches the rest of the pipeline).
        now = datetime.now()
        row = {
            'GuidanceStep': step,
            'TrainFileName': train_filename,
            'TestFile': test_filename,
            'ColumnName': column_name,
            'ModelType': 'lgbm',
            'TrainSize': int(len(cur_train_y)),
            'TrainPos': int(cur_train_y.sum()),
            'TestSize': int(len(cur_test_y)),
            'TestPos': int(cur_test_y.sum()),
            'NConfirmedPos_MovedToTrain': 0,
            'ModelPath': model_path,
            'Date': now.date().isoformat(),
            'Time': now.strftime("%H:%M:%S"),
        }
        for k, v in metrics.items():
            row[f"Test_{k}"] = v
        rows.append(row)

        # If more guidance rounds are coming, move the top-N true positives to train.
        if step < n_guidance_steps and len(cur_test_y) > 0:
            n_take = min(top_n, len(cur_test_y))
            top_indices = np.argsort(-y_proba)[:n_take]
            confirmed_pos_indices = top_indices[cur_test_y[top_indices] == 1]
            n_confirmed = int(len(confirmed_pos_indices))
            print(f"[ModelGuidedScreening] Step {step}: {n_confirmed}/{n_take} top-{top_n} predictions are true positives "
                  f"→ moving them from test to train.")
            row['NConfirmedPos_MovedToTrain'] = n_confirmed

            if n_confirmed > 0:
                cur_train_X = np.vstack([cur_train_X, cur_test_X[confirmed_pos_indices]])
                cur_train_y = np.concatenate([cur_train_y, cur_test_y[confirmed_pos_indices]])
                cur_train_smiles = np.concatenate([cur_train_smiles, cur_test_smiles[confirmed_pos_indices]])

                keep_mask = np.ones(len(cur_test_y), dtype=bool)
                keep_mask[confirmed_pos_indices] = False
                cur_test_X = cur_test_X[keep_mask]
                cur_test_y = cur_test_y[keep_mask]
                cur_test_smiles = cur_test_smiles[keep_mask]

    results_path = os.path.join(out_dir, "results_guided_screening.csv")
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\n[ModelGuidedScreening] Wrote {len(rows)} step row(s) to {results_path}")
    print(f"[ModelGuidedScreening] Models     : {models_dir}")
    print(f"[ModelGuidedScreening] Predictions: {preds_dir}")


if __name__ == "__main__":
    run_guided_screening()
