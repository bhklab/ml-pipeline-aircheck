"""
SelfTrainingLoop4.py — tighter-positive variant of SelfTrainingLoop3.

Same as SelfTrainingLoop3 except the positive side is intentionally limited
to a small top-N:

  Positives:
      Top TOP_N_POS = 100 test rows by predicted probability, then
      KEEP only those with prob > POS_THRESHOLD = 0.2 (safety floor).
      The motivation: in Loop3 with TOP_N_POS = 1000 every picked row already
      had a score > 0.8 — so the 0.2 threshold wasn't really filtering
      anything, and we were pulling in low-confidence rows as positives.
      Cutting to top-100 keeps only the most confidently-positive
      predictions and reduces pseudo-label noise.

  Negatives (UNCHANGED from SelfTrainingLoop3 — REPLACE, not append):
      Identify test rows with prob < NEG_PROB_THRESHOLD = 0.001.
      RANDOMLY sample ceil(NEG_REPLACE_FRACTION * #current_train_negatives)
      of them. RANDOMLY remove the same number of negatives from the current
      training pool and append the new ones in their place. Net train-negative
      count is roughly preserved; composition shifts toward what the model
      is currently most confident about.

The test set itself is fixed across iterations.

Outputs (under {RunFolderName}/SelfTrainingLoop4/):
  models/model_iter{0..N}.pkl
  predictions/iter{0..N}_predictions.parquet           (un-sorted)
  predictions/iter{0..N}_predictions_sorted.parquet    (sorted by y_prob desc)
  results_self_training_loop4.csv                      (one row per iteration)

Run:
  python SelfTrainingLoop4.py
"""
import os
import math
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from data_utils import load_data
from eval_utils import calculate_metrics
from config_utils import read_config


# --- Tunables (override here, or by passing kwargs to run_self_training_loop4) ---
N_ITERATIONS         = 9      # number of self-training rounds AFTER the baseline (iter 0)

# Positive-side selection — tightened compared to Loop3 (was 1000):
TOP_N_POS            = 100
POS_THRESHOLD        = 0.2     # safety floor; only top-ranked rows with prob > this are kept

# Negative-side replacement (same as Loop3): candidates are test rows with prob < this.
NEG_PROB_THRESHOLD   = 0.001
# Fraction of current train negatives to REPLACE per iteration (random selection on both sides).
NEG_REPLACE_FRACTION = 0.5

INCLUDE_NEGATIVES    = True    # False = skip negative replacement, positives-only
RANDOM_SEED          = 42      # reproducible random sampling


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
    df = pd.read_parquet(path, columns=[smiles_col])
    if isinstance(nrows, int) and nrows > 0:
        df = df.head(nrows)
    return df[smiles_col].values


def run_self_training_loop4(config_name="config.yaml",
                            n_iterations=N_ITERATIONS,
                            top_n_pos=TOP_N_POS,
                            pos_threshold=POS_THRESHOLD,
                            neg_prob_threshold=NEG_PROB_THRESHOLD,
                            neg_replace_fraction=NEG_REPLACE_FRACTION,
                            include_negatives=INCLUDE_NEGATIVES,
                            random_seed=RANDOM_SEED):
    config, RunFolderName = read_config(config_name)

    out_dir = os.path.join(RunFolderName, "SelfTrainingLoop4")
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

    print(f"[SelfTrainingLoop4] Train file           : {train_path}")
    print(f"[SelfTrainingLoop4] Test  file           : {test_path}")
    print(f"[SelfTrainingLoop4] Column               : {column_name}")
    print(f"[SelfTrainingLoop4] LGBM params          : {lgbm_params}")
    print(f"[SelfTrainingLoop4] n_iterations         : {n_iterations}")
    print(f"[SelfTrainingLoop4] top_n_pos            : {top_n_pos}  (then filter prob > {pos_threshold})")
    print(f"[SelfTrainingLoop4] neg_prob_threshold   : {neg_prob_threshold}  (candidates below this)")
    print(f"[SelfTrainingLoop4] neg_replace_fraction : {neg_replace_fraction}  (random replacement)")
    print(f"[SelfTrainingLoop4] include_negatives    : {include_negatives}")
    print(f"[SelfTrainingLoop4] random_seed          : {random_seed}")

    # --- Load train + test once; free the pandas dataframes after stacking. ---
    X_train_df, Y_train_df = load_data(train_path, [column_name], label_col_train, nrows_train)
    train_X = np.stack(X_train_df[column_name])
    train_y = np.stack(Y_train_df.iloc[:, 0]).astype(int)
    train_smiles = _read_smiles_aligned(train_path, smiles_col, nrows_train)
    del X_train_df, Y_train_df

    X_test_df, Y_test_df = load_data(test_path, [column_name], label_col_test, nrows_test)
    test_X = np.stack(X_test_df[column_name])
    test_y = np.stack(Y_test_df.iloc[:, 0]).astype(int)
    test_smiles = _read_smiles_aligned(test_path, smiles_col, nrows_test)
    del X_test_df, Y_test_df

    # Rebind without copying — saves memory; vstack/concatenate produce fresh arrays anyway.
    cur_train_X = train_X
    cur_train_y = train_y
    cur_train_smiles = train_smiles
    del train_X, train_y, train_smiles

    pseudo_labeled = set()  # remember which test rows have been used so far

    train_filename = os.path.basename(train_path).split('.')[0]
    test_filename = os.path.basename(test_path).split('.')[0]
    rng = np.random.default_rng(random_seed)

    rows = []
    for it in range(n_iterations + 1):
        print(f"\n[SelfTrainingLoop4] === Iter {it} | "
              f"train={len(cur_train_y)} ({int(cur_train_y.sum())} pos) | "
              f"test={len(test_y)} ({int(test_y.sum())} pos, fixed) ===")

        model = _train_lgbm(cur_train_X, cur_train_y, lgbm_params)
        model_path = os.path.join(models_dir, f"model_iter{it}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        y_pred, y_proba = _predict(model, test_X)

        metrics = calculate_metrics(
            test_X, test_y, y_pred, y_proba,
            area_hits_K=area_hits_K, regression_target=None,
        )

        pred_df = pd.DataFrame({
            smiles_col: test_smiles,
            'LABEL': test_y,
            'y_pred': y_pred,
            'y_prob': y_proba,
        })
        pred_df.to_parquet(os.path.join(preds_dir, f"iter{it}_predictions.parquet"), index=False)
        pred_df.sort_values('y_prob', ascending=False).to_parquet(
            os.path.join(preds_dir, f"iter{it}_predictions_sorted.parquet"), index=False,
        )

        now = datetime.now()
        row = {
            'Iteration': it,
            'TrainFileName': train_filename,
            'TestFile': test_filename,
            'ColumnName': column_name,
            'ModelType': 'lgbm',
            'TrainSize': int(len(cur_train_y)),
            'TrainPos': int(cur_train_y.sum()),
            'TestSize': int(len(test_y)),
            'TestPos': int(test_y.sum()),
            'PseudoLabeledThisIter_Pos': 0,
            'NegReplacedThisIter': 0,
            'TopN_Pos': top_n_pos,
            'PosThreshold': pos_threshold,
            'NegProbThreshold': neg_prob_threshold,
            'NegReplaceFraction': neg_replace_fraction,
            'IncludeNegatives': include_negatives,
            'RandomSeed': random_seed,
            'ModelPath': model_path,
            'Date': now.date().isoformat(),
            'Time': now.strftime("%H:%M:%S"),
        }
        for k, v in metrics.items():
            row[f"Test_{k}"] = v
        rows.append(row)

        if it < n_iterations:
            available_mask = np.ones(len(test_y), dtype=bool)
            if pseudo_labeled:
                available_mask[list(pseudo_labeled)] = False

            # --- Positives (append): top TOP_N_POS by prob, filter prob > POS_THRESHOLD ---
            avail_indices = np.where(available_mask)[0]
            if len(avail_indices) > 0:
                avail_proba = y_proba[avail_indices]
                top_order = np.argsort(-avail_proba)[:top_n_pos]
                top_pos_candidates = avail_indices[top_order]
                pos_keep = top_pos_candidates[y_proba[top_pos_candidates] > pos_threshold]
            else:
                pos_keep = np.array([], dtype=int)
            n_pos_added = int(len(pos_keep))

            # Append the positives first.
            if n_pos_added > 0:
                cur_train_X = np.vstack([cur_train_X, test_X[pos_keep]])
                cur_train_y = np.concatenate([cur_train_y, np.ones(n_pos_added, dtype=int)])
                cur_train_smiles = np.concatenate([cur_train_smiles, test_smiles[pos_keep]])
                pseudo_labeled.update(pos_keep.tolist())

            # --- Negatives (REPLACE half of current train negatives, random both sides) ---
            n_replaced = 0
            if include_negatives:
                # Refresh available_mask after the positive picks above.
                neg_avail_mask = np.ones(len(test_y), dtype=bool)
                if pseudo_labeled:
                    neg_avail_mask[list(pseudo_labeled)] = False
                neg_candidates = np.where(neg_avail_mask & (y_proba < neg_prob_threshold))[0]

                # Number to replace = ceil(fraction * current train negatives).
                cur_train_neg_indices = np.where(cur_train_y == 0)[0]
                n_neg_train = int(len(cur_train_neg_indices))
                n_target = int(math.ceil(neg_replace_fraction * n_neg_train))

                # Cap by what's actually available (candidates and existing negatives).
                n_replace = min(n_target, len(neg_candidates), n_neg_train)

                if n_replace > 0:
                    chosen_test_idx = rng.choice(neg_candidates, size=n_replace, replace=False)
                    chosen_train_neg_idx = rng.choice(cur_train_neg_indices, size=n_replace, replace=False)

                    keep_mask = np.ones(len(cur_train_y), dtype=bool)
                    keep_mask[chosen_train_neg_idx] = False
                    cur_train_X = cur_train_X[keep_mask]
                    cur_train_y = cur_train_y[keep_mask]
                    cur_train_smiles = cur_train_smiles[keep_mask]

                    cur_train_X = np.vstack([cur_train_X, test_X[chosen_test_idx]])
                    cur_train_y = np.concatenate([cur_train_y, np.zeros(n_replace, dtype=int)])
                    cur_train_smiles = np.concatenate([cur_train_smiles, test_smiles[chosen_test_idx]])

                    pseudo_labeled.update(chosen_test_idx.tolist())
                    n_replaced = int(n_replace)

            row['PseudoLabeledThisIter_Pos'] = n_pos_added
            row['NegReplacedThisIter'] = n_replaced

            print(f"[SelfTrainingLoop4] Iter {it}: appended {n_pos_added} positives "
                  f"(top-{top_n_pos} & prob>{pos_threshold}); replaced {n_replaced} train "
                  f"negatives with randomly-chosen test rows where prob<{neg_prob_threshold}.")

    results_path = os.path.join(out_dir, "results_self_training_loop4.csv")
    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"\n[SelfTrainingLoop4] Wrote {len(rows)} iter row(s) to {results_path}")
    print(f"[SelfTrainingLoop4] Models     : {models_dir}")
    print(f"[SelfTrainingLoop4] Predictions: {preds_dir}")


if __name__ == "__main__":
    run_self_training_loop4()
