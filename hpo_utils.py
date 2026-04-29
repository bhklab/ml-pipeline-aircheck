"""
Optuna-based hyperparameter search.

Evaluates each trial on the same CV folds saved at {RunFolderName}/CVFolds/cv_splits.pkl
(produced by model_utils.create_cv_folds), so HPO is consistent with the rest
of the pipeline and never sees the test set.

Single source of truth for per-model search spaces is `suggest_params`. Edit a
model's branch there to widen / narrow its range — one place to change.
"""
import os
import pickle
import warnings

import numpy as np
import optuna
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
    precision_score,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _precision_at_k(y_true, y_score, k):
    k = min(k, len(y_true))
    if k <= 0:
        return 0.0
    order = np.argsort(y_score)[::-1][:k]
    return float(np.sum(y_true[order] == 1) / k)


def _hits_at_k(y_true, y_score, k):
    k = min(k, len(y_true))
    if k <= 0:
        return 0
    order = np.argsort(y_score)[::-1][:k]
    return int(np.sum(y_true[order] == 1))


def _ideal_area_hits_at_k(P, K):
    P = int(P); K = int(K)
    if P >= K:
        return K * (K + 1) // 2
    return P * (P + 1) // 2 + P * (K - P)


def _area_hits_at_k(y_true, y_score, K):
    """Σ_{k=1..K} hits@k. Mirrors eval_utils.area_hits_at_k."""
    K = int(K)
    if K <= 0 or len(y_true) == 0:
        return 0
    K = min(K, len(y_true))
    order = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true)[order][:K].astype(int)
    return int(np.cumsum(y_sorted).sum())


def _area_hits_at_k_norm(y_true, y_score, K):
    P = int(np.sum(np.asarray(y_true) == 1))
    ideal = _ideal_area_hits_at_k(P, K)
    if ideal == 0:
        return 0.0
    return float(_area_hits_at_k(y_true, y_score, K) / ideal)


def _resolve_k(metric, default_k):
    """Pull K off the tail of `metric` (e.g. 'area_hits_at_50' -> 50);
    fall back to `default_k` when the suffix is the literal placeholder 'k'."""
    tail = metric.rsplit('_', 1)[-1]
    if tail.isdigit():
        return int(tail)
    if default_k is None:
        raise ValueError(
            f"Metric {metric!r} needs a K but none was provided "
            "(set optuna_K in config or use an inline form like 'area_hits_at_100')."
        )
    return int(default_k)


def _score(y_true, y_pred, y_proba, metric, default_k=None):
    """Return a scalar score for the given metric name (higher is better)."""
    metric = str(metric).lower()
    if metric == 'f1':
        return f1_score(y_true, y_pred, zero_division=0)
    if metric == 'roc_auc':
        return roc_auc_score(y_true, y_proba)
    if metric in ('average_precision', 'auc_pr', 'pr_auc'):
        return average_precision_score(y_true, y_proba)
    if metric == 'precision':
        return precision_score(y_true, y_pred, zero_division=0)
    if metric.startswith('area_hits_at_') and metric.endswith('_norm'):
        # 'area_hits_at_k_norm' or 'area_hits_at_100_norm'
        inner = metric[:-len('_norm')]
        k = _resolve_k(inner, default_k)
        return _area_hits_at_k_norm(y_true, y_proba, k)
    if metric.startswith('area_hits_at_'):
        k = _resolve_k(metric, default_k)
        return _area_hits_at_k(y_true, y_proba, k)
    if metric.startswith('precision_at_'):
        k = _resolve_k(metric, default_k)
        return _precision_at_k(y_true, y_proba, k)
    if metric.startswith('hits_at_'):
        k = _resolve_k(metric, default_k)
        return _hits_at_k(y_true, y_proba, k)
    raise ValueError(f"Unknown HPO metric: {metric}")


# ---------------------------------------------------------------------------
# Per-model search-space registry
# ---------------------------------------------------------------------------

def suggest_params(trial, model_name):
    """Reasonable default Optuna search space per model.

    Edit a branch to widen / narrow that model's ranges. Keys that don't depend
    on the trial (e.g. solver=saga for LR, verbosity=-1 for LGBM) are baked in
    here so they always reach the model.
    """
    if model_name == 'lgbm':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 5000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 4, 6, 8, 10, 14]),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 20.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 50.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 20.0),
            'verbosity': -1,
            'n_jobs': 1,
            'random_state': 42,
        }

    if model_name == 'lgbmregressor':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 200, 5000),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 255),
            'max_depth': trial.suggest_categorical('max_depth', [-1, 4, 6, 8, 10, 14]),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 20.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 50.0, log=True),
            'verbosity': -1,
            'n_jobs': 1,
            'random_state': 42,
        }

    if model_name == 'rf':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_categorical('max_depth', [None, 8, 16, 24, 40]),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'n_jobs': 1,
            'random_state': 42,
        }

    if model_name == 'lr':
        return {
            'C': trial.suggest_float('C', 1e-3, 100.0, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'solver': 'saga',
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'max_iter': 2000,
            'n_jobs': 1,
            'random_state': 42,
        }

    if model_name == 'sgd':
        return {
            'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
            'max_iter': 2000,
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'random_state': 42,
        }

    if model_name == 'svc':
        return {
            'C': trial.suggest_float('C', 1e-2, 100.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'random_state': 42,
        }

    if model_name == 'nb':
        return {
            'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True),
        }

    if model_name == 'dt':
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
            'random_state': 42,
        }

    if model_name == 'knn':
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'p': trial.suggest_categorical('p', [1, 2]),
            'n_jobs': 1,
        }

    if model_name == 'gb':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'random_state': 42,
        }

    if model_name == 'ada':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0, log=True),
            'random_state': 42,
        }

    if model_name == 'bag':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 10, 200),
            'max_samples': trial.suggest_float('max_samples', 0.3, 1.0),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
            'n_jobs': 1,
            'random_state': 42,
        }

    if model_name == 'mlp':
        arch = trial.suggest_categorical(
            'hidden_layer_sizes', ['64', '128', '64-32', '128-64', '128-64-32'],
        )
        layers = tuple(int(x) for x in arch.split('-'))
        return {
            'hidden_layer_sizes': layers,
            'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'max_iter': 500,
            'early_stopping': True,
            'random_state': 42,
        }

    if model_name == 'catboost':
        return {
            'iterations': trial.suggest_int('iterations', 200, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 30.0, log=True),
            'random_seed': 42,
            'silent': True,
        }

    if model_name == 'tf_ff':
        arch = trial.suggest_categorical('hidden_units', ['64', '128-64', '256-128-64'])
        return {
            'hidden_units': [int(x) for x in arch.split('-')],
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        }

    if model_name == 'tf_cnn1D':
        conv_choice = trial.suggest_categorical(
            'conv_layers', ['64-3', '32-3,16-3', '64-5,32-3'],
        )
        ff_choice = trial.suggest_categorical('ff_layers', ['32', '64-32', '128-64'])
        return {
            'conv_layers': [tuple(int(x) for x in c.split('-')) for c in conv_choice.split(',')],
            'ff_layers': [int(x) for x in ff_choice.split('-')],
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        }

    raise ValueError(f"No Optuna search space defined for model: {model_name!r}")


# ---------------------------------------------------------------------------
# Search driver
# ---------------------------------------------------------------------------

def _model_predict_proba(model, X):
    """Positive-class probabilities, supporting sklearn classifiers and Keras models."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.predict(X).flatten()


def _restore_param_types(params):
    """Optuna user_attrs round-trip through JSON, so tuples become lists.
    Some sklearn / Keras builders care; restore the few known tuple/list shapes.
    """
    out = dict(params)
    if 'hidden_layer_sizes' in out and isinstance(out['hidden_layer_sizes'], list):
        out['hidden_layer_sizes'] = tuple(out['hidden_layer_sizes'])
    if 'conv_layers' in out and isinstance(out['conv_layers'], list):
        out['conv_layers'] = [tuple(x) for x in out['conv_layers']]
    return out


def load_cv_splits(RunFolderName):
    """Load CV splits saved by create_cv_folds; raise a clear error if missing."""
    path = os.path.join(RunFolderName, "CVFolds", "cv_splits.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CV splits not found at {path}. Run create_cv_folds first "
            "(Step 2.5 of aircheck_pipeline)."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def optuna_hyperparameter_search(
    model_name,
    X,
    y,
    cv_splits,
    get_model,
    train_model,
    n_trials=30,
    metric='area_hits_at_k',
    k=100,
    n_jobs=1,
    storage_path=None,
    study_name=None,
    random_state=42,
):
    """Run Optuna TPE search using the supplied CV splits.

    Each trial: trains the model on every (train_idx, test_idx) split, scores
    via `metric`, returns the mean across folds. MedianPruner aborts under-
    performing trials at intermediate fold boundaries. If `storage_path` is
    given the study is persisted to SQLite (resumable across runs).

    Returns: (best_params: dict, study: optuna.Study).
    """
    n_samples = X.shape[0]
    max_idx = max(int(max(t.max(), s.max())) for t, s in cv_splits)
    if max_idx >= n_samples:
        raise ValueError(
            f"CV split indices reference up to {max_idx} but X has only {n_samples} rows. "
            "The CV folds were built from a different version of the train file — "
            "rerun create_cv_folds to refresh them."
        )

    # TensorFlow models are not thread-safe inside Optuna's parallel objective —
    # Keras layer-name registration and global session state race across threads.
    # Clamp to 1 trial at a time for tf_ff / tf_cnn1D so users don't hit cryptic errors.
    if model_name in ('tf_ff', 'tf_cnn1D') and n_jobs > 1:
        warnings.warn(
            f"optuna_n_jobs={n_jobs} is unsafe for TensorFlow model {model_name!r}; "
            "clamping to 1 to avoid Keras / TF threading errors."
        )
        n_jobs = 1

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    storage_url = None
    if storage_path:
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        storage_url = f"sqlite:///{storage_path}"

    input_shape = X.shape[1]

    def objective(trial):
        params = suggest_params(trial, model_name)
        # Persist the full param dict (incl. fixed kwargs) for retrieval after the study.
        json_safe = {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}
        trial.set_user_attr('full_params', json_safe)

        fold_scores = []
        for fold_idx, (tr_idx, te_idx) in enumerate(cv_splits):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            try:
                model = get_model(model_name, dict(params), input_shape)
                model = train_model(model, X_tr, y_tr)
                y_proba = _model_predict_proba(model, X_te)
                y_pred = (y_proba > 0.5).astype(int)
                fold_scores.append(_score(y_te, y_pred, y_proba, metric, default_k=k))
            except Exception as e:
                warnings.warn(f"Trial {trial.number} fold {fold_idx + 1} failed: {e}")
                fold_scores.append(0.0)

            trial.report(float(np.mean(fold_scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        study_name=study_name or f"hpo_{model_name}",
        storage=storage_url,
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True)

    full_params = study.best_trial.user_attrs.get('full_params')
    best_params = _restore_param_types(full_params) if full_params else dict(study.best_trial.params)

    print(
        f"[Optuna] {model_name}: best mean {metric} = {study.best_value:.4f} "
        f"(trial #{study.best_trial.number} of {len(study.trials)})"
    )
    return best_params, study
