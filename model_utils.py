from json import load
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
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier
from nn_models import build_simple_ffnn,  build_configurable_cnn_1d

import warnings
warnings.filterwarnings("ignore")
#==============================================================================
#==============================================================================
def get_model(model_name, best_params={}, input_shape=20):
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
    elif model_name == 'lgbm':
        return LGBMClassifier(**best_params) if best_params else LGBMClassifier()
    elif model_name == 'lgbmregressor':
        return LGBMRegressor(**best_params) if best_params else LGBMRegressor()
    elif model_name == 'catboost':
        return CatBoostClassifier(silent=True, **best_params) if best_params else CatBoostClassifier(silent=True)
    elif model_name == 'tf_ff':
        #return build_simple_ffnn(input_shape=2048, hidden_units=[64,32], learning_rate=0.001)
        best_params['input_shape'] = input_shape
        return build_simple_ffnn(**best_params) if best_params else build_simple_ffnn()
    elif model_name == 'tf_cnn1D':
        best_params['input_shape'] = input_shape
        #return build_configurable_cnn_1d(input_shape=2048, conv_layers=[(64, 3)], ff_layers=[32], learning_rate=0.001)
        return build_configurable_cnn_1d(**best_params) if best_params else build_configurable_cnn_1d()
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
        print(train_path)
        # Extract the filename without extension for naming
        train_filename = os.path.basename(train_path).split('.')[0]
        
        total_columns = column_names
        if config['feature_fusion_method'] != "None":
            X_train, Y_train = load_data(train_path, column_names, label_column_train, nrows_train)
            #Y_train_array = np.stack(Y_train.iloc[:, 0])
            #X_train_array = np.stack(X_train[column_names_j])
            # === Feature Fusion (optional) ===
            X_train, fused_column_name = fuse_columns(X_train, column_names , feature_fusion_method)
            total_columns = fused_column_name

        for model_name_i in model_names:
            for column_names_j in total_columns:
                
                # Train data
                if config['feature_fusion_method'] == "None":
                    X_train, Y_train = load_data(train_path, [column_names_j], label_column_train, nrows_train)
                    #Y_train_array = np.stack(Y_train.iloc[:, 0])
                    #X_train_array = np.stack(X_train[column_names_j])
                Y_train_array = np.stack(Y_train.iloc[:, 0])
                X_train_array = np.stack(X_train[column_names_j])
                #print(X_train_array.shape[1])
                
                # Model subfolder includes train filename
                model_subfolder = os.path.join(RunFolderName, f"{train_filename}_{model_name_i}_{column_names_j}")
                print(model_subfolder)
                os.makedirs(model_subfolder, exist_ok=True)

                # === Hyperparameter Tuning ===
                # 'none' -> use config['hyperparameters'][model_name_i] as-is.
                # 'bayesian' -> existing scikit-optimize BayesSearchCV (legacy).
                # 'optuna' -> Optuna TPE search using saved CV folds (hpo_utils.py).
                hpo_mode = str(hyperparameters_tuning).lower()
                if hpo_mode in ('y', 'bayesian'):
                    best_params = bayesian_hyperparameter_search(model_name_i, X_train_array, Y_train_array)
                elif hpo_mode == 'optuna':
                    from hpo_utils import optuna_hyperparameter_search, load_cv_splits
                    cv_splits = load_cv_splits(RunFolderName)
                    storage_path = os.path.join(
                        RunFolderName, 'HPO', f'{model_name_i}_{column_names_j}.db'
                    )
                    best_params, _ = optuna_hyperparameter_search(
                        model_name=model_name_i,
                        X=X_train_array,
                        y=Y_train_array,
                        cv_splits=cv_splits,
                        get_model=get_model,
                        train_model=train_model,
                        n_trials=int(config.get('optuna_n_trials', 30)),
                        metric=str(config.get('optuna_metric', 'area_hits_at_k')),
                        k=int(config.get('optuna_K', 100)),
                        n_jobs=int(config.get('optuna_n_jobs', 1)),
                        storage_path=storage_path,
                        study_name=f"hpo_{model_name_i}_{column_names_j}",
                    )
                elif hpo_mode in ('n', 'none', 'no'):
                    best_params = hyperparameters.get(model_name_i, {})
                else:
                    raise ValueError(
                        f"hyperparameters_tuning must be 'none', 'bayesian', or 'optuna'; got {hyperparameters_tuning!r}"
                    )

                # === Cross Validation ===
                avg_metrics = cross_validate_and_save_models(
                    config,
                    X_train_array=X_train_array,
                    Y_train_array=Y_train_array,
                    model_name=model_name_i,
                    model_subfolder=model_subfolder,
                    Nfold=Nfold,
                    get_model=get_model,
                    train_model=train_model,
                    evaluate_model=evaluate_model,
                    best_params=best_params,
                    train_path=train_path,
                    train_filename=train_filename,
                    column_name=column_names_j,
                )

                # === Final Model Training ===
                train_and_save_final_model(
                    config,
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
VALID_DESCRIPTORS = ["ECFP4", "ECFP6", "FCFP4", "FCFP6", "TOPTOR", "MACCS", "RDK", "AVALON", "ATOMPAIR"]


def _tanimoto_distance_matrix_binary(X_bin):
    """Pairwise Tanimoto distance matrix for binary fingerprints."""
    X_bin = (X_bin > 0).astype(np.uint8)
    intersection = X_bin @ X_bin.T
    bit_counts = X_bin.sum(axis=1, keepdims=True)
    union = bit_counts + bit_counts.T - intersection
    sim = np.divide(
        intersection, union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=(union != 0),
    ).astype(np.float32)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def _build_positive_test_groups_cluster_exclusion(
    cluster_to_pos_indices, target_size, max_groups, random_state
):
    """Greedy cluster-held-out positive group builder."""
    rng = np.random.default_rng(random_state)
    unused_clusters = set(cluster_to_pos_indices.keys())
    test_pos_groups = []
    excluded_pos_groups = []
    group_descriptions = []

    while len(test_pos_groups) < max_groups:
        available = [(cid, cluster_to_pos_indices[cid]) for cid in unused_clusters]
        if not available:
            break
        available_desc = sorted(available, key=lambda x: len(x[1]), reverse=True)
        remaining_total = sum(len(m) for _, m in available_desc)
        if remaining_total < target_size:
            break

        seed_cid, seed_members = available_desc[0]
        selected_cluster_ids = [seed_cid]
        combined_members = list(seed_members)

        if len(combined_members) < target_size:
            other = [(cid, cluster_to_pos_indices[cid]) for cid in unused_clusters if cid != seed_cid]
            other = sorted(other, key=lambda x: len(x[1]))  # smallest first
            for cid, members in other:
                if len(combined_members) >= target_size:
                    break
                selected_cluster_ids.append(cid)
                combined_members.extend(members)

        if len(combined_members) < target_size:
            break

        combined_members = np.array(combined_members)
        sampled_test = rng.choice(combined_members, size=target_size, replace=False)
        test_pos_groups.append(np.sort(sampled_test))
        excluded_pos_groups.append(np.sort(combined_members))
        group_descriptions.append({
            "fold": len(test_pos_groups),
            "cluster_ids": str(selected_cluster_ids),
            "n_excluded_pos_total": len(combined_members),
            "n_test_pos_final": target_size,
        })
        for cid in selected_cluster_ids:
            unused_clusters.discard(cid)

    return test_pos_groups, excluded_pos_groups, group_descriptions


def _build_stratified_folds(y, n_splits, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return [(train_idx, test_idx) for train_idx, test_idx in skf.split(np.zeros((len(y), 1)), y)]


def _pick_cluster_feature(train_path):
    """Prefer ECFP4 if present in the parquet; otherwise fall back to any known descriptor."""
    import pyarrow.parquet as pq
    schema_names = pq.ParquetFile(train_path).schema_arrow.names
    if "ECFP4" in schema_names:
        return "ECFP4"
    for c in VALID_DESCRIPTORS:
        if c in schema_names:
            return c
    raise ValueError(
        f"None of the known descriptor columns {VALID_DESCRIPTORS} found in {train_path}; "
        "cannot perform cluster-based fold creation."
    )


def _build_cluster_folds(X_features, y, folds_dir):
    """Cluster-held-out CV splits: AgglomerativeClustering on Tanimoto distance of positives."""
    from sklearn.cluster import AgglomerativeClustering

    # Hard-coded experiment parameters (intentionally not in config.yaml).
    n_clusters = 20                # Number of agglomerative clusters built from the positives.
                                   # Chosen so that ~target_n_folds groups of clusters of size
                                   # ~target_test_pos_per_fold can be carved out (≈2 clusters/fold).
    linkage = "average"            # AgglomerativeClustering linkage on the precomputed Tanimoto
                                   # distance matrix. 'average' balances chaining vs. tight clusters.
    target_n_folds = 10            # Maximum number of CV folds to build. Fewer may be produced if
                                   # the cluster pool runs out of unused positives before reaching this.
    target_test_pos_per_fold = 100 # Number of positives held out as the test set of each fold.
                                   # The full cluster(s) supplying those positives are excluded
                                   # from the fold's training set (cluster-held-out evaluation).
    test_neg_fraction = 0.25       # Fraction of all negatives randomly sampled (without replacement)
                                   # for each fold's test set. Sampled fresh per fold, so negatives
                                   # may overlap across folds. Training gets the remaining negatives.
    random_state = 42              # Seed for the RNG used both to pick test positives within a
                                   # cluster group and to sample test negatives — ensures the same
                                   # folds are reproduced on re-runs.

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < target_test_pos_per_fold:
        raise ValueError("Not enough positives to create even one cluster-based fold.")

    print("Computing Tanimoto distance matrix for positives...")
    dist_pos = _tanimoto_distance_matrix_binary(X_features[pos_idx])
    print("Running AgglomerativeClustering...")
    agg = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage=linkage)
    pos_cluster_labels = agg.fit_predict(dist_pos)

    cluster_to_pos_indices = {}
    for cid in np.unique(pos_cluster_labels):
        members_local = np.where(pos_cluster_labels == cid)[0]
        cluster_to_pos_indices[cid] = pos_idx[members_local]

    pd.DataFrame([
        {"cluster_id": cid, "size": len(members)}
        for cid, members in cluster_to_pos_indices.items()
    ]).sort_values("size", ascending=False).reset_index(drop=True).to_csv(
        os.path.join(folds_dir, "positive_cluster_sizes.csv"), index=False
    )

    # Per-positive cluster id (orig_idx is the row index in the dropna'd train table).
    pd.DataFrame({
        "orig_idx": pos_idx,
        "cluster_id": pos_cluster_labels,
    }).to_csv(os.path.join(folds_dir, "positive_cluster_labels.csv"), index=False)

    test_pos_groups, excluded_pos_groups, group_descriptions = _build_positive_test_groups_cluster_exclusion(
        cluster_to_pos_indices=cluster_to_pos_indices,
        target_size=target_test_pos_per_fold,
        max_groups=target_n_folds,
        random_state=random_state,
    )
    if not test_pos_groups:
        raise ValueError("No positive test folds could be built from agglomerative clusters.")

    pd.DataFrame(group_descriptions).to_csv(
        os.path.join(folds_dir, "positive_test_group_summary.csv"), index=False
    )

    rng = np.random.default_rng(random_state)
    n_test_neg_per_fold = int(round(len(neg_idx) * test_neg_fraction))

    cv_splits = []
    for test_pos_idx, excluded_pos_idx in zip(test_pos_groups, excluded_pos_groups):
        test_neg_idx = rng.choice(neg_idx, size=n_test_neg_per_fold, replace=False)
        test_idx = np.sort(np.concatenate([test_pos_idx, test_neg_idx]))
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_neg_idx] = False
        train_mask[excluded_pos_idx] = False
        train_idx = np.where(train_mask)[0]
        cv_splits.append((train_idx, test_idx))

    return cv_splits


def create_cv_folds(config, RunFolderName):
    """Build CV folds and persist them under {RunFolderName}/CVFolds/.

    config keys read:
      - cv_method:          'stratified' or 'cluster'
      - train_data:         list of train paths (uses the first)
      - label_column_train: list with the label column name
      - nrows_train:        int or None
      - Nfold:              only used for stratified

    Cluster-mode parameters (n_clusters=20, target_n_folds=10,
    target_test_pos_per_fold=100, test_neg_fraction=0.25) are hard-coded inside
    _build_cluster_folds. The clustering feature is ECFP4 when present,
    otherwise the first available column from VALID_DESCRIPTORS.
    """
    cv_method = config.get('cv_method', 'stratified').lower()
    train_path = config['train_data'][0]
    label_col = config['label_column_train'][0]
    nrows = config['nrows_train']
    if not (isinstance(nrows, int) and nrows > 0):
        nrows = None

    folds_dir = os.path.join(RunFolderName, 'CVFolds')
    os.makedirs(folds_dir, exist_ok=True)

    if cv_method == 'stratified':
        df_label = pd.read_parquet(train_path, columns=[label_col])
        if nrows is not None:
            df_label = df_label.head(nrows)
        y = df_label[label_col].astype(int).values
        cv_splits = _build_stratified_folds(y, n_splits=config['Nfold'])

    elif cv_method == 'cluster':
        cluster_feature = _pick_cluster_feature(train_path)
        print(f"Cluster CV: using '{cluster_feature}' as the clustering feature.")
        df = pd.read_parquet(train_path, columns=[cluster_feature, label_col])
        if nrows is not None:
            df = df.head(nrows)
        df = df.dropna(subset=[cluster_feature, label_col]).reset_index(drop=True)
        first_val = df[cluster_feature].iloc[0]
        if isinstance(first_val, str):
            X_features = np.vstack(df[cluster_feature].apply(
                lambda x: np.fromstring(x, sep=',', dtype=np.float32)
            ).values)
        else:
            X_features = np.vstack(df[cluster_feature].apply(np.array).values).astype(np.float32)
        y = df[label_col].astype(int).values
        cv_splits = _build_cluster_folds(X_features, y, folds_dir)

    else:
        raise ValueError(f"cv_method must be 'stratified' or 'cluster', got: {cv_method}")

    with open(os.path.join(folds_dir, 'cv_splits.pkl'), 'wb') as f:
        pickle.dump(cv_splits, f)

    summary_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        pd.DataFrame({'train_idx': train_idx}).to_csv(
            os.path.join(folds_dir, f'fold_{fold_idx}_train_idx.csv'), index=False
        )
        pd.DataFrame({'test_idx': test_idx}).to_csv(
            os.path.join(folds_dir, f'fold_{fold_idx}_test_idx.csv'), index=False
        )
        y_train = y[train_idx]
        y_test = y[test_idx]
        summary_rows.append({
            'fold': fold_idx,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_pos': int((y_train == 1).sum()),
            'train_neg': int((y_train == 0).sum()),
            'test_pos': int((y_test == 1).sum()),
            'test_neg': int((y_test == 0).sum()),
        })
    pd.DataFrame(summary_rows).to_csv(os.path.join(folds_dir, 'fold_summary.csv'), index=False)

    print(f"Saved {len(cv_splits)} {cv_method} CV folds to {folds_dir}")


def plot_cv_folds_umap(config, RunFolderName):
    """Generate per-fold UMAP plots (Jaccard distance on binary fingerprints).

    Reads {RunFolderName}/CVFolds/cv_splits.pkl (produced by create_cv_folds),
    computes ONE 2-D UMAP embedding over a positive-heavy reduced sample of
    the training file (all positives + neg_ratio*#pos random negatives), then
    saves one plot per fold showing train positives vs test positives so you
    can verify they're cleanly separated.

    Output: {RunFolderName}/CVFolds/umap_fold_{i}.png
    """
    import umap
    import matplotlib.pyplot as plt

    train_path = config['train_data'][0]
    label_col = config['label_column_train'][0]
    nrows = config['nrows_train']
    if not (isinstance(nrows, int) and nrows > 0):
        nrows = None

    folds_dir = os.path.join(RunFolderName, 'CVFolds')
    cv_splits_path = os.path.join(folds_dir, 'cv_splits.pkl')
    if not os.path.exists(cv_splits_path):
        raise FileNotFoundError(
            f"CV splits not found at {cv_splits_path}. Run create_cv_folds first."
        )

    feature_column = _pick_cluster_feature(train_path)
    print(f"UMAP plot: using '{feature_column}' as the embedding feature.")

    df = pd.read_parquet(train_path, columns=[feature_column, label_col])
    if nrows is not None:
        df = df.head(nrows)
    df = df.dropna(subset=[feature_column, label_col]).reset_index(drop=True)
    df['orig_idx'] = df.index  # row index in the original train table — matches cv_splits indices

    # Reduce to a positive-heavy subset so UMAP is tractable.
    random_seed = 42
    neg_ratio = 1
    df_pos = df[df[label_col] == 1]
    df_neg = df[df[label_col] == 0]
    n_neg_sample = min(len(df_neg), neg_ratio * len(df_pos))
    df_neg_sampled = df_neg.sample(n=n_neg_sample, random_state=random_seed)
    df_umap = pd.concat([df_pos, df_neg_sampled], axis=0).reset_index(drop=True)

    first_val = df_umap[feature_column].iloc[0]
    if isinstance(first_val, str):
        X = np.vstack(df_umap[feature_column].apply(
            lambda x: np.fromstring(x, sep=',', dtype=np.float32)
        ).values)
    else:
        X = np.vstack(df_umap[feature_column].apply(np.array).values)
    X = (X > 0).astype(np.uint8)
    y = df_umap[label_col].astype(int).values
    orig_idx = df_umap['orig_idx'].values

    print(f"Computing UMAP embedding (Jaccard) on {X.shape[0]} samples...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='jaccard',
        n_components=2,
        random_state=42,
        low_memory=True,
        verbose=True,
    )
    X_umap = reducer.fit_transform(X)

    with open(cv_splits_path, 'rb') as f:
        cv_splits = pickle.load(f)

    # Optional: agglomerative-cluster colouring of positives (cluster mode only).
    cluster_labels_path = os.path.join(folds_dir, 'positive_cluster_labels.csv')
    if os.path.exists(cluster_labels_path):
        cluster_df = pd.read_csv(cluster_labels_path)
        idx_to_cluster = dict(zip(cluster_df['orig_idx'], cluster_df['cluster_id']))

        pos_mask = (y == 1)
        X_umap_pos = X_umap[pos_mask]
        pos_orig_idx = orig_idx[pos_mask]
        pos_clusters = np.array([idx_to_cluster.get(i, -1) for i in pos_orig_idx])

        plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap('tab20')
        unique_clusters = sorted(c for c in np.unique(pos_clusters) if c != -1)
        for i, cid in enumerate(unique_clusters):
            mask = (pos_clusters == cid)
            plt.scatter(
                X_umap_pos[mask, 0], X_umap_pos[mask, 1],
                s=20, alpha=0.85, color=cmap(i % cmap.N),
                label=f"Cluster {cid}",
            )
        plt.title("Agglomerative clusters of positives (UMAP)")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.legend(fontsize=8, ncol=2, bbox_to_anchor=(1.02, 1.0), loc='upper left')
        plt.tight_layout()
        cluster_plot_path = os.path.join(folds_dir, "umap_positive_clusters.png")
        plt.savefig(cluster_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {cluster_plot_path}")

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        train_mask = np.isin(orig_idx, train_idx)
        test_mask = np.isin(orig_idx, test_idx)

        y_train = y[train_mask]
        y_test = y[test_mask]
        X_umap_train = X_umap[train_mask]
        X_umap_test = X_umap[test_mask]

        plt.figure(figsize=(8, 6))
        plt.scatter(
            X_umap_train[y_train == 1, 0], X_umap_train[y_train == 1, 1],
            c='darkblue', s=18, alpha=0.8, label='Train Positive',
        )
        plt.scatter(
            X_umap_test[y_test == 1, 0], X_umap_test[y_test == 1, 1],
            c='darkred', s=28, alpha=1.0, label='Test Positive',
        )
        plt.title(f"UMAP - Fold {fold_idx}")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(folds_dir, f"umap_fold_{fold_idx}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved: {out_path}")


#==============================================================================
#==============================================================================
def cross_validate_and_save_models(config, X_train_array, Y_train_array, model_name, model_subfolder, Nfold, get_model, train_model, evaluate_model, best_params,
                                   train_path=None, train_filename=None, column_name=None):
    """
    Perform cross-validation using pre-built fold splits, save fold models, and return average metrics.

    The CV splits are loaded from {RunFolderName}/CVFolds/cv_splits.pkl,
    which is produced by create_cv_folds.

    Per-fold and mean-across-folds prediction parquets are written to
    {RunFolderName}/CVResults/, named with train_filename, model_name and
    column_name (mirroring the test-pipeline naming).
    """
    tf_models = config['tf_models']
    tf_dnn = True if model_name in tf_models else False

    RunFolderName = os.path.dirname(model_subfolder)
    cv_splits_path = os.path.join(RunFolderName, 'CVFolds', 'cv_splits.pkl')
    with open(cv_splits_path, 'rb') as f:
        cv_splits = pickle.load(f)

    cv_results_dir = os.path.join(RunFolderName, 'CVResults')
    os.makedirs(cv_results_dir, exist_ok=True)

    # Load SMILES + label aligned with X_train_array (same head(nrows_train) view
    # that load_data used in train_pipeline), so cv_splits indices map correctly.
    df_meta_train = None
    if train_path is not None:
        smiles_col = config.get('smiles_column', 'SMILES')
        label_col = config['label_column_train'][0]
        nrows_train = config.get('nrows_train', None)
        if not (isinstance(nrows_train, int) and nrows_train > 0):
            nrows_train = None
        df_meta_train = pd.read_parquet(train_path, columns=[smiles_col, label_col])
        if nrows_train is not None:
            df_meta_train = df_meta_train.head(nrows_train)
        df_meta_train = df_meta_train.reset_index(drop=True)

    fold_metrics = []
    input_shape = X_train_array.shape[1]
    # Accumulate per-sample fold predictions to build the mean prediction parquet.
    n_samples = X_train_array.shape[0]
    sum_prob = np.zeros(n_samples, dtype=np.float64)
    count_prob = np.zeros(n_samples, dtype=np.int32)

    # Optional per-fold negative downsampling. If set, keep all positives in the fold's
    # training set and randomly subsample negatives so that #neg = ratio * #pos.
    # Test fold is unaffected.
    cluster_fold_train_neg_ratio = config.get('cluster_fold_train_neg_ratio', None)
    if isinstance(cluster_fold_train_neg_ratio, str) and cluster_fold_train_neg_ratio.lower() == 'none':
        cluster_fold_train_neg_ratio = None

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        CrossVal_data_train, CrossVal_data_test = X_train_array[train_idx], X_train_array[test_idx]
        CrossVal_label_train, CrossVal_label_test = Y_train_array[train_idx], Y_train_array[test_idx]

        if cluster_fold_train_neg_ratio is not None:
            rng = np.random.default_rng(42 + fold_idx)
            pos_local_idx = np.where(CrossVal_label_train == 1)[0]
            neg_local_idx = np.where(CrossVal_label_train == 0)[0]
            n_pos = len(pos_local_idx)
            n_neg_target = int(cluster_fold_train_neg_ratio * n_pos)
            if len(neg_local_idx) > n_neg_target:
                sampled_neg_local_idx = rng.choice(neg_local_idx, size=n_neg_target, replace=False)
                keep_local_idx = np.concatenate([pos_local_idx, sampled_neg_local_idx])
                rng.shuffle(keep_local_idx)
                CrossVal_data_train = CrossVal_data_train[keep_local_idx]
                CrossVal_label_train = CrossVal_label_train[keep_local_idx]

        model_fold = get_model(model_name , best_params, input_shape)
        model_fold = train_model(model_fold, CrossVal_data_train, CrossVal_label_train)
        
        # Save model for this fold           
        fold_model_path = os.path.join(model_subfolder, f"model_fold{fold_idx + 1}")    
        if tf_dnn:
            fold_model_path = fold_model_path + ".h5"
            model_fold.save(fold_model_path)  # TensorFlow model
        else:
            fold_model_path = fold_model_path + ".pkl"
            with open(fold_model_path , 'wb') as f:
                pickle.dump(model_fold, f)


        metrics, ypred_i, yprob_i = evaluate_model(
            fold_model_path, CrossVal_data_test, CrossVal_label_test,
            area_hits_K=int(config.get('area_hits_K', 500)),
        )
        fold_metrics.append(metrics)

        # Per-fold prediction parquets (sorted + unsorted) under CVResults/.
        if df_meta_train is not None and train_filename is not None and column_name is not None:
            fold_meta = df_meta_train.iloc[test_idx].reset_index(drop=True).copy()
            fold_meta["y_pred"] = ypred_i
            fold_meta["y_prob"] = yprob_i

            base = f"{train_filename}_{model_name}_{column_name}_fold{fold_idx + 1}"
            out_path = os.path.join(cv_results_dir, f"{base}_predictions.parquet")
            sorted_path = os.path.join(cv_results_dir, f"{base}_predictions_sorted.parquet")
            fold_meta.to_parquet(out_path, index=False)
            fold_meta.sort_values("y_prob", ascending=False).to_parquet(sorted_path, index=False)

            sum_prob[test_idx] += yprob_i
            count_prob[test_idx] += 1

    # Mean-across-folds prediction parquet (one row per sample that appeared
    # in at least one test fold; y_prob is averaged over the folds it appeared in).
    if df_meta_train is not None and train_filename is not None and column_name is not None:
        appeared = count_prob > 0
        if appeared.any():
            mean_prob = np.zeros(n_samples, dtype=np.float64)
            mean_prob[appeared] = sum_prob[appeared] / count_prob[appeared]

            mean_df = df_meta_train.loc[appeared].reset_index(drop=True).copy()
            mean_df["y_prob"] = mean_prob[appeared]
            mean_df["y_pred"] = (mean_df["y_prob"] > 0.5).astype(int)
            mean_df["fold_count"] = count_prob[appeared]

            base = f"{train_filename}_{model_name}_{column_name}_mean"
            out_path = os.path.join(cv_results_dir, f"{base}_predictions.parquet")
            sorted_path = os.path.join(cv_results_dir, f"{base}_predictions_sorted.parquet")
            mean_df.to_parquet(out_path, index=False)
            mean_df.sort_values("y_prob", ascending=False).to_parquet(sorted_path, index=False)

    # Average metrics across folds
    avg_metrics = {metric: np.mean([fold[metric] for fold in fold_metrics]) for metric in fold_metrics[0]}
    return avg_metrics
#==============================================================================
#==============================================================================
def train_and_save_final_model(config, X_train_array, Y_train_array, model_name, model_subfolder, get_model, train_model, best_params):
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
    input_shape = X_train_array.shape[1]
    
    model = get_model(model_name , best_params, input_shape)
    model = train_model(model, X_train_array, Y_train_array)
    
    tf_models = config['tf_models']  
    tf_dnn = True if model_name in tf_models else False
    
    model_path = os.path.join(model_subfolder, "model")
    if tf_dnn:
        model_path = model_path + ".h5"
        model.save(model_path)  # TensorFlow model
    else:
        model_path = model_path + ".pkl"
        with open(model_path , 'wb') as f:
            pickle.dump(model, f)
 
#==============================================================================
#==============================================================================
def bayesian_hyperparameter_search(
    model_name, X_train, y_train,
    cv=5, n_iter=50, scoring='average_precision', random_state=42, n_jobs=1,
):
    """Scikit-optimize BayesSearchCV with sensible per-model search spaces.

    Defaults reflect imbalanced binary classification on fingerprint-style data:
      - StratifiedKFold(shuffle=True) so class ratios are preserved across folds.
      - scoring='average_precision' (AUC-PR), which is more discriminative than
        accuracy or AUC-ROC for imbalanced data. Override via the `scoring` arg.
      - n_iter=50 — TPE-style search needs ≥30 iters to outperform random.

    Ranges mirror those in hpo_utils.suggest_params (single source of truth for
    Optuna). Edit there if you want both tuners to share new ranges.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if model_name == 'rf':
        model = RandomForestClassifier(random_state=random_state, n_jobs=1)
        param_space = {
            'n_estimators': (100, 800),
            'max_depth': (5, 40),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': ['sqrt', 'log2'],
            'class_weight': [None, 'balanced'],
        }
    elif model_name == 'lr':
        model = LogisticRegression(
            solver='saga', max_iter=2000, random_state=random_state, n_jobs=1,
        )
        param_space = {
            'C': (1e-3, 100.0, 'log-uniform'),
            'penalty': ['l1', 'l2', 'elasticnet'],
            'l1_ratio': (0.0, 1.0, 'uniform'),  # ignored by sklearn unless penalty='elasticnet'
            'class_weight': [None, 'balanced'],
        }
    elif model_name == 'ridge':
        model = RidgeClassifier(random_state=random_state)
        param_space = {
            'alpha': (1e-3, 100.0, 'log-uniform'),
            'class_weight': [None, 'balanced'],
        }
    elif model_name == 'sgd':
        model = SGDClassifier(loss='log_loss', max_iter=2000, random_state=random_state)
        param_space = {
            'alpha': (1e-6, 1e-1, 'log-uniform'),
            'penalty': ['l1', 'l2', 'elasticnet'],
            'l1_ratio': (0.0, 1.0, 'uniform'),
            'class_weight': [None, 'balanced'],
        }
    elif model_name == 'perceptron':
        model = Perceptron(max_iter=2000, random_state=random_state)
        param_space = {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'alpha': (1e-6, 1e-1, 'log-uniform'),
            'class_weight': [None, 'balanced'],
        }
    elif model_name == 'svc':
        # probability=True so BayesSearchCV's AP/AUC scoring can call predict_proba.
        model = SVC(probability=True, random_state=random_state)
        param_space = {
            'C': (1e-2, 100.0, 'log-uniform'),
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'class_weight': [None, 'balanced'],
        }
    elif model_name == 'nb':
        model = GaussianNB()
        param_space = {
            'var_smoothing': (1e-12, 1e-6, 'log-uniform'),
        }
    elif model_name == 'dt':
        model = DecisionTreeClassifier(random_state=random_state)
        param_space = {
            'max_depth': (3, 30),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'criterion': ['gini', 'entropy'],
            'class_weight': [None, 'balanced'],
        }
    elif model_name == 'knn':
        model = KNeighborsClassifier(n_jobs=1)
        param_space = {
            'n_neighbors': (3, 50),
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        }
    elif model_name == 'gb':
        model = GradientBoostingClassifier(random_state=random_state)
        param_space = {
            'n_estimators': (100, 500),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'max_depth': (3, 10),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'subsample': (0.5, 1.0, 'uniform'),
        }
    elif model_name == 'ada':
        model = AdaBoostClassifier(random_state=random_state)
        param_space = {
            'n_estimators': (50, 500),
            'learning_rate': (0.01, 2.0, 'log-uniform'),
        }
    elif model_name == 'bag':
        model = BaggingClassifier(random_state=random_state, n_jobs=1)
        param_space = {
            'n_estimators': (10, 200),
            'max_samples': (0.3, 1.0, 'uniform'),
            'max_features': (0.3, 1.0, 'uniform'),
        }
    elif model_name == 'mlp':
        model = MLPClassifier(max_iter=500, early_stopping=True, random_state=random_state)
        param_space = {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            'activation': ['relu', 'tanh'],
            'alpha': (1e-6, 1e-1, 'log-uniform'),
            'learning_rate_init': (1e-4, 1e-1, 'log-uniform'),
        }
    elif model_name == 'lgbm':
        model = LGBMClassifier(random_state=random_state, verbosity=-1, n_jobs=1)
        param_space = {
            'n_estimators': (200, 5000),
            'learning_rate': (0.005, 0.2, 'log-uniform'),
            'num_leaves': (15, 255),
            'min_child_samples': (5, 300),
            'subsample': (0.5, 1.0, 'uniform'),
            'colsample_bytree': (0.3, 1.0, 'uniform'),
            'reg_alpha': (1e-4, 20.0, 'log-uniform'),
            'reg_lambda': (1e-4, 50.0, 'log-uniform'),
            'scale_pos_weight': (1.0, 20.0, 'uniform'),
        }
    elif model_name == 'catboost':
        model = CatBoostClassifier(silent=True, random_seed=random_state)
        param_space = {
            'iterations': (200, 2000),
            'learning_rate': (0.01, 0.3, 'log-uniform'),
            'depth': (4, 10),
            'l2_leaf_reg': (1.0, 30.0, 'log-uniform'),
        }
    elif model_name in ('tf_ff', 'tf_cnn1D', 'lgbmregressor'):
        raise NotImplementedError(
            f"Bayesian search via BayesSearchCV is not supported for {model_name!r}; "
            "use hyperparameters_tuning='optuna' for this model."
        )
    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")

    search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=n_iter,
        cv=skf,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=False,
    )
    search.fit(X_train, y_train)

    print(f"[Bayesian] {model_name}: best mean {scoring} = {search.best_score_:.4f}")
    return dict(search.best_params_)
#==============================================================================
#==============================================================================

