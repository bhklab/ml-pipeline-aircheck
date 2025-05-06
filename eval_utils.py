import pandas as pd
import pickle
import os
import numpy as np



# top 100, 2000 metrics??
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
    classification_report
)
#==============================================================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_test, y_proba),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_test, y_pred)
    }

    # Optionally print classification report
    # print("Classification Report:\n", classification_report(y_test, y_pred))

    return metrics
#==============================================================================


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
    FeatureFusionAll):
    
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
        if FeatureFusionAll.lower() == 'y':
            X_test, fused_column_name = fuse_columns(X_test, column_names)

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

# NetShape results and save the figure
# Jmaes Metrics
# Luca Metrics