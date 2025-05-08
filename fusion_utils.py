

import pandas as pd
import os
import warnings
import shutil
import os

import pickle
import numpy as np
import matplotlib.pyplot as plt
from eval_utils import calculate_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score
#==============================================================================
# Function to plot radar chart for model metrics
def plot_model_metrics_radar(df, metric_columns, save_path=""):
    """
    Plots a radar chart for model metrics.

    Parameters:
    - df (pd.DataFrame): DataFrame containing model evaluation metrics.
    - metric_columns (list): List of metric column names to plot.
    - save_path (str, optional): Path to save the radar chart image.

    Returns:
    - None (Displays radar chart and saves if save_path is provided)
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

    # Add labels
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_columns, fontsize=12)
    ax.set_title("Model Metrics", fontsize=16, fontweight='bold')

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Radar chart saved at {save_path}")

    plt.show()
#-----------------------------------------------------------------------------

def select_best_models(
    RunFolderName,
    trainfile_for_modelselection = '',
    evaluationfile_for_modelselection='',
    evaluation_column=None,
    num_top_models=5,
    Fusion = 'N'):
    
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

    # Sort by specified metric columns in descending order
    df_sorted = df.sort_values(by=evaluation_column, ascending=[False]*len(evaluation_column))

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
    

    # Write model paths to output txt file
    with open(output_txt_path, 'w') as f:
        for path in best_models['ModelPath']:
            f.write(f"{path}\n")
            
    # Save Best Models to BestModels Folder 
    best_models_folder = os.path.join(RunFolderName, "BestModels")
    os.makedirs(best_models_folder, exist_ok=True)
    for path in best_models['ModelPath']:
        model_name = os.path.basename(path)
        model_file = os.path.join(path, "model.pkl") 
        
        if os.path.exists(model_file):
            # Create the new name with the folder name prefix
            new_model_name = f"{model_name}_model.pkl"
            model_save_path = os.path.join(best_models_folder, new_model_name)
            
            # Copy the model file with the new name
            shutil.copy(model_file, model_save_path)
            print(f"Saved: {model_save_path}")
        else:
            print(f"Model file not found in {path}. Skipping.")


    print(f"Top {num_top_models} distinct model paths written to {output_txt_path}")
    
    if evaluation_column is not None:
        # Calculate mean values for the specified evaluation columns
        radar_data = best_models[evaluation_column].mean().to_dict()
        
        # Remove "Test_" prefix from the metric names for the radar plot
        radar_data_clean = {col.replace("Test_", ""): value for col, value in radar_data.items()}
        radar_df = pd.DataFrame([radar_data_clean])
    
        radar_chart_path = os.path.join(RunFolderName, "RadarChart_TopModels.png")
        plot_model_metrics_radar(radar_df, list(radar_data_clean.keys()), save_path=radar_chart_path)
#==============================================================================


#==============================================================================
def fusion_pipeline(
    RunFolderName,
    test_paths,
    column_names,
    label_column_test,
    nrows_test,
    load_data,
    fuse_columns,
    evaluate_model,
    feature_fusion_method):
    
    # Path to Best Models
    best_models_folder = os.path.join(RunFolderName, "BestModels")
    model_files = [
        os.path.join(best_models_folder, f) 
        for f in os.listdir(best_models_folder) 
        if f.endswith(".pkl")
    ]

    if not model_files:
        raise ValueError("No models found in BestModels folder.")

    fusion_results = []

    for test_path in test_paths:
        # Load test data
        X_test, Y_test = load_data(test_path, column_names, label_column_test, nrows_test)
        Y_test_array = np.stack(Y_test.iloc[:, 0])

        # === Feature Fusion (if specified) ===
        X_test, fused_column_name = fuse_columns(X_test, column_names, feature_fusion_method)

        # Store predictions for fusion
        y_preds = []
        y_probas = []

        # Run each model
        for model_file in model_files:
            model_name = os.path.basename(model_file).replace(".pkl", "")

            with open(model_file, 'rb') as f:
                model = pickle.load(f)

            # Make predictions
            column_name = model_name.split("_")[-2] # Extract the column name from the model name
            X_test_array = np.stack(X_test[column_name])


            y_pred = model.predict(X_test_array)
            y_proba = model.predict_proba(X_test_array)[:, 1]

            y_preds.append(y_pred)
            y_probas.append(y_proba)

        # Average predictions and probabilities across all models
        y_pred_fusion = np.mean(y_preds, axis=0).round().astype(int)
        y_proba_fusion = np.mean(y_probas, axis=0)
        
        # Calculate metrics on the fusion results
        metrics = calculate_metrics(X_test_array, Y_test_array, y_pred_fusion, y_proba_fusion)

        # Save results for this test file
        fusion_result = {
            "TestFile": os.path.basename(test_path)
        }
        fusion_result.update(metrics)
        fusion_results.append(fusion_result)

    # Save the fusion results in a CSV file
    fusion_results_df = pd.DataFrame(fusion_results)
    fusion_results_csv_path = os.path.join(RunFolderName, "FusionResults.csv")
    fusion_results_df.to_csv(fusion_results_csv_path, index=False)

    print(f"Fusion results saved to {fusion_results_csv_path}")
