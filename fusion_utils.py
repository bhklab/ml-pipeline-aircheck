

import pandas as pd
import os
import warnings
#==============================================================================
def select_best_models(
    RunFolderName,
    evaluation_file_name='',
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
    if evaluation_file_name:
        df = df[df['TestFile'] == evaluation_file_name]

    # Sort by specified metric columns in descending order
    df_sorted = df.sort_values(by=evaluation_column, ascending=[False]*len(evaluation_column))

    # Drop duplicates based on ModelPath to get k distinct models
    df_sorted_unique = df_sorted.drop_duplicates(subset=['ModelPath'])

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

    print(f"Top {num_top_models} distinct model paths written to {output_txt_path}")
#==============================================================================
