import pandas as pd
import numpy as np
import os
import shutil



def read_parquet_file(file_path, columns=None, nrows=None):
    df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
    if nrows is not None:
        df = df.head(nrows)
    return df

def process_column_to_array(df, column_name):
    if isinstance(df[column_name].iloc[0], str):
        # Column is string, needs conversion
        return np.stack(df[column_name].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32)))
    else:
        # Column is already array-like
        return np.stack(df[column_name])


def convert_columns_to_array(df, column_names):
    for col in column_names:
        if isinstance(df[col].iloc[0], str):
            # Convert comma-separated strings to NumPy arrays in-place
            df[col] = df[col].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32))
        else:
            # Already array-like (e.g., list), convert to np.array for consistency
            df[col] = df[col].apply(np.array)
    return df

#==============================================================================            
def process_data(X, column_name):
    return np.stack(X[column_name].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32)))
#==============================================================================                
def load_data_strings(path, column_names, label_column):
    df = pd.read_parquet(path)
    X_train = df[column_names].str.split(',', expand=True).apply(pd.to_numeric, errors='coerce')
    X = [f'{column_names}_{i}' for i in range(X_train.shape[1])]
    Y = df[label_column]   
    return X, Y
#==============================================================================
def read_parquet_file(file_path, columns=None, nrows=None):
    df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
    if nrows is not None:
        df = df.head(nrows)
    return df
#==============================================================================
def process_column_to_array(df, column_name):
    if isinstance(df[column_name].iloc[0], str):
        # Column is string, needs conversion
        return np.stack(df[column_name].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32)))
    else:
        # Column is already array-like
        return np.stack(df[column_name])
#==============================================================================
'''def load_data(path, column_names, label_column, nrows):
    X = read_parquet_file(path, columns=column_names, nrows=None)
    Y = read_parquet_file(path, columns=label_column, nrows=None)
    return X,Y'''

def load_data(path, column_names, label_column, nrows):
    # Validate nrows
    if not isinstance(nrows, int) or nrows <= 0:
        nrows = None  # Read all rows if nrows is invalid

    # Read the full data to check the maximum row count
    total_rows = len(pd.read_parquet(path, columns=[column_names[0]]))  # Get row count of the first column
    
    if nrows is not None and nrows > total_rows:
        print(f"Warning: Specified nrows ({nrows}) exceeds total rows ({total_rows}). Using all rows instead.")
        nrows = None  # Use all rows if nrows exceeds available rows

    # Load data with the corrected nrows
    X = read_parquet_file(path, columns=column_names, nrows=nrows)
    Y = read_parquet_file(path, columns=label_column, nrows=nrows)
    
    return X, Y

#==============================================================================
def fuse_columns(X, column_names, feature_fusion_method=None):
    """
    Fuses multiple array-like columns from a DataFrame into a single new column.

    Parameters:
    - X: pd.DataFrame
    - column_names: list of str — column names to be fused
    - feature_fusion_method: str or None — "All" (fuse all) or "Pairwise" (each column with all others)

    Returns:
    - X: updated DataFrame with new fused column(s)
    - fused_column_names: list of new fused column names
    """
    # If method is None, do nothing and return the original DataFrame
    if feature_fusion_method.lower() == "none":
        return X, column_names

    # Convert method to lowercase to be case-insensitive
    feature_fusion_method = feature_fusion_method.lower()
    fused_column_names = []

    if feature_fusion_method.lower() == "all":
        # Fuse all columns into one
        fused_column_name = "_".join(column_names)
        for j, column_name in enumerate(column_names):  
            current_array = np.stack(X[column_name])
            if j == 0:
                fused_array = current_array
            else:
                fused_array = np.concatenate((fused_array, current_array), axis=1)

        X[fused_column_name] = list(fused_array)
        fused_column_names.append(fused_column_name)

    elif feature_fusion_method.lower() == "pairwise":
        # Fuse each column with all other columns one by one
        for i, col1 in enumerate(column_names):
            for j, col2 in enumerate(column_names):
                if col1 != col2:  # Avoid self-fusion
                    fused_col_name = f"{col1}_{col2}"
                    fused_array = np.concatenate(
                        (np.stack(X[col1]), np.stack(X[col2])),
                        axis=1
                    )
                    X[fused_col_name] = list(fused_array)
                    fused_column_names.append(fused_col_name)
    
    else:
        raise ValueError("Invalid feature_fusion_method. Choose 'All', 'Pairwise', or None.")

    return X, fused_column_names


#==============================================================================
# Changing negative ratio, check if that ratio is possible
'''def create_balanced_datasets(
    train_paths,
    label_column,
    balance_ratios=[1, 2, 4],
    balance_flag=True):'''
def create_balanced_datasets(config):
    
    train_paths = config['train_data']
    label_column = config['label_column_train']
    balance_ratios = config['balance_ratios']
    balance_flag = config['balance_flag']
    
    if not balance_flag:
        return train_paths  # Return the original list if not balancing

    # List to store all paths (including new balanced datasets)
    all_paths = list(train_paths)  # Start with the original paths
    
    # Determine the main directory and create the BalancedTrain folder
    main_dir = os.path.dirname(train_paths[0])
    balanced_dir = os.path.join(main_dir, "BalancedTrain")
    # Remove the BalancedTrain folder if it already exists
    if os.path.exists(balanced_dir):
        shutil.rmtree(balanced_dir)  # Delete the entire directory and its contents
    os.makedirs(balanced_dir, exist_ok=True)
        

    for train_path in train_paths:
        # Load the data
        df = pd.read_parquet(train_path)
    
        # Separate positive and negative samples
        positive_samples = df[df[label_column[0]] == 1]
        negative_samples = df[df[label_column[0]] == 0]  # Fixed condition for negatives
    
        # Generate balanced datasets for each ratio
        for ratio in balance_ratios:
            if ratio == 1:
                # Equal number of positive and negative samples
                min_count = min(len(positive_samples), len(negative_samples))
                balanced_pos = positive_samples.sample(min_count, replace=False)
                balanced_neg = negative_samples.sample(min_count, replace=False)
    
            elif ratio > 1:
                # More negatives than positives based on the ratio
                balanced_pos = positive_samples  # Keep all positives
                max_neg_count = min(len(negative_samples), len(positive_samples) * ratio)
                balanced_neg = negative_samples.sample(max_neg_count, replace=False)
    
            # Concatenate and shuffle
            balanced_df = pd.concat([balanced_pos, balanced_neg]).sample(frac=1).reset_index(drop=True)
    
            # Save the new dataset in the BalancedTrain folder
            ratio_suffix = f"balanced_{ratio}x"
            new_path = os.path.join(
                balanced_dir,
                f"{os.path.basename(train_path).split('.')[0]}_{ratio_suffix}.parquet"
            )
            balanced_df.to_parquet(new_path, index=False)
            all_paths.append(new_path)


    return all_paths
#==============================================================================

    