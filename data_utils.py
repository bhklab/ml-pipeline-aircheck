import pandas as pd
import numpy as np
import os
import shutil


def stack_to_uint8(series, batch_size=10000):
    """Stack a pandas Series of 1-D numpy arrays into an (N_rows, N_bits) uint8 matrix,
    built in chunks so we never allocate the whole float32 source at once.

    Memory: peak ~ batch_size * N_bits * 4 bytes (the temporary per-chunk float32
    stack) + the destination (N_rows * N_bits bytes). For a 460k x 2048 file the
    destination is ~0.9 GiB instead of ~3.5 GiB at float32.

    Safe for binary fingerprints (0/1) and count fingerprints with values <= 255.
    For count fingerprints with values > 255 this will silently overflow — in that
    case use np.stack(...).astype(np.int16) or the wider-dtype helper instead.
    """
    n_rows = len(series)
    if n_rows == 0:
        return np.empty((0, 0), dtype=np.uint8)
    n_bits = len(series.iloc[0])
    out = np.empty((n_rows, n_bits), dtype=np.uint8)
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        out[start:end] = np.stack(series.iloc[start:end].values).astype(np.uint8)
    return out


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
    
    # Convert all relevant columns
    for col in column_names:
        if isinstance(X[col].iloc[0], str):
            X[col] = X[col].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32))
        else:
            X[col] = X[col].apply(np.array)
    
    return X, Y

#==============================================================================
def fuse_columns(X, column_names, feature_fusion_method=None):
    """
    Fuses multiple array-like columns from a DataFrame into one or more new columns.

    Parameters:
    - X: pd.DataFrame
    - column_names: list of str — base column names available for fusing
    - feature_fusion_method: str or list of str — any combination of 'None', 'All', 'Pairwise'.
        - 'None'     : keep the original columns
        - 'All'      : add one column that is the concatenation of all base columns
        - 'Pairwise' : add one column for every pair of base columns
        Multiple modes can be combined, e.g. ['None', 'All', 'Pairwise'].

    Returns:
    - X: updated DataFrame, possibly with new fused columns added
    - fused_column_names: list of column names to use downstream (originals and/or fused)
    """
    # Normalize to a list of lowercase method names
    if isinstance(feature_fusion_method, str):
        methods = [feature_fusion_method]
    else:
        methods = list(feature_fusion_method)
    methods = [m.lower() for m in methods]

    valid = {"none", "all", "pairwise"}
    invalid = [m for m in methods if m not in valid]
    if invalid:
        raise ValueError(f"Invalid feature_fusion_method values: {invalid}. Valid options: None, All, Pairwise.")

    fused_column_names = []

    if "none" in methods:
        fused_column_names.extend(column_names)

    if "all" in methods:
        # Fuse all columns into one. Use '-' separator so base descriptors can be recovered later.
        fused_column_name = "-".join(column_names)
        for j, column_name in enumerate(column_names):
            current_array = np.stack(X[column_name])
            if j == 0:
                fused_array = current_array
            else:
                fused_array = np.concatenate((fused_array, current_array), axis=1)

        X[fused_column_name] = list(fused_array)
        fused_column_names.append(fused_column_name)

    if "pairwise" in methods:
        # Fuse each column with all other columns one by one. Use '-' separator.
        for i, col1 in enumerate(column_names):
            for j, col2 in enumerate(column_names):
                if i < j:
                    if col1 != col2:  # Avoid self-fusion
                        fused_col_name = f"{col1}-{col2}"
                        fused_array = np.concatenate(
                            (np.stack(X[col1]), np.stack(X[col2])),
                            axis=1
                        )
                        X[fused_col_name] = list(fused_array)
                        fused_column_names.append(fused_col_name)

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
        #df = pd.read_parquet(train_path, columns=["LABEL", "ECFP4"])
    
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

    