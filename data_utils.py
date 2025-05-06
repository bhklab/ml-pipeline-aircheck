import pandas as pd
import numpy as np
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
def load_data(path, column_names, label_column, nrows):
    X = read_parquet_file(path, columns=column_names, nrows=None)
    Y = read_parquet_file(path, columns=label_column, nrows=None)
    return X,Y
#==============================================================================
def fuse_columns(X, column_names):
    """
    Fuses multiple array-like columns from a DataFrame into a single new column.

    Parameters:
    - X: pd.DataFrame
    - column_names: list of str — column names to be fused

    Returns:
    - X: updated DataFrame with new fused column
    - fused_column_name: name of the new fused column
    """
    fused_column_name = "_".join(column_names)
    
    for j, column_name in enumerate(column_names):  
        current_array = np.stack(X[column_name])
        if j == 0:
            fused_array = current_array
        else:
            fused_array = np.concatenate((fused_array, current_array), axis=1)
    
    X[fused_column_name] = list(fused_array)
    return X, fused_column_name
#==============================================================================
# Combining data?

#==============================================================================
# reading other file formats?



#==============================================================================
# Changing negative ratio, check if that ratio is possible
'''
# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created successfully.")
else:
    print(f"Folder '{folder_path}' already exists.")


# Define the ratio of negative samples (N times more than positives)
N = 1  # Adjust this value as needed

# Select all rows where DELLabel == 1 (positive samples)
positive_samples = df_train[df_train["DELLabel"] == 1]

# Select N times more rows where DELLabel == 0 (negative samples)
negative_samples = df_train[df_train["DELLabel"] == 0].sample(n=len(positive_samples) * N, random_state=42)

# Combine both subsets to create a balanced dataset with the desired ratio
df_new = pd.concat([positive_samples, negative_samples])

# Define the output path with a descriptive filename
TrainData_path = '/content/drive/My Drive/AircheckWorkshopData/'
output_folder = os.path.dirname(TrainData_path)  # Same folder as the input file
output_filename = f"Trainset_Aircheck_custom.parquet"
output_path = os.path.join(output_folder, output_filename)

# Save the balanced dataset as a new Parquet file
df_new.to_parquet(output_path, index=False)
print(f"Balanced dataset saved as '{output_filename}' in '{output_folder}'")

df_train=df_new'''
#==============================================================================