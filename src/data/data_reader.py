import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """A utility class for loading and processing data from parquet files."""
    
    @staticmethod
    def read_parquet_file(file_path: str | Path, columns: list[str] | None = None, 
                         nrows: int | None = None) -> pd.DataFrame:
        """Read a parquet file with specified columns and row limits.

        Args:
            file_path: Path to the parquet file.
            columns: List of column names to read. If None, reads all columns.
            nrows: Number of rows to read. If None, reads all rows.

        Returns:
            DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If nrows is negative or invalid columns are specified.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found at: {file_path}")
        
        if nrows is not None and (not isinstance(nrows, int) or nrows < 0):
            raise ValueError("nrows must be a non-negative integer or None")

        try:
            df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')
        except Exception as e:
            raise ValueError(f"Failed to read parquet file: {str(e)}")

        if nrows is not None:
            df = df.head(nrows)
        
        return df

    @staticmethod
    def process_column_to_array(df: pd.DataFrame, column_name: str) -> np.ndarray:
        """Convert a DataFrame column to a numpy array.

        Args:
            df: Input DataFrame.
            column_name: Name of the column to process.

        Returns:
            Numpy array of the processed column.

        Raises:
            KeyError: If the specified column is not in the DataFrame.
            ValueError: If the column data cannot be converted to a numpy array.
        """
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame")

        try:
            if isinstance(df[column_name].iloc[0], str):
                return np.stack(df[column_name].apply(
                    lambda x: np.fromstring(x, sep=',', dtype=np.float32)))
            return np.stack(df[column_name].apply(np.array))
        except Exception as e:
            raise ValueError(f"Failed to convert column '{column_name}' to array: {str(e)}")

    @staticmethod
    def load_data_strings(path: str | Path, column_names: list[str], 
                         label_column: str) -> tuple[list[str], pd.Series]:
        """Load string data from a parquet file and convert to numerical format.

        Args:
            path: Path to the parquet file.
            column_names: List of feature column names.
            label_column: Name of the label column.

        Returns:
            Tuple of feature names and labels.

        Raises:
            ValueError: If input parameters are invalid or data processing fails.
        """
        if not column_names or not isinstance(column_names, list):
            raise ValueError("column_names must be a non-empty list")
        if not isinstance(label_column, str):
            raise ValueError("label_column must be a string")

        try:
            df = DataLoader.read_parquet_file(path, columns=column_names + [label_column])
            X_train = df[column_names].str.split(',', expand=True).apply(pd.to_numeric, errors='coerce')
            feature_names = [f'{col}_{i}' for col in column_names for i in range(X_train.shape[1] // len(column_names))]
            Y = df[label_column]
            return feature_names, Y
        except Exception as e:
            raise ValueError(f"Failed to load and process string data: {str(e)}")

    @staticmethod
    def load_data(path: str | Path, column_names: list[str], label_column: str, 
                  nrows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load and process data from a parquet file.

        Args:
            path: Path to the parquet file.
            column_names: List of feature column names.
            label_column: Name of the label column.
            nrows: Number of rows to read. If None, reads all rows.

        Returns:
            Tuple of processed features and labels.

        Raises:
            ValueError: If input parameters are invalid or data processing fails.
            FileNotFoundError: If the parquet file does not exist.
        """
        if not column_names or not isinstance(column_names, list):
            raise ValueError("column_names must be a non-empty list")
        if not isinstance(label_column, str):
            raise ValueError("label_column must be a string")
        
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found at: {file_path}")
        
        if nrows is not None:
            if not isinstance(nrows, int) or nrows <= 0:
                nrows = None
            else:
                total_rows = len(pd.read_parquet(file_path, columns=[column_names[0]], engine='pyarrow'))
                if nrows > total_rows:
                    print(f"Warning: Specified nrows ({nrows}) exceeds total rows ({total_rows}). Using all rows.")
                    nrows = None

        try:
            X = DataLoader.read_parquet_file(file_path, columns=column_names, nrows=nrows)
            Y = DataLoader.read_parquet_file(file_path, columns=[label_column], nrows=nrows)

            for col in column_names:
                # X[col] = DataLoader.process_column_to_array(X, col)
                if isinstance(X[col].iloc[0], str):
                    X[col] = X[col].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32))
                else:
                    X[col] = X[col].apply(np.array)
            return X, Y
        except Exception as e:
            raise ValueError(f"Failed to load and process data: {str(e)}")