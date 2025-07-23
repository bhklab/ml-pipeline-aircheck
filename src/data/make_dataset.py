import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

class DataProcessor:
    """A utility class for processing and transforming DataFrame data."""

    @staticmethod
    def convert_columns_to_array(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
        """Convert specified DataFrame columns to numpy arrays in-place.

        Args:
            df: Input DataFrame.
            column_names: List of column names to convert.

        Returns:
            DataFrame with converted columns.

        Raises:
            KeyError: If any specified column is not in the DataFrame.
            ValueError: If column_names is empty or data conversion fails.
        """
        if not column_names or not isinstance(column_names, list):
            raise ValueError("column_names must be a non-empty list")

        missing_cols = [col for col in column_names if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

        try:
            for col in column_names:
                if isinstance(df[col].iloc[0], str):
                    df[col] = df[col].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32))
                else:
                    df[col] = df[col].apply(np.array)
            return df
        except Exception as e:
            raise ValueError(f"Failed to convert columns to arrays: {str(e)}")

    @staticmethod
    def process_data(X: pd.DataFrame, column_name: str) -> np.ndarray:
        """Process a single DataFrame column to a stacked numpy array.

        Args:
            X: Input DataFrame.
            column_name: Name of the column to process.

        Returns:
            Stacked numpy array of the processed column.

        Raises:
            KeyError: If the specified column is not in the DataFrame.
            ValueError: If the column data cannot be converted to a numpy array.
        """
        if column_name not in X.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame")

        try:
            return np.stack(X[column_name].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32)))
        except Exception as e:
            raise ValueError(f"Failed to process column '{column_name}' to array: {str(e)}")

    @staticmethod
    def process_column_to_array(df: pd.DataFrame, column_name: str) -> np.ndarray:
        """Convert a DataFrame column to a stacked numpy array, handling string or array-like data.

        Args:
            df: Input DataFrame.
            column_name: Name of the column to process.

        Returns:
            Stacked numpy array of the processed column.

        Raises:
            KeyError: If the specified column is not in the DataFrame.
            ValueError: If the column data cannot be converted to a numpy array.
        """
        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in DataFrame")

        try:
            if isinstance(df[column_name].iloc[0], str):
                return np.stack(df[column_name].apply(lambda x: np.fromstring(x, sep=',', dtype=np.float32)))
            return np.stack(df[column_name].apply(np.array))
        except Exception as e:
            raise ValueError(f"Failed to convert column '{column_name}' to array: {str(e)}")

    @staticmethod
    def fuse_columns(X: pd.DataFrame, column_names: list[str], feature_fusion_method: str | None = None) -> tuple[pd.DataFrame, list[str]]:
        """Fuse multiple array-like columns into a single new column.

        Args:
            X: Input DataFrame.
            column_names: List of column names to fuse.
            feature_fusion_method: Fusion method ('all', 'pairwise', or None).

        Returns:
            Tuple of updated DataFrame with fused column(s) and list of new fused column names.

        Raises:
            ValueError: If column_names is empty, fusion method is invalid, or data processing fails.
            KeyError: If any specified column is not in the DataFrame.
        """
        if not column_names or not isinstance(column_names, list):
            raise ValueError("column_names must be a non-empty list")

        missing_cols = [col for col in column_names if col not in X.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

        feature_fusion_method = feature_fusion_method.lower() if feature_fusion_method else "none"
        if feature_fusion_method not in ["all", "pairwise", "none"]:
            raise ValueError("Invalid feature_fusion_method. Choose 'all', 'pairwise', or None.")

        if feature_fusion_method == "none":
            return X, column_names

        fused_column_names = []
        try:
            if feature_fusion_method == "all":
                fused_column_name = "_".join(column_names)
                for j, column_name in enumerate(column_names):
                    current_array = np.stack(X[column_name])
                    if j == 0:
                        fused_array = current_array
                    else:
                        fused_array = np.concatenate((fused_array, current_array), axis=1)
                X[fused_column_name] = list(fused_array)
                fused_column_names.append(fused_column_name)

            elif feature_fusion_method == "pairwise":
                for i, col1 in enumerate(column_names):
                    for j, col2 in enumerate(column_names):
                        if i < j:
                            if col1 != col2:
                                fused_col_name = f"{col1}_{col2}"
                                fused_array = np.concatenate((np.stack(X[col1]), np.stack(X[col2])), axis=1)
                                X[fused_col_name] = list(fused_array)
                                fused_column_names.append(fused_col_name)

            return X, fused_column_names
        except Exception as e:
            raise ValueError(f"Failed to fuse columns: {str(e)}")

    @staticmethod
    def create_balanced_datasets(
        config: dict | None = None
    ) -> list[str]:
        """Create balanced datasets by adjusting the positive-to-negative sample ratio.

        Args:
            train_paths: List of paths to training parquet files.
            label_column: Name of the label column.
            balance_ratios: List of negative-to-positive ratios for balancing.
            balance_flag: If True, create balanced datasets; if False, return original paths.
            config: Optional configuration dictionary with keys 'train_data', 'label_column_train',
                   'balance_ratios', 'balance_flag'. Overrides other parameters if provided.

        Returns:
            List of paths to original and new balanced datasets.

        Raises:
            ValueError: If inputs are invalid or data processing fails.
            FileNotFoundError: If any input file does not exist.
        """
        if config:
            train_paths = config.get('train_data')
            label_column = config.get('label_column_train')
            balance_ratios = config.get('balance_ratios')
            balance_flag = config.get('balance_flag')

        if not train_paths or not isinstance(train_paths, list):
            raise ValueError("train_paths must be a non-empty list")
        if not isinstance(label_column, str):
            raise ValueError("label_column must be a string")
        if not isinstance(balance_ratios, list) or not balance_ratios:
            raise ValueError("balance_ratios must be a non-empty list")
        if any(not isinstance(r, (int, float)) or r <= 0 for r in balance_ratios):
            raise ValueError("balance_ratios must contain positive numbers")

        if not balance_flag:
            return [str(path) for path in train_paths]

        all_paths = [str(path) for path in train_paths]
        if not all_paths:
            raise ValueError("No valid training paths provided")

        main_dir = os.path.dirname(all_paths[0])
        balanced_dir = os.path.join(main_dir, "BalancedTrain")
        if os.path.exists(balanced_dir):
            shutil.rmtree(balanced_dir)
        os.makedirs(balanced_dir, exist_ok=True)

        try:
            for train_path in train_paths:
                train_path = Path(train_path)
                if not train_path.exists():
                    raise FileNotFoundError(f"Parquet file not found at: {train_path}")

                df = pd.read_parquet(train_path)
                if label_column not in df.columns:
                    raise KeyError(f"Label column '{label_column}' not found in DataFrame")

                positive_samples = df[df[label_column] == 1]
                negative_samples = df[df[label_column] == 0]

                for ratio in balance_ratios:
                    if ratio == 1:
                        min_count = min(len(positive_samples), len(negative_samples))
                        balanced_pos = positive_samples.sample(min_count, replace=False)
                        balanced_neg = negative_samples.sample(min_count, replace=False)
                    elif ratio > 1:
                        balanced_pos = positive_samples
                        max_neg_count = min(len(negative_samples), int(len(positive_samples) * ratio))
                        balanced_neg = negative_samples.sample(max_neg_count, replace=False)
                    else:
                        raise ValueError(f"Invalid balance ratio: {ratio}")

                    balanced_df = pd.concat([balanced_pos, balanced_neg]).sample(frac=1).reset_index(drop=True)
                    ratio_suffix = f"balanced_{ratio}x"
                    new_path = os.path.join(
                        balanced_dir,
                        f"{train_path.stem}_{ratio_suffix}.parquet"
                    )
                    balanced_df.to_parquet(new_path, index=False)
                    all_paths.append(new_path)

            return all_paths
        except Exception as e:
            raise ValueError(f"Failed to create balanced datasets: {str(e)}")