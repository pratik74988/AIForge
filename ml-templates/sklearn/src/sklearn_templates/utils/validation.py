import pandas as pd


def validate_dataset(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names.")

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least two columns.")

    return True


def validate_target(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    if df[target_column].isna().all():
        raise ValueError("Target column contains only missing values.")

    return True