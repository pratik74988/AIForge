import pandas as pd


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
):
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y