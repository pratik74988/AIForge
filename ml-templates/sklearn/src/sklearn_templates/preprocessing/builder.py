from sklearn.compose import ColumnTransformer

from .numerical import build_numerical_pipeline
from .categorical import build_categorical_pipeline


def build_preprocessor(
    numerical_features,
    categorical_features,
    numerical_config=None,
    categorical_config=None,
):
    numerical_config = numerical_config or {}
    categorical_config = categorical_config or {}

    transformers = []

    if numerical_features:
        transformers.append(
            (
                "numerical",
                build_numerical_pipeline(**numerical_config),
                numerical_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                build_categorical_pipeline(**categorical_config),
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )