from typing import Any

from sklearn.compose import ColumnTransformer

from sklearn_templates.preprocessing.builder import build_preprocessor


SUPPORTED_IMPUTATION_STRATEGIES = {
    "mean",
    "median",
    "most_frequent",
    "constant",
}

SUPPORTED_SCALING_METHODS = {
    "standard",
    "minmax",
    "robust",
    "none",
}


def _resolve_columns(
    columns: str | list[str] | None,
    numerical_features: list[str],
    categorical_features: list[str],
) -> tuple[set[str], set[str]]:
    """
    Resolve a TrainingSpec column selector into numerical and categorical
    feature sets.
    """

    numerical = set(numerical_features)
    categorical = set(categorical_features)

    if columns is None:
        raise ValueError(
            "Preprocessing columns must be specified."
        )

    if columns == "numeric":
        return numerical, set()

    if columns == "categorical":
        return set(), categorical

    if columns == "all":
        return numerical, categorical

    if isinstance(columns, list):
        requested = set(columns)

        unknown = requested - numerical - categorical

        if unknown:
            raise ValueError(
                f"Unknown preprocessing columns: {sorted(unknown)}"
            )

        return (
            requested & numerical,
            requested & categorical,
        )

    raise ValueError(
        f"Unsupported column selector: {columns}"
    )


def build_training_preprocessor(
    preprocessing_steps: list[Any],
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build an sklearn preprocessor from TrainingSpec preprocessing steps."""

    numerical_config: dict[str, Any] = {}
    categorical_config: dict[str, Any] = {}

    for step in preprocessing_steps:
        operation = step.operation
        columns = step.columns
        parameters = dict(step.parameters)

        selected_numerical, selected_categorical = _resolve_columns(
            columns=columns,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        if operation == "impute":
            strategy = parameters.get("strategy")

            if strategy not in SUPPORTED_IMPUTATION_STRATEGIES:
                raise ValueError(
                    f"Unsupported imputation strategy: {strategy}"
                )

            if selected_numerical:
                if selected_numerical != set(numerical_features):
                    raise ValueError(
                        "Partial numerical feature selection is not "
                        "supported for imputation."
                    )

                numerical_config["imputer"] = strategy

            if selected_categorical:
                if selected_categorical != set(categorical_features):
                    raise ValueError(
                        "Partial categorical feature selection is not "
                        "supported for imputation."
                    )

                categorical_config["imputer"] = strategy

            if not selected_numerical and not selected_categorical:
                raise ValueError(
                    "Imputation preprocessing must target at least "
                    "one feature."
                )

        elif operation == "encode":
            method = parameters.get("method")

            if method != "one_hot":
                raise ValueError(
                    f"Unsupported categorical encoding method: {method}"
                )

            if selected_numerical:
                raise ValueError(
                    "Encoding preprocessing cannot target numerical columns."
                )

            if not selected_categorical:
                raise ValueError(
                    "Encoding preprocessing must target categorical columns."
                )

            if selected_categorical != set(categorical_features):
                raise ValueError(
                    "Partial categorical feature selection is not "
                    "supported for encoding."
                )

        elif operation == "scale":
            method = parameters.get("method")

            if method not in SUPPORTED_SCALING_METHODS:
                raise ValueError(
                    f"Unsupported numerical scaling method: {method}"
                )

            if selected_categorical:
                raise ValueError(
                    "Scaling preprocessing cannot target categorical columns."
                )

            if not selected_numerical:
                raise ValueError(
                    "Scaling preprocessing must target numerical columns."
                )

            if selected_numerical != set(numerical_features):
                raise ValueError(
                    "Partial numerical feature selection is not "
                    "supported for scaling."
                )

            numerical_config["scaler"] = method

        else:
            raise ValueError(
                f"Unsupported preprocessing operation: {operation}"
            )

    return build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        numerical_config=numerical_config,
        categorical_config=categorical_config,
    )