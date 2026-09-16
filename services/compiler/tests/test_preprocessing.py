import pandas as pd
import pytest

from compiler.preprocessing import build_training_preprocessor
from planner.models.training_spec import PreprocessingStep


def get_transformer(preprocessor, name):
    for transformer_name, transformer, _ in preprocessor.transformers:
        if transformer_name == name:
            return transformer
    raise AssertionError(f"Transformer {name!r} was not found.")


def get_columns(preprocessor, name):
    for transformer_name, _, columns in preprocessor.transformers:
        if transformer_name == name:
            return columns
    raise AssertionError(f"Transformer {name!r} was not found.")


def test_build_training_preprocessor():
    steps = [
        PreprocessingStep(
            name="numeric_imputation",
            operation="impute",
            columns="numeric",
            parameters={"strategy": "median"},
        ),
        PreprocessingStep(
            name="categorical_encoding",
            operation="encode",
            columns="categorical",
            parameters={"method": "one_hot"},
        ),
        PreprocessingStep(
            name="numerical_scaling",
            operation="scale",
            columns="numeric",
            parameters={"method": "standard"},
        ),
    ]

    preprocessor = build_training_preprocessor(
        preprocessing_steps=steps,
        numerical_features=["age", "income"],
        categorical_features=["city", "occupation"],
    )

    assert "numerical" in [
        name for name, _, _ in preprocessor.transformers
    ]
    assert "categorical" in [
        name for name, _, _ in preprocessor.transformers
    ]

    assert get_transformer(preprocessor, "numerical") is not None
    assert get_transformer(preprocessor, "categorical") is not None


def test_unsupported_operation_is_rejected():
    steps = [
        PreprocessingStep(
            name="unknown",
            operation="normalize",
            columns="numeric",
            parameters={},
        )
    ]

    with pytest.raises(
        ValueError,
        match="Unsupported preprocessing operation",
    ):
        build_training_preprocessor(
            preprocessing_steps=steps,
            numerical_features=["age"],
            categorical_features=[],
        )


def test_imputation_and_scaling_configuration_is_applied():
    steps = [
        PreprocessingStep(
            name="numeric_imputation",
            operation="impute",
            columns="numeric",
            parameters={"strategy": "median"},
        ),
        PreprocessingStep(
            name="numerical_scaling",
            operation="scale",
            columns="numeric",
            parameters={"method": "standard"},
        ),
    ]

    preprocessor = build_training_preprocessor(
        preprocessing_steps=steps,
        numerical_features=["age", "income"],
        categorical_features=[],
    )

    numerical = get_transformer(preprocessor, "numerical")

    assert numerical.named_steps["imputer"].strategy == "median"
    assert (
        numerical.named_steps["scaler"].__class__.__name__
        == "StandardScaler"
    )

    X = pd.DataFrame(
        {
            "age": [20, None, 40],
            "income": [20_000, 30_000, 40_000],
        }
    )

    transformed = numerical.fit_transform(X)

    assert transformed.shape == (3, 2)
    assert not pd.isna(transformed).any()


def test_one_hot_encoding_is_applied():
    steps = [
        PreprocessingStep(
            name="categorical_encoding",
            operation="encode",
            columns="categorical",
            parameters={"method": "one_hot"},
        ),
    ]

    preprocessor = build_training_preprocessor(
        preprocessing_steps=steps,
        numerical_features=[],
        categorical_features=["city"],
    )

    categorical = get_transformer(preprocessor, "categorical")

    assert (
        categorical.named_steps["encoder"].__class__.__name__
        == "OneHotEncoder"
    )
    assert categorical.named_steps["encoder"].handle_unknown == "ignore"

    X = pd.DataFrame(
        {
            "city": ["Pune", "Mumbai", "Pune"],
        }
    )

    transformed = categorical.fit_transform(X)

    assert transformed.shape == (3, 2)


def test_unsupported_scaling_method_is_rejected():
    steps = [
        PreprocessingStep(
            name="numerical_scaling",
            operation="scale",
            columns="numeric",
            parameters={"method": "unknown_scaler"},
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Unsupported numerical scaling method",
    ):
        build_training_preprocessor(
            preprocessing_steps=steps,
            numerical_features=["age"],
            categorical_features=[],
        )


def test_unsupported_encoding_method_is_rejected():
    steps = [
        PreprocessingStep(
            name="categorical_encoding",
            operation="encode",
            columns="categorical",
            parameters={"method": "ordinal"},
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Unsupported categorical encoding method",
    ):
        build_training_preprocessor(
            preprocessing_steps=steps,
            numerical_features=[],
            categorical_features=["city"],
        )


def test_missing_imputation_strategy_is_rejected():
    steps = [
        PreprocessingStep(
            name="numeric_imputation",
            operation="impute",
            columns="numeric",
            parameters={},
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Unsupported imputation strategy",
    ):
        build_training_preprocessor(
            preprocessing_steps=steps,
            numerical_features=["age"],
            categorical_features=[],
        )


def test_missing_scaling_method_is_rejected():
    steps = [
        PreprocessingStep(
            name="numerical_scaling",
            operation="scale",
            columns="numeric",
            parameters={},
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Unsupported numerical scaling method",
    ):
        build_training_preprocessor(
            preprocessing_steps=steps,
            numerical_features=["age"],
            categorical_features=[],
        )


def test_numeric_column_group_is_resolved():
    steps = [
        PreprocessingStep(
            name="numeric_scaling",
            operation="scale",
            columns="numeric",
            parameters={"method": "standard"},
        )
    ]

    preprocessor = build_training_preprocessor(
        preprocessing_steps=steps,
        numerical_features=["age", "income"],
        categorical_features=["city"],
    )

    numerical = get_transformer(preprocessor, "numerical")

    assert numerical is not None


def test_all_columns_can_be_used_for_imputation():
    steps = [
        PreprocessingStep(
            name="imputation",
            operation="impute",
            columns="all",
            parameters={"strategy": "most_frequent"},
        )
    ]

    preprocessor = build_training_preprocessor(
        preprocessing_steps=steps,
        numerical_features=["age"],
        categorical_features=["city"],
    )

    assert len(preprocessor.transformers) == 2

    numerical = get_transformer(preprocessor, "numerical")
    categorical = get_transformer(preprocessor, "categorical")

    assert numerical.named_steps["imputer"].strategy == "most_frequent"
    assert categorical.named_steps["imputer"].strategy == "most_frequent"


def test_explicit_columns_are_resolved():
    steps = [
        PreprocessingStep(
            name="numeric_scaling",
            operation="scale",
            columns=["age", "income"],
            parameters={"method": "standard"},
        )
    ]

    preprocessor = build_training_preprocessor(
        preprocessing_steps=steps,
        numerical_features=["age", "income"],
        categorical_features=["city"],
    )

    numerical_columns = get_columns(preprocessor, "numerical")

    assert numerical_columns == ["age", "income"]


def test_unknown_explicit_column_is_rejected():
    steps = [
        PreprocessingStep(
            name="numeric_scaling",
            operation="scale",
            columns=["does_not_exist"],
            parameters={"method": "standard"},
        )
    ]

    with pytest.raises(
        ValueError,
        match="Unknown preprocessing columns",
    ):
        build_training_preprocessor(
            preprocessing_steps=steps,
            numerical_features=["age"],
            categorical_features=["city"],
        )

