from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def build_categorical_pipeline(
    imputer="most_frequent",
    handle_unknown="ignore",
):
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy=imputer)
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown=handle_unknown,
                sparse_output=True
            )
        ),
    ])