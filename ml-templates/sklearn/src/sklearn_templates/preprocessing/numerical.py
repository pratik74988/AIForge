from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def build_numerical_pipeline(
    imputer="median",
    scaler="standard",
):
    steps = [
        ("imputer", SimpleImputer(strategy=imputer))
    ]

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "none": "passthrough",
    }

    steps.append(("scaler", scalers[scaler]))

    return Pipeline(steps)