from .linear_regression import LinearRegressionTemplate
from .ridge import RidgeTemplate
from .lasso import LassoTemplate
from .random_forest import RandomForestRegressionTemplate


REGRESSION_REGISTRY = {
    "linear_regression": LinearRegressionTemplate,
    "ridge": RidgeTemplate,
    "lasso": LassoTemplate,
    "random_forest": RandomForestRegressionTemplate,
}


def get_regression_template(model_name, preprocessor, model_config):
    if model_name not in REGRESSION_REGISTRY:
        raise ValueError(
            f"Unknown regression model '{model_name}'. "
            f"Available: {list(REGRESSION_REGISTRY.keys())}"
        )

    template_cls = REGRESSION_REGISTRY[model_name]

    return template_cls(
        preprocessor=preprocessor,
        model_config=model_config,
    )