# __init__.py
from .logistic_regression import LogisticRegressionTemplate
from .random_forest import RandomForestTemplate
from .svm import SVMTemplate

CLASSIFICATION_REGISTRY = {
    "logistic_regression": LogisticRegressionTemplate,
    "random_forest": RandomForestTemplate,
    "svm": SVMTemplate,
}


def get_classification_template(model_name: str, preprocessor, model_config: dict):
    if model_name not in CLASSIFICATION_REGISTRY:
        raise ValueError(
            f"Unknown classification model '{model_name}'. "
            f"Available: {list(CLASSIFICATION_REGISTRY.keys())}"
        )
    template_cls = CLASSIFICATION_REGISTRY[model_name]
    return template_cls(preprocessor=preprocessor, model_config=model_config)