from importlib import import_module


TEMPLATE_REGISTRY = {
    "sklearn": {
        "classification": {
            "logistic_regression": (
                "sklearn_templates.classification",
                "LogisticRegressionTemplate",
            ),
            "random_forest": (
                "sklearn_templates.classification",
                "RandomForestTemplate",
            ),
            "svm": (
                "sklearn_templates.classification",
                "SVMTemplate",
            ),
        },
        "regression": {
            "linear_regression": (
                "sklearn_templates.regression",
                "LinearRegressionTemplate",
            ),
            "ridge": (
                "sklearn_templates.regression",
                "RidgeTemplate",
            ),
            "lasso": (
                "sklearn_templates.regression",
                "LassoTemplate",
            ),
            "random_forest": (
                "sklearn_templates.regression",
                "RandomForestRegressionTemplate",
            ),
        },
        "clustering": {
            "kmeans": (
                "sklearn_templates.clustering",
                "KMeansTemplate",
            ),
            "dbscan": (
                "sklearn_templates.clustering",
                "DBSCANTemplate",
            ),
        },
    },
}


def get_template_class(
    framework: str,
    task: str,
    model: str,
):
    try:
        module_name, class_name = TEMPLATE_REGISTRY[
            framework
        ][task][model]
    except KeyError:
        raise ValueError(
            f"Unknown template: "
            f"{framework}/{task}/{model}"
        )

    module = import_module(module_name)
    return getattr(module, class_name)