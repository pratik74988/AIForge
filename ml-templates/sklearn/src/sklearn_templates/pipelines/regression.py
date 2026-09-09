from ..regression import get_regression_template


def build_regression_pipeline(
    model_name,
    preprocessor,
    model_config=None,
):
    model_config = model_config or {}

    template = get_regression_template(
        model_name=model_name,
        preprocessor=preprocessor,
        model_config=model_config,
    )

    return template.build_pipeline()