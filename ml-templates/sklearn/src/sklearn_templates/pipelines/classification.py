from ..classification import get_classification_template


def build_classification_pipeline(
    model_name,
    preprocessor,
    model_config=None,
):
    model_config = model_config or {}

    template = get_classification_template(
        model_name=model_name,
        preprocessor=preprocessor,
        model_config=model_config,
    )

    return template.build_pipeline()