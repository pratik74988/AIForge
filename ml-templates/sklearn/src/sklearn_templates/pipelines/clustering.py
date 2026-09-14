from ..clustering import get_clustering_template


def build_clustering_pipeline(
    model_name,
    preprocessor,
    model_config=None,
):
    model_config = model_config or {}

    template = get_clustering_template(
        model_name=model_name,
        preprocessor=preprocessor,
        model_config=model_config,
    )

    return template.build_pipeline()