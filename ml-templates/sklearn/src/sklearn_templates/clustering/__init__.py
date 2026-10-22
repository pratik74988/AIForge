from .kmeans import KMeansTemplate
from .dbscan import DBSCANTemplate


CLUSTERING_REGISTRY = {
    "kmeans": KMeansTemplate,
    "dbscan": DBSCANTemplate,
}


def get_clustering_template(model_name, preprocessor, model_config):
    if model_name not in CLUSTERING_REGISTRY:
        raise ValueError(
            f"Unknown clustering model '{model_name}'. "
            f"Available: {list(CLUSTERING_REGISTRY.keys())}"
        )

    return CLUSTERING_REGISTRY[model_name](
        preprocessor=preprocessor,
        model_config=model_config,
    )