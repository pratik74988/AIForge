from sklearn.cluster import KMeans
from .base import ClusteringTemplate


class KMeansTemplate(ClusteringTemplate):

    def build_estimator(self):
        return KMeans(
            n_clusters=self.model_config.get("n_clusters", 8),
            init=self.model_config.get("init", "k-means++"),
            n_init=self.model_config.get("n_init", 10),
            max_iter=self.model_config.get("max_iter", 300),
            random_state=self.model_config.get("random_state", 42),
        )