from sklearn.cluster import DBSCAN
from .base import ClusteringTemplate


class DBSCANTemplate(ClusteringTemplate):

    def build_estimator(self):
        return DBSCAN(
            eps=self.model_config.get("eps", 0.5),
            min_samples=self.model_config.get("min_samples", 5),
            metric=self.model_config.get("metric", "euclidean"),
            algorithm=self.model_config.get("algorithm", "auto"),
        )