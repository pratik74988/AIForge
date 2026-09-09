from sklearn.linear_model import LogisticRegression
from .base import ClassificationTemplate


class LogisticRegressionTemplate(ClassificationTemplate):

    def build_estimator(self):
        return LogisticRegression(
            C=self.model_config.get("C", 1.0),
            l1_ratio=self.model_config.get("l1_ratio", 0),   # 0 = l2, 1 = l1, None = elasticnet
            solver=self.model_config.get("solver", "lbfgs"),
            max_iter=self.model_config.get("max_iter", 1000),
            class_weight=self.model_config.get("class_weight", None),
            random_state=self.model_config.get("random_state", 42),
        )