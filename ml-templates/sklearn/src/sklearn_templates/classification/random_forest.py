from sklearn.ensemble import RandomForestClassifier
from .base import ClassificationTemplate


class RandomForestTemplate(ClassificationTemplate):

    def build_estimator(self):
        return RandomForestClassifier(
            n_estimators=self.model_config.get("n_estimators", 100),
            max_depth=self.model_config.get("max_depth", None),
            min_samples_split=self.model_config.get("min_samples_split", 2),
            min_samples_leaf=self.model_config.get("min_samples_leaf", 1),
            max_features=self.model_config.get("max_features", "sqrt"),
            class_weight=self.model_config.get("class_weight", None),
            random_state=self.model_config.get("random_state", 42),
            n_jobs=self.model_config.get("n_jobs", -1),
        )