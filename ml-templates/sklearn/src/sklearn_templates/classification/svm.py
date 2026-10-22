# svm.py
from sklearn.svm import SVC
from .base import ClassificationTemplate
class SVMTemplate(ClassificationTemplate):
    def build_estimator(self):
        return SVC(
            C=self.model_config.get("C", 1.0),
            kernel=self.model_config.get("kernel", "rbf"),
            gamma=self.model_config.get("gamma", "scale"),
            probability=self.model_config.get("probability", True),
            class_weight=self.model_config.get("class_weight", None),
            random_state=self.model_config.get("random_state", 42),
        )