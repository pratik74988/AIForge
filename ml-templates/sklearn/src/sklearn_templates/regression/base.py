from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline


class RegressionTemplate(ABC):

    def __init__(self, preprocessor, model_config: dict):
        self.preprocessor = preprocessor
        self.model_config = model_config
        self.pipeline: Pipeline | None = None

    @abstractmethod
    def build_estimator(self):
        """Return an unfitted sklearn regressor."""
        pass

    def build_pipeline(self) -> Pipeline:
        self.pipeline = Pipeline([
            ("preprocessing", self.preprocessor),
            ("model", self.build_estimator()),
        ])
        return self.pipeline

    def fit(self, X, y):
        if self.pipeline is None:
            self.build_pipeline()

        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been built.")

        return self.pipeline.predict(X)