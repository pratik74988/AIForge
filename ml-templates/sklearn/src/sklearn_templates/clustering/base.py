from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline


class ClusteringTemplate(ABC):

    def __init__(self, preprocessor, model_config: dict):
        self.preprocessor = preprocessor
        self.model_config = model_config
        self.pipeline: Pipeline | None = None

    @abstractmethod
    def build_estimator(self):
        """Return an unfitted sklearn clustering estimator."""
        pass

    def build_pipeline(self) -> Pipeline:
        self.pipeline = Pipeline([
            ("preprocessing", self.preprocessor),
            ("model", self.build_estimator()),
        ])
        return self.pipeline

    def fit(self, X):
        if self.pipeline is None:
            self.build_pipeline()

        self.pipeline.fit(X)
        return self

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("Pipeline has not been built.")

        model = self.pipeline.named_steps["model"]
        if not hasattr(model, "predict"):
            raise NotImplementedError(
                f"{model.__class__.__name__} does not support predict() on new data; "
                f"use fit_predict() instead."
            )

        return self.pipeline.predict(X)

    def fit_predict(self, X):
        if self.pipeline is None:
            self.build_pipeline()

        return self.pipeline.fit_predict(X)