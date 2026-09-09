from sklearn.ensemble import RandomForestRegressor
from .base import RegressionTemplate


class RandomForestRegressionTemplate(RegressionTemplate):

    def build_estimator(self):
        return RandomForestRegressor(
            n_estimators=self.model_config.get("n_estimators", 100),
            max_depth=self.model_config.get("max_depth", None),
            min_samples_split=self.model_config.get("min_samples_split", 2),
            min_samples_leaf=self.model_config.get("min_samples_leaf", 1),
            max_features=self.model_config.get("max_features", 1.0),
            random_state=self.model_config.get("random_state", 42),
            n_jobs=self.model_config.get("n_jobs", -1),
        )