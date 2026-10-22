from sklearn.linear_model import Ridge
from .base import RegressionTemplate


class RidgeTemplate(RegressionTemplate):

    def build_estimator(self):
        return Ridge(
            alpha=self.model_config.get("alpha", 1.0),
            fit_intercept=self.model_config.get("fit_intercept", True),
            solver=self.model_config.get("solver", "auto"),
            random_state=self.model_config.get("random_state", 42),
        )