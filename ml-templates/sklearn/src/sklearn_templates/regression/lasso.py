from sklearn.linear_model import Lasso
from .base import RegressionTemplate


class LassoTemplate(RegressionTemplate):

    def build_estimator(self):
        return Lasso(
            alpha=self.model_config.get("alpha", 1.0),
            fit_intercept=self.model_config.get("fit_intercept", True),
            max_iter=self.model_config.get("max_iter", 1000),
            random_state=self.model_config.get("random_state", 42),
        )