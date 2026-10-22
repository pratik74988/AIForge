from sklearn.linear_model import LinearRegression
from .base import RegressionTemplate


class LinearRegressionTemplate(RegressionTemplate):

    def build_estimator(self):
        return LinearRegression(
            fit_intercept=self.model_config.get("fit_intercept", True),
            n_jobs=self.model_config.get("n_jobs", -1),
        )