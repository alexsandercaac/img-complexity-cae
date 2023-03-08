"""
    This module defines a custom estimator for the scikit-learn API.
    This estimator is used to adjust the threshold of the CAE reconstruction
    error to make predictions.
"""
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from skopt import BayesSearchCV
from skopt.space import Real


class ScikitCaeEstimator(BaseEstimator):
    """
    This class defines a custom estimator for the scikit-learn API.
    This estimator is used to adjust the threshold of the CAE reconstruction
    error to make predictions.
    """

    def __init__(self, threshold: float = None, score_func=None):
        self.threshold = threshold
        self.score_func = score_func

    def predict(self, X, y=None):
        return np.where(X > self.threshold, 1, 0)

    def fit(self, X, y=None):
        return self

    def score(self, X, y=None):
        predictions = self.predict(X)
        if self.score_func is not None:
            return self.score_func(y, predictions)
        else:
            return np.mean(predictions == y)


def bayesian_search_th(inputs: npt.ArrayLike,
                       labels: npt.ArrayLike,
                       th_min: float, th_max: float,
                       prior: str = 'uniform',
                       score_func: callable = None,
                       n_iter: int = 25
                       ) -> dict:
    """
    This function performs a bayesian search to find the optimal threshold
    for a given score function.

    All the data is used for calculating the score function that is used
    for the bayesian search.

    Args:
        inputs(float): The inputs to the model.
        labels(float): The labels of the inputs.
        th_min(float): The minimum threshold value.
        th_max(float): The maximum threshold value.
        prior(str): The prior distribution of the threshold.
            Defaults to 'uniform'.
        score_func(callable): The score function to be used.
            Defaults to None, in which case the accuracy is used for scoring.
        n_iter(int): The number of iterations for the bayesian search.
            Defaults to 25.

    Returns:
        dict: The best threshold and its associated score.
    """

    params_grid = {
        'threshold': Real(th_min, th_max, prior=prior)
    }

    # The bayesian search is performed using a single fold, and all data
    # is used in calculating the score.
    cv_splits = (
        ([0], list(range(len(labels)))) for _ in range(1)
    )

    clf = BayesSearchCV(
        ScikitCaeEstimator(score_func=score_func),
        params_grid, n_jobs=-1, cv=cv_splits, n_iter=n_iter
    )

    search = clf.fit(inputs, labels)

    best_threshold = search.best_params_['threshold']
    best_score = search.best_score_

    return {'threshold': best_threshold, 'score': best_score}
