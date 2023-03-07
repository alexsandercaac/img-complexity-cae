"""
    This module defines a custom estimator for the scikit-learn API.
    This estimator is used to adjust the threshold of the CAE reconstruction
    error to make predictions.
"""
import numpy as np
from sklearn.base import BaseEstimator


class ScikitCaeEstimator(BaseEstimator):
    """
    This class defines a custom estimator for the scikit-learn API.
    This estimator is used to adjust the threshold of the CAE reconstruction
    error to make predictions.
    """
    def __init__(self, threshold: float = None):
        self.threshold = threshold

    def predict(self, X, y=None):
        return np.where(X > self.threshold, 1, 0)
    
    def fit(self, X, y=None):
        return self

    def score(self, X, y=None, score_func=None):
        predictions = self.predict(X)
        if score_func is not None:
            return score_func(y, predictions)
        else:
            return np.mean(predictions == y)
