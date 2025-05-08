import numpy as np

from src.models.OnlineIForest import OnlineIForest
from src.templates.model import Model


class OIF(Model):
    """Implementation of Online Isolation Forest (OIF) algorithm."""

    def __init__(self, observation_period, **kwargs):
        oiforest_params: dict = {'num_trees': 128,
                                 'max_leaf_samples': 8,
                                 'window_size': kwargs.get('window_size', 1024),
                                 'type': 'adaptive'}
        self.oiforest: OnlineIForest = OnlineIForest.create(**oiforest_params)

    def train(self, **kwargs):
        pass

    def update(self, X: np.ndarray):
        pass

    def predict(self, X: np.ndarray):
        self.oiforest.learn_batch(X)
        score = self.oiforest.score_batch(X)
        return score
