import numpy as np
import src.utils.pamsearch as pams
from src.templates.model import Model


class SWKNN(Model):

    def __init__(self, observation_period, **kwargs):
        super().__init__(observation_period=observation_period)
        self.idx = kwargs.get('run')

    def train(self, X_train, **kwargs):
        labels = kwargs.get('labels', None)
        blocksize = kwargs.get('window_size', 1000)
        self.training_len = X_train.shape[0]
        self.scores = np.zeros(X_train.shape[0])
        timestamps = np.arange(0, X_train.shape[0])
        timestamps = timestamps[:, None]
        best_param = pams.parameter_selection(
            X_train, labels, timestamps, self.training_len, blocksize, 'swknn', self.observation_period)
        print('Best parameter:', best_param)
        self.detector = pams.adjust_algorithm(
            'swknn', best_param,  self.idx, self.observation_period, blocksize)

        # TRAINING
        print('Training...')
        self.scores[0:self.training_len] = self.detector.fit_predict(X_train)
        return self

    def update(self, X):
        pass

    def predict(self, X):
        self.scores = self.detector.fit_predict(X)
        return self.scores
