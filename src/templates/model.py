import numpy as np

from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract class for model wrapping.

    Attributes:
        None

    Methods:
        fit: Abstract method for fitting the model.
        score: Abstract method for outputting the anomaly score.
        update: Abstract method for updating the model with new data.
        get_params: Abstract method for returning the model parameters.
    """

    def __init__(self, observation_period: int, **kwargs):
        self.observation_period = observation_period

    @abstractmethod
    def train(self, X_train: np.ndarray):
        """Abstract method for fitting the model. Trains the model on "historical data".

        Args: 
            X (np.): dataframe containing the time series data.
        """
        pass

    @abstractmethod
    def update(self, X):
        """Updates the model with new data.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """Outputs the anomaly score.
        """
        pass
