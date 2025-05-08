from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """Abstract class for metrics.
    """

    def __init__(self) -> None:
        """Initializes the metric by defining the label list and anomaly score list.
        """
        self.y_true_list = []
        self.anomaly_score_list = []

    def __str__(self) -> str:
        return self.__class__.__name__

    def update(self, label: int, anomaly_score: list) -> 'Metric':
        """Updates the metric with the next instance.
        Args:
            label: The true label of the instance.
            anomaly_score: The anomaly score of the instance.

        Returns:
            self
        """
        self.y_true_list.extend(label)
        self.anomaly_score_list.extend(anomaly_score)
        return self

    @abstractmethod
    def get_score(self, **kwargs) -> float:
        """Returns the score of the model on a time series.
        """
        pass
