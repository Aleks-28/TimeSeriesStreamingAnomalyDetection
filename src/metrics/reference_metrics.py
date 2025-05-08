import numpy as np

import src.utils.pamsearch as pams
from src.utils.indices import get_indices
from src.templates.metric import Metric


class RefMetrics(Metric):
    """
    Reference metrics from the Comparison and evaluation study paper. [https://doi.org/10.1016/j.eswa.2023.120994.]
    """

    def __init__(self, window_size: int = 5):
        super().__init__()

    def __name__(self):
        return 'RefMetrics'

    def __str__(self):
        return super().__str__()

    def get_score(self, **kwargs) -> float:
        """
        Compute
        """
        labels = np.array(self.y_true_list)
        scores = np.array(self.anomaly_score_list)
        perf = get_indices(labels,
                           pams.transform_scores(scores))
        return perf
