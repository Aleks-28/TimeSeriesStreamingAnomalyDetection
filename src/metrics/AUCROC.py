import numpy as np

from src.templates.metric import Metric
from sklearn.metrics import roc_auc_score


class AUCROC(Metric):
    """
    sklearn.metrics.roc_auc_score wrapper class.
    """

    def __init__(self):
        super().__init__()

    def __name__(self):
        return 'AUCROC'

    def __str__(self):
        return super().__str__()

    def get_score(self, **kwargs) -> float:
        """
        Compute the AUC-ROC score using sklearn.metrics.roc_auc_score and round it to 3 decimal places.
        :return: AUC-ROC score rounded to 3 decimal places.
        """
        transformed_scores = -1 / (1 + np.array(self.anomaly_score_list))
        # Min-max normalization
        self.normalized_scores = (transformed_scores - np.min(transformed_scores)) / \
            (np.max(transformed_scores) - np.min(transformed_scores))
        return float(round(roc_auc_score(self.y_true_list, self.normalized_scores).item(), 2))
