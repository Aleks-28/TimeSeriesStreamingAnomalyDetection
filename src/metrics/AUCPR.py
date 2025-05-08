from src.templates.metric import Metric
from sklearn.metrics import average_precision_score


class AUCPR(Metric):
    """sklean's average_precision_score"""
    def __init__(self):
        super().__init__()

    def __name__(self):
        return 'AUCPR'

    def __str__(self):
        return super().__str__()

    def get_score(self, **kwargs) -> float:
        return float(round(average_precision_score(self.y_true_list, self.anomaly_score_list).item(), 2))
