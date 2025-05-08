import numpy as np

from src.templates.metric import Metric
from TSB_UAD.vus.metrics import get_metrics


class VUSPR(Metric):
    """VUSPR metric from TSB-UAD repo https://github.com/TheDatumOrg/TSB-UAD"""

    def __init__(self):
        super().__init__()

    def __name__(self):
        return 'VUSPR'

    def __str__(self):
        return super().__str__()

    def get_score(self, **kwargs) -> float:
        slidingWindow = kwargs.get("vus_window", 5)
        anomaly_score_list = np.array(self.anomaly_score_list)
        y_true_list = np.array(self.y_true_list)
        results = get_metrics(anomaly_score_list, y_true_list, metric="vus",
                              slidingWindow=slidingWindow)
        return float(round(results["VUS_PR"], 2))
