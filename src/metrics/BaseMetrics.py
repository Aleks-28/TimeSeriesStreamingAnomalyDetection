from src.templates.metric import Metric
from sklearn.metrics import f1_score, precision_score, recall_score


class Recall(Metric):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return super().__str__()

    def get_score(self):
        return recall_score(self.y_true_list, self.anomaly_score_list)


class Precision(Metric):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return super().__str__()

    def get_score(self):
        return precision_score(self.y_true_list, self.anomaly_score_list)


class F1(Metric):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return super().__str__()

    def get_score(self):
        return f1_score(self.y_true_list, self.anomaly_score_list)