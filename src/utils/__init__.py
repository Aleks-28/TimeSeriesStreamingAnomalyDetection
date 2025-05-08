import re
from collections import defaultdict

import numpy as np


def get_optimal_n_bins(X, upper_bound=None, epsilon=1):
    """ Determine optimal number of bins for a histogram using the Birge 
    Rozenblac method (see :cite:`birge2006many` for details.)

    See  https://doi.org/10.1051/ps:2006001 

    Parameters 
    ---------- 
    X : array-like of shape (n_samples, n_features) 
        The samples to determine the optimal number of bins for. 

    upper_bound :  int, default=None 
        The maximum value of n_bins to be considered. 
        If set to None, np.sqrt(X.shape[0]) will be used as upper bound. 

    epsilon : float, default = 1 
        A stabilizing term added to the logarithm to prevent division by zero. 

    Returns 
    ------- 
    optimal_n_bins : int 
        The optimal value of n_bins according to the Birge Rozenblac method 
    """
    if upper_bound is None:
        upper_bound = int(np.sqrt(X.shape[0]))

    n = X.shape[0]
    maximum_likelihood = np.zeros((upper_bound - 1, 1))

    for i, b in enumerate(range(1, upper_bound)):
        histogram, _ = np.histogram(X, bins=b)

        maximum_likelihood[i] = np.sum(
            histogram * np.log(b * histogram / n + epsilon) - (
                b - 1 + np.power(np.log(b), 2.5)))

    return np.argmax(maximum_likelihood) + 1


def get_anomalies_ranges(anomalies_index):
    """Extracts the ranges of anomalies from the anomalies index"""
    ranges = []
    start = None
    end = None
    for i, idx in enumerate(anomalies_index):
        if i == 0:
            start = idx
            end = idx
        else:
            if idx == end + 1:
                end = idx
            else:
                ranges.append((start, end))
                start = idx
                end = idx
    ranges.append((start, end))
    return ranges


def compute_avg_scores(runs: int, res_dict: dict) -> dict:
    """Computes the average scores of the runs.
    Args:
        runs (int): number of runs
        res_dict (dict): dictionary containing the results of all metrics of each run
    Returns:
        dict: dictionary containing the average scores
    """

    avg_dict = defaultdict(list)
    for run in range(runs):
        score_dict = res_dict[f"run_{run}"][0]
        for metric in score_dict.keys():
            avg_dict[metric].extend(score_dict[metric])

    for metric in avg_dict.keys():
        if metric != 'RefMetrics':
            avg_dict[metric] = [x for x in avg_dict[metric] if x is not None]
            avg_dict[metric] = round(
                sum(avg_dict[metric]) / len(avg_dict[metric]), 2)
    return avg_dict


def display_results(res_dict: defaultdict, runs: int) -> None:
    """Displays the results of the evaluation
    Args:
        res_dict (defaultdict): dictionary containing the results of the evaluation
    Returns:
        None
    """
    print("\n" + "*" * 100)
    print(f"Evaluation results for : {runs} runs")
    dict = res_dict['avg_scores']
    for metric in dict.keys():
        if metric != 'RefMetrics':
            print(f"{metric} : {dict[metric]}")


def extract_numbers(input_string):
    # Regular expression to find the number before 'x' and the number after the last '_'
    match = re.search(r'comut_(\d+)x(\d+)_(\d+)', input_string)
    if match:
        # Extract both numbers
        return match.group(1), match.group(2), match.group(3)
    else:
        return None, None
