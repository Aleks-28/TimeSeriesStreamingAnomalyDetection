import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import *

from src.datasets import *

def get_results(observation_period, window_factor, dataset, alg_list):
    """Returns a dictionary with the averaged results for 10 runs of all algorithms for a given dataset and a set of parameters (observation_period, window_factor)"""
    result_dict = {}
    # scan folder for all alg results
    for alg in alg_list:
        try:
            with open(os.path.join('results', 'pkl', f"{alg}_{dataset}_window_{observation_period}_{window_factor}.pkl"), 'rb') as f:
                results = pickle.load(f)
                result_dict[alg] = results
        except Exception as e:
            print(f"Couldn't load results for {alg} on {dataset} with obs={observation_period} and swf={window_factor}: {e}")
            print("Trying to load a metric instead")
            try:
                with open(os.path.join('results', 'pkl', f"RefMetrics_{dataset}_window_{observation_period}_{window_factor}.pkl"), 'rb') as f:
                    results = pickle.load(f)
                    result_dict[alg] = [results['AUCPR'], results['AUCROC'], results['VUSPR'], results['VUSROC']]
            except Exception as e:
                print(f"Couldn't load metric for {alg} on {dataset} with obs={observation_period} and swf={window_factor}: {e}")
            continue
    return result_dict

def calculate_early_detection(anomaly_score, anomaly_mask):
    """Calculates the distance between the start of the anomaly and the first time the anomaly score is above a certain threshold"""
    distance_list = []
    thresholds = np.linspace(0, 0.2*max(anomaly_score), max(anomaly_score))
    anomaly_ranges = get_anomalies_ranges(anomaly_mask) # outputs a list of tuples with the start and end of each anomaly
    for threshold in thresholds:
        for anomaly_range in range(len(anomaly_ranges)):
            # check if anomaly is point anomaly or range anomaly
            if anomaly_range[0] == anomaly_range[1]:
                for i in range(anomaly_range[0],len(anomaly_score),1):
                    if anomaly_score[i] > threshold:
                        distance_list.append(i-anomaly_range[0])
                        break
            else:
                for i in range(anomaly_range[1],len(anomaly_score),1):
                    if anomaly_score[i] > threshold:
                        distance_list.append(i-anomaly_range[1])
                        break
    # compute area under the curve of the distance list: the lower the better
    auc = np.trapz(distance_list)
    return auc

def main():
    observation_period = [100, 500, 1000, 5000]
    sliding_window_factor = [0.005, 0.01, 0.1, 0.2]
    alg_list = ['xStream', 'LODA', 'SWKNN', 'SDOs', 'RSHASH', 'LODASALMON']
    dataset_list = ['comut4', 'comut8', 'comut16', 'insectsAbr', 'insectsIncr', 'insectsIncrGrd', 'insectsIncrRecr', 'swan']
    for dataset in dataset_list:
        path_df = Dataset(dataset_name=dataset).path_df
        for _, row in path_df.iterrows():
                    sample = pd.read_csv(row["path"])
                    anomaly_mask = sample[sample.iloc[:, -1] == 1].index.to_list()
                    result_dict = get_results(100, 0.01, 'swan', alg_list)
                    for alg in alg_list:
                        for run in range(10):
                            for sample in range(len(result_dict[alg][f"run_{run}"][1])):
                                anomaly_mask = result_dict[alg][f"run_{run}"][1][f"ano_score_sample_{sample}"]
                                anomaly_score = result_dict[alg][f"run_{run}"][0][sample]
    distance_list = calculate_early_detection(result_dict['xStream'][0], result_dict['xStream'][1])




if __name__ == "__main__":
    main()