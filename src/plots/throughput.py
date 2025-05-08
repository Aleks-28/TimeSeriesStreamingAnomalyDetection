# calculate real throughtput
import os
import pickle
import sys
from tkinter import E
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import plotly.graph_objects as go
from collections import defaultdict
import plotly.express as px

from src.datasets import *

def get_results(observation_period, window_factor, dataset, alg_list):
    result_dict = {}
    # scan folder for all alg results
    for alg in alg_list:
        try:
            with open(os.path.join('results', 'pkl', f"{alg}_{dataset}_window_{observation_period}_{window_factor}.pkl"), 'rb') as f:
                results = pickle.load(f)
                result_dict[alg] = results
        except Exception as e:
            print(f"Error in {alg}: {e}")
    return result_dict


def get_processing_times(results_dict):
    """Averages throughput of each window for each run"""
    processing_times = {}
    try:
        results_dict.pop('avg_scores')
    except Exception as e:
        print(f"Error in: {e}")
    for key in results_dict.keys():
        processing_times[key] = results_dict[key][2]['thgrpt_sample_0']
    # average throughtput
    processing_times = pd.DataFrame(processing_times).mean(axis=1)
    processing_times = processing_times[:-1] # drop the last window as it is smaller
    return processing_times

def throughput(processing_times):
    """
    Compute the number of subsequences processed in one second.
    
    Parameters:
        processing_times (list of float): List of processing times for each subsequence.
    
    Returns:
        int: Number of subsequences that can be processed in one second.
    """
    total_time = 0.0
    count = 0.0
    for t in processing_times:
        # If processing the next subsequence would be within the time limit,
        # count it fully.
        if total_time + t <= 1.0:
            total_time += t
            count += 1.0
        else:
            # Otherwise, add the fractional subsequence that fits in the remaining time.
            remaining_time = 1.0 - total_time
            count += remaining_time / t  # This fraction represents the proportion processed.
            break
    return round(count, 2)

def plot_throughput(dataset, obs, sliding_window_list, alg_list):
    """Line plot of throughput for a given tuple (dataset, observation_period, swf) for all algorithms."""
    thgrt = defaultdict(list)
    for swf in sliding_window_list:
        results_dict = get_results(obs, swf, dataset, alg_list)
        for alg in alg_list:
            processing_times = get_processing_times(results_dict[alg])
            through = throughput(processing_times)
            thgrt[f"{alg}"].append(through)
    
    fig = go.Figure()
    for alg in alg_list:
        fig.add_trace(go.Scatter
                        (x=sliding_window_factor, y=thgrt[alg], mode='markers+lines', name=alg))
    fig.update_layout(title=f"Throughput for {dataset}",
                        xaxis_title='Sliding window factor', yaxis_title='Throughput (subsequences per second)', template='simple_white')
    
    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
    fig.write_html(os.path.join('results', 'plots', f"throughput_{dataset}_{obs}.html"), auto_open=True)
    fig.write_image(os.path.join("images", f"throughput_{dataset}_{obs}.svg"))
    

if __name__ == "__main__":
    alg_list = ['xStream', 'LODA', 'SWKNN', 'SDOs', 'RSHASH', 'LODASALMON']
    dataset_list = ['swan', 'insectsAbr', 'insectsIncr', 'insectsIncrRecr', 'insectsIncrGrd', 'comut4', 'comut8', 'comut16']
    observation_list = [100, 500, 1000, 5000]
    sliding_window_factor = [0.005, 0.01, 0.1, 0.2]
    for obs in observation_list:
        plot_throughput('comut4', obs, sliding_window_factor, alg_list)
    
    

