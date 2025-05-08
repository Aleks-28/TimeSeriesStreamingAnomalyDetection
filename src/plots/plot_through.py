# calculate real throughtput
import os
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os

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

def get_avg_throughtput_list(data):
    """Averages throughput of each window for each run"""
    throughtput = {}
    data.pop('avg_scores')
    for key in data.keys():
        throughtput[key] = data[key][2]['thgrpt_sample_0']
    # average throughtput
    throughtput = pd.DataFrame(throughtput).mean(axis=1)
    throughtput = throughtput[:-1]
    return throughtput

def throughput_dashboard(observation_period, sliding_window_factor, dataset_name, alg_list):
    # Create a subplot grid: rows for observation_period, columns for sliding_window_factor
    fig = make_subplots(
        rows=len(observation_period),
        cols=len(sliding_window_factor),
        subplot_titles=[
            f"Obs: {obs}, SWF: {swf}" for obs in observation_period for swf in sliding_window_factor
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.1
    )

    for row_idx, observation in enumerate(observation_period, start=1):
        for col_idx, swf in enumerate(sliding_window_factor, start=1):
            dataset = Dataset(dataset_name=dataset_name)
            for _, ts_data, _, _, _, _, _ in dataset.get_ts(0.2): 
                window_size = int(swf * ts_data.shape[0])
                break
            results = get_results(observation, swf, dataset_name)
            throughtput_dict = {}
            
            for alg in alg_list:
                try:
                    throughtput_dict[alg] = window_size / get_avg_throughtput_list(results[alg])
                except Exception as e:
                    print(f"Error for {alg}: {e}")

            for alg in alg_list:
                try:
                    fig.add_trace(
                        go.Scatter(
                            x=throughtput_dict[alg].index,
                            y=throughtput_dict[alg].values,
                            mode='lines',   
                            name=alg if row_idx == 1 and col_idx == 1 else None,  # Only show legend for the first subplot
                            legendgroup=alg,  # Group legends by algorithm to avoid duplicates
                            showlegend=(row_idx == 1 and col_idx == 1),  # Only show legend for the first subplot
                            line=dict(color=px.colors.qualitative.Plotly[alg_list.index(alg) % len(px.colors.qualitative.Plotly)])
                        ),
                        row=row_idx,
                        col=col_idx
                    )
                except Exception as e:
                    print(f"Error for {alg}: {e}")

    # Update layout for the entire figure
    fig.update_layout(
        title='Throughput Comparison for Different Observation Periods and Sliding Window Factors',
        height=800,  # Adjust figure height
        width=1200,  # Adjust figure width
        showlegend=True,
        xaxis_title='Elapsed Time',
        yaxis_title='Throughput',
    )

    # Save the plot as an HTML file
    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
    fig.write_html('results/plots/throughput_subplots.html', auto_open=True)

def throughput_per_dataset(observation_period, sliding_window_factor, dataset_name, alg_list, dataset_list):
    """Plots throughput per dataset."""
    for dataset_name in dataset_list:
        fig = go.Figure()
        dataset = Dataset(dataset_name=dataset_name)
        for _, ts_data, _, _, _, _, _ in dataset.get_ts(0.2): 
            window_size = int(sliding_window_factor[0] * ts_data.shape[0])
            break
        results = get_results(observation_period[0], sliding_window_factor[0], dataset_name, alg_list)
        throughtput_dict = {}
        
        for alg in alg_list:
            try:
                throughtput_dict[alg] = window_size / get_avg_throughtput_list(results[alg])
            except Exception as e:
                print(f"Error for {alg}: {e}")

        for alg in alg_list:
            try:
                fig.add_trace(
                    go.Scatter(
                        x=throughtput_dict[alg].index,
                        y=throughtput_dict[alg].values,
                        mode='lines',   
                        name=alg,  # Only show legend for the first subplot
                        legendgroup=alg,  # Group legends by algorithm to avoid duplicates
                        showlegend=True,  # Only show legend for the first subplot
                        line=dict(color=px.colors.qualitative.Plotly[alg_list.index(alg) % len(px.colors.qualitative.Plotly)])
                    )
                )
            except Exception as e:
                print(f"Error for {alg}: {e}")

        # Update layout for the entire figure
        fig.update_layout(
            title=f'Throughput Comparison for {dataset_name}',
            height=800,  # Adjust figure height
            width=1200,  # Adjust figure width
            showlegend=True,
            xaxis_title='Elapsed Time',
            yaxis_title='Throughput',
        )

        # Save the plot as an HTML file
        if not os.path.exists('results/plots'):
            os.makedirs('results/plots')
        fig.write_html(f'results/plots/throughput_datasets_{dataset_name}.html', auto_open=True)

def avg_throughput(observation_period_list, sliding_window_factor, dataset_name, alg_list, dataset_list):
    """Plots average throughput for each dataset. X is the observation period and Y is the average throughput."""
    for dataset_name in dataset_list:
        avg_throughput_list = []
        dataset = Dataset(dataset_name=dataset_name)
        fig = go.Figure()
        for observation_period in observation_period_list:
            for _, ts_data, _, _, _, _, _ in dataset.get_ts(0.2): 
                window_size = int(sliding_window_factor[0] * ts_data.shape[0])
                break
            results = get_results(observation_period, sliding_window_factor[0], dataset_name, alg_list)
            throughtput_dict = {}
            for alg in alg_list:
                try:
                    throughtput_dict[alg] = window_size / get_avg_throughtput_list(results[alg])
                    avg_throughput_list.append(throughtput_dict[alg].mean())
                except Exception as e:
                    print(f"Error for {alg}: {e}")

        for alg in alg_list:
            try:
                fig.add_trace(
                    go.Scatter(
                        x=observation_period_list,
                        y=avg_throughput_list,
                        mode='lines+markers',   
                        name=alg,  # Only show legend for the first subplot
                        legendgroup=alg,  # Group legends by algorithm to avoid duplicates
                        showlegend=True,  # Only show legend for the first subplot
                        line=dict(color=px.colors.qualitative.Plotly[alg_list.index(alg) % len(px.colors.qualitative.Plotly)])
                    )
                )
            except Exception as e:
                print(f"Error for {alg}: {e}")

            # Update layout for the entire figure
            fig.update_layout(
                title=f'Average Throughput Comparison for {dataset_name}',
                height=800,  # Adjust figure height
                width=1200,  # Adjust figure width
                showlegend=True,
                xaxis_title='Observation Period',
                yaxis_title='Average Throughput',
            )

            fig.write_html(f'results/plots/avg_throughput_{dataset_name}.html', auto_open=True)

if __name__ == "__main__":
    dataset_name = 'comut4'
    alg_list = ['xStream', 'LODA', 'SWKNN', 'SDOs', 'RSHASH', 'LODASALMON']
    dataset_list = ['swan', 'insectsAbr', 'insectsIncr', 'insectsIncrRecr', 'insectsIncrGrd', 'comut4', 'comut8', 'comut16']
    observation_period = [100, 500, 1000, 5000]
    sliding_window_factor = [0.005, 0.01, 0.1, 0.2]
    throughput_per_dataset(observation_period, sliding_window_factor, dataset_name, alg_list, dataset_list)