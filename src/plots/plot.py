import os
from tempfile import template
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pickle
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.datasets import *

def get_results(observation_period, window_factor, dataset, alg_list):
    """Returns a dictionary with the averaged results for 10 runs of all algorithms for a given dataset and a set of parameters (observation_period, window_factor)"""
    result_dict = {}
    # scan folder for all alg results
    for alg in alg_list:
        try:
            with open(os.path.join('results', 'pkl', f"{alg}_{dataset}_window_{observation_period}_{window_factor}.pkl"), 'rb') as f:
                results = pickle.load(f)
                result_dict[alg] = results['avg_scores']
        except Exception as e:
            print(f"Couldn't load results for {alg} on {dataset} with obs={observation_period} and swf={window_factor}: {e}")
            print("Trying to load a metric instead")
            try:
                with open(os.path.join('results', 'pkl', f"{alg}_{dataset}_window_{observation_period}_{window_factor}.pkl"), 'rb') as f:
                    results = pickle.load(f)
                    result_dict[alg] = [results['AUCPR'], results['AUCROC'], results['VUSPR'], results['VUSROC']]
            except Exception as e:
                print(f"Couldn't load metric for {alg} on {dataset} with obs={observation_period} and swf={window_factor}: {e}")
            continue
    return result_dict

def get_results_dataset(dataset_list, observation_period, sliding_window_factor, alg_list):
    """Outputs a dataframe with the results of all algorithms for all datasets, for a given observation period and sliding window factor.
    Also outputs a list of errors that occurred during the process.
    Args:
        dataset_list (list): list of dataset names
        observation_period (int): observation period
        sliding_window_factor (float): sliding window factor
        alg_list (list): list of algorithm names
    Returns:
        res_df (pd.DataFrame): dataframe with the results of all algorithms for all datasets. Columns: dataset, alg, VUSPR, VUSROC, AUCROC, AUCPR
        error_list (list): list of errors that occurred during the process
    """
    error_list = []
    res_df = pd.DataFrame(columns=['dataset', 'alg', 'VUSPR', 'VUSROC', 'AUCROC', 'AUCPR'])
    for dataset in dataset_list:
        results = get_results(observation_period, sliding_window_factor, dataset, alg_list)
        for alg in alg_list:
            if 'RefMetrics' in results.keys():
                results.pop('RefMetrics')
            try:
                res_df = pd.concat([res_df, pd.DataFrame([{'dataset': dataset, 'alg': alg, 'VUSPR': results[alg]['VUSPR'], 'VUSROC': results[alg]['VUSROC'], 'AUCROC': results[alg]['AUCROC'], 'AUCPR': results[alg]['AUCPR']}])], ignore_index=True)
            except Exception as e:
                error_list.append((dataset, alg, e))
                res_df = pd.concat([res_df, pd.DataFrame([{'dataset': dataset, 'alg': alg, 'VUSPR': None, 'VUSROC': None, 'AUCROC': None, 'AUCPR': None}])], ignore_index=True)
    return res_df, error_list

# def boxplot(fig, df, col):
#     """Plots a boxplot for each metric (VUSPR, VUSROC, AUCROC, AUCPR) for all algorithms in the dataframe df.
#     Args:
#         fig (plotly.graph_objects.Figure): figure where the boxplots will be added
#         df (pd.DataFrame): dataframe with the results of all algorithms for all datasets. Columns: dataset, alg, VUSPR, VUSROC, AUCROC, AUCPR
#         col (int): column where the boxplots will be added
#     """
#     metrics = ["VUSPR", "VUSROC", "AUCROC", "AUCPR"]
#     df_melted = df.melt(id_vars=["dataset", "alg"], value_vars=metrics,
#                         var_name="metric", value_name="score")
#     
#     for i, metric in enumerate(metrics):
#         metric_df = df_melted[df_melted["metric"] == metric]
#         fig.add_trace(
#             go.Box(x=metric_df["alg"], y=metric_df["score"], name=metric, boxmean=True),
#             row=i + 1, col=col  
#         )
#         fig.update_yaxes(range=[0, 1], row=i + 1, col=col)

def aggregate_results(observation_period, sliding_window_factor, alg_list, dataset_list):
    """
    Builds a dataframe obtaining the results over all sets (observation, swf) for each dataset.
    Plots the results in a dashboard of boxplots, where datasets are arranged as columns and metrics as rows.
    """
    all_results = []
    
    # Collect all results
    for obs in observation_period:
        for swf in sliding_window_factor:
            res_df, _ = get_results_dataset(dataset_list, obs, swf, alg_list)
            res_df["Observation Period"] = obs
            res_df["Sliding Window Factor"] = swf
            all_results.append(res_df)
    
    # Combine all results into a single dataframe
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Extract unique datasets and metrics
    datasets = final_df["dataset"].unique()
    metrics = [col for col in final_df.columns if col not in ["dataset", "alg", "Observation Period", "Sliding Window Factor"]]
    
    # Create a subplot figure with datasets as columns and metrics as rows
    fig = make_subplots(rows=len(metrics), cols=len(datasets),
                           subplot_titles=[f"{metric} - {dataset}" for metric in metrics for dataset in datasets],
                           vertical_spacing=0.2, horizontal_spacing=0.05)
    
    # Populate the subplots with boxplots
    for row_idx, metric in enumerate(metrics, start=1):
        for col_idx, dataset in enumerate(datasets, start=1):
            df_filtered = final_df[final_df["dataset"] == dataset].reset_index(drop=True)
            boxplot = px.box(df_filtered, x="alg", y=metric, color="alg")
            
            for trace in boxplot.data:
                fig.add_trace(trace, row=row_idx, col=col_idx)
    fig.update_xaxes(tickangle=45)
    # Save the plot as an HTML file
    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
    fig.write_html('results/plots/aggregate_boxplots.html', auto_open=True)

def plot_all_boxplots(observation_period, sliding_window_factor, alg_list, dataset_list):
    """Plot boxplots for VUSPR, AUCPR for the given dataset and all algorithms."""
    all_results = []
    for obs in observation_period:
        for swf in sliding_window_factor:
            res_df, _ = get_results_dataset(dataset_list, obs, swf, alg_list)
            res_df["Observation Period"] = obs
            res_df["Sliding Window Factor"] = swf
            all_results.append(res_df)
    
    final_df = pd.concat(all_results, ignore_index=True)
    fig = go.Figure()

    metrics = ["AUCPR", "VUSPR"]
    plot_df = final_df[["dataset", "AUCPR", "VUSPR"]].groupby("dataset").mean().reset_index()
    for metric in metrics:
        fig.add_trace(go.Box(
            y=plot_df[metric],
            x=plot_df["dataset"],
            name=metric,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            marker_size=2,
            line_width=2)
            )
        
    fig.update_layout(
        xaxis_title="Dataset",
        yaxis_title="Score",
        showlegend=True
    )

    # Save the plot as an HTML file
    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
    fig.write_html('results/plots/all_boxplots.html', auto_open=True)
    fig.write_image('images/all_boxplots.svg')

def overall_results(observation_period, sliding_window_factor, alg_list, dataset_list):
    """Plots a group of bar plots with the average AUCPR score for each algorithm and dataset."""
    all_results = []
    for obs in observation_period:
        for swf in sliding_window_factor:
            res_df, _ = get_results_dataset(dataset_list, obs, swf, alg_list)
            res_df["Observation Period"] = obs
            res_df["Sliding Window Factor"] = swf
            all_results.append(res_df)
    
    final_df = pd.concat(all_results, ignore_index=True)
    metric = ["AUCPR", "AUCROC", "VUSPR", "VUSROC"]
    for m in metric:
        plot_df = final_df[[m, "dataset", "alg"]].groupby(["dataset", "alg"]).mean().reset_index()
        fig = go.Figure()
        for alg in alg_list:
            plot_df_alg = plot_df[plot_df["alg"] == alg]
            fig.add_trace(go.Bar(
                x=plot_df_alg["dataset"],
                y=plot_df_alg[m],
                name=alg
            ))
        fig.update_layout(
            xaxis_title="Dataset",
            yaxis_title=f"{m} Score",
            showlegend=True
        )
        # increase font size of ticks and axis labels
        fig.update_layout(
            xaxis_title_font=dict(size=23),
            yaxis_title_font=dict(size=23),
            xaxis_tickfont=dict(size=20),
            yaxis_tickfont=dict(size=20),
            title_font=dict(size=25),
            legend_font=dict(size=20),
            template='simple_white'
        )
    
        # Save the plot as an HTML file
        if not os.path.exists('results/plots'):
            os.makedirs('results/plots')
        fig.write_html(f'results/plots/overall_results_{m}.html', auto_open=True)
        fig.write_image(f'images/overall_results_{m}.svg')

def heatmap(alg_list, dataset_list, observation_period, sliding_window_factor):
    """Plots a heatmap with the AUCPR score for each algorithm and dataset."""
    all_results = []
    for obs in observation_period:
        for swf in sliding_window_factor:
            res_df, _ = get_results_dataset(dataset_list, obs, swf, alg_list)
            res_df["Observation Period"] = obs
            res_df["Sliding Window Factor"] = swf
            all_results.append(res_df)
    
    final_df = pd.concat(all_results, ignore_index=True)
    metric = ["AUCPR", "AUCROC", "VUSPR", "VUSROC"]
    for m in metric:
        plot_df = final_df[[m, "dataset", "alg"]].groupby(["dataset", "alg"]).mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=plot_df[m],
            x=plot_df["alg"],
            y=plot_df["dataset"],
            colorscale='Viridis',
            text=plot_df[m].round(2),
            texttemplate='%{text}', 
            showscale=True
        ))
        fig.update_layout(
            xaxis_title="Algorithm",
            yaxis_title="Dataset",
            showlegend=True
        )
        # make text bigger
        fig.update_traces(textfont_size=30)
        fig.update_layout(
            xaxis_title_font=dict(size=45),
            yaxis_title_font=dict(size=45),
            xaxis_tickfont=dict(size=45),
            yaxis_tickfont=dict(size=45),
        )
        # Save the plot as an HTML file
        if not os.path.exists('results/plots'):
            os.makedirs('results/plots')
        fig.write_html(f'results/plots/heatmap_{m}.html', auto_open=True)
        fig.write_image(f'results/plots/heatmap_{metric}.svg')

def obs_plot(observation_period, sliding_window_factor, alg_list, dataset_list):
    """Plots one rainbow plot per algorithm, with x being a pair of observation period and sliding window factor, and y being the AUCPR score."""
    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
    for i,alg in enumerate(alg_list):    
        fig = go.Figure()
        all_results = []
        for obs in observation_period:
            for swf in sliding_window_factor:
                res_df, _ = get_results_dataset(dataset_list, obs, swf, [alg])
                res_df["Observation Period"] = obs
                res_df["Sliding Window Factor"] = swf
                all_results.append(res_df)
        
        obs_df = pd.concat(all_results, ignore_index=True)
        fig.add_trace(go.Box(
            y=obs_df["AUCPR"],
            x=obs_df["Observation Period"], 
            boxpoints='all',
            boxmean=True,
            jitter=0.5,
            whiskerwidth=1,
            marker_color=colors[i],
            marker_size=2,
            line_width=1,
            showlegend=False)
            )
        # add line for average AUCPR
        avg = obs_df.groupby("Observation Period", as_index=False)["AUCPR"].mean()
        fig.add_trace(go.Scatter(x=avg['Observation Period'], y=avg['AUCPR'], 
                                 mode='lines', name='Average', line=dict(color='red', width=2)))

        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
            yaxis=dict(zeroline=False, gridcolor='white'),
            template='simple_white',
            xaxis_title="Observation Period",
            yaxis_title="AUCPR Score",
            showlegend=True,
            title_font=dict(size=35),
            legend_font=dict(size=30),
            xaxis_title_font=dict(size=30),
            yaxis_title_font=dict(size=30),
            xaxis_tickfont=dict(size=30),
            yaxis_tickfont=dict(size=30),
            width=1200,
            height=800
        )

        if not os.path.exists('results/plots'):
            os.makedirs('results/plots')
        fig.write_html(os.path.join('results', 'plots', f'obs_plot_{alg}.html'), auto_open=True)
        fig.write_image(os.path.join("images", f"obs_plot_{alg}.svg"))
    
def swf_plot(observation_period, sliding_window_factor, alg_list, dataset_list):
    """Plots one rainbow plot per algorithm, with x being a pair of observation period and sliding window factor, and y being the AUCPR score."""
    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
          'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
    for i,alg in enumerate(alg_list):    
        fig = go.Figure()
        all_results = []
        for obs in observation_period:
            for swf in sliding_window_factor:
                res_df, _ = get_results_dataset(dataset_list, obs, swf, [alg])
                res_df["Observation Period"] = obs
                res_df["Sliding Window Factor"] = swf
                all_results.append(res_df)
        
        obs_df = pd.concat(all_results, ignore_index=True)
        obs_df = obs_df.dropna(subset=["AUCPR"])
        fig.add_trace(go.Box(
            y=obs_df["AUCPR"],
            x=obs_df["Sliding Window Factor"], 
            boxpoints='all',
            boxmean=True,
            jitter=0.5,
            whiskerwidth=1,
            fillcolor=colors[i],
            marker_size=2,
            line_width=1,
            showlegend=False)
            )
        # add line for average AUCPR
        avg = obs_df[['Sliding Window Factor','AUCPR']].groupby('Sliding Window Factor').mean().reset_index()
        fig.add_trace(
            go.Scatter(x=avg['Sliding Window Factor'], y=avg['AUCPR'], mode='lines', name='Average', 
                       line=dict(color='red', width=2)))
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(zeroline=False, gridcolor='white'),
            template='simple_white',
            xaxis_title="Sliding Window Factor",
            yaxis_title="AUCPR Score",
            showlegend=True,
            title_font=dict(size=35),
            legend_font=dict(size=30),
            xaxis_title_font=dict(size=30),
            yaxis_title_font=dict(size=30),
            xaxis_tickfont=dict(size=30),
            yaxis_tickfont=dict(size=30),
            width=1200,
            height=800
        )
        if not os.path.exists('results/plots'):
            os.makedirs('results/plots')
        fig.write_html(os.path.join('results', 'plots', f'swf_plot_{alg}.html'), auto_open=True)
        fig.write_image(os.path.join("images", f"swf_plot_{alg}.svg"))

def cont_rate_plot(observation_period, sliding_window_factor, alg_list, dataset_list):
    """Plot AUCPR vs contamination rate for each algorithm."""
    # get stat dict for all datasets
    dicts = {dataset_name: None for dataset_name in dataset_list}
    for dataset_name in dicts.keys():
        dataset = Dataset(dataset_name=dataset_name, stats=True)
        distribs = dataset.get_distribs()
        dicts[dataset_name] = distribs

    # get contamination rates for all datasets in a dict with dataset name as key and mean contamination rate as value
    contamination_rates = {dataset_name: np.mean(list(dicts[dataset_name]['cont_rate'].keys())) for dataset_name in dicts.keys()}
    anomaly_lengths = {dataset_name: np.mean([int(length) for length in dicts[dataset_name]['ano_lengths'].keys()]) for dataset_name in dicts.keys()}
    # Sort datasets by contamination rate
    sorted_datasets = sorted(contamination_rates, key=contamination_rates.get)

    fig = go.Figure()

    # Line plot: one line per algorithm, x is dataset (ordered by contamination rate), y is AUCPR
    for alg in alg_list:
        all_results = []
        for obs in observation_period:
            for swf in sliding_window_factor:
                res_df, _ = get_results_dataset(dataset_list, obs, swf, [alg])
                res_df["Observation Period"] = obs
                res_df["Sliding Window Factor"] = swf
                all_results.append(res_df)

        obs_df = pd.concat(all_results, ignore_index=True)
        # add contamination rate to obs_df
        obs_df['cont_rate'] = obs_df['dataset'].map(contamination_rates)
        plot_df = obs_df[['dataset', 'AUCPR', 'cont_rate']].groupby('dataset').mean().reset_index()
        plot_df = plot_df.set_index('dataset').loc[sorted_datasets].reset_index()  # Order by contamination rate

        fig.add_trace(go.Scatter(
            x=plot_df['cont_rate'],
            y=plot_df['AUCPR'],
            mode='lines+markers',
            name=alg)
        )

    fig.update_layout(
        title="AUCPR Score vs Contamination Rate",
        xaxis_title="Contamination Rate",
        yaxis_title="AUCPR Score",
        showlegend=True
    )

    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
    fig.write_html(os.path.join('results', 'plots', 'cont_rate_plot.html'), auto_open=True)

def ano_length_plot(observation_period, sliding_window_factor, alg_list, dataset_list):
    """Plot AUCPR vs anomaly length for each algorithm."""
    # get stat dict for all datasets
    dicts = {dataset_name: None for dataset_name in dataset_list}
    for dataset_name in dicts.keys():
        dataset = Dataset(dataset_name=dataset_name, stats=True)
        distribs = dataset.get_distribs()
        dicts[dataset_name] = distribs

    # get average anomaly lengths for all datasets in a dict with dataset name as key and mean anomaly length as value
    anomaly_lengths = {dataset_name: np.mean([int(length) for length in dicts[dataset_name]['ano_lengths'].keys()]) for dataset_name in dicts.keys()}
    # Sort datasets by anomaly length
    sorted_datasets = sorted(anomaly_lengths, key=anomaly_lengths.get)
    fig = go.Figure()
    for i,alg in enumerate(alg_list):    
        all_results = []
        for obs in observation_period:
            for swf in sliding_window_factor:
                res_df, _ = get_results_dataset(dataset_list, obs, swf, [alg])
                res_df["Observation Period"] = obs
                res_df["Sliding Window Factor"] = swf
                all_results.append(res_df)
        plot_df = pd.concat(all_results, ignore_index=True)
        # comput average AUCPR for each dataset
        plot_df['ano_length'] = plot_df['dataset'].map(anomaly_lengths)
        plot_df = plot_df[['dataset', 'AUCPR', 'ano_length']].groupby('dataset').mean().reset_index()
        plot_df = plot_df.set_index('dataset').loc[sorted_datasets].reset_index()  # Order by ano_length
        # add line plot for AUCPR for each alg : x is ano_length, with name of corresponding dataset, y is AUCPR
        fig.add_trace(go.Scatter(
            x=plot_df['ano_length'],
            y=plot_df['AUCPR'],
            mode='lines+markers',
            name=alg)
        )
    fig.update_layout(
        title="AUCPR Score vs Anomaly Length",
        xaxis_title="Anomaly Length",
        yaxis_title="AUCPR Score",
        showlegend=True
    )
    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
    fig.write_html(os.path.join('results', 'plots', 'ano_length_plot.html'), auto_open=True)

def dim_plot(observation_period, sliding_window_factor, alg_list, dataset_list):
    """Plot AUCPR vs dimensionality for each algorithm."""
    # get stat dict for all datasets
    dicts = {dataset_name: None for dataset_name in dataset_list}
    for dataset_name in dicts.keys():
        dataset = Dataset(dataset_name=dataset_name, stats=True)
        distribs = dataset.get_distribs()
        dicts[dataset_name] = distribs

    # get contamination rates for all datasets in a dict with dataset name as key and mean contamination rate as value
    dimensionality = {dataset_name: np.mean(list(dicts[dataset_name]['dimensions'].keys())) for dataset_name in dicts.keys()}
    # Sort datasets by contamination rate
    sorted_datasets = sorted(dimensionality, key=dimensionality.get)

    fig = go.Figure()
    # Line plot: one line per algorithm, x is dataset (ordered by contamination rate), y is AUCPR
    for alg in alg_list:
        all_results = []
        for obs in observation_period:
            for swf in sliding_window_factor:
                res_df, _ = get_results_dataset(dataset_list, obs, swf, [alg])
                res_df["Observation Period"] = obs
                res_df["Sliding Window Factor"] = swf
                all_results.append(res_df)

        obs_df = pd.concat(all_results, ignore_index=True)
        # add ano_length  to obs_df
        obs_df['dimensionality'] = obs_df['dataset'].map(dimensionality)
        # for datasets with same dimensionality, group and average the AUCPR
        plot_df = obs_df[['dataset', 'AUCPR', 'dimensionality']].groupby('dataset').mean().reset_index()
        plot_df = plot_df.set_index('dataset').loc[sorted_datasets].reset_index()  # Order by dimensionality

        fig.add_trace(go.Scatter(
            x=plot_df['dimensionality'],
            y=plot_df['AUCPR'],
            mode='lines+markers',
            marker=dict(size=10),
            name=alg,
        ))

        fig.update_layout(
        title="AUCPR Score vs dimensionality",
        xaxis_title="dimensionality",
        yaxis_title="AUCPR Score",
        title_font=dict(size=40),
        legend_font=dict(size=30),
        xaxis=dict(title='dimensionality', title_font=dict(size=25)),
        yaxis=dict(title='AUCPR Score', title_font=dict(size=25)),
        showlegend=True,
        template='simple_white',
        )

    if not os.path.exists('results/plots'):
        os.makedirs('results/plots')
    fig.write_html(os.path.join('results', 'plots', 'ano_dimensionality_plot.html'), auto_open=True)
    fig.write_image(os.path.join("images", "ano_dimensionality_plot.svg"))

def heatmap(alg_list, dataset_list, observation_period, sliding_window_factor):
    """Plots a heatmap with the AUCPR score for each algorithm and dataset. The scores are averaged over all observation periods and sliding window factors."""
    all_results = []
    for obs in observation_period:
        for swf in sliding_window_factor:
            res_df, _ = get_results_dataset(dataset_list, obs, swf, alg_list)
            res_df["Observation Period"] = obs
            res_df["Sliding Window Factor"] = swf
            all_results.append(res_df)
    
    final_df = pd.concat(all_results, ignore_index=True)
    metric = ["AUCPR", "AUCROC", "VUSPR", "VUSROC"]
    for m in metric:
        plot_df = final_df[[m, "dataset", "alg"]].groupby(["dataset", "alg"]).mean().reset_index()
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=plot_df[m],
            x=plot_df["alg"],
            y=plot_df["dataset"],
            colorscale='Viridis',
            text=plot_df[m].round(2),  # Add text annotations
            texttemplate='%{text}',  # Show 2 decimal places
            hoverinfo='text+z',  # Show both text and z value on hover
            showscale=True
        ))
        fig.update_layout(
            title=f"{m} averaged for Each Algorithm and Dataset",
            xaxis_title="Algorithm",
            yaxis_title="Dataset",
            showlegend=True,
            title_font=dict(size=20),
            legend_font=dict(size=15)
        )

        # Save the plot as an HTML file
        if not os.path.exists('results/plots'):
            os.makedirs('results/plots')
        fig.write_html(f'results/plots/heatmap_{m}.html', auto_open=True)
        fig.write_image(f'images/heatmap_{m}.png')

def main():
    observation_period = [100, 500, 1000, 5000]
    sliding_window_factor = [0.005, 0.01, 0.1, 0.2]
    alg_list = ['xStream', 'LODA', 'SWKNN', 'SDOs', 'RSHASH']
    dataset_list = ['comut4', 'comut8', 'comut16', 'insectsAbr',
                'insectsIncr', 'insectsIncrGrd', 'insectsIncrRecr', 'swan']
    # plot_all_boxplots(observation_period, sliding_window_factor, alg_list, dataset_list)
    # obs_plot(observation_period, sliding_window_factor, alg_list, dataset_list)
    # swf_plot(observation_period, sliding_window_factor, alg_list, dataset_list)
    # overall_results(observation_period, sliding_window_factor, alg_list, dataset_list)
    # heatmap(alg_list, dataset_list, observation_period, sliding_window_factor)
    # cont_rate_plot(observation_period, sliding_window_factor, alg_list, dataset_list)
    # ano_length_plot(observation_period, sliding_window_factor, alg_list, dataset_list)
    dim_plot(observation_period, sliding_window_factor, alg_list, dataset_list)
    overall_results(observation_period, sliding_window_factor, alg_list, dataset_list)
if __name__ == '__main__':
    main()