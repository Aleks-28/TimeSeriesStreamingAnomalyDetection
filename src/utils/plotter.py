from pydoc import text
import re
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import get_anomalies_ranges

__all__ = ["Plot"]


class Plot:
    """Class for plotting the anomaly scores and time series data with marked anomalies.

    Attributes:
        dataset (Dataset): Dataset object
        anomaly_scores (dict): Dictionary containing the anomaly scores for each sample
        ts_data (np.ndarray): Time series data
        ts_label (np.ndarray): Time series labels
        fit_portion (float): Portion of the data used for fitting the model
        window_size (int): Size of the sliding window
        model (Model): Model object
        params (dict): Model parameters
        score_list (list): List of anomaly scores

    Methods:
        plot_stats: Plot the statistics of the dataset
        plot_anomaly_scores: Plot the anomaly scores and corresponding time series with marked anomaly for a given sample number
    """

    def __init__(self, dataset, anomaly_score_list, ts_data, label, fit_portion, window_size, model, params, score_list):
        self.dataset = dataset
        self.anomaly_scores = anomaly_score_list
        self.ts_data = ts_data
        self.ts_label = label
        self.fit_portion = fit_portion
        self.window_size = window_size
        self.model = model
        self.params = params
        self.score_list = score_list

    def plot_stats(self):
        distrib_dict = self.dataset.get_distribs()
        # Create a figure
        print(distrib_dict)
        for distrib in distrib_dict.keys():
            print(distrib)
            # Add traces for each key in the data
            fig = go.Figure()
            if distrib == "mad":

                fig = make_subplots(rows=1, cols=len(distrib_dict[distrib].keys()), subplot_titles=[
                                    f"{key}" for key in distrib_dict[distrib].keys()])
                for idx, (key, values) in enumerate(distrib_dict[distrib].items(), start=1):
                    fig.add_trace(
                        go.Histogram(
                            x=values,
                            name=key,
                        ),
                        row=1,
                        col=idx,
                    )

            if distrib == "ano_pos":
                # order the dictionary by key ascending
                distrib_dict[distrib] = dict(
                    sorted(distrib_dict[distrib].items()))
                for key in distrib_dict[distrib].keys():
                    value = [distrib_dict[distrib][key]]
                    fig.add_trace(go.Bar(x=[str(key)], y=value))
            else:
                for key in distrib_dict[distrib].keys():
                    value = [distrib_dict[distrib][key]]
                    print(value)
                    fig.add_trace(go.Bar(x=[str(key)], y=value))

            fig.update_layout(
                xaxis_title=f"{distrib}",
                yaxis_title="Number of samples",
                barmode="group",
            )

            fig.show(renderer="browser")

    def plot_anomaly_scores(self, sample_nbr: int, run: int, tot_runs: int) -> None:
        """ Plot the anomaly scores and corresponding time series with marked anomaly for a given sample number.
        Opens a local window with the plot and saves the plot as an html file.

        Args:
            sample_nbr (int): sample number to plot the anomaly scores for
            run (int): run number

        Returns:
            None
        """
        save_path_html = "results/html/"
        save_path_png = "results/images/"
        dimension = len(self.ts_data[1])
        df = pd.DataFrame(self.ts_data)
        ts_label = pd.DataFrame(self.ts_label)
        anomalies_index = ts_label[ts_label[0] == 1].index
        anomaly_ranges = get_anomalies_ranges(anomalies_index)
        fig = make_subplots(rows=dimension + 1, cols=1,
                            shared_xaxes=True, vertical_spacing=0.02)

        for i in range(dimension, 0, 3):
            print(i)
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df.iloc[:, i], mode='lines', name=f'value-{i}'),
                row=i+1, col=1
            )
            for anomaly in anomaly_ranges:
                print(f"Anomaly: {anomaly}")
                fig.add_shape(
                    type="rect",
                    x0=df.index[anomaly[0]], y0=min(df.iloc[:, i]), x1=df.index[anomaly[1]], y1=max(df.iloc[:, i]),
                    line=dict(color="Red"),
                    fillcolor="Red",
                    opacity=0.2,
                    row=i+1, col=1,
                    layer="below",
                    name='anomaly' if i == 0 else None
                )
        anomaly_score_list = self.anomaly_scores["ano_score_sample_" + str(
            sample_nbr)]
        start_index = int(self.fit_portion * len(df))

        fig.add_trace(
            go.Scatter(x=df.index[start_index:],
                       y=anomaly_score_list, mode="lines", name="anomaly score"),
            row=dimension+1, col=1
        )

        params = [self.params if self.params else "Default"]
        fig.update_layout(
            height=1000, title_text=f"Anomaly score Visualization: run {run}/{tot_runs} | {self.model.__class__.__name__} on {str(self.dataset)} | "
                                    f"sample {sample_nbr} | Params: {params}, Window size: {self.window_size} | AUCROC: {self.score_list['AUCROC'][sample_nbr]}, "
                                    f"AUCPR: {self.score_list['AUCPR'][sample_nbr]}"
        )

        sanitized_params = re.sub(r'[<>:"/\\|?*]', '_', str(self.params))
        # make sure the path exists
        if not os.path.exists(save_path_html):
            os.makedirs(save_path_html)
        fig.write_html(
            f"{save_path_html}{str(self.dataset)}_{str(self.model.__class__.__name__)}_run_{run}_p_{sanitized_params}_w_{str(self.window_size)}_{sample_nbr}.html")
        # fig.show(renderer="browser")
        return print(f"Anomaly score plot saved at {save_path_html} and {save_path_png}")
