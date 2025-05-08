import importlib
import inspect
import os
import pickle
import time

import numpy as np
from tqdm import tqdm

from src.datasets import Dataset
from src.models import *
from src.streamer import Streamer
from src.templates.metric import Metric
from src.templates.model import Model
from src.utils import compute_avg_scores
from src.utils.plotter import Plot

np.random.seed(0)


class Runner():
    """Runner class for evaluating the model on the dataset.

    Attributes:
        model (Model): Model to be evaluated
        metrics (str): Metric or metrics to be used for evaluation
        dataset (Dataset): Dataset to be used for evaluation. Needs a dataloader in src/utils/data_loaders.py.
        fit_portion (float): Portion of the dataset to be used for training
        observation_period (int): Size of the sliding window used for training in number of points
        runs (int): Number of runs for the evaluation
        metrics (list): List of metric classes
        plot (bool): Whether to plot the anomaly scores
        sliding_window_factor (float): Portion of the total sample length to be used as the sliding window during predict
        runs_res (dict): Dictionary containing the scores for each metric, the anomaly scores for each sample and the throughput for each sample
        score_dict (dict): Dictionary containing the scores for each metric
        anomaly_score_dict (dict): Dictionary containing the anomaly scores for each sample
        throughput (dict): Dictionary storing the time taken to compute the anomaly scores for each sample for each window.

    Methods:
        _load_metrics: Loads all the metrics in the metrics module.
        evaluate: Evaluates the model on the dataset.
        _init_score_dict: Initializes the score dictionary by assigning an empty list to each metric instance.
        _init_evaluation: Initializes the evaluation process.
        _fit_score: Predicts the anomaly scores on the data stream, updates the model and add current points and labels to metric instance for later metrics computation.
        _plot_anomaly_score: Plots the anomaly scores and corresponding time series using the Plotter class.
        _init_loading_bar: Initializes a loading bar.
        _compute_score: Computes the score for each metric instance.
        _save_res: Calculates the average score over each runs and saves the dictionnary result as pickle file.
    """

    def __init__(self, model: Model, dataset: Dataset, observation_period: int = 100, fit_portion: float = 0.2, runs: float = 1, **kwargs) -> None:

        self.params = kwargs if kwargs else {}
        model = globals()[model]
        self.model = model(observation_period, **kwargs)
        self.dataset = dataset
        self.metrics = self._load_metrics()
        self.fit_portion = fit_portion
        self.observation_period = observation_period
        self.runs = runs
        self.plot = kwargs.get('plot', False)
        self.sliding_window_factor = kwargs.get('sliding_window_factor', 0.01)
        self.runs_res = {}
        self.score_dict = {}
        self.anomaly_score_dict = {}
        self.throughput = {}

    def _load_metrics(self) -> list:
        """Loads all the metrics in the metrics module.

        Args:
            None

        Returns:
            metrics: list of metric classes
        """
        loaded_metrics = []
        metrics_module = importlib.import_module('src.metrics')
        for _, obj in inspect.getmembers(metrics_module):
            if inspect.isclass(obj) and issubclass(obj, Metric) and obj is not Metric:
                loaded_metrics.append(obj())
        print("\n" + "-" * 100)
        print(
            f"Loaded metrics: {[str(loaded_metrics) for loaded_metrics in loaded_metrics]}")
        return loaded_metrics

    def evaluate(self) -> dict:
        """Evaluate the model on the dataset.

        Args:
            None

        Returns:
            runs_res (list[dict]): list of dictionaries with one key per run named 'run_i' where i is the run number, and one key named 'avg_scores' containing the averaged metrics over all runs.: 
                runs_res['run_i'] contains a list with three elements:
                    1) Metrics score (dict): a dictionary with the metric name as key and the score as value
                    2) Anomaly scores (dict): ano_score_sample_i a dictionary with the sample number as key and a list with the anomaly scores for each data point as value
                    3) Throughput (dict): a dictionary with the sample number as key and a list with the time taken to compute the anomaly scores for each window as value
        """
        for run in range(self.runs):
            total_ts = sum(1 for _ in self.dataset.get_ts(self.fit_portion))
            outer_pbar = self._init_loading_bar(
                total_ts, 'Dataset progression', 0, True)
            self._init_score_dict()
            for sample_nbr, ts_data, ts_label, ts_train, ts_test, ts_label_train, ts_label_test in self.dataset.get_ts(self.fit_portion):

                # Initialize the evaluation process
                self, data_stream = self._init_evaluation(
                    sample_nbr, ts_data, ts_train, ts_test, ts_label_train, ts_label_test, run)

                # Initialize the loading bar
                total_iterations = sum(
                    1 for _ in data_stream.iterate_window())
                inner_pbar = self._init_loading_bar(
                    total_iterations, 'Detecting anomalies', 1, False)

                # Fit the model on the data stream and calculate the anomaly scores and throughput
                self._fit_score(sample_nbr, data_stream, inner_pbar)
                inner_pbar.close()
                self._compute_score()

                if self.plot == True:
                    self._plot_anomaly_score(
                        sample_nbr, ts_data, ts_label, run, self.runs)

                outer_pbar.update(1)
            outer_pbar.close()
            self.runs_res[f"run_{run}"] = [
                self.score_dict, self.anomaly_score_dict, self.throughput]
            print("\n" + "-" * 100)
            print(
                f"\n Run {run} completed \n")
        self._save_res()
        return self.runs_res

    def _init_score_dict(self) -> 'Runner':
        """Initializes the score dictionary by assigning an empty list to each metric instance.
        """
        for metric_instance in self.metrics:
            self.score_dict[str(metric_instance)] = []
        return self

    def _init_evaluation(self, sample_nbr: int, ts_data: np.ndarray, ts_train: np.ndarray, ts_test: np.ndarray, ts_label_train: np.ndarray, ts_label_test: np.ndarray, run: int) -> tuple:
        """Initializes the evaluation process :
        1. Fits the model on the train portion.
        2. Initializes the data stream. 
        3. Initializes the metric instances. 

        Args:
            sample (int): Sample number
            ts_data (np.ndarray): Time series data to be evaluated of dimension (n_samples, n_features).
            ts_label (np.ndarray): Time series labels (n_samples,).

        Returns:
            self: Runner object
            data_stream: Streamer object
        """
        self.anomaly_score_dict["ano_score_sample_" + str(sample_nbr)] = []
        self.throughput["thgrpt_sample_" + str(sample_nbr)] = []
        window_size = int(self.sliding_window_factor*ts_data.shape[0])
        # Initialize the model for each sample
        self.model.__init__(
            observation_period=self.observation_period, run=run, window_size=window_size, **self.params)
        self.model.train(X_train=ts_train,
                         labels=ts_label_train, data=ts_data, window_size=window_size)
        data_stream = Streamer(
            ts_test=ts_test, labels=ts_label_test, window_size=window_size)
        for metric_instance in self.metrics:
            metric_instance.__init__()
        return self, data_stream

    def _fit_score(self, sample_nbr: int, data_stream: Streamer, inner_pbar: tqdm) -> 'Runner':
        """Predicts the anomaly scores on the data stream, updates the model and add current points
        and labels to metric instance for later metrics computation.

        Args:
            sample_nbr (int): Sample number
            data_stream (Streamer): Streamer object
            inner_pbar (tqdm): tqdm object

        Returns:
            self: Runner object
        """
        for window, window_labels in data_stream.iterate_window():
            start_time = time.time()
            anomaly_score = self.model.predict(X=window)
            self.model.update(X=window)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.anomaly_score_dict["ano_score_sample_" + str(sample_nbr)].extend(
                anomaly_score)
            self.throughput["thgrpt_sample_" + str(sample_nbr)].append(
                elapsed_time)
            for metric_instance in self.metrics:
                metric_instance.update(window_labels, anomaly_score)
            inner_pbar.update(1)
        return self

    def _plot_anomaly_score(self, sample_nbr: int, ts_data: np.ndarray, ts_label: np.ndarray, run: int, runs: int) -> None:
        """Plots the anomaly scores and corresponding time series using the Plotter class.

        Args:
            ts_data (np.ndarray): Time series data of dimension (n_samples, n_features).
            ts_label (np.ndarray): Time series labels (n_samples,).
            sample_nbr (int): Sample number

        Returns:
            None
        """
        plotter = Plot(self.dataset, self.anomaly_score_dict,
                       ts_data, ts_label, self.fit_portion, self.observation_period, self.model, self.params, self.score_dict)
        plotter.plot_anomaly_scores(sample_nbr, run, runs)
        return None

    def _init_loading_bar(self, total, desc, position, leave) -> 'tqdm':
        """Initializes a loading bar.

        Args:
            total: total number of iterations
            desc: description of the loading bar
            position: position of the loading bar

        Returns:
            tqdm: tqdm object
        """
        return tqdm(total=total, desc=desc, position=position, leave=leave)

    def _compute_score(self) -> 'Runner':
        """Computes the score for each metric instance.

        Args:
            ts_data (np.ndarray): Current evaluated time series data of dimension (n_points, n_features).

        Returns:
            self: Runner object
        """
        try:
            for _, metric_instance in enumerate(self.metrics):
                score = metric_instance.get_score(vus_window=5)
                print(f"\n{str(metric_instance)}: {score}")
                self.score_dict[str(metric_instance)].append(score)
        except Exception as e:
            print(f"\nUndefined score: {e}")
            self.score_dict[str(metric_instance)].append(None)
        return self

    def _save_res(self):
        """Calculates the average score over each runs and saves the dictionnary result as pickle file.

        Args:
            None

        Returns:
            None
        """
        try:
            avg_scores = compute_avg_scores(self.runs, self.runs_res)
            self.runs_res["avg_scores"] = avg_scores
        except Exception as e:
            print(f"\nError computing average scores: {e}")
        save_folder = os.path.join('results', 'pkl')
        try:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            filename = f"{str(self.model.__class__.__name__)}_{str(self.dataset)}_window_{str(self.observation_period)}_{self.sliding_window_factor}.pkl"
            save_path = os.path.join(save_folder, filename)
            with open(f'{save_path}', 'wb') as f:
                pickle.dump(self.runs_res, f)
            print(f"\nResults saved in {save_path}")
        except Exception as e:
            print(f"\nError saving results: {e}")
        return None
