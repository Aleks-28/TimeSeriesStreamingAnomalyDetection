from collections import defaultdict

import numpy as np
import pandas as pd


class Statistics_extractor():
    """Compute statistics from a dataset
    :param df: pandas dataframe
    """

    def __init__(self, path_df: pd.DataFrame) -> None:
        """Constructor for the Statistics_extractor class.

        Args:
            path_df (pd.DataFrame): dataframe containing the paths to the data
            dataset_name (str): name of the dataset
        Returns:
            None
        """
        super().__init__()
        self.dimensions = defaultdict(int)
        self.ts_lengths = defaultdict(int)
        self.cont_rate = defaultdict(float)
        self.ano_lengths = defaultdict(int)
        self.ano_types = defaultdict(int)
        self.ano_pos = []
        self.mad_dict = defaultdict(list)
        self.drift = defaultdict(list)
        self.path_df = path_df
        self.stats_df = path_df.copy()
        self.num_samples = len(path_df)
        self.stats = {
            'dimensions': self.dimensions,
            'num_samples': self.num_samples,
            'ts_lengths': self.ts_lengths,
            'cont_rate': self.cont_rate,
            'ano_lengths': self.ano_lengths,
            'ano_types': self.ano_types,
            'ano_pos': self.ano_pos,
            'mad': self.mad_dict,
            'drift': self.drift
        }

    def _update_dimensions(self, column: pd.Series) -> 'Statistics_extractor':
        """Update the number of dimensions. Its is assumed that the first column is
        the timestamp and the last the label.

        args:
            column (pd.Series): pandas series

        return:
            self
        """
        self.dimensions[len(column)-2] += 1
        return self

    def _update_ano_type_length(self, anomaly_mask: list) -> tuple:
        """
        Detects anomalies in a time series based on a boolean anomaly mask and classifies them as
        point anomalies or anomalous subsequences.

        Parameters:
        - anomaly_mask: array-like, boolean mask where True indicates an anomaly

        Returns:
        - anomalies: list of tuples (start, end), where start == end for point anomalies
        """
        anomaly_indices = np.where(anomaly_mask)[0]
        # Group consecutive anomalies into subsequences
        anomalies = []
        start = None
        for i in range(len(anomaly_indices)):
            if start is None:
                start = anomaly_indices[i]
            if i == len(anomaly_indices) - 1 or anomaly_indices[i] + 1 != anomaly_indices[i + 1]:
                end = anomaly_indices[i]
                anomalies.append((start, end))
                start = None
        for start, end in anomalies:
            length = end - start + 1
            if length <= 1:
                self.ano_types["point"] += 1
                self.ano_lengths["1"] += 1
            else:
                self.ano_types["subseq"] += 1
                self.ano_lengths[f"{length}"] += 1
        return anomalies

    def _update_ano_pos(self, anomaly_mask: list, sample: int) -> 'Statistics_extractor':
        """Updates the anomaly position dict

        args:
            anomaly_mask (list): list of indexes where the anomalies are located
            ano_pos (dict): dictionary to store the position of the anomalies

        return:
            dict: updated dictionary
        """
        for idx in anomaly_mask:
            relative_pos = [round(idx/len(sample), 2)]
            self.ano_pos.extend(relative_pos)
        return self

    @ staticmethod
    def _mean_absolute_deviation(s: pd.Series) -> list:
        """Compute the Median Absolute Deviation a data series.
        args:
            s (pd.Series): pandas series
        return:
            list: list of the median absolute deviation
        """
        return float(round(np.mean(np.abs(s - np.mean(s))), 3))

    def _update_ts_lengths(self, length: int) -> 'Statistics_extractor':
        """Update the number of time series lengths
        args:
            dict (defaultdict): dictionary containing the number of time series lengths
            length (int): length of the time series
        return:
            defaultdict: updated dictionary
        """
        self.ts_lengths[length] += 1
        return self

    def _update_cont_rate(self, anomaly_mask: list, df: pd.DataFrame) -> 'Statistics_extractor':
        """Update the contamination rate
        args:
            dict (defaultdict): dictionary containing the contamination rate
            anomaly_mask (list): list of indexes where the anomalies are located
            df (pd.Dataframe): pandas dataframe
        return:
            defaultdict: updated dictionary
        """
        self.cont_rate[len(anomaly_mask)/len(df)] += 1
        return self

    def _update_stats(self, samples: bool = False) -> None:
        """Update the stats dict for the dataset.
        args:
            None
        return:
            None
        """
        if samples:
            self.stats = {
                'dimensions': self.dimensions,
                'num_samples': self.num_samples,
                'ts_lengths': self.ts_lengths,
                'cont_rate': self.cont_rate,
                'ano_lengths': self.ano_lengths,
                'ano_types': self.ano_types,
                'ano_pos': self.ano_pos,
                'mad': self.mad_dict,
                'drift': None
            }
        else:
            self.stats = {
                'dimensions': self.dimensions,
                'ts_lengths': self.ts_lengths,
                'cont_rate': self.cont_rate,
                'ano_lengths': self.ano_lengths,
                'ano_types': self.ano_types,
                'ano_pos': self.ano_pos,
                'mad': self.mad_dict,
                'drift': None
            }

    def _retrieve_df_stats(self, anomaly_mask: list, row_idx: int, stats: dict, current_ts: pd.DataFrame) -> 'Statistics_extractor':
        """Creates new columns for the stats dataframe to accomodate for each 'global 'statistic:
        - min_anomaly_length
        - max_anomaly_length
        - median_anomaly_length
        - avg_anomaly_length
        - cont_rate

        args:
            anomaly_mask (list): list of indexes where the anomalies are located
            row_idx (int): row index of the dataframe
            stats (dict): dictionary containing the statistics
            current_ts (pd.DataFrame): pandas dataframe

        return:
            self
        """
        self.stats_df.at[row_idx, "min_anomaly_length"] = min(
            keys for keys, _ in stats["ano_lengths"].items())
        self.stats_df.at[row_idx, "max_anomaly_length"] = max(
            keys for keys in stats["ano_lengths"].keys())
        self.stats_df.at[row_idx, "median_anomaly_length"] = np.median(
            [int(key) for key in stats["ano_lengths"].keys()])
        self.stats_df.at[row_idx, "avg_anomaly_length"] = np.mean(
            [int(key) for key in stats["ano_lengths"].keys()])
        self.stats_df.at[row_idx, "cont_rate"] = len(
            anomaly_mask)/len(current_ts)
        return self

    def _get_mad(self, df: pd.DataFrame) -> float:
        """Compute the Median Absolute Deviation of a data series for each dimension,
        assuming the first dimension is the timestamp and the last the label.
        args:
            df (pd.DataFrame): pandas dataframe
        return:
            None
        """
        df = df.iloc[:, 1:-1]
        for column in df.columns:
            mad_value = self._mean_absolute_deviation(df[column])
            self.mad_dict[column].append(mad_value)
        return None

    def _drift_detection(self, df: pd.DataFrame) -> float:
        """== Not implemented ==
        Compute the drift detection for a given time series using ADWIN from the river library.
        args:
            df (pd.DataFrame): pandas dataframe
        return:
            dict: dictionary containing the drift detection
        """
        i = 0
        # number of columns in df
        num_columns = len(df.columns)
        self.drift = defaultdict(lambda: [False]*num_columns)
        # Create a drift detector
        # adwin = drift.ADWIN()
        # iterate over each dimensions except the the label
        # for column in df.columns[:-1]:
        #     df_temp = df[column]
        #     for idx, value in df_temp.items():
        #         adwin.update(value)
        #         if adwin.drift_detected:
        #             self.drift[f"{idx}"][i] = True
        #     i += 1
        return "Not implemented yet"

    def get_stats(self) -> 'Statistics_extractor':
        """Update the instance with computed statistics for the dataset:
        - Number of dimensions
        - Number of samples
        - Time series lengths
        - Contamination rate
        - Anomaly lengths
        - Anomaly types
        - Anomaly position
        - Median Absolute Deviation
        args:
            None
        return:
            Statistics_extractor: updated instance of the class
        """
        for idx, row in self.path_df.iterrows():
            sample = pd.read_csv(row["path"])
            anomaly_mask = sample[sample.iloc[:, -1] == 1].index.to_list()
            self._update_dimensions(sample.columns)
            self._update_ts_lengths(len(sample))
            self._update_cont_rate(anomaly_mask, sample)
            self._update_ano_type_length(anomaly_mask)
            self._update_ano_pos(anomaly_mask, sample)
            self._get_mad(sample)
            # self._drift_detection(sample)
            self._update_stats()
            self._retrieve_df_stats(anomaly_mask, idx, self.stats, sample)

        self._update_stats(samples=True)
        return self

    def get_stats_df(self) -> pd.DataFrame:
        """Get the stats dataframe: the stats dataframe contains the paths to the data and
        some basic stats associated with it. For more details, please refer to the
        _retrieve_df_stats method.
        args:
            None
        return:
            stats_df (pd.DataFrame): dataframe containing the stats
        """
        if self.stats_df is None:
            raise ValueError(
                "Stats dataframe is not defined. Please run the get_stats method first.")
        else:
            return self.stats_df

    def get_stats_dict(self) -> dict:
        """Get the stats dict
        args:
            None
        return:
            stats (dict): dictionary containing the stats
        """
        return self.stats
