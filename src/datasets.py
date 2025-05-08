import os
import pprint

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src import get_loader
from src.statistics_extractor import Statistics_extractor


class Dataset():
    """ Wrapper class for handling datasets
    Attributes:
        dataset_name (str): name of the dataset
        datapath (str): path to the dataset
        path_df (pd.DataFrame): dataframe used for storing the paths to each sample (.csv file).
        stats (bool): whether to compute statistics on the dataset. See Statistics_extractor for details.
        stats_df (pd.DataFrame): dataframe with general statistics.
        nbr_samples (tuple): number of samples in the dataset.
        _stats (dict): dictionary with statistics.

    Methods:
        __init__(dataset_name: str, stats=False, datapath: str = 'data'): Constructor
        __repr__(): Representation of the class
        __str__(): String representation of the class
        _get_path_df() -> pd.DataFrame: Load the dataset and return a dataframe with paths to the data.
        _compute_statistics(): Compute statistics on the dataset.
        dimensions(): Return the number of dimensions in the dataset.
        num_samples(): Return the number of samples in the dataset.
        ts_lengths(): Return the lengths of the time series in the dataset.
        cont_rate(): Return the rate of continuous data in the dataset.
        ano_lengths(): Return the lengths of the anomalies in the dataset.
        ano_types(): Return the types of the anomalies in the dataset.
        ano_pos(): Return the positions of the anomalies in the dataset.
        mad(): Return the mean absolute deviation of the dataset.
        drift(): Return the drift of the dataset.
        print_stats(): Print the statistics of the dataset.
        get_stats_dict(): Return the statistics of the dataset as a dictionary.
        get_distribs() -> dict: Return the distributions of the dataset in a dict.
        get_ts(fit_portion: float = 0.2): Generator that yields the time series data and labels as (n_samples, n_features).
    """

    def __init__(self, dataset_name: str, stats=False, datapath: str = 'data'):

        self.dataset_name = dataset_name
        self.datapath = os.path.join(datapath, dataset_name)
        self.path_df = self._get_path_df()
        self.nbr_samples = len(self.path_df)
        if stats:
            self._stats, self.stats_df = self._compute_statistics()

    def __repr__(self):
        return f"Dataset({self.dataset_name})"

    def __str__(self):
        return f"{self.dataset_name}"

    def _get_path_df(self) -> pd.DataFrame:
        """Load the dataset and return a dataframe with paths to the data.
        Uses a data_loader that must be customized for each dataset.
        Args:
            None
        Returns:
            path_df (pd.DataFrame): dataframe with paths to the data
        """
        data_loader = get_loader(
            self.dataset_name, self.datapath)
        path_df = data_loader.get_path_df()
        return path_df

    def _compute_statistics(self):
        stats = Statistics_extractor(self.path_df)
        stats.get_stats()
        return stats.get_stats_dict(), stats.get_stats_df()

    @property
    def dimensions(self):
        return self._stats['dimensions']

    @property
    def num_samples(self):
        return self._stats['num_samples']

    @property
    def ts_lengths(self):
        return self._stats['ts_lengths']

    @property
    def cont_rate(self):
        return self._stats['cont_rate']

    @property
    def ano_lengths(self):
        return self._stats['ano_lengths']

    @property
    def ano_types(self):
        return self._stats['ano_types']

    @property
    def ano_pos(self):
        return self._stats['ano_pos']

    @property
    def mad(self):
        return self._stats['mad']

    @property
    def drift(self):
        return self._stats['drift']

    def print_stats(self):
        pprint.pprint(self._stats)

    def get_stats_dict(self):
        return self._stats

    def get_distribs(self) -> dict:
        """Return the distributions of the dataset in a dict
        """
        return {
            'dimensions': self.dimensions,
            'ts_lengths': self.ts_lengths,
            'cont_rate': self.cont_rate,
            'ano_lengths': self.ano_lengths,
            'ano_types': self.ano_types,
            'ano_pos': self.ano_pos,
            'mad': self.mad,
            'drift': self.drift
        }

    def get_ts(self, fit_portion: float = 0.2):
        """Generator that yields the time series data and labels for training and testing.
        It is assumed that the first column of the input data is the timestamp and the last column is the label.

        Args:
            fit_portion (float): portion of the dataset to use for training

        Yields:
            idx (int): index of sample in the dataset
            ts_data (np.array): time series data
            ts_label (np.array): time series labels
            ts_train (np.array): time series training data
            ts_test (np.array): time series testing data
            ts_label_train (np.array): time series training labels
            ts_label_test (np.array): time series testing
        """
        for idx, row in self.path_df.iterrows():
            ts_data = pd.read_csv(row["path"])
            train_len = int(len(ts_data) * fit_portion)
            ts_data = ts_data.to_numpy().astype('float64')
            ts_label = ts_data[:, -1].astype('int')
            ts_data = MinMaxScaler().fit_transform(ts_data)
            ts_label_train = ts_data[:train_len, -1].astype('int')
            ts_label_test = ts_data[train_len:, -1].astype('int')
            ts_test = ts_data[train_len:, 1:-1].astype('float64')
            if "train_path" in row:
                ts_train = pd.read_csv(row["train_path"])
                ts_train = ts_train.to_numpy().astype('float64')
                ts_train = MinMaxScaler().fit_transform(ts_train)
                ts_train = ts_train[:train_len, 1:-1]
            else:
                ts_train = ts_data[:train_len, 1:-1].astype('float64')
            ts_data = ts_data[:, 1:-1]
            yield idx, ts_data, ts_label, ts_train, ts_test, ts_label_train, ts_label_test

    
