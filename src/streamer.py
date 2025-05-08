
from typing import Iterator, Tuple
import numpy as np


class Streamer():
    """Class that contains the methods for streaming the data in windows of size window_size.

    Attributes:
        data: numpy array of shape (n_samples, n_features)
            The data to be streamed.
        labels: numpy array of shape (n_samples,)
            The labels of the data.
        window_size: int
            The size of the sliding window
    """

    def __init__(self, ts_test: np.ndarray, labels: np.ndarray, window_size: int) -> None:
        self.ts_test = ts_test
        self.labels = labels
        self.window_size = window_size

    def iterate_window(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generator that iterates over the data in windows of size window_size 
        and return the window data and labels as numpy arrays.

        Args:
            None

        Yields:
            window_data: numpy array of shape (window_size, n_features)
                The window data.
            window_labels: numpy array of shape (window_size,)
                The window labels.
        """
        num_samples = len(self.ts_test)
        for start_idx in range(0, num_samples, self.window_size):
            end_idx = start_idx + self.window_size
            window_data = self.ts_test[start_idx:end_idx]
            window_labels = self.labels[start_idx:end_idx]
            # Make labels numpy.uint8
            window_labels = window_labels.astype('uint8')
            yield window_data, window_labels

    @staticmethod
    def iterate_point(window: np.ndarray, labels: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generator that iterates over the data in a window and return the data and labels as numpy arrays.

        Args:
            window (_type_): window data
            labels (_type_): window labels

        Yields:
            data: data point
            label: label point
        """
        for data, label in zip(window, labels):
            yield data, label
