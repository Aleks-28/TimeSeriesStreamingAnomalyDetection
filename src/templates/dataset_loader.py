import pandas as pd

from abc import ABC, abstractmethod


class DatasetLoader(ABC):
    """Abstract class for loading datasets. It should be inherited by all dataset loaders in data_loader.py.
    One data loader must be implemented for each dataset. The data loader should be named {dataset_name}Loader, where 
    {dataset_name} is the name of the folder containing the dataset, in data/.

    Attributes:
        data_path (str): path to the dataset.

    Methods:
        get_path_df: Abstract method for handling datasets. It should output a pandas dataframe containing the paths to the time series.
    """

    def __init__(self, data_path: str,):
        self.data_path = data_path

    @abstractmethod
    def get_path_df(self) -> pd.DataFrame:
        """Abstract method for handling datasets. It should output 
        a pandas dataframe containing the paths to the time series. Also where the datasets should be formatted according
        to the following instructions :
        - Each sample should be stored in a csv file.
        - The first column should contain the timestamps.
        - The last column of the csv file should contain the labels.

        args:
            None

        return:
            pd.DataFrame: dataframe containing the paths to time series, assumed to be stored in a csv file.
        """
        pass
