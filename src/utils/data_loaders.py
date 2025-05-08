import os

import numpy as np
import pandas as pd

from src.templates.dataset_loader import DatasetLoader
from src.utils import extract_numbers

# Add the dataset loaders here. Don't forget to update __all__.
__all__ = ["comut4Loader", "comut8Loader", "comut16Loader", "insectsAbrLoader", "insectsIncrLoader", "insectsIncrGrdLoader", "insectsIncrRecrLoader",
           "arrhytmiaLoader", "bloodLoader", "swanLoader", "smdLoader", "edfLoader"]


class arrhytmiaLoader(DatasetLoader):

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)

    def get_path_df(self):
        df = pd.read_csv(self.data_path + "arrhythmia.csv")
        df = df.rename(columns={df.columns[-1]: "label"})
        df.to_csv(self.data_path + "arrhythmia_bench.csv", index=False)
        path_df = pd.DataFrame({"path": [self.data_path + "arrhythmia.csv"]})
        return path_df


class comut4Loader(DatasetLoader):
    """Loads the comut dataset. Selects time series based on requested dimensions and contamination rate.

        Args:
            data_path (str): path to the data
            cont_rate (int, optional): contamination rate of the time series. Defaults to 4.
            win (bool, optional): whether to use windows path. Defaults to True.
        return:
            pd.DataFrame: dataframe containing the paths to time series, assumed to be stored in a csv file.
    """

    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.cont_rate = 40

    def get_path_df(self) -> pd.DataFrame:
        files = []
        for root, _, file in os.walk("data/comut4"):
            for f in file:
                files.append(os.path.join(root, f))
        path_df = pd.DataFrame(files, columns=['path'])

        for i, row in path_df.iterrows():
            extract_cont_rate, dimension, sample_nbr = extract_numbers(
                row['path'])
            if int(extract_cont_rate) != self.cont_rate:
                print(f'Anomaly in {row["path"]}')
                path_df.drop(i, inplace=True)
                continue
            else:
                if 'no_anomaly' in row['path']:
                    train_path = row['path']
                    path_df.drop(i, inplace=True)
                    # find indexes of corresponding test and train
                    indexes = path_df[path_df['path'].str.contains(
                        f'comut_{self.cont_rate}x{dimension}_{sample_nbr}')].index
                    path_df.loc[indexes, 'train_path'] = train_path
                else:
                    continue
        return path_df


class comut8Loader(DatasetLoader):
    """Loads the comut dataset. Selects time series based on requested dimensions and contamination rate.

        Args:
            data_path (str): path to the data
            cont_rate (int, optional): contamination rate of the time series. Defaults to 4.
            win (bool, optional): whether to use windows path. Defaults to True.
        return:
            pd.DataFrame: dataframe containing the paths to time series, assumed to be stored in a csv file.
    """

    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.cont_rate = 80

    def get_path_df(self) -> pd.DataFrame:
        files = []
        for root, _, file in os.walk("data/comut8"):
            for f in file:
                files.append(os.path.join(root, f))
        path_df = pd.DataFrame(files, columns=['path'])

        for i, row in path_df.iterrows():
            extract_cont_rate, dimension, sample_nbr = extract_numbers(
                row['path'])
            if int(extract_cont_rate) != self.cont_rate:
                print(f'Anomaly in {row["path"]}')
                path_df.drop(i, inplace=True)
                continue
            else:
                if 'no_anomaly' in row['path']:
                    train_path = row['path']
                    path_df.drop(i, inplace=True)
                    # find indexes of corresponding test and train
                    indexes = path_df[path_df['path'].str.contains(
                        f'comut_{self.cont_rate}x{dimension}_{sample_nbr}')].index
                    path_df.loc[indexes, 'train_path'] = train_path
                else:
                    continue
        return path_df


class comut16Loader(DatasetLoader):
    """Loads the comut dataset. Selects time series based on requested dimensions and contamination rate.

        Args:
            data_path (str): path to the data
            cont_rate (int, optional): contamination rate of the time series. Defaults to 4.
            win (bool, optional): whether to use windows path. Defaults to True.
        return:
            pd.DataFrame: dataframe containing the paths to time series, assumed to be stored in a csv file.
    """

    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.cont_rate = 160

    def get_path_df(self) -> pd.DataFrame:
        files = []
        for root, _, file in os.walk("data/comut16"):
            for f in file:
                files.append(os.path.join(root, f))
        path_df = pd.DataFrame(files, columns=['path'])

        for i, row in path_df.iterrows():
            extract_cont_rate, dimension, sample_nbr = extract_numbers(
                row['path'])
            if int(extract_cont_rate) != self.cont_rate:
                print(f'Anomaly in {row["path"]}')
                path_df.drop(i, inplace=True)
                continue
            else:
                if 'no_anomaly' in row['path']:
                    train_path = row['path']
                    path_df.drop(i, inplace=True)
                    # find indexes of corresponding test and train
                    indexes = path_df[path_df['path'].str.contains(
                        f'comut_{self.cont_rate}x{dimension}_{sample_nbr}')].index
                    path_df.loc[indexes, 'train_path'] = train_path
                else:
                    continue
        return path_df


class insectsAbrLoader(DatasetLoader):
    """Loads the insects dataset. Dataset version from METER (Yoon et al. 2022. https://doi.org/10.1145/3534678.3539348).
    Get it here : https://github.com/zjiaqi725/METER/tree/main/datasets
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)

    def get_path_df(self):
        header = ["feature" + str(i) for i in range(1, 34)] + ["label"]
        path = os.path.join(self.data_path, "INSECTS_Abr.csv")
        save_path = os.path.join(self.data_path, "INSECTS_Abr_bench.csv")
        df = pd.read_csv(path, names=header)
        df.columns = header
        df.insert(0, "timestamp", range(0, len(df)))
        df.to_csv(save_path, index=False)
        path_df = pd.DataFrame(
            {"path": [save_path]})
        return path_df


class insectsIncrLoader(DatasetLoader):
    """Loads the insects dataset. Dataset version from METER (Yoon et al. 2022. https://doi.org/10.1145/3534678.3539348).
    Get it here : https://github.com/zjiaqi725/METER/tree/main/datasets
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)

    def get_path_df(self):
        header = ["feature" + str(i) for i in range(1, 34)] + ["label"]
        path = os.path.join(self.data_path, "INSECTS_Incr.csv")
        save_path = os.path.join(self.data_path, "INSECTS_Incr_bench.csv")
        df = pd.read_csv(path, names=header)
        df.columns = header
        df.insert(0, "timestamp", range(0, len(df)))
        df.to_csv(save_path, index=False)
        path_df = pd.DataFrame(
            {"path": [save_path]})
        return path_df


class insectsIncrGrdLoader(DatasetLoader):
    """Loads the insects dataset. Dataset version from METER (Yoon et al. 2022. https://doi.org/10.1145/3534678.3539348).
    Get it here : https://github.com/zjiaqi725/METER/tree/main/datasets
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)

    def get_path_df(self):
        header = ["feature" + str(i) for i in range(1, 34)] + ["label"]
        path = os.path.join(self.data_path, "INSECTS_IncrGrd.csv")
        save_path = os.path.join(self.data_path, "INSECTS_IncrGrd_bench.csv")
        df = pd.read_csv(path, names=header)
        df.columns = header
        df.insert(0, "timestamp", range(0, len(df)))
        df.to_csv(save_path, index=False)
        path_df = pd.DataFrame(
            {"path": [save_path]})
        return path_df


class insectsIncrRecrLoader(DatasetLoader):
    """Loads the insects dataset. Dataset version from METER (Yoon et al. 2022. https://doi.org/10.1145/3534678.3539348).
    Get it here : https://github.com/zjiaqi725/METER/tree/main/datasets
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)

    def get_path_df(self):
        header = ["feature" + str(i) for i in range(1, 34)] + ["label"]
        path = os.path.join(self.data_path, "INSECTS_IncrRecr.csv")
        save_path = os.path.join(self.data_path, "INSECTS_IncrRecr_bench.csv")
        df = pd.read_csv(path, names=header)
        df.columns = header
        df.insert(0, "timestamp", range(0, len(df)))
        df.to_csv(save_path, index=False)
        path_df = pd.DataFrame(
            {"path": [save_path]})
        return path_df


class bloodLoader(DatasetLoader):
    """Loads the blood-transfusion dataset. You can get it from
    "https://www.kaggle.com/datasets/whenamancodes/blood-transfusion-dataset?resource=download"
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)

    def get_path_df(self):
        df = pd.read_csv(self.data_path)
        if 'label' not in df.columns:
            df = df.rename(
                columns={'whether he/she donated blood in March 2007': 'label'})
            df.to_csv('data\\blood-transfusion\\transfusion.csv', index=True)
        path_df = pd.DataFrame(
            {"path": [os.path.join(self.data_path, "blood-transfusion", "transfusion.csv")]})
        return path_df


class swanLoader(DatasetLoader):
    """Loads the swan dataset. Preprocessing was done according to the instruction from
    "https://github.com/CN-TU/py-outlier-detection-stream-data".
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)

    def get_path_df(self):

        from scipy.io.arff import loadarff
        arffdata = loadarff(os.path.join("data", "swan", "swan.arff"))
        df_data = pd.DataFrame(arffdata[0])
        if (df_data['class'].dtypes == 'object'):
            df_data['class'] = df_data['class'].map(
                lambda x: x.decode("utf-8").lstrip('b').rstrip(''))

        df_data['class'].rename('label', inplace=True)
        # remove the att11 and att12 column
        df_data.drop(columns=['att11', 'att12'], inplace=True)
        df_data.to_csv(os.path.join(
            "data", "swan", "swan_bench.csv"), index=True)
        path_df = pd.DataFrame(
            {"path": [os.path.join(
                "data", "swan", "swan_bench.csv"), os.path.join(
                "data", "swan", "swan.csv")]})
        return path_df


class pathLoader(DatasetLoader):
    """From https://github.com/lcs-crr/PATH. Paper: https://arxiv.org/abs/2411.13951
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)
        print("PATH dataset is not implemented")

    def get_path_df(self):
        pass


class smdLoader(DatasetLoader):
    """SMD dataset, get it from https://github.com/NetManAIOps/OmniAnomaly/tree/master. Selection and preprocessing according to 
    Wagner et al, TimeSeAD: Benchmarking Deep Multivariate Time-Series Anomaly Detection, 2023. link: https://openreview.net/forum?id=iMmsCI0JsS
    """

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)

    def get_path_df(self):
        FILENAMES = [
            'machine-1-2.txt',
            'machine-1-7.txt',
            'machine-2-1.txt',
            'machine-2-2.txt',
            'machine-2-3.txt',
            'machine-2-4.txt',
            'machine-2-6.txt',
            'machine-2-7.txt',
            'machine-2-9.txt',
            'machine-3-1.txt',
            'machine-3-2.txt',
            'machine-3-3.txt',
            'machine-3-6.txt',
            'machine-3-8.txt',
            'machine-3-9.txt'
        ]
        data_path = 'data'
        path_df = []
        for root, _, file in os.walk(os.path.join(data_path, 'smd/test')):
            for f in file:
                if f in FILENAMES:
                    df = pd.read_csv(os.path.join(root, f), sep=',', header=None)
                    # compute autocorrelation and remove features with low autocorrelation
                    for column in df.columns:
                        if df[column].autocorr() < 0.1:
                            df.drop(column, axis=1, inplace=True)
                    labels = pd.read_csv(os.path.join(root + '_label', f), header=None)
                    df['label'] = labels
                    df.insert(0, 'timestamp', df.index)
                    filename = f.replace('.txt', '.csv')
                    save_path = os.path.join(data_path, 'smd/bench')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    df.to_csv(os.path.join(save_path,filename), index=False, header=True)
                    path_df.append(os.path.join(save_path,filename))
        path_df = pd.DataFrame(path_df, columns=['path'])
        return path_df

class edfLoader(DatasetLoader):

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path)

    def get_path_df(self):
        df_rupture = pd.read_csv(os.path.join('data','edf','UC_rupture.csv'), sep=';')
        df_rupture['times'] = pd.to_datetime(df_rupture['times'], dayfirst=True)
        # order by time
        df_rupture = df_rupture.sort_values('times')
        df_rupture = df_rupture[(df_rupture['times'] > '1995-05-10') & (df_rupture['times'] < '2022-12-27')] # select ranges where data is relevant
        df_rupture = df_rupture.drop(columns=['Sensor_2'])
        df_rupture = df_rupture.drop(columns=['Sensor_4'])
        df_rupture = df_rupture.drop(columns=['Sensor_3'])
        # back fill
        df_rupture.bfill(inplace=True)

        # add label column, from 04/2011 to 31/10/2011
        df_rupture['label'] = 0
        df_rupture.loc[(df_rupture['times'] > '2011-04-01') & (df_rupture['times'] < '2011-10-31'), 'label'] = 1
        df_rupture.loc[(df_rupture['times'] > '2011-10-31'), 'label'] = 0

        # add timestamp column to the left
        df_rupture.insert(0, 'timestamp', range(0, len(df_rupture)))
        
        # drop time
        df_rupture.drop(columns=['times'], inplace=True)

        # save to csv
        df_rupture.to_csv('data/edf/edf_bench.csv', index=False)
        path_df = pd.DataFrame({"path": ['data/edf/edf_bench.csv']})
        return path_df
