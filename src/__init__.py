from src.utils.data_loaders import *


def get_loader(dataset_name: str, data_path: str) -> object:
    """Returns the loader for the given dataset. Each new dataset should have a 
    custom loader addociated with it that outputs a dataframe with the path to each sample.
    Args:
        dataset_name (str): name of the dataset
        data_path (str): path to the data
    Returns:
        object: loader object
    """
    try:
        loader_class = globals().get(f"{dataset_name}Loader")
        if loader_class is None:
            raise ValueError(
                f"No loader found for '{dataset_name}'! Please create a new loader or test another dataset.")
        return loader_class(data_path=data_path)
    except Exception as e:
        raise e
