import argparse

from src.datasets import Dataset
from src.runner import Runner
from src.utils import display_results

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(runs: int = 1, model: str = "xStream", dataset_name: str = "swan", observation_period: int = 100, sliding_window_factor: int = 0.01, plot: bool = False) -> None:
    """Runs the benchmark for the given model and dataset.
    Args:
        runs (int, optional): Number of runs for the evaluation. Defaults to 1.
        model (str, optional): Model to be used for the evaluation. Defaults to "xStream".
        dataset_name (str, optional): Dataset to be used for the evaluation. Defaults to "swan".
        observation_period (int, optional): Size of the sliding window in number of points fro Training. Defaults to 100.
        sliding_window_factor (int, optional): Portion of the total sample length to be used as the sliding window during predict. Defaults to 0.01.

    Returns:
        None
    """
    dataset = Dataset(dataset_name=dataset_name, stats=False)
    runner = Runner(model=model,
                    dataset=dataset,
                    fit_portion=0.2, observation_period=observation_period, runs=runs, sliding_window_factor=sliding_window_factor, plot=plot)
    res_dict = runner.evaluate()
    display_results(res_dict, runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streaming TSAD benchmark launcher. Outputs a .pkl file containing a dictionnary with run_number as key.")
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs for the evaluation')
    parser.add_argument('--model', type=str, default="LODA",
                        help='Model to be used for the evaluation')
    parser.add_argument('--dataset_name', type=str, default="swan",
                        help='Name of the dataset to be used for the evaluation')
    parser.add_argument('--observation_period', type=int, default=100,
                        help='Size of the sliding window in number of points.')
    parser.add_argument('--sliding_window_factor', type=float, default=0.01,
                        help='Portion of the total sample length to be used as the sliding window during predict.')
    args = parser.parse_args()
    main(
        runs=args.runs,
        model=args.model,
        dataset_name=args.dataset_name,
        observation_period=args.observation_period,
        sliding_window_factor=args.sliding_window_factor,
    )
