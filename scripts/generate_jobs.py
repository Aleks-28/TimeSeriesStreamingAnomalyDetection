import itertools
import os

def check_existing_jos(model, dataset_name, observation_period, sliding_window_factors):
    """Check if there are any existing jobs in the directory.

    Returns:
        bool: True if there are existing jobs, False otherwise.
    """
    res: bool = False
    for file in os.listdir("results/pkl"):
        if f"{model}_{dataset_name}_window_{observation_period}_{sliding_window_factors}.pkl" in file:
            res = True
            break
    return res


def main():
    """Scans the results folder for existing jobs and generates a shell script to run the missing ones.

    Args:
        model (str): The model to use. Default is 'OIF'.
    """

    models = ["xStream", "LODA", "LODASALMON", "RSHASH", "SDOs", "SWKNN", "OIF"]  # In case you want to extend it, keep it as a list
    dataset_names = ["swan", "insectsAbr", "insectsIncr", "insectsIncrRecr", 
                     "insectsIncrGrd", "comut4", "comut8", "comut16"]
    observation_periods = [100, 500, 1000, 5000]
    sliding_window_factors = [0.005, 0.01, 0.1, 0.2]

    with open(f"scripts/run_jobs.sh", "w") as file:
        file.write("#!/bin/bash\n\n")
        for combination in itertools.product(models, dataset_names, observation_periods, sliding_window_factors):
            model, dataset_name, observation_period, sliding_window_factor = combination
            if check_existing_jos(model, dataset_name, observation_period, sliding_window_factor) == True:
                continue
            else:
                command = (
                    f"python run.py --runs 10 --model {model} --dataset_name {dataset_name} "
                    f"--observation_period {observation_period} --sliding_window_factor {sliding_window_factor}\n"
                )
                file.write(command)


if __name__ == "__main__":
    main()
