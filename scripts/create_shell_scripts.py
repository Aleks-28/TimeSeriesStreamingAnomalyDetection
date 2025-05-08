"""
@who: Emmanouil Sylligardos
@where: Ecole Normale Superieur (ENS), Paris, France
@when: PhD Candidate, 1st year (2024)
@what: MSAD-E
"""

import itertools
import os
import numpy as np
from templates import sh_templates


def main():
    current_dir = "scripts"
    experiment_desc = {
        "job_name": "streaming_benchmark",
        "environment": "streamtsad_bench_env",
        "script_name": "run.py",
        "args": {
            "runs": [10],
            "model": ['LODA', 'RSHASH', 'SDOs', 'xStream', 'SWKNN', 'LODASALMON'],
            "dataset_name": ['swan', 'insectsAbr', 'insectsIncr', 'insectsIncrRecr', 'insectsIncrGrd', 'comut4', 'comut8', 'comut16'],
            "observation_period": [100, 500, 1000, 5000],
            "sliding_window_factor": [0.005, 0.01, 0.1, 0.2],
        },
        "gpu_required": "1 if \"model_class\" == \"raw\" else 1"
    }
    # experiment_desc = {
    #     "job_name": "regressors_proof",
    #     "environment": "msadt",
    #     "script_name": "src/regressors_proof.py",
    #     "args": {
    #         "dataset": ['all', 'SVDB', 'Genesis', 'GHL', 'SensorScope', 'ECG', 'OPPORTUNITY', 'SMD', 'KDD21', 'Daphnet', 'NAB', 'YAHOO', 'Dodgers', 'MITDB', 'IOPS', 'Occupancy', 'MGAB'],
    #         "saving_path": ["experiments/regressors_proof_16_01_2025"],
    #     },
    #     "gpu_required": "0"
    # }
    template = sh_templates['cleps_cpu']

    # Analyse json
    saving_dir = experiment_desc['job_name']
    environment = experiment_desc["environment"]
    script_name = experiment_desc["script_name"]
    args = experiment_desc["args"]
    args_saving_path = 'results'
    arg_names = list(args.keys())
    arg_values = list(args.values())
    gpu_required = experiment_desc["gpu_required"]

    # Generate all possible combinations of arguments
    combinations = list(itertools.product(*arg_values))

    # Create the commands
    jobs = set()
    for combination in combinations:
        cmd = f"{script_name}"
        job_name = experiment_desc["job_name"]

        supervised_flag = 0
        for name, value in zip(arg_names, combination):
            if name == 'experiment' and value == 'supervised':
                supervised_flag = 1
            if supervised_flag and name == 'split':
                continue
            cmd += f" --{name} {value}"
            if name == 'saving_path':
                continue

            job_name += f"_{value}"

            if isinstance(gpu_required, str) and name in gpu_required:
                gpu_required = int(
                    eval(gpu_required.replace(name, str(value))))

        # Create saving dir if doesn't exist
        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)

        # Write the .sh file
        with open(os.path.join(saving_dir, f'{job_name}.sh'), 'w') as rsh:
            rsh.write(template.format(job_name, args_saving_path,
                      args_saving_path, environment, cmd))

        jobs.add(job_name)

    # Create sh file to conduct all experiments
    run_all_sh = ""
    jobs = list(jobs)
    jobs.sort()
    for job in jobs:
        run_all_sh += f"sbatch {os.path.join(current_dir, saving_dir, f'{job}.sh')}\n"

    with open(os.path.join(saving_dir, f'conduct_{experiment_desc["job_name"]}.sh'), 'w') as rsh:
        rsh.write(run_all_sh)


if __name__ == "__main__":
    main()
