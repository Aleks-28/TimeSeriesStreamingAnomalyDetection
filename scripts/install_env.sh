#!/bin/bash

# Exit on any error
set -e
eval "$(conda shell.bash hook)"

conda create -n stsadbench_env_salmon python=3.8 -y

conda create -n stsadbench_env_oif python=3.9 -y

conda activate stsadbench_env_salmon

pip install -r ../requirements_py_version38.txt
cd ../src/utils/xStream

pip install .

conda deactivate
cd ../../..
conda activate stsadbench_env_oif
pip install -r requirements_py_version39.txt
conda deactivate

conda cr

echo "Environment setup complete!"
