@echo off

:: Exit on error
setlocal ENABLEEXTENSIONS
setlocal ENABLEDELAYEDEXPANSION

:: Initialize conda
CALL conda init bash

:: Create environments
CALL conda create -n stsadbench_env_salmon python=3.8 -y
CALL conda create -n stsadbench_env_oif python=3.9 -y

:: Activate and install for stsadbench_env_salmon
CALL conda activate stsadbench_env_salmon
pip install -r ..\requirements.txt
pip install wheel setuptools setuptools_scm
cd ..\src\utils\xStream
pip install .
CALL conda deactivate

:: Activate and install for stsadbench_env_oif
cd ..\..\..
CALL conda activate stsadbench_env_oif
pip install -r requirements_OIF.txt

@echo Environment setup complete!
pause
