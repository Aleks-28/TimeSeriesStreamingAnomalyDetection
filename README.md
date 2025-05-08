## Streaming Multivariate Anomaly Detection Benchmark

This repositery contains code for the streaming anomaly detection benchmark conducted during an internship at EDF R&D.

# Requirements
- g++ compile or C++ buildtools for windows (library is untested on windows)
- conda for environment management (mandatory to use the included installation scripts)

To reproduce the results, install the three environments (one for plotting, one for dSalmon library and one for Online Isolation Forest) using the scritps/install_env.sh.

Instructions for downloading the data are given in utils/data_loaders.

The repo contains the following folders :
- *images* are generated images for the result analysis
- *notebooks* contains various notebooks used during the project. The most thorough one is result_analysis that show an example of result extraction from .pkl files.
- *scripts* contains small scripts used for the INRIA server job scheduling, as well as scripts for analyzing logs and generating missing jobs command lines.
- *src* contains the source code for the benchmark main framework. The main loop for testing is in src/runner.py.


To run the benchmark, run the following command: `python run.py --runs 10 --model SWKNN --dataset_name comut4 --observation_period 100 --sliding_window_factor 0.01`
For launching parallel experiments on multiple cores, first generate jobs with scripts/generate_jobs.py, then launch them with run_multiproc.py.

# Adding a model

To add a model, first create a .py file in the src/models, following the template in src/templates/model.py
The model name must then be added in the src/models/__init__.py import list .
The experiments can afterwards be launched normally using the name of the class (case sensitive) as model name.

# Adding a metric
For adding a metric, create a .py files named with metric following the src/templates/metric.py template.
Import the newly added metric to the list: __all__ = ['AUCPR', 'AUCROC', 'VUSPR', 'VUSROC', 'RefMetrics', *'MyNewMetric'*] in src/metrics/__init__.py.