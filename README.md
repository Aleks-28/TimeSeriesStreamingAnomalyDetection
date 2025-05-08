# Streaming Multivariate Anomaly Detection Benchmark

This repository contains the code and resources for a **streaming anomaly detection benchmark** developed during my internship at **EDF R&D**. The benchmark evaluates different models for detecting anomalies in multivariate time-series data under streaming constraints.

---

## 📦 Requirements

- `g++` compiler or C++ build tools (Note: the library is **untested on Windows**)
- [Anaconda](https://www.anaconda.com/) or Miniconda (mandatory to use provided environment scripts)

---

## 🛠 Installation

To set up the required environments, run the following script:

```bash
scripts/install_env.sh
```

This will install three conda environments:
- One for plotting
- One for the `dSalmon` C++ library
- One for Online Isolation Forest

**Data Download:**  
Instructions for downloading the datasets are available in `utils/data_loaders/`.

---

## 📁 Repository Structure

```text
.
├── images/              # Generated plots and result visualizations
├── notebooks/           # Jupyter notebooks for analysis (see `result_analysis.ipynb`)
├── scripts/             # Job scheduling, log analysis, and batch processing scripts
├── src/                 # Core benchmarking framework (main loop in `runner.py`)
├── utils/               # Utilities including dataset loaders
├── run.py               # Main entry point for running experiments
├── run_multiproc.py     # Script for launching parallel jobs
└── README.md            # You're here!
```

---

## 🚀 Running the Benchmark

Use the following command to run the benchmark:

```bash
python run.py --runs 10 --model SWKNN --dataset_name comut4 --observation_period 100 --sliding_window_factor 0.01
```

---

## 🧵 Running Parallel Experiments

To launch experiments in parallel:

1. **Generate jobs**:
   ```bash
   python scripts/generate_jobs.py
   ```

2. **Run the jobs** using multiprocessing:
   ```bash
   python run_multiproc.py
   ```

---

## ➕ Adding a New Model

To include a new model:

1. Create a new Python file in `src/models/` following the template:
   ```bash
   src/templates/model.py
   ```
2. Register the model in `src/models/__init__.py`:
   ```python
   from .MyNewModel import MyNewModel
   ```

3. Use the class name as the `--model` argument in `run.py`.

> **Note:** The model class name is **case-sensitive**.

---

## 📊 Adding a New Metric

To define a new evaluation metric:

1. Create a new file in `src/metrics/` based on:
   ```bash
   src/templates/metric.py
   ```

2. Add your metric to the `all` list in `src/metrics/__init__.py`:
   ```python
   all = ['AUCPR', 'AUCROC', 'VUSPR', 'VUSROC', 'RefMetrics', 'MyNewMetric']
   ```

---

## 📓 Notebooks

Exploratory analysis and result visualization are available in the `notebooks/` directory. The most comprehensive one is:

- `result_analysis.ipynb`: Demonstrates result extraction from `.pkl` files.

---

## 🧑‍💻 Contributors

- Developed during a research internship at EDF R&D

---

## 📄 License

This project may be subject to EDF's internal licensing. Please consult with the maintainers before reuse or distribution.

---

## 📬 Contact

For questions or contributions, please contact the original developer or open an issue on this repository.
