# BayesCRNInference

Bayesian inference framework for **stochastic chemical reaction networks (CRNs)** using **exact trajectory data** under **mass-action kinetics**.

This repository accompanies a research manuscript and is intended to support **methodological transparency and reproducibility**, rather than to function as a general-purpose inference library.

---

## 📖 Overview

This code implements Bayesian inference for stochastic biochemical reaction networks using **fully observed stochastic trajectories** (e.g., Gillespie simulations).

The central scientific goal is to infer:
- **Reaction rate parameters**
- **Network structure** (distinguishing absent reactions from reactions with very small rates)

from **exact discrete-state trajectories**, without diffusion or moment approximations.

Inference is performed using MCMC, leveraging a decomposition of the likelihood into contributions from **local stoichiometric changes**.

---

## 🔬 Scientific Scope and Assumptions

This code assumes:

- Discrete molecular counts
- Continuous-time Markov jump processes
- Mass-action kinetics
- Fully observed state trajectories (event times and states)

It is **not** designed for:
- Partial observations
- Time-discretized data
- Deterministic or diffusion approximations
- Black-box CRN inference

Users are expected to be familiar with stochastic CRNs and Bayesian inference.

---

## 📂 Repository Structure

```text
BayesCRNInference/
│
├── CRN_Simulation/          # External CRN simulation code (prior work)
│   ├── CRN.py
│   ├── MatrixExponentialKrylov.py
│   ├── DistributionOfSystems.py
│   ├── MarginalDistribution.py
│   └── __init__.py
│
├── src/                     # Inference and likelihood construction
│   ├── parsing.py           # Trajectory parsing and sufficient statistics
│   ├── mcmc.py              # MCMC algorithms
│   └── inference.py         # High-level inference utilities
│
├── examples/                # Jupyter notebooks reproducing manuscript analyses
│   ├── example1.ipynb
│   ├── example2.ipynb
│   └── example3.ipynb
│
├── data/                    # Example stochastic trajectories (tracked)
│   ├── example1_crn1_trajectory.json
│   ├── example2_crn2_trajectory.json
│   └── example3_trajectory.json
│
├── check_requirements.ipynb # Environment and dependency check
│
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🧩 External CRN Simulation Code

The directory `CRN_Simulation/` contains **supporting simulation code adapted from prior work** and is included to enable trajectory generation and visualization.

This code is **not the focus of the repository**.  
All inference logic, likelihood construction, and MCMC algorithms are implemented in `src/`.

---

## 🚀 Getting Started

### Requirements

- Python 3.9+
- Jupyter Notebook

Core dependencies include:
- numpy
- scipy
- matplotlib
- seaborn
- scikit-learn
- torch

Install dependencies via:

```bash
pip install -r requirements.txt
```

To verify your environment, run:

```text
check_requirements.ipynb
```

---

## 📊 Example Trajectories

The `data/` directory contains **example stochastic trajectories** used in the accompanying notebooks.

These trajectories:
- Are generated from known CRNs
- Serve as illustrative inputs for inference
- Are included to support transparency and ease of use

They should **not** be interpreted as canonical datasets.

---

## 📓 Reproducing Analyses

The notebooks in `examples/` demonstrate the inference pipeline used in the manuscript:

1. Load or generate a stochastic trajectory
2. Parse trajectory data into sufficient statistics
3. Construct local likelihood components
4. Run MCMC for rate and/or structure inference
5. Visualize posterior distributions

Because inference relies on stochastic simulation and MCMC:
- Exact numerical results may vary between runs
- Qualitative posterior behavior and conclusions should be consistent

---

## 🔁 Reproducibility Philosophy

This repository prioritizes:

- **Methodological reproducibility**
- **Likelihood transparency**
- **Clear separation of modeling assumptions**

It does **not** guarantee:
- Identical MCMC traces
- Bitwise reproducibility of figures
- Exact posterior samples across environments

Random seeds are used where appropriate to improve stability, but stochastic variation is expected.

---

## ⚖️ License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

You are free to use, modify, and distribute this code, provided that any derivative work is released under the same license.

---

## ✨ Acknowledgments

- Code developed by **Suzanne Sindi** (ssindi(at)ucmerced.edu)
- CRN simulation code adapted from **Zhou Fang** (zhfang(at)amss.ac.cn)
- Developed to accompany a manuscript on Bayesian inference for stochastic biochemical reaction networks with full trajectory data
