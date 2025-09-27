# BayesCRNInference

Bayesian inference framework for chemical reaction networks (CRNs) using MCMC.  
This code supports **exact stochastic trajectory data** under **mass-action kinetics** with **discrete molecular counts**, and allows joint inference of both **network structure** and **reaction rates**.

---

## 📖 Overview
This repository contains code used in our manuscript to demonstrate Bayesian MCMC inference on stochastic CRN trajectories.  

Key features:
- Inference from **exact stochastic trajectories** (not approximations).
- Models assume **mass-action kinetics** with discrete counts.
- Supports inference of both **reaction rates** and **network structure** (detecting reactions that are absent vs. very small).
- Includes reproducible **example notebooks** from the manuscript.

---

## 📂 Repository Structure
```text
BayesCRNInference/
│
├── CRN/                 # Supporting CRN code (adapted from collaborator)
│
├── src/                 # Core source code (functions)
│   ├── parsing.py       # Functions for parsing stochastic trajectory data
│   ├── mcmc.py          # Functions for MCMC sampling and inference
│   └── inference.py     # (Optional) high-level wrappers combining parsing + MCMC
│
├── examples/            # Reproducible Jupyter notebooks
│   ├── example1.ipynb   # Simple toy example
│   ├── example2.ipynb   # Intermediate example
│   ├── example3.ipynb   # Full manuscript example
│
├── main_examples.ipynb  # Clean single notebook demonstrating manuscript runs
│
├── README.md            # Project description
├── LICENSE              # GNU GPL v3 license
└── .gitignore           # Ignore Python build artifacts, notebooks, etc.
````

---

## 🚀 Getting Started

### Requirements

* Python 3.9+
* Jupyter
* Standard scientific Python stack: `numpy`, `scipy`, `matplotlib`
* (Optional) `tqdm` for progress bars

You can install dependencies via:

```bash
pip install -r requirements.txt
```


### Running Examples

To reproduce results from the manuscript:

1. Open `main_examples.ipynb` in Jupyter.
2. Run all cells top-to-bottom.
3. To explore more detail, see individual notebooks in `examples/`.

---

## ⚖️ License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.
You are free to use, share, and modify this code, but **any derivative work must also remain open-source under the same license**.

---

## ✨ Acknowledgments

* Core CRN code adapted from Zhou Fang (zhfang(at)amss.ac.cn).
* Code package developed to accompany tentatively titled: Bayesian Inference in Stochastic Biochemical Reaction Systems with Full Trajectory Data.
```


