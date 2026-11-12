# Examples

Each notebook is meant to be run from this folder (`examples/`); paths are relative to it. Notebooks that write
trajectories put them in `../data/`, MCMC results in `../results/`, LASSO runs in `../sparse_runs/`, and LaTeX
tables in `../writing/`. `data/` and `results/` are not tracked by git (see the top-level README), so on a fresh
clone run the notebooks in the order listed for each example.

Common settings of the Bayesian inference (all examples): spike-and-slab prior with spike weight `pi = 0.75` and
Gamma slab (`alpha = 2`, rate `beta = 0.25`), adaptive Metropolis–Hastings on the working parameters, one independent
inference per stoichiometric channel with at least one observed event. Every example uses 500,000 iterations per
channel, burn-in 50,000 and thinning 100 (4,501 retained draws); the proposal covariance is adapted from the
states after iteration 25,000. All examples call `run_channel_mcmc` in `src/mcmc.py`,
which seeds each channel's chain (base seed = example number), saves the chain
(`results/<example>/..._chain.npz`, every 10th iteration) and writes `mcmc_summary.xlsx` with posterior means,
95% intervals, Pr(off), the effective sample size (ESS) of the retained draws and the acceptance rate. A channel takes
seconds to a minute, so each example's MCMC runs in minutes. Under mass action, a reaction $2X \to \cdots$ has propensity $k\,X(X-1)/2$.

---

## Section 5.1: non-identifiable networks

### `example1.ipynb` — conservation law
Two networks on $X_1, X_2$ with $X_1 + X_2 = 2$ that produce identical dynamics (system 1:
$X_1 \rightleftharpoons X_2$; system 2: four bimolecular reactions). Simulates both ($T = 100$), compares their
likelihoods along an interpolation of the parameters, and runs the MCMC and an $\ell_1$ path.
Writes `results/example1/...`.

### `example2.ipynb` — catalyst species
$\varnothing \to X_1$, $X_1 \to 2X_1$, $X_1 + X_2 \to X_2$ with $X_2$ never changing, so the propensities of
$X_1 \to \varnothing$ and $X_1 + X_2 \to X_2$ are proportional along every trajectory. Writes `results/example2/`.

---

## Section 5.2: predator–prey

### `example4.ipynb` — Bayesian inference
Lotka–Volterra system $A \to 2A$ (1.0), $A \to \varnothing$ (0.1), $A + B \to 2B$ (0.01), $B \to \varnothing$ (0.5),
simulated from $(100, 50)$ to $T = 40$ (seed 42; prey extinction near $t = 39.8$).
Writes `data/example4_network.json`, `data/example4_T40_trajectory.json`, `results/example4_T40/mcmc_summary.xlsx`.
MCMC: seed 4, a few minutes.

### `example4_sparse_massaction.ipynb` — LASSO comparison and the predator–prey table
Runs the sparse-learning-CRN LASSO ($\lambda = 0.01$) on the same trajectory in the black-box and the mass-action
configuration (see the top-level README), checks every fit against an independent optimum of the same objective, and
writes the manuscript's predator–prey table `writing/table_predprey_detail_massaction.tex`
(Bayesian columns from `results/example4_T40/mcmc_summary.xlsx`).
Run folders: `sparse_runs/example4_T40_lam0p01_blackbox/`, `sparse_runs/example4_T40_lam0p01_massaction/`.
* `RUN_LASSO = False` (default): post-processing only, about a minute.
* `RUN_LASSO = True`: re-runs both fits with the compiled programs, about 30 minutes.

---

## Section 5.3: random 4-species network

### `example5.ipynb` — Bayesian inference
20 reactions sampled at random (seed 19) from all unimolecular and bimolecular reactions of $(A, B, X, Y)$, rates from
Gamma(2.6, 0.4); simulated from $(5,5,5,5)$ for $T = 100, 200, 400$ (seed 42).
Writes `data/example5_network.json`, `data/example5_T{100,200,400}_trajectory.json`,
`results/example5_T{100,200,400}/mcmc_summary.xlsx`. MCMC: seed 5, a few minutes per $T$.

### `example_sparse.ipynb` — LASSO runs
Exports the example 5 trajectories for sparse-learning-CRN and prints the terminal commands that run it
(`sparse_runs/example5_T*_lam*/`), for $\lambda = 0.01$ at each $T$ and a sweep of $\lambda$ at $T = 200$.

The LASSO fits use the long stopping rule of the other examples (`tot_step 300000`, stop after 100000 steps without a
new minimum). Earlier fits with their default stopping rule had not converged; they are kept in
`sparse_runs/_previous_default_stopping/`. LASSO-MA fits (candidate reactions only) are in
`sparse_runs/example5_T{100,200,400}_lam0p01_massaction/`. All fits can be rerun with `scripts/run_lasso_overnight.sh`
(see below).

The Section 5.3 tables are built by `../writing/comparison_tables.ipynb` (Bayes, LASSO-MA and LASSO-BB $L^2$ errors;
$\lambda$ sweep of LASSO-BB).

---

## Ergodic example: antithetic integral controller

Network: $\varnothing \xrightarrow{k} X$, $X \xrightarrow{\gamma} \varnothing$, $Z_1 \xrightarrow{k_p} Z_1 + X$,
$\varnothing \xrightarrow{\mu} Z_1$, $X \xrightarrow{\theta} X + Z_2$, $Z_1 + Z_2 \xrightarrow{\eta} \varnothing$
with $k = 2$, $\gamma = 1$, $k_p = 1$, $\mu = 6$, $\theta = 2$, $\eta = 1$. Ergodic because $\mu/\theta > k/\gamma$
(Briat, Gupta & Khammash, Cell Systems 2016), so the consistency results of the manuscript apply. Stationary
identities used as checks: $E[X] = \mu/\theta = 3$, $E[Z_1] = (\gamma\mu/\theta - k)/k_p = 1$, $E[Z_1 Z_2] = \mu/\eta = 6$.
Two active reactions, $\varnothing \to X$ and $Z_1 \to Z_1 + X$, share the stoichiometric change $(+1, 0, 0)$.

Run in this order:

### `example6_aic.ipynb` — simulation and Bayesian inference at $T = 5000$
Simulates one trajectory from $(0,0,0)$ to $T = 5000$ (seed 42, about 120,000 events), plots it, checks the stationary
identities with time-weighted averages (a plain average over recorded states is biased), and runs the MCMC
(seed 6).
Writes `data/example6_network.json`, `data/example6_trajectory.json`, `results/example6/`. A few minutes.

### `example6_aic_Tsweep.ipynb` — posterior concentration as $T$ grows
Uses prefixes of the same trajectory up to $T = 100, 500, 1000, 5000$ (written to
`data/example6_T{T}_trajectory.json`; $T = 5000$ is the full trajectory) and runs the same MCMC for each.
Writes `results/example6_T{T}/`, `results/example6_Tsweep/` (CSV and figure `posterior_vs_T.png`) and
`writing/table_aic_bayes_Tsweep.tex`.
* `RUN_MCMC = False` (default): post-processing only, seconds.
* `RUN_MCMC = True`: re-runs the MCMC for every $T$, a few minutes ($T = 5000$
  reproduces `results/example6/` exactly).

### `example6_aic_lasso.ipynb` — LASSO comparison
Runs the sparse-learning-CRN LASSO ($\lambda = 0.01$) on the same prefixes in the black-box and the mass-action
configuration. All fits are computed with their program, run to convergence, and checked against the independent solver
of `src/sparse_learning_compare.py`. The long fits ($T = 1000, 5000$) take 1–15 hours each; they were run with
`scripts/run_lasso_overnight.sh`.
Run folders: `sparse_runs/example6_T{T}_lam0p01_{massaction,blackbox}/`. Writes `results/example6_lasso/` (including
the $L^2$ propensity errors of Bayes, LASSO-MA and LASSO-BB, `l2_errors.csv`) and
`writing/table_aic_lasso_comparison.tex`, `writing/table_aic_lasso_blackbox.tex`.
* `RUN_LASSO = False` (default): post-processing only, about a minute.
* `RUN_LASSO = True`: re-runs their programs one channel after another, about a day; use
  `scripts/run_lasso_overnight.sh` to run the channels in parallel.

---

## Checks

### `prior_sensitivity.ipynb` — sensitivity to the prior (manuscript appendix)
Reruns the Bayesian inference for the ergodic example ($T = 100$), predator–prey ($T = 40$) and the random network
($T = 100$) under the paper's prior and four alternatives (spike weight 0.5 or 0.9, a flatter or a tighter Gamma slab),
with the paper's seeds. Writes `results/prior_sensitivity/`, `writing/table_prior_sensitivity.tex` and
`writing/fig_prior_sensitivity.pdf`.
* `RUN_MCMC = False` (default): reads the saved summaries, seconds.
* `RUN_MCMC = True`: reruns all 15 combinations, about 35 minutes; finished combinations are skipped, so an interrupted
  run can be resumed.

### `coverage_study.ipynb` — coverage of the credible intervals (antithetic integral controller)
Simulates 100 independent trajectories of the controller up to $T = 500$ and runs the paper's sampler (same prior and
settings, seeded) on each. Reports, for each active rate, how often the 95% credible interval contains the true value
(with a Clopper–Pearson interval), misses below/above, mean bias and width, the joint-region coverage for the two rates
of channel $+1X$, and how often the median probability model is the true network. Writes `results/coverage_study/`
(one CSV per trajectory) and `writing/table_coverage.tex`.
* `RUN = False` (default): reads the saved CSVs, seconds.
* `RUN = True`: about 1.5 hours on 2 cores; finished trajectories are skipped, so an interrupted run can be resumed.

### `verify_mcmc_speedup.ipynb` — new vs original sampler
Compares `adaptive_mcmc_spike_slab` with the original implementation (`adaptive_mcmc_spike_slab_original`) on real
data: same likelihood and prior values, same posterior, run times, and exact reproduction of a saved chain. About 8 minutes.

---

## Scripts (`../scripts/`)

* `build_sparse_learning_mac.sh` — builds Zhang et al.'s `sparse_learning` program (their source, unchanged) on a Mac
  without Homebrew, into `~/sparse-learning-build/`; the libconfig source it needs is in `scripts/vendor/`.
* `run_lasso_overnight.sh` — runs the LASSO fits listed in the script with their program, one process per channel, in
  parallel (`SPARSE_EXE=... caffeinate -i bash scripts/run_lasso_overnight.sh`). Then
  `merge_lasso_overnight.sh` copies the results into the run folders.

---

## Other

* `example3.ipynb` — random sparse network on 3 species ($T = 100, 200, 400$); not in the manuscript.
* `run_example4_index6.py` — one-off script that filled in a channel skipped by an earlier version of `example4.ipynb`.
