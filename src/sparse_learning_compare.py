"""
Helpers for running the sparse-learning-CRN LASSO of Zhang et al. (2019)
(https://github.com/zwpku/sparse-learning-CRN) on BayesCRNInference trajectories,
in two configurations, without modifying their C++ code:

* black box   : every channel uses all polynomial basis functions of order <= 2
                (their default, xx_basis_flag = 1, i.e. quadratic basis x*x);
* mass action : xx_basis_flag = 0 (quadratic basis x(x-1)) and each channel's basis file
                is cut down to the propensities of the candidate reactions of that channel
                (the same candidate set as the Bayesian analysis).

The module also re-implements their objective in Python so that every fit can be checked
independently (see `check_run`).

Their objective for one channel (utils.cpp, minus_log_likelihood_partial + penalty):

    f(c) = (1/T_obs) [ - sum_{jumps n in channel} log G(a_c(x_n)) + sum_i tau_i G(a_c(x_i)) ]
           + lambda * sum_j |c_j|,     a_c(x) = sum_j c_j phi_j(x),

with G(x) = eps * log(1 + exp(x/eps)) a smoothed positive part and T_obs the total holding
time in the exported trajectory file (the time of the last event).

Used by examples/example6_aic_lasso.ipynb.
"""
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.parsing import export_for_sparse_learning

# Longer stopping rule than their defaults (tot_step 20000, max_step_since_prev_min_cost 1000):
# with the defaults some channels stop before convergence.
DEFAULT_STOP = {"tot_step": 300000, "max_step_since_prev_min_cost": 100000, "cost_stop_tol": 1e-12}


# --------------------------------------------------------------------------------------------
# Running their programs
# --------------------------------------------------------------------------------------------
def prepare_run(run_dir, traj_file, T, lam, xx_flag, stop=DEFAULT_STOP, overwrite=False):
    """Create a working directory for their programs: trajectory in their format + config.

    xx_flag = 1 : quadratic basis x*x (their default, black box)
    xx_flag = 0 : quadratic basis x(x-1) (mass action)
    """
    if os.path.exists(run_dir):
        if not overwrite:
            raise RuntimeError(f"{run_dir} exists; pass overwrite=True to replace it")
        shutil.rmtree(run_dir)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        export_for_sparse_learning(trajectory_files=[traj_file], working_dir=run_dir, T=T, poly_order=2,
                                   regular_lambda=lam, tot_step=stop["tot_step"],
                                   cost_stop_tol=stop["cost_stop_tol"])
    path = os.path.join(run_dir, "sparse_learning.cfg")
    cfg = open(path).read()
    cfg = re.sub(r"^xx_basis_flag = \d ;$", f"xx_basis_flag = {xx_flag} ;", cfg, flags=re.M)
    cfg = re.sub(r"^max_step_since_prev_min_cost = \d+ ;$",
                 f"max_step_since_prev_min_cost = {stop['max_step_since_prev_min_cost']} ;", cfg, flags=re.M)
    open(path, "w").write(cfg)


def run_program(exe, run_dir):
    """Run one of their programs (prepare or sparse_learning) inside run_dir."""
    p = subprocess.run([exe], cwd=run_dir, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr)
        raise RuntimeError(f"{os.path.basename(exe)} failed in {run_dir} (exit code {p.returncode})")


# --------------------------------------------------------------------------------------------
# Reading their files
# --------------------------------------------------------------------------------------------
def cfg_value(run_dir, key):
    cfg = open(os.path.join(run_dir, "sparse_learning.cfg")).read()
    return float(re.search(rf"^{key} = ([^ ;]+) ;", cfg, re.M).group(1))


def read_channel_info(run_dir):
    """Channels (stoichiometric changes) and their event counts, in their channel order."""
    tok = open(os.path.join(run_dir, "output", "channel_info.txt")).read().split()
    n_ch, n_sp = int(tok[0]), int(tok[1])
    vals = list(map(int, tok[2:]))
    changes = [tuple(vals[i*n_sp:(i+1)*n_sp]) for i in range(n_ch)]
    return changes, vals[n_ch*n_sp:n_ch*n_sp + n_ch]


def read_basis_info(run_dir):
    """Basis functions as tuples of species indices: () = 1, (i,) = x_i, (i, j) = x_i x_j or x_i(x_i - 1)."""
    tok = list(map(int, open(os.path.join(run_dir, "output", "basis_funct_info.txt")).read().split()))
    basis, pos = [], 2
    for _ in range(tok[1]):
        k = tok[pos]
        basis.append(tuple(tok[pos+1:pos+1+k]))
        pos += 1 + k
    return basis


def read_basis_file(path):
    """Their basis_of_channel_<i>.txt: n; indices; weights; initial values; (min; max)."""
    tok = open(path).read().split()
    n = int(tok[0])
    v = tok[1:]
    return n, [v[k*n:(k+1)*n] for k in range(len(v) // n)]


def channel_basis(run_dir, ch_i):
    _, fields = read_basis_file(os.path.join(run_dir, "output", f"basis_of_channel_{ch_i}.txt"))
    return [int(x) for x in fields[0]]


def read_omega(run_dir, ch_i):
    tok = open(os.path.join(run_dir, "output", f"omega_vec_for_channel_{ch_i}.txt")).read().split()
    return np.array(list(map(float, tok[1:1 + int(tok[0])])))


def basis_label(b, species, xx_flag):
    if len(b) == 0:
        return "1"
    if len(b) == 1:
        return species[b[0]]
    i, j = b
    if i == j:
        return f"{species[i]}({species[i]}-1)" if xx_flag == 0 else f"{species[i]}^2"
    return species[i] + species[j]


def reactant_key(reactant_vector):
    """Reactant complex as a sorted tuple of species indices, e.g. 2A -> (0, 0), A+B -> (0, 1)."""
    return tuple(sorted(s for s, n in enumerate(reactant_vector) for _ in range(int(n))))


def rate_factor(reactant_vector):
    """Coefficient c on x(x-1) corresponds to rate k = 2c under the propensity k x(x-1)/2."""
    return 2.0 if any(int(n) == 2 for n in reactant_vector) else 1.0


# --------------------------------------------------------------------------------------------
# Mass-action restriction
# --------------------------------------------------------------------------------------------
def restrict_to_candidates(run_dir, reactant_matrix, compatible_reactions, write=True):
    """Cut each channel's basis file down to the propensities of its candidate reactions.

    Returns {change: [(basis index, reaction index), ...]} in candidate order.
    write=False only checks that the saved files already hold exactly this restriction.
    The original files are kept as basis_of_channel_<i>_full.txt.
    """
    changes, _ = read_channel_info(run_dir)
    basis = read_basis_info(run_dir)
    keep_all = {}
    for ch_i, change in enumerate(changes):
        path = os.path.join(run_dir, "output", f"basis_of_channel_{ch_i}.txt")
        keep = [(basis.index(reactant_key(reactant_matrix[:, gi])), gi) for gi in compatible_reactions[change]]
        keep_all[change] = keep
        if write:
            full = path.replace(".txt", "_full.txt")
            shutil.copy2(path, full)
            n, fields = read_basis_file(full)
            idx_full = [int(x) for x in fields[0]]
            pos = [idx_full.index(b) for b, _ in keep]
            with open(path, "w") as f:
                f.write(f"{len(pos)} \n")
                for k, row in enumerate(fields):
                    f.write(" ".join(row[p] for p in pos) + " \n")
                    if k == 2:
                        f.write("\n")
        assert channel_basis(run_dir, ch_i) == [b for b, _ in keep], f"basis file of channel {change} does not match"
    return keep_all


# --------------------------------------------------------------------------------------------
# Checking convergence
# --------------------------------------------------------------------------------------------
def termination_summary(run_dir):
    """How each channel's fit stopped, from their log (None if no log is present)."""
    log_path = os.path.join(run_dir, "log", "sparse_infer.log")
    if not os.path.exists(log_path):
        return None
    changes, _ = read_channel_info(run_dir)
    blocks = re.split(r"Solving coefficients for the reaction channel (\d+)\.\.\. finished\.", open(log_path).read())
    rows = []
    for k in range(1, len(blocks), 2):
        ch, body = int(blocks[k]), blocks[k+1]
        reason = re.search(r"Reason of termination:\s*(.*)", body)
        steps = re.search(r"Total steps=(\d+),.*?min_cost=(\S+) \(at Step (\d+)\)", body)
        rows.append({"channel": changes[ch], "reason": reason.group(1).strip() if reason else "?",
                     "steps": int(steps.group(1)) if steps else None})
    return pd.DataFrame(rows)


OPTIMUM_FILE = "independent_optimum.json"   # saved in <run_dir>/output/ by check_run(..., save=True)


def check_run(run_dir, species, n_starts=4, save=False):
    """Re-implement their objective, evaluate it at their coefficients, minimise it independently.

    Returns a DataFrame with, per channel: their logged cost (if a log exists), the objective at their
    coefficients, the objective at our optimum, the gap, and both coefficient vectors.
    If their program has not been run in run_dir (no omega_vec files), only our optimum is reported.

    The optimum is saved to <run_dir>/output/independent_optimum.json when save=True. If that file exists, the
    saved optimum is used for the reported coefficients (so tables do not depend on the scipy version), and the
    optimisation is still repeated: column "re-solve diff" is the objective at a fresh optimum minus the objective
    at the saved one (near zero, or slightly negative if the fresh solve is marginally better).
    States are grouped (identical states share holding time and jump counts), which makes this fast
    even for long trajectories.
    """
    eps, gcut = cfg_value(run_dir, "eps"), cfg_value(run_dir, "g_cut")
    lam, xx = cfg_value(run_dir, "regular_lambda"), int(cfg_value(run_dir, "xx_basis_flag"))
    d = np.loadtxt(os.path.join(run_dir, "traj_data", "traj_0.txt"), skiprows=1)
    X, tau = d[:, 1:-1], d[:, -1]
    T_obs = tau.sum()
    dX = np.diff(X, axis=0)
    states, inv = np.unique(X, axis=0, return_inverse=True)
    inv = inv.ravel()
    tau_s = np.bincount(inv, weights=tau, minlength=len(states))
    basis = read_basis_info(run_dir)

    def phi(b, Z):
        if len(b) == 0:
            return np.ones(len(Z))
        if len(b) == 1:
            return Z[:, b[0]]
        i, j = b
        return Z[:, i] * (Z[:, i] - 1) if (i == j and xx == 0) else Z[:, i] * Z[:, j]

    PHI = np.stack([phi(b, states) for b in basis], 1)
    G = lambda x: np.where(x / eps >= gcut, x, eps * np.log1p(np.exp(np.minimum(x / eps, gcut))))

    def logG(x):
        t = x / eps
        with np.errstate(divide="ignore", invalid="ignore"):
            small = np.log(np.log1p(np.exp(np.minimum(t, gcut)))) + np.log(eps)
            small = np.where(np.isinf(small), t + np.log(eps), small)
            big = np.log(np.abs(x)) + np.log1p(np.log1p(np.exp(-np.abs(t))) / np.abs(t))
        return np.where(t > gcut, big, small)

    logcost = {}
    log_path = os.path.join(run_dir, "log", "sparse_infer.log")
    if os.path.exists(log_path):
        for m in re.finditer(r"channel (\d+)\.\.\. finished\..*?Corresponding cost : (\S+)", open(log_path).read(), re.S):
            logcost[int(m.group(1))] = float(m.group(2))

    changes, _ = read_channel_info(run_dir)
    opt_path = os.path.join(run_dir, "output", OPTIMUM_FILE)
    saved = None
    if os.path.exists(opt_path) and not save:
        import json
        saved = {tuple(d["channel"]): np.array(d["coefficients"]) for d in json.load(open(opt_path))}
    rows = []
    for i, change in enumerate(changes):
        idx = channel_basis(run_dir, i)
        n = len(idx)
        P = PHI[:, idx]
        jump = np.all(dX == np.array(change), axis=1)
        n_s = np.bincount(inv[:-1][jump], minlength=len(states)).astype(float)
        has = n_s > 0

        def f(c):
            a = P @ c
            return (-(n_s[has] * logG(a[has])).sum() + (tau_s * G(a)).sum()) / T_obs + lam * np.abs(c).sum()

        has_theirs = os.path.exists(os.path.join(run_dir, "output", f"omega_vec_for_channel_{i}.txt"))
        theirs = read_omega(run_dir, i) if has_theirs else np.zeros(n)
        sc = np.maximum(np.abs(P).mean(0), 1e-12)
        fs = lambda z: f(z[:n] / sc - z[n:] / sc)
        starts = [np.r_[np.maximum(theirs, 0), np.maximum(-theirs, 0)] * np.r_[sc, sc], np.full(2 * n, 0.01)]
        starts += [np.random.default_rng(s).uniform(0, 2, 2 * n) for s in range(n_starts)]
        best = min((minimize(fs, s0, method="L-BFGS-B", bounds=[(0, None)] * (2 * n),
                             options=dict(maxiter=20000, maxfun=200000, ftol=1e-15, gtol=1e-12)) for s0 in starts),
                   key=lambda r: r.fun)
        fresh = best.x[:n] / sc - best.x[n:] / sc
        ours = saved[change] if saved is not None else fresh
        f_ours = f(ours)
        rows.append({"channel": change,
                     "their cost (log)": logcost.get(i, np.nan),
                     "f(their coeffs)": f(theirs) if has_theirs else np.nan,
                     "f(our optimum)": f_ours,
                     "gap": f(theirs) - f_ours if has_theirs else np.nan,
                     "re-solve diff": best.fun - f_ours,
                     "basis": " ".join(basis_label(basis[b], species, xx) for b in idx),
                     "their coeffs": theirs if has_theirs else None,
                     "our coeffs": ours})
    if save:
        import json
        json.dump([{"channel": list(map(int, r["channel"])), "basis": r["basis"],
                    "coefficients": [float(v) for v in r["our coeffs"]], "objective": float(r["f(our optimum)"])}
                   for r in rows], open(opt_path, "w"), indent=1)
    return pd.DataFrame(rows)
