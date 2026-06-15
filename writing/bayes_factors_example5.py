"""
Bayes factor summary for example5 MCMC results (T=100, T=200, T=400).

For each stoichiometric channel and each candidate reaction:
  - Prob(on)  = 1 - Prob_Off
  - Prob(off) = Prob_Off
  - BF(on:off) = [Prob(on)/Prob(off)] / [pi/(1-pi)]
                 with prior pi=0.75 (as used in adaptive_mcmc_spike_slab)

Usage:
    python writing/bayes_factors_example5.py
"""

import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.parsing import load_reaction_network

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NETWORK_FILE = os.path.join(REPO, 'data', 'example5_network.json')
RESULTS_DIRS = {
    'T=100': os.path.join(REPO, 'results', 'example5_T100'),
    'T=200': os.path.join(REPO, 'results', 'example5_T200'),
    'T=400': os.path.join(REPO, 'results', 'example5_T400'),
}
PRIOR_PI = 0.75   # probability of being "on" in the spike-and-slab prior
PRIOR_ODDS = PRIOR_PI / (1 - PRIOR_PI)   # = 3

# ── Load network to get reaction names ──────────────────────────────────────
(_, _, CRN_reaction_names, CRN_parameter_names,
 trueTheta, parameter_values, sampled_indices,
 reactant_matrix, unique_changes, compatible_reactions,
 species_names) = load_reaction_network(NETWORK_FILE)

# Build lookup: (run_index, param_index) -> reaction name
rxn_name_lookup = {}
for ch_idx, deltaX in enumerate(unique_changes):
    for param_idx, full_rxn_idx in enumerate(compatible_reactions[deltaX]):
        # Find this full_rxn_idx in sampled_indices to get the CRN name
        if full_rxn_idx in sampled_indices:
            crn_pos = sampled_indices.index(full_rxn_idx)
            name = CRN_reaction_names[crn_pos].rstrip(':')
        else:
            name = f'rxn_{full_rxn_idx}'
        rxn_name_lookup[(ch_idx, param_idx)] = name

def bayes_factor(prob_off, clip=1e-10):
    """BF for 'on' vs 'off', corrected for prior odds."""
    prob_on  = 1.0 - prob_off
    prob_off_c = max(prob_off, clip)
    prob_on_c  = max(prob_on,  clip)
    posterior_odds = prob_on_c / prob_off_c
    return posterior_odds / PRIOR_ODDS

def bf_str(bf):
    if bf >= 1e6:  return ">1e6"
    if bf <= 1e-6: return "<1e-6"
    return f"{bf:.2f}"

# ── Print results ────────────────────────────────────────────────────────────
print("=" * 90)
print("BAYES FACTORS — example5 (4-species sparse CRN)")
print(f"Prior: π = {PRIOR_PI} (each reaction), BF = [Prob(on)/Prob(off)] / {PRIOR_ODDS:.1f}")
print("BF >> 1 → strong evidence ON  |  BF << 1 → strong evidence OFF")
print("=" * 90)

T_values = list(RESULTS_DIRS.keys())

# Load all dataframes
dfs = {}
for T, rdir in RESULTS_DIRS.items():
    xlsx = os.path.join(rdir, 'mcmc_summary.xlsx')
    dfs[T] = pd.read_excel(xlsx)

# Get all (run_index, param_index) combinations that appear in any T
all_keys = set()
for df in dfs.values():
    for _, row in df.iterrows():
        all_keys.add((int(row['Run_Index']), int(row['Param_Index'])))

# Group by run_index
run_indices = sorted(set(k[0] for k in all_keys))

for run_idx in run_indices:
    # Get param indices for this channel
    params = sorted(k[1] for k in all_keys if k[0] == run_idx)

    # Get count and true theta from T=100 (as reference)
    ref_df = dfs['T=100']
    ch_rows = ref_df[ref_df['Run_Index'] == run_idx]
    if ch_rows.empty:
        continue
    count_100 = int(ch_rows.iloc[0]['Count'])

    print(f"\nChannel {run_idx}  (T=100 obs: {count_100})")
    hdr = f"  {'Reaction':<25} {'True θ':>8} | " + " | ".join(f"{T:>16}" for T in T_values)
    print(hdr)
    print("  " + "-" * (25 + 8 + 3 + len(T_values) * 19))

    for p_idx in params:
        rxn_name = rxn_name_lookup.get((run_idx, p_idx), f'param_{p_idx}')

        # Get true theta from any df
        true_theta = None
        for df in dfs.values():
            row = df[(df['Run_Index'] == run_idx) & (df['Param_Index'] == p_idx)]
            if not row.empty:
                true_theta = row.iloc[0]['True_Theta']
                break

        active = true_theta is not None and abs(true_theta) > 1e-6
        marker = "✓" if active else " "

        cells = []
        for T in T_values:
            df = dfs[T]
            row = df[(df['Run_Index'] == run_idx) & (df['Param_Index'] == p_idx)]
            if row.empty:
                cells.append(f"{'—':>16}")
            else:
                poff = float(row.iloc[0]['Prob_Off'])
                pon  = 1.0 - poff
                bf   = bayes_factor(poff)
                cells.append(f"Pon={pon:.4f} BF={bf_str(bf):>6}")

        theta_str = f"{true_theta:.4f}" if true_theta is not None else "  N/A  "
        print(f"{marker} {rxn_name:<25} {theta_str:>8} | " + " | ".join(cells))

print("\n" + "=" * 90)
print("SUMMARY: reactions with True_θ > 0 should have Pon ≈ 1 and BF >> 1")
print("         reactions with True_θ = 0 should have Pon ≈ 0 and BF << 1")
