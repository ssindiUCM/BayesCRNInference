"""
Run MCMC for Example 4, index 6 only — channel (-1,+1) = A+B→2B.

This was skipped in example4.ipynb because norm_theta == 0.01 exactly hit
the strict `> 0.01` threshold (now fixed to `>= 0.01` in the notebook).
Run this script once to fill in the missing result, then re-run the
comparison table cells in example4.ipynb.
"""
import sys, os, numpy as np
sys.path.append(os.path.abspath(".."))

from src.parsing import (load_reaction_network, load_trajectory,
                          generate_reactions, build_CRN_byNameSelection,
                          extract_local_data)
from src.inference import local_log_likelihood
from src.mcmc import (adaptive_mcmc_spike_slab, summarize_chains,
                       plot_mcmc_samples, plot_mcmc_chain,
                       calc_network_posteriors,
                       plot_network_and_parameter_posteriors)
from CRN_Simulation.CRN import CRN

# ── Replicate notebook setup ────────────────────────────────────────────────
species_names = ["A", "B"]
complexes = np.array([
    [0, 1, 0, 2, 0, 1],   # A
    [0, 0, 1, 0, 2, 1],   # B
])

from src.parsing import generate_reactions
(reactant_matrix, product_matrix, stoichiometric_matrix,
 reaction_names, parameter_names, unique_changes, compatible_reactions) = generate_reactions(
    complexes, species_names)

reactions_sys = ["A_to_2A", "A_to_Empty", "A+B_to_2B", "B_to_Empty"]
rates_sys     = [1.0, 0.1, 0.01, 0.5]

from src.parsing import build_CRN_byNameSelection
(CRN_stoichiometric_matrix, CRN_reaction_names, CRN_parameter_names,
 CRN_propensities, trueTheta, parameter_values, sampled_indices) = build_CRN_byNameSelection(
    reactant_matrix, product_matrix, stoichiometric_matrix,
    reaction_names, parameter_names,
    reactions_sys, rates_sys, species_names)

from CRN_Simulation.CRN import CRN
reactionNetwork = CRN(CRN_stoichiometric_matrix, CRN_propensities,
                       CRN_reaction_names, CRN_parameter_names)

# ── Load trajectory ─────────────────────────────────────────────────────────
from src.parsing import load_trajectory, parse_trajectory
time_list, state_list = load_trajectory("../data/example4_T40_trajectory.json")

from src.parsing import parse_trajectory
unique_states, jump_counts, waiting_times, propensities = parse_trajectory(
    state_list, time_list, reactant_matrix, unique_changes, compatible_reactions,
    verbose=False)

# ── Run MCMC for index 6 only ───────────────────────────────────────────────
TARGET_INDEX = 6
results_dir  = "../results/example4_T40"
os.makedirs(results_dir, exist_ok=True)

NIterates = 500_000
Burnin    = 50_000
Thinout   = 100

index   = TARGET_INDEX
deltaX  = unique_changes[index]
print(f"\nProcessing Index: {index}, Stoichiometric Change: {deltaX}")

(local_counts, local_waiting_times, local_propensities, selected_deltaX) = extract_local_data(
    jump_counts, waiting_times, propensities, unique_changes,
    index=index, deltaX=deltaX, verbose=True)

localTheta    = trueTheta[compatible_reactions[selected_deltaX]]
num_reactions = len(localTheta)
norm_theta    = np.linalg.norm(localTheta, ord=2)
total_count   = sum(local_counts.values())
print(f"Local True Theta: {localTheta}  (Norm: {norm_theta:.4f})")
print(f"Total Count = {total_count}")

filtered_reactions = [reaction_names[i] for i in compatible_reactions[selected_deltaX]]
print(f"Reaction Names = {filtered_reactions}")

np.random.seed(123)
theta_init = np.random.uniform(0.001, 2.0, size=localTheta.shape)

print("Running Adaptive MCMC with spike-and-slab prior...")
AdaptiveThetaChain = adaptive_mcmc_spike_slab(
    local_counts, local_waiting_times, local_propensities,
    theta_init, trueTheta, num_iterations=NIterates,
    alpha=2, beta=0.25, pi=0.75, burn_in=Burnin, adapt_every_n=10,
    printEveryNSteps=10_000)

filenameKDE       = os.path.join(results_dir, f"AdaptiveMCMC_plot_Index_{index}_TotalCount_{total_count}_kde.png")
filenameChain     = os.path.join(results_dir, f"AdaptiveMCMC_plot_Index_{index}_TotalCount_{total_count}_chain.png")
filenamePosterior = os.path.join(results_dir, f"AdaptiveMCMC_plot_Index_{index}_TotalCount_{total_count}_posterior.png")

plot_mcmc_samples(AdaptiveThetaChain, localTheta, epsilon=1e-5,
                  burnin=Burnin, thinout=Thinout, filename=filenameKDE)
plot_mcmc_chain(AdaptiveThetaChain, filename=filenameChain)
if num_reactions >= 2:
    calc_network_posteriors(AdaptiveThetaChain, epsilon=1e-5,
                            prob_cutoff=0.05, burnin=Burnin, thinout=Thinout)
    plot_network_and_parameter_posteriors(AdaptiveThetaChain, localTheta, epsilon=1e-5,
                                          prob_cutoff=0.05, burnin=Burnin, thinout=Thinout,
                                          filename=filenamePosterior)

summarize_chains([AdaptiveThetaChain], localTheta, ["Adaptive Spike & Slab"],
                 results_dir=results_dir, filename="mcmc_summary.xlsx",
                 burnin=Burnin, thinout=Thinout, alpha_ci=0.05, epsilon=1e-3,
                 run_index=index, count=total_count)

print(f"\nDone. Results appended to {results_dir}/mcmc_summary.xlsx")
print("Now re-run the comparison table cells in example4.ipynb.")
