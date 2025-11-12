import numpy as np
import random
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt

def is_whole_number(val):
    """
    Check if a value is an integer or a float representing an exact integer.

    Args:
        val: Any numeric value

    Returns:
        True if val is an integer (Python int, NumPy int) or a float equal to an integer.
        False otherwise.
    """
    # Python int or NumPy integer
    if isinstance(val, (int, np.integer)):
        return True
    # Float that is a whole number
    if isinstance(val, float) and val.is_integer():
        return True
    return False


def extract_local_data(jump_counts, waiting_times, propensities, unique_changes, index=None, deltaX=None, verbose=True):
    """
    Extract the local counts, waiting times, and propensities for a single stoichiometric change.

    Args:
        jump_counts (dict): Dictionary mapping state -> array of counts for each unique stoichiometric change
        waiting_times (dict): Dictionary mapping state -> array of cumulative waiting times for each unique stoichiometric change
        propensities (dict): Dictionary mapping state -> list of propensity arrays for each unique stoichiometric change
        unique_changes (list of tuples): List of all unique stoichiometric change vectors
        index (int, optional): Index of the desired unique change in unique_changes
        deltaX (tuple or array-like, optional): Specific stoichiometric change vector to extract
        verbose (bool, default True): Print progress info

    Returns:
        local_counts (dict): counts for the selected stoichiometric change for each state
        local_waiting_times (dict): waiting times for the selected stoichiometric change for each state
        local_propensities (dict): propensity arrays for the selected stoichiometric change for each state
        selected_deltaX (tuple): the ΔX vector that was used
    """

    # -------------------------------------------
    # Determine which stoichiometric change to use
    # -------------------------------------------
    if deltaX is not None and index is not None:
        selected_deltaX = tuple(int(x) for x in deltaX)
        if index < 0 or index >= len(unique_changes):
            raise IndexError(f"Index {index} out of bounds for unique_changes.")
        unique_at_index = tuple(int(x) for x in unique_changes[index])
        if selected_deltaX != unique_at_index:
            raise ValueError(
                f"Mismatch: provided deltaX {selected_deltaX} does not match "
                f"unique_changes[{index}] = {unique_at_index}"
            )
        index_to_extract = index
    elif deltaX is not None:
        selected_deltaX = tuple(int(x) for x in deltaX)
        if selected_deltaX not in unique_changes:
            raise ValueError(f"Specified deltaX {selected_deltaX} not found in unique_changes.")
        index_to_extract = unique_changes.index(selected_deltaX)
    elif index is not None:
        if index < 0 or index >= len(unique_changes):
            raise IndexError(f"Index {index} out of bounds for unique_changes.")
        index_to_extract = index
        selected_deltaX = tuple(int(x) for x in unique_changes[index_to_extract])
    else:
        raise ValueError("You must specify either an index or a deltaX vector.")

    if verbose:
        formatted_deltaX = ", ".join(str(x) for x in selected_deltaX)
        print(f"Extracting local data for stoichiometric change [{formatted_deltaX}] at index {index_to_extract}")

    # -------------------------------------------
    # Extract the relevant information for each state
    # -------------------------------------------
    local_counts = {state: counts[index_to_extract] for state, counts in jump_counts.items()}
    local_waiting_times = {state: times[index_to_extract] for state, times in waiting_times.items()}
    local_propensities = {state: props[index_to_extract] for state, props in propensities.items()}

    # -------------------------------------------
    # Sanity checks
    # -------------------------------------------
    keys_counts = set(local_counts.keys())
    keys_wait   = set(local_waiting_times.keys())
    keys_prop   = set(local_propensities.keys())

    if keys_counts != keys_wait or keys_counts != keys_prop:
        raise ValueError(
            "Mismatch in state keys! The keys of local_counts, "
            "local_waiting_times, and local_propensities must be identical. "
            "Please check your trajectory data."
        )

    # -------------------------------------------
    # Additional sanity checks
    # -------------------------------------------
    warnings_found = False
    prop_length_ref = None

    for state in local_counts.keys():
        count_val = local_counts[state]
        wait_val  = local_waiting_times[state]
        prop_vals = local_propensities[state]
    
        # Check propensity: list length consistent; non-negative whole numbers
        prop_len = len(prop_vals)
        if prop_length_ref is None:
            prop_length_ref = prop_len  # use first state as reference
        elif prop_len != prop_length_ref:
            warnings_found = True
            print(f"WARNING: State {state} has propensity length {prop_len}, expected {prop_length_ref}")
    
        if prop_len == 0:
            warnings_found = True
            print(f"WARNING: State {state} has empty propensity list!")
        
        for val in prop_vals:
            if val < 0:
                warnings_found = True
                print(f"WARNING: negative propensity {val} for state {state}")
            if not is_whole_number(val):
                warnings_found = True
                print(f"WARNING: propensity {val} for state {state} is not integer-valued")

        # Check count_val (non-negative integers)
        if count_val < 0:
            warnings_found = True
            print(f"WARNING: Negative count detected for state {state}: {count_val}")
        if not is_whole_number(count_val):
            warnings_found = True
            print(f"WARNING: Count for state {state} is not an integer: {count_val}")
        
        #Check waiting times (>0 if counts>0 and non-negative)
        if count_val > 0 and wait_val <= 0:
            warnings_found = True
            print(f"WARNING: State {state} has count > 0 ({count_val}) but non-positive waiting time ({wait_val})")
        if wait_val < 0:
            warnings_found = True
            print(f"WARNING: Negative waiting time detected for state {state}: {wait_val}")

    # Final message
    if not warnings_found:
        print("✅ All states processed successfully — no empty propensities, consistent lengths, no negative counts, all waiting times valid.")
    else:
        print("⚠️ Local data extraction completed, but warnings were detected. Review output above.")
    if verbose:
        print(f"Local data extraction complete. {len(local_counts)} states processed.")

    return local_counts, local_waiting_times, local_propensities, selected_deltaX


def local_log_likelihood(local_counts, local_waiting_times, local_propensities, theta):
    """
    Compute the LOCAL log-likelihood function for a given stoichiometric change.

    Vectorized over all states.

    Args:
        local_counts:        dict mapping state x -> counts n_{x,l} for the selected change
        local_waiting_times: dict mapping state x -> cumulative waiting time tau_x
        local_propensities:  dict mapping state x -> propensities g_j(x) (for each compatible reaction)
        theta:               vector of parameters theta_j for each propensity (must be non-negative)
    
    Returns:
        log-likelihood value (float)
    """
    theta = np.asarray(theta)

    # Stack everything into arrays
    states = list(local_propensities.keys())
    G = np.vstack([local_propensities[s] for s in states])       # shape (num_states, num_props)
    N = np.array([local_counts[s] for s in states])              # shape (num_states,)
    T = np.array([local_waiting_times[s] for s in states])      # shape (num_states,)

    # Basic checks
    if len(theta) != G.shape[1]:
        raise ValueError(f"Theta length {len(theta)} does not match propensity length {G.shape[1]}")
    if np.any(theta < 0):
        raise ValueError("All theta values must be non-negative")

    # Compute normalization for all states
    norms = G @ theta  # shape (num_states,)

    # Handle zero normalization
    if np.any(norms == 0):
        zero_mask = (norms == 0)
        if np.any(N[zero_mask] > 0):
            # Observed events but zero propensity -> log-likelihood = -inf
            return -np.inf
        else:
            # No events for these states, log(0) term = 0
            norms[zero_mask] = 1  # temporarily safe for log

    # Vectorized log-likelihood computation
    log_terms = N * np.log(norms)
    exp_terms = T * norms
    log_likelihood_value = np.sum(log_terms - exp_terms)

    return log_likelihood_value

def plot_likelihood_vs_theta_multiplier(local_counts, local_waiting_times, local_propensities, localTheta,
                                        delta=0.5, num_points=50, title=None):
    """
    For a given local theta vector, vary it by a multiplier and plot the log-likelihood.

    Parameters
    ----------
    local_counts, local_waiting_times, local_propensities : arrays
        Local data for the selected stoichiometric change.
    localTheta : np.ndarray
        True parameter vector for this local reaction set.
    delta : float
        Range around 1 to vary the multiplier (1 +/- delta). Default 0.5 gives 0.5x to 1.5x.
    num_points : int
        Number of multiplier points to evaluate. Default 50.
    title : str
        Optional title for the plot.

    Returns
    -------
    multipliers : np.ndarray
        Multipliers evaluated.
    log_likelihood_values : list
        Log-likelihood values for each multiplier.
    max_ll : float
        Maximum log-likelihood observed.
    best_multiplier : float
        Multiplier achieving maximum log-likelihood.
    best_theta : np.ndarray
        Theta corresponding to maximum log-likelihood.
    """
    # Compute total observed counts for this local stoichiometric change
    total_counts = sum(local_counts.values())
    print(f"Total observed jumps for this local stoichiometric change: {total_counts}")


    # Generate the range of multipliers around 1
    multipliers = np.linspace(1 - delta, 1 + delta, num_points)

    # Store log-likelihood values
    log_likelihood_values = []

    # Track the best result
    max_ll = -np.inf
    best_multiplier = None
    best_theta = None

    print(f"Local True Theta = {localTheta}")
    true_ll = local_log_likelihood(local_counts, local_waiting_times, local_propensities, localTheta)
    print(f"Log-Likelihood at True Theta = {true_ll}")

    # Iterate over multipliers
    for rho in multipliers:
        theta_scaled = localTheta * rho  # scale the entire local theta
        ll = local_log_likelihood(local_counts, local_waiting_times, local_propensities, theta_scaled)
        log_likelihood_values.append(ll)

        # Track maximum
        if ll > max_ll:
            max_ll = ll
            best_multiplier = rho
            best_theta = theta_scaled

    print(f"Maximum Log-Likelihood = {max_ll}")
    print(f"Best Multiplier = {best_multiplier}")
    print(f"Best Theta = {best_theta}")

    # Plot results
    plt.figure(figsize=(6,4))
    plt.plot(multipliers, log_likelihood_values, marker='o', label='Log-Likelihood')
    plt.axvline(1.0, color='red', linestyle='--', label='True Theta Multiplier')
    plt.xlabel('Multiplier of Local True Theta')
    plt.ylabel('Log Likelihood')
    plt.title(title if title is not None else 'Log Likelihood vs Multiplier of Local True Theta')
    plt.grid(True)
    plt.legend()
    plt.show()

    return multipliers, log_likelihood_values, max_ll, best_multiplier, best_theta

def get_positive_deltaX_indices_and_values(jump_counts_dict, unique_changes):
    """
    Identify the indices of unique stoichiometric changes that have any positive counts across all states,
    and return the corresponding deltaX values.

    Parameters
    ----------
    jump_counts_dict : dict
        Keys: states (tuples)
        Values: arrays of counts, length = len(unique_changes)
    unique_changes : list or array
        List of unique stoichiometric changes, aligned with count arrays

    Returns
    -------
    positive_indices : list of int
        Indices into unique_changes that have at least one positive count
    positive_deltaX : list
        Entries from unique_changes corresponding to positive_indices
    """
    positive_indices_set = set()
    
    for counts in jump_counts_dict.values():
        positive_indices_set.update(np.nonzero(counts > 0)[0])
    
    positive_indices = sorted(positive_indices_set)
    positive_deltaX = [unique_changes[i] for i in positive_indices]
    
    return positive_indices, positive_deltaX

