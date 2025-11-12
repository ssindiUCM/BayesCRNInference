import numpy as np
import random
import json
import re
from collections import defaultdict

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


#def local_log_likelihood(local_counts, local_waiting_times, local_propensities, theta):
#    """
#    Compute the LOCAL log-likelihood function for a given stoichiometric change.
#
#    Args:
#        local_counts:        dict mapping state x -> counts n_{x,l} for the selected change
#        local_waiting_times: dict mapping state x -> cumulative waiting time tau_x
#        local_propensities:  dict mapping state x -> propensities g_j(x) (for each compatible reaction)
#
#        Parameter: theta: vector of parameters theta_j for each propensity
#    
#    Returns:
#        log-likelihood value (float)
#    """
#    log_likelihood_value = 0
#
#    numEvents = sum(local_counts.values())
#
#    # Check for zero theta
#    if np.linalg.norm(theta, ord=2) == 0:
#        if numEvents == 0:
#            return 0       # no events observed, log-likelihood = log(1)
#        else:
#            return -np.inf # events observed, but zero rates -> log-likelihood = -infinity
#
#    # Loop over all states
#    for state in local_propensities:
#        # Make sure the state exists in all dictionaries
#        if state in local_counts and state in local_waiting_times:
#            g_x   = local_propensities[state]   # propensities g_j(x)
#            n_xl  = local_counts[state]         # observed counts n_{x,l}
#            tau_x = local_waiting_times[state]  # cumulative waiting time tau_x
#
#            # Normalization term: sum_j theta_j * g_j(x)
#            normalization = np.dot(theta, g_x)
#
#            # Handle zero normalization
#            if normalization == 0 and n_xl > 0:
#                # Observed events but zero propensity -> log(0)
#                return -np.inf
#            elif normalization == 0 and n_xl == 0:
#                term1 = 0
#            else:
#                term1 = n_xl * np.log(normalization)
#            
#            term2 = tau_x * normalization
#            log_likelihood_value += term1 - term2
#
#    return log_likelihood_value

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
