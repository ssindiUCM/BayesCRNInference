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

    # Print out all states and the length of their propensity lists
    print("\nSanity check: propensity lengths for each state:")
    for state in local_propensities.keys():
        prop_len = len(local_propensities[state])
        print(f"State {state}: propensity length = {prop_len}")
        if prop_len == 0:
            print(f"WARNING: State {state} has empty propensity list!")

    # Additional sanity checks
    for state, count_val in local_counts.items():
        if count_val < 0:
            print(f"WARNING: Negative count detected for state {state}: {count_val}")
    for state, wait_val in local_waiting_times.items():
        if wait_val < 0:
            print(f"WARNING: Negative waiting time detected for state {state}: {wait_val}")

    if verbose:
        print(f"\nLocal data extraction complete. {len(local_counts)} states processed.")

    return local_counts, local_waiting_times, local_propensities, selected_deltaX


def local_log_likelihood(local_counts, local_waiting_times, local_propensities, theta):
    """
    Compute the LOCAL log-likelihood function for a given stoichiometric change.

    Args:
        local_counts:        dict mapping state x -> counts n_{x,l} for the selected change
        local_waiting_times: dict mapping state x -> cumulative waiting time tau_x
        local_propensities:  dict mapping state x -> propensities g_j(x) (for each compatible reaction)

        Parameter: theta: vector of parameters theta_j for each propensity
    
    Returns:
        log-likelihood value (float)
    """
    log_likelihood_value = 0

    numEvents = sum(local_counts.values())

    # Check for zero theta
    if np.linalg.norm(theta, ord=2) == 0:
        if numEvents == 0:
            return 0       # no events observed, log-likelihood = log(1)
        else:
            return -np.inf # events observed, but zero rates -> log-likelihood = -infinity

    # Loop over all states
    for state in local_propensities:
        # Make sure the state exists in all dictionaries
        if state in local_counts and state in local_waiting_times:
            g_x   = local_propensities[state]   # propensities g_j(x)
            n_xl  = local_counts[state]         # observed counts n_{x,l}
            tau_x = local_waiting_times[state]  # cumulative waiting time tau_x

            # Normalization term: sum_j theta_j * g_j(x)
            normalization = np.dot(theta, g_x)

            # Handle zero normalization
            if normalization == 0 and n_xl > 0:
                # Observed events but zero propensity -> log(0)
                return -np.inf
            elif normalization == 0 and n_xl == 0:
                term1 = 0
            else:
                term1 = n_xl * np.log(normalization)
            
            term2 = tau_x * normalization
            log_likelihood_value += term1 - term2

    return log_likelihood_value
