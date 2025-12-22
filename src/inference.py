import numpy as np
import random
import json
import re
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.optimize import minimize


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
    #local_waiting_times = {state: times[index_to_extract] for state, times in waiting_times.items()}
    local_waiting_times = {state: np.sum(times) for state, times in waiting_times.items()}
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


def local_log_likelihood(Filtered_X_Counts,Filtered_T_Vals, Filtered_X_Propensities, theta):
    """
    Compute the LOCAL log-likelihood function given the data and parameters theta.

    **Precomputed**
    Filtered_X_Counts:       Dictionary of counts for each state x (n_{x,l}.
    Filtered_T_Vals:         Dictionary of target values for each state x. (tau_{x})
    Filtered_X_Propensities: Dictionary of propensities g_j(x) for each state x. (lambda_{x,l})

    **Parameter**
    theta: Vector of parameters (theta_j) for each propensity.
    
    Returns the log-likelihood value.
    """
    log_likelihood_value = 0

    numEvents = sum(Filtered_X_Counts.values())

    if np.linalg.norm(theta, ord=2) == 0:
        if numEvents == 0: #We have no observations!
            return 0       #Return log(1)
        else:               #We observe data! But no rates
            return -np.inf #Return log(0)
    
    # Iterate over all states in the dictionary
    for state in Filtered_X_Propensities:
        # Make sure that the state exists in all dictionaries
        if state in Filtered_X_Counts and state in Filtered_T_Vals:
            g_x   = Filtered_X_Propensities[state]  # propensities g_j(x)
            n_xl  = Filtered_X_Counts[state]        # observed counts n_{x, l}
            tau_x = Filtered_T_Vals[state]          # target values tau_{x}
        
            # Compute the normalization term: sum_{j} theta_j * g_j(x)
            normalization = np.dot(theta, g_x)  # This is the sum_{j} theta_j * g_j(x)

            # Handle zero or negative normalization explicitly
            if normalization == 0 and n_xl > 0:
                #print(f"We should never be here!")
                #print(f"State = {state}")
                #print(f"Theta = {theta}")
                #print(f"g_x = {g_x}")
                #print(f"n_xl = {n_xl}")
                return -np.inf  # log(0) is -infinity, and this is proper behavior
            elif normalization == 0 and n_xl == 0:
                term1 = 0
            else:
                term1 = n_xl * np.log(normalization)
            
            term2 = tau_x * normalization
            log_likelihood_value += term1 - term2

            # Calculate the log-likelihood for this state
            #if normalization > 0:
            #    term1 = n_xl * np.log(normalization)  # n_{x,l} * log( sum_j theta_j g_j(x) )
            #    term2 = tau_x * normalization         # tau_{x} * sum_j theta_j g_j(x)
            #    log_likelihood_value += term1 - term2
    
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

def plot_likelihood_vs_theta_interpolation(local_counts, local_waiting_times, local_propensities,
                                           theta1, theta2, num_points=50, title=None):
    """
    Plot the log-likelihood as a function of alpha, where
    theta(alpha) = alpha * theta1 + (1 - alpha) * theta2.

    Parameters
    ----------
    local_counts, local_waiting_times, local_propensities : arrays/dicts
        Local data for the selected stoichiometric change.
    theta1, theta2 : np.ndarray
        Two true parameter vectors to interpolate between.
    num_points : int
        Number of alpha points to evaluate. Default 50.
    title : str
        Optional title for the plot.

    Returns
    -------
    alphas : np.ndarray
        Alpha values evaluated (0 to 1).
    log_likelihood_values : list
        Log-likelihood values for each alpha.
    max_ll : float
        Maximum log-likelihood observed.
    best_alpha : float
        Alpha achieving maximum log-likelihood.
    best_theta : np.ndarray
        Theta corresponding to maximum log-likelihood.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate alpha values between 0 and 1
    alphas = np.linspace(0, 1, num_points)
    log_likelihood_values = []

    # Track the best result
    max_ll = -np.inf
    best_alpha = None
    best_theta = None

    print(f"Theta1 = {theta1}")
    print(f"Theta2 = {theta2}")

    for alpha in alphas:
        theta_alpha = alpha * theta1 + (1 - alpha) * theta2
        ll = local_log_likelihood(local_counts, local_waiting_times, local_propensities, theta_alpha)
        log_likelihood_values.append(ll)

        if ll > max_ll:
            max_ll = ll
            best_alpha = alpha
            best_theta = theta_alpha

    print(f"Maximum Log-Likelihood = {max_ll}")
    print(f"Best Alpha = {best_alpha}")
    print(f"Best Theta = {best_theta}")

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(alphas, log_likelihood_values, marker='o', label='Log-Likelihood')
    plt.axvline(1.0, color='red', linestyle='--', label='Theta1')
    plt.axvline(0.0, color='green', linestyle='--', label='Theta2')
    plt.xlabel(r'Alpha ($\theta(\alpha) = \alpha \theta_1 + (1-\alpha) \theta_2$)')
    plt.ylabel('Log Likelihood')
    plt.title(title if title is not None else 'Log Likelihood vs Theta Interpolation')
    plt.grid(True)
    plt.legend()
    plt.show()

    return alphas, log_likelihood_values, max_ll, best_alpha, best_theta


def get_positive_deltaX_indices_and_values(jump_counts_dict, unique_changes, verbose=True):
    positive_indices_set = set()
    
    for state_idx, counts in enumerate(jump_counts_dict.values()):
        pos_indices_this_state = np.nonzero(counts > 0)[0]
        if verbose and len(pos_indices_this_state) > 0:
            print(f"State {state_idx}: positive indices = {pos_indices_this_state}, counts = {counts[pos_indices_this_state]}")
        positive_indices_set.update(pos_indices_this_state)
    
    if verbose:
        print(f"\tLength of unique_changes:", len(unique_changes))
        print(f"\nAll positive indices set across all states (unsorted): {positive_indices_set}")
    
    positive_indices = sorted(positive_indices_set)
    positive_deltaX = [unique_changes[i] for i in positive_indices]
    
    if verbose:
        print(f"Sorted positive indices: {positive_indices}")
        print(f"Corresponding ΔX values: {positive_deltaX}\n")
    
    return positive_indices, positive_deltaX

def plot_l1_path(lambdas, theta_path, obj_values, results_dir, index, reaction_names=None, true_theta=None):
    """
    Plot L1 path of parameters vs lambda and the objective value.
    
    Parameters
    ----------
    lambdas : array-like
        Array of lambda values used in L1 sweep
    theta_path : array-like, shape (num_lambdas, num_parameters)
        Optimized theta for each lambda
    obj_values : array-like
        Objective function values for each lambda
    results_dir : str
        Directory to save plots
    index : int
        Index of the stoichiometric change
    reaction_names : list of str, optional
        Names of reactions
    true_theta : array-like, optional
        True parameter values for plotting as dashed lines
    """
    n_params = theta_path.shape[1]
    fig, axes = plt.subplots(1, n_params + 1, figsize=(4*(n_params+1), 4))

    # Plot each parameter vs lambda
    for i in range(n_params):
        axes[i].plot(lambdas, theta_path[:, i], marker='o', label=f"θ_{i}")
        axes[i].set_xscale('log')
        axes[i].set_xlabel('Lambda')
        axes[i].set_ylabel('Parameter Value')
        axes[i].grid(True, alpha=0.3)
        if reaction_names is not None:
            axes[i].set_title(reaction_names[i])
        # Plot horizontal line for true theta
        if true_theta is not None:
            axes[i].axhline(true_theta[i], color='red', linestyle='--', label='True θ')
        axes[i].legend()

    # Plot objective value
    axes[-1].plot(lambdas, obj_values, marker='o', color='black', label='Objective')
    axes[-1].set_xscale('log')
    axes[-1].set_xlabel('Lambda')
    axes[-1].set_ylabel('Objective Value')
    axes[-1].grid(True, alpha=0.3)
    axes[-1].legend()

    plt.tight_layout()
    filename = os.path.join(results_dir, f"L1_result_index_{index}.png")
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def run_l1_path(
    local_counts,
    local_waiting_times,
    local_propensities,
    theta_init,
    lambdas
):
    """
    Runs an L1-regularized optimization path.
    """
    num_lambdas = len(lambdas)
    dim = len(theta_init)

    theta_path = np.zeros((num_lambdas, dim))
    obj_values = np.zeros(num_lambdas)

    theta_current = theta_init.copy()

    bounds = [(0.0, None)] * dim  # enforce non-negativity

    for i, lam in enumerate(lambdas):
        res = minimize(
            l1_objective,
            theta_current,
            args=(local_counts, local_waiting_times, local_propensities, lam),
            method="L-BFGS-B",
            bounds=bounds
        )

        theta_current = res.x
        theta_path[i, :] = theta_current
        obj_values[i] = -res.fun  # convert back to max objective

        # Optional early stopping: everything is zero
        if np.all(theta_current < 1e-8):
            theta_path = theta_path[:i+1]
            obj_values = obj_values[:i+1]
            lambdas = lambdas[:i+1]
            break

    return lambdas, theta_path, obj_values

def l1_objective(theta, counts, waiting_times, propensities, lam):
    """
    Negative penalized log-likelihood
    """
    ll = local_log_likelihood(counts, waiting_times, propensities, theta)
    penalty = lam * np.sum(np.abs(theta))
    return -(ll - penalty)
