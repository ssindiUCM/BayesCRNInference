import numpy as np
import random
import json
import re
from collections import defaultdict

def generate_reactions(complexes, species_names=None):
    """
    Generate all possible reactions from a given set of complexes, 
    and identify reactions that share the same stoichiometric change.

    Parameters
    ----------
    complexes : np.ndarray
        Matrix of shape (num_species, num_complexes) specifying complexes.
        Each column is a complex, rows correspond to species counts.
    species_names : list of str, optional
        Names of the species. If None, defaults to ['S1', 'S2', ...].

    Returns
    -------
    reactant_matrix : np.ndarray
        Matrix of reactants for all reactions (num_species x num_reactions).
    product_matrix : np.ndarray
        Matrix of products for all reactions (num_species x num_reactions).
    stoichiometric_matrix : np.ndarray
        Stoichiometric matrix (product - reactant).
    reaction_names : list of str
        Names of reactions in "A+B_to_2C" format.
    parameter_names : list of str
        Names of the reaction parameters, e.g., ["k0", "k1", ...].
    compatible_reactions : dict
        Dictionary mapping stoichiometric change tuple to list of reaction indices.
        Example: {(1,-1): [0, 3]} → reactions at columns 0 and 3 have the same net change.
    """
    num_species, num_complexes = complexes.shape

    # Default species names
    if species_names is None:
        species_names = [f"S{i+1}" for i in range(num_species)]

    # Compute total number of reactions: all pairwise transitions between complexes
    num_reactions = num_complexes * (num_complexes - 1)  # skip i==j
    reactant_matrix = np.zeros((num_species, num_reactions))
    product_matrix  = np.zeros((num_species, num_reactions))
    reaction_names  = []
    parameter_names = []

    count = 0
    for i in range(num_complexes):  # Loop over reactant complexes (LHS)
        for j in range(num_complexes):  # Loop over product complexes (RHS)
            if i == j:
                continue  # skip trivial reaction: complex -> itself

            # Fill reactant and product matrices for reaction `count`
            reactant_matrix[:, count] = complexes[:, i]
            product_matrix[:, count]  = complexes[:, j]

            # Construct readable reaction names for users
            LHS_species = [
                f"{int(reactant_matrix[idx, count])}{species_names[idx]}"
                if reactant_matrix[idx, count] > 1 else species_names[idx]
                for idx in range(num_species) if reactant_matrix[idx, count] > 0
            ]
            LHSName = "+".join(LHS_species) if LHS_species else "Empty"

            RHS_species = [
                f"{int(product_matrix[idx, count])}{species_names[idx]}"
                if product_matrix[idx, count] > 1 else species_names[idx]
                for idx in range(num_species) if product_matrix[idx, count] > 0
            ]
            RHSName = "+".join(RHS_species) if RHS_species else "Empty"

            # Save reaction and parameter names
            reaction_names.append(f"{LHSName}_to_{RHSName}:")
            parameter_names.append(f"k{count}")

            count += 1

    # Compute stoichiometric matrix: net change for each reaction
    stoichiometric_matrix = product_matrix - reactant_matrix

    # -------------------------------
    # Determine compatible reactions
    # -------------------------------
    # Map each unique stoichiometric change to the list of reaction indices
    compatible_reactions = defaultdict(list)
    for col_idx in range(stoichiometric_matrix.shape[1]):
        # Convert stoichiometric change vector to a hashable tuple
        change_tuple = tuple(stoichiometric_matrix[:, col_idx].astype(int))
        compatible_reactions[change_tuple].append(col_idx)

    # Convert defaultdict to a regular dict for clarity
    compatible_reactions = dict(compatible_reactions)

    return (reactant_matrix, product_matrix, stoichiometric_matrix,
            reaction_names, parameter_names, compatible_reactions)



def get_reaction_indices(reaction_names_full, reactions_to_select):
    """
    Given a full list of reaction names and a list of reactions to select,
    return the indices corresponding to the selected reactions (exact match).

    Parameters
    ----------
    reaction_names_full : list of str
        All reaction names in the full CRN.
    reactions_to_select : list of str
        Subset of reaction names to select (exact match).

    Returns
    -------
    indices : list of int
        Indices of reactions_to_select in reaction_names_full.
    """
    indices = []
    for r in reactions_to_select:
        try:
            i = reaction_names_full.index(r + ":")  # add colon to match your format
            indices.append(i)
        except ValueError:
            raise ValueError(f"Reaction '{r}' not found in reaction_names_full")
    return indices


_valid_ident_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

def _safe_ident(name):
    """Return True if `name` is a safe Python identifier we allow in generated code."""
    return bool(_valid_ident_re.match(name))

def sample_reactions(reactant_matrix, product_matrix, stoichiometric_matrix,
                     reaction_names, parameter_names, species_names,
                     N=20, alpha=2.6, beta=0.4, seed=None):
    """
    Randomly sample a subset of reactions from a full CRN and construct propensity functions.

    PARAMETERS
    ----------
    reactant_matrix : np.ndarray, shape (num_species, num_reactions)
        Stoichiometric coefficients of reactants for all reactions.
    product_matrix : np.ndarray, shape (num_species, num_reactions)
        Stoichiometric coefficients of products for all reactions.
    stoichiometric_matrix : np.ndarray, shape (num_species, num_reactions)
        Stoichiometric changes for all reactions (product - reactant).
    reaction_names : list of str, length num_reactions
        Names of all reactions, e.g., "X1_to_X2".
    parameter_names : list of str, length num_reactions
        Names of the parameters corresponding to each reaction, e.g., "k0", "k1".
    species_names : list of str, length num_species
        Names of all species in the system.
    N : int, optional (default=20)
        Number of reactions to randomly sample.
    alpha : float, optional (default=2.6)
        Shape parameter for gamma distribution used to generate reaction rate constants.
    beta : float, optional (default=0.4)
        Scale parameter for gamma distribution used to generate reaction rate constants.
    seed : int or None, optional
        Random seed for reproducibility.

    RETURNS
    -------
    CRN_stoichiometric_matrix : np.ndarray, shape (num_species, N)
        Stoichiometric matrix for the sampled reactions.
    CRN_reaction_names : list of str, length N
        Names of the sampled reactions.
    CRN_parameter_names : list of str, length N
        Parameter names for the sampled reactions.
    CRN_propensities : list of callable, length N
        Propensity functions for each sampled reaction.
        Each function takes arguments: (parameter, species variables)
        and returns the propensity value based on the reactants.

        Example:
            Reaction:  X1 + X2 -> 2 X2
            Reactant vector: [1, 1]   # 1 X1, 1 X2
            Parameter: k0
            Generated propensity: lambda k0, X1, X2: k0 * X1 * X2

    parameter_values : dict
        Dictionary mapping parameter names to sampled gamma-distributed values.
        e.g., {"k0": 1.23, "k7": 0.45}
    sampled_indices : list of int, length N
        Indices of the sampled reactions in the original full CRN.
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Basic sanity checks
    for pname in parameter_names:
        if not _safe_ident(pname):
            raise ValueError(f"Unsafe parameter name: {pname}")
    for s in species_names:
        if not _safe_ident(s):
            raise ValueError(f"Unsafe species name: {s}")

    # Sample reaction indices
    sampled_indices = random.sample(range(len(reaction_names)), N)

    # Extract sampled stoichiometric matrix and names
    CRN_stoichiometric_matrix = stoichiometric_matrix[:, sampled_indices]
    CRN_reaction_names = [reaction_names[i] for i in sampled_indices]
    CRN_parameter_names = [parameter_names[i] for i in sampled_indices]

    CRN_propensities = []
    parameter_values = {}

    # For debugging/traceback clarity: number of species
    num_species = reactant_matrix.shape[0]

    for idx in sampled_indices:
        reactants = reactant_matrix[:, idx]
        par_name = parameter_names[idx]   # e.g., "k0"
        # sample parameter value (true value)
        pVal = np.random.gamma(alpha, beta)
        parameter_values[par_name] = pVal

        # Build propensity lambda string depending on reactants
        nz = np.nonzero(reactants)[0]
        propensity_string = ""

        if len(nz) == 0:
            # zero reactants, lambda k0: k0
            propensity_string = f"lambda {par_name}: {par_name}"

        elif len(nz) == 1:
            i = nz[0]
            sname = species_names[i]
            multiplicity = int(reactants[i])
            if multiplicity == 1:
                # lambda k0, X: k0*X
                propensity_string = f"lambda {par_name}, {sname}: {par_name}*{sname}"
            elif multiplicity == 2:
                # lambda k0, X: k0*X*(X-1)/2
                propensity_string = (f"lambda {par_name}, {sname}: "
                                     f"{par_name}*{sname}*({sname}-1)/2")
            else:
                # general multiplicity fallback using falling factorial
                # (rare given bi-molecular assumption)
                # produce product like X*(X-1)*... for count terms
                terms = "*".join([f"({sname}-{j})" for j in range(multiplicity)])
                propensity_string = f"lambda {par_name}, {sname}: {par_name}*({terms})/{np.math.factorial(multiplicity)}"

        elif len(nz) == 2:
            i, j = nz
            s1, s2 = species_names[i], species_names[j]
            # lambda k0, X, Y: k0*X*Y
            propensity_string = f"lambda {par_name}, {s1}, {s2}: {par_name}*{s1}*{s2}"

        else:
            raise ValueError("Only up to bi-molecular reactants supported here (<=2 distinct reactant species).")

        # Extra safety: verify all identifiers in propensity_string are allowed
        # Extract tokens that look like identifiers and ensure they are either
        # parameter names or species names
        tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', propensity_string)
        for t in tokens:
            if t in {'lambda', 'return'}:  # python keywords that may appear
                continue
            if t not in parameter_names and t not in species_names:
                # allow numeric names like '2' are not matched by regex; so any token here must be known
                raise ValueError(f"Disallowed identifier in propensity: {t} (prop string: {propensity_string})")

        # Create the function object by eval
        # Safe-ish because we validated all identifiers used are from our allowed lists.
        propensity_func = eval(propensity_string)
        CRN_propensities.append(propensity_func)

    return (CRN_stoichiometric_matrix, CRN_reaction_names, CRN_parameter_names,
            CRN_propensities, parameter_values, sampled_indices)

def build_subCRN_from_names(reactant_matrix, product_matrix, stoichiometric_matrix,
                            reaction_names, parameter_names, species_names,
                            selected_names, rates=None,
                            alpha=2.6, beta=0.4, seed=None):
    """
    Construct a sub-CRN given a selection of reaction names and optional rates.

    This function selects reactions by name from the full CRN and generates
    the associated stoichiometric submatrix, reaction names, parameter names,
    propensities, and parameter values. If no rates are provided, parameters
    are sampled from a Gamma(alpha, beta) distribution.

    Parameters
    ----------
    reactant_matrix : ndarray (num_species x num_reactions)
        Full reactant matrix for the CRN.
    product_matrix : ndarray (num_species x num_reactions)
        Full product matrix for the CRN.
    stoichiometric_matrix : ndarray (num_species x num_reactions)
        Full stoichiometric matrix for the CRN.
    reaction_names : list of str
        Names of all reactions in the full CRN.
    parameter_names : list of str
        Names of all parameters in the full CRN.
    species_names : list of str
        Names of species in the CRN.
    selected_names : list of str
        Names of reactions to select for the sub-CRN.
    rates : list of float, optional
        Rates corresponding to selected reactions. If None, gamma prior is used.
    alpha : float, optional
        Shape parameter of Gamma distribution used if rates=None. Default=2.6.
    beta : float, optional
        Scale parameter of Gamma distribution used if rates=None. Default=0.4.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    CRN_stoichiometric_matrix : ndarray
        Stoichiometric matrix for selected reactions.
    CRN_reaction_names : list of str
        Names of selected reactions.
    CRN_parameter_names : list of str
        Names of parameters corresponding to selected reactions.
    CRN_propensities : list of callables
        Propensity functions for each selected reaction.
    CRN_parameter_values : dict
        Dictionary mapping parameter names to their numeric values.
    CRN_indices : list of int
        Indices of selected reactions in the full CRN.
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Get exact indices of the selected reactions using the robust helper
    CRN_indices = get_reaction_indices(reaction_names, selected_names)

    # Extract submatrices and names
    CRN_stoichiometric_matrix = stoichiometric_matrix[:, CRN_indices]
    CRN_reaction_names = [reaction_names[i] for i in CRN_indices]
    CRN_parameter_names = [parameter_names[i] for i in CRN_indices]

    # Prepare outputs
    CRN_propensities = []
    CRN_parameter_values = {}

    for idx, par_name in zip(CRN_indices, CRN_parameter_names):
        reactants = reactant_matrix[:, idx]

        # Determine parameter value: use provided rates or gamma prior
        if rates is not None:
            rate_val = rates[CRN_indices.index(idx)]  # match index to rates list
        else:
            rate_val = np.random.gamma(alpha, beta)
        CRN_parameter_values[par_name] = rate_val

        # Build propensity function depending on reactants
        nz = np.nonzero(reactants)[0]  # species with non-zero stoichiometry
        propensity_string = ""

        if len(nz) == 0:
            # Zero reactants: lambda k: k
            propensity_string = f"lambda {par_name}: {par_name}"
        elif len(nz) == 1:
            i = nz[0]
            sname = species_names[i]
            multiplicity = int(reactants[i])
            if multiplicity == 1:
                propensity_string = f"lambda {par_name}, {sname}: {par_name}*{sname}"
            elif multiplicity == 2:
                propensity_string = f"lambda {par_name}, {sname}: {par_name}*{sname}*({sname}-1)/2"
            else:
                terms = "*".join([f"({sname}-{j})" for j in range(multiplicity)])
                propensity_string = f"lambda {par_name}, {sname}: {par_name}*({terms})/{np.math.factorial(multiplicity)}"
        elif len(nz) == 2:
            i, j = nz
            s1, s2 = species_names[i], species_names[j]
            propensity_string = f"lambda {par_name}, {s1}, {s2}: {par_name}*{s1}*{s2}"
        else:
            raise ValueError("Only bi-molecular reactions with up to 2 distinct reactant species supported.")

        # Extra safety: verify identifiers
        tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', propensity_string)
        for t in tokens:
            if t in {'lambda', 'return'}:
                continue
            if t not in parameter_names and t not in species_names:
                raise ValueError(f"Disallowed identifier in propensity: {t}")

        # Create function object
        CRN_propensities.append(eval(propensity_string))

    return (CRN_stoichiometric_matrix,
            CRN_reaction_names,
            CRN_parameter_names,
            CRN_propensities,
            CRN_parameter_values,
            CRN_indices)

def generate_single_trajectory(rn, parameter_values, species_names,
                               finalTime=120, minVal=5, maxVal=5, seed=None):
    """
    Generate a single stochastic trajectory for a given CRN, allowing species-specific initial bounds.

    Parameters
    ----------
    rn : CRN object
        The CRN object created from sampled reactions.
    parameter_values : dict
        Parameter values for the CRN.
    species_names : list of str
        Names of the species.
    finalTime : float
        Time until which the trajectory is simulated.
    minVal : int or list/array
        Minimum initial count per species. If int, applied to all species. If list/array, must match number of species.
    maxVal : int or list/array
        Maximum initial count per species. If int, applied to all species. If list/array, must match number of species.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    time_list : list
        Times at which states are recorded.
    state_list : list of dict
        State of each species at each time point.
    """
    if seed is not None:
        np.random.seed(seed)

    num_species = len(species_names)

    # Convert single int to array for species-specific bounds
    if np.isscalar(minVal):
        minVal_array = np.full(num_species, minVal, dtype=int)
    else:
        minVal_array = np.array(minVal, dtype=int)
        if len(minVal_array) != num_species:
            raise ValueError("Length of minVal array must match number of species.")

    if np.isscalar(maxVal):
        maxVal_array = np.full(num_species, maxVal, dtype=int)
    else:
        maxVal_array = np.array(maxVal, dtype=int)
        if len(maxVal_array) != num_species:
            raise ValueError("Length of maxVal array must match number of species.")

    # Generate random initial counts per species
    initial_counts = np.array([
        np.random.randint(low=min_val, high=max_val + 1)
        for min_val, max_val in zip(minVal_array, maxVal_array)
    ])
    initial_state = dict(zip(species_names, initial_counts))
    print(f"Initial state: {initial_state}")

    # Run SSA simulation
    time_list, state_list = rn.SSA(initial_state, parameter_values, 0, finalTime)

    # Plot trajectory
    rn.plot_trajectories(time_list, state_list)

    return time_list, state_list


def save_trajectory(time_list, state_list, filename):
    """
    Save SSA trajectory to a JSON file.

    time_list: list or ndarray of time points
    state_list: list or ndarray of states (each state can be ndarray)
    """
    # Convert numpy arrays to lists
    time_list_serializable = [float(t) for t in time_list]
    state_list_serializable = [s.tolist() if isinstance(s, np.ndarray) else s for s in state_list]

    data = {
        "time": time_list_serializable,
        "states": state_list_serializable
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Trajectory saved to {filename}")


def load_trajectory(filename):
    """
    Load a SSA trajectory from a JSON file.

    Returns:
        time_list: numpy array of time points
        state_list: list of numpy arrays for each state
    """
    with open(filename, "r") as f:
        data = json.load(f)

    time_list = np.array(data["time"], dtype=float)
    state_list = [np.array(state, dtype=float) for state in data["states"]]

    return time_list, state_list

def propensity_values(x, reactant_matrix, j_values):
    """
    Compute propensities for a vector of reactions (j_values) given the current state.

    Args:
        x (array-like): Current state vector containing populations of each species.
        reactant_matrix (2D array-like): Matrix with stoichiometric coefficients for reactants.
        j_values (array-like): Indices of reactions to compute propensities for.

    Returns:
        np.ndarray: Vector of propensities corresponding to each j in j_values.
    """
    x = np.asarray(x)  # Ensure state vector is NumPy array
    reactant_matrix = np.asarray(reactant_matrix)
    
    propensities = np.zeros(len(j_values))

    for idx, j in enumerate(j_values):
        propensity = 1
        # Loop over species instead of hard-coding 3
        for i in range(len(x)):
            r = reactant_matrix[i, j]
            if r == 2:
                propensity *= x[i] * (x[i] - 1) / 2  # combinatorial factor
            else:
                propensity *= x[i] ** r  # 0^0 = 1, 1^1 = x[i]
        propensities[idx] = propensity

    return propensities

def parse_trajectory(state_list, time_list, reactant_matrix, compatible_reactions, verbose=True):
    """
    Parse a single stochastic trajectory to compute, for each unique state:
      - counts of each unique stoichiometric change (jump_counts)
      - cumulative waiting times for each change (waiting_times)
      - propensity arrays for each change at that state (propensities)

    Assumes a single trajectory (state_list, time_list). If you have multiple
    trajectories, call this function for each and combine the results in post-processing.

    Args:
        state_list (list[np.ndarray] or np.ndarray): ordered sequence of state vectors.
        time_list  (list[float] or np.ndarray): ordered sequence of times corresponding to state_list.
        reactant_matrix (np.ndarray): reactant stoichiometry matrix (num_species x num_reactions).
        compatible_reactions (dict): mapping from stoichiometric change tuple -> list of reaction indices.
        verbose (bool): print progress messages (default True).

    Returns:
        unique_changes (list[tuple]): list of unique stoichiometric change vectors (tuples), order preserved.
        unique_states  (list[tuple]): list of unique visited states (tuples).
        jump_counts    (defaultdict): state_key -> np.array counts of length len(unique_changes).
        waiting_times  (defaultdict): state_key -> np.array waiting times of length len(unique_changes).
        propensities   (defaultdict): state_key -> list (length len(unique_changes)) of propensity arrays
                                      (each entry is an array of propensities for compatible reactions).
    """
    import numpy as np
    from collections import defaultdict

    # ---------- Input validation ----------
    if state_list is None or time_list is None:
        raise ValueError("state_list and time_list must be provided (single trajectory).")

    # Convert to numpy-friendly sequences (but keep original list semantics)
    # Accept both Python lists and numpy arrays.
    if isinstance(state_list, np.ndarray):
        # expect shape (T, num_species) or (T,) of arrays; convert to list of 1-D arrays
        state_list = [np.asarray(s) for s in state_list]
    else:
        state_list = [np.asarray(s) for s in state_list]

    if isinstance(time_list, np.ndarray):
        time_list = list(time_list)
    else:
        time_list = list(time_list)

    if len(state_list) != len(time_list):
        raise ValueError("state_list and time_list must have equal length.")

    if len(state_list) < 2:
        raise ValueError("Trajectory too short: need at least two timepoints.")

    # ---------- Prepare unique changes ----------
    unique_changes = list(compatible_reactions.keys())
    num_unique_changes = len(unique_changes)
    if verbose:
        print(f"Tracking {num_unique_changes} unique stoichiometric changes (single trajectory).")
        print(f"Trajectory length: {len(state_list)} timepoints.  Will iterate to len-2 to avoid final non-jump.")

    # ---------- Initialize containers ----------
    jump_counts   = defaultdict(lambda: np.zeros(num_unique_changes, dtype=int))
    waiting_times = defaultdict(lambda: np.zeros(num_unique_changes, dtype=float))
    propensities  = defaultdict(lambda: [[] for _ in range(num_unique_changes)])

    # ---------- Main loop over states (stop at len-2) ----------
    # (we do not compute the delta that involves the final printed state at t_final)
    for i in range(len(state_list) - 2):   # <-- your desired stopping point
        Xcurr = np.asarray(state_list[i])
        Xnext = np.asarray(state_list[i + 1])
        deltaX = Xnext - Xcurr
        deltaT = time_list[i + 1] - time_list[i]

        # normalize delta to ints for dictionary keys
        try:
            deltaX_tuple = tuple(deltaX.astype(int))
        except Exception:
            # fallback in case deltaX is list-like
            deltaX_tuple = tuple(int(x) for x in deltaX)

        if deltaX_tuple in compatible_reactions:
            state_key = tuple(int(x) for x in Xcurr)
            idx = unique_changes.index(deltaX_tuple)   # index in unique_changes
            jump_counts[state_key][idx] += 1
            waiting_times[state_key][idx] += deltaT
        else:
            if verbose:
                print(f"\tStep {i}: deltaX {deltaX_tuple} not in compatible_reactions (ignored)")

    # ---------- Compute propensities for each observed state ----------
    # We only compute propensities for states that appeared in jump_counts
    for state_key in list(jump_counts.keys()):
        for idx, deltaX_tuple in enumerate(unique_changes):
            reaction_indices = compatible_reactions[deltaX_tuple]  # list of reaction indices (order preserved)
            # compute propensity vector for this state and this stoich-change (preserves order)
            propensities[state_key][idx] = propensity_values(state_key, reactant_matrix, reaction_indices)

    unique_states = list(jump_counts.keys())

    if verbose:
        print(f"Finished parsing single trajectory. Observed {len(unique_states)} unique states.")
    return unique_changes, unique_states, jump_counts, waiting_times, propensities

