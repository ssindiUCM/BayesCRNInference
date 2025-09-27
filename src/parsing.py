import numpy as np
import random
import json
import re


def generate_reactions(complexes, species_names=None):
    """
    Generate all possible reactions from a given set of complexes.

    Parameters
    ----------
    complexes : np.ndarray
        Matrix of shape (num_species, num_complexes) specifying complexes.
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
    """
    num_species, num_complexes = complexes.shape

    # Default species names
    if species_names is None:
        species_names = [f"S{i+1}" for i in range(num_species)]
    
    num_reactions = num_complexes * (num_complexes - 1)
    reactant_matrix = np.zeros((num_species, num_reactions))
    product_matrix  = np.zeros((num_species, num_reactions))
    reaction_names  = []
    parameter_names = []

    count = 0
    for i in range(num_complexes):  # Reactant (LHS)
        for j in range(num_complexes):  # Product (RHS)
            if i == j:
                continue

            reactant_matrix[:, count] = complexes[:, i]
            product_matrix[:, count]  = complexes[:, j]

            # Construct LHS name
            LHS_species = [
                f"{int(reactant_matrix[idx, count])}{species_names[idx]}"
                if reactant_matrix[idx, count] > 1 else species_names[idx]
                for idx in range(num_species) if reactant_matrix[idx, count] > 0
            ]
            LHSName = "+".join(LHS_species) if LHS_species else "Empty"

            # Construct RHS name
            RHS_species = [
                f"{int(product_matrix[idx, count])}{species_names[idx]}"
                if product_matrix[idx, count] > 1 else species_names[idx]
                for idx in range(num_species) if product_matrix[idx, count] > 0
            ]
            RHSName = "+".join(RHS_species) if RHS_species else "Empty"

            # Reaction and parameter names
            reaction_names.append(f"{LHSName}_to_{RHSName}:")
            parameter_names.append(f"k{count}")

            count += 1

    stoichiometric_matrix = product_matrix - reactant_matrix

    return reactant_matrix, product_matrix, stoichiometric_matrix, reaction_names, parameter_names


_valid_ident_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

def _safe_ident(name):
    """Return True if `name` is a safe Python identifier we allow in generated code."""
    return bool(_valid_ident_re.match(name))

def sample_reactions(reactant_matrix, product_matrix, stoichiometric_matrix,
                     reaction_names, parameter_names, species_names,
                     N=20, alpha=2.6, beta=0.4, seed=None):
    """
    Construct a random sub-CRN by sampling N reactions from the full set and
    create propensity functions whose signatures match parameter/species names.

    This version builds the same kind of lambda strings your original code used,
    but checks that parameter and species names are safe identifiers before eval.
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


def generate_single_trajectory(rn, parameter_values, species_names,
                               finalTime=120, minVal=5, maxVal=5, seed=None):
    """
    Generate a single stochastic trajectory for a given CRN.

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
    minVal : int
        Minimum initial count per species.
    maxVal : int
        Maximum initial count per species.
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

    # Generate random initial counts
    initial_counts = np.random.randint(low=minVal, high=maxVal + 1, size=len(species_names))
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
