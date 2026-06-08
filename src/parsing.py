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
    unique_changes : list of tuples (unique stochoimetric changes)
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
    # Determine unique_changes and compatible_reactions
    # -------------------------------
    unique_changes = []
    compatible_reactions = {}

    for col_idx in range(stoichiometric_matrix.shape[1]):

        # Convert stoichiometric change vector to a hashable tuple
        change_tuple = tuple(stoichiometric_matrix[:, col_idx].astype(int))

        # First time we see this ΔX → create new entry
        if change_tuple not in compatible_reactions:
            compatible_reactions[change_tuple] = []
            unique_changes.append(change_tuple)

        # Add this reaction index to the ΔX entry
        compatible_reactions[change_tuple].append(col_idx)

    return (reactant_matrix, product_matrix, stoichiometric_matrix,
            reaction_names, parameter_names, unique_changes, compatible_reactions)



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

def build_CRN_bySamplingReactions(reactant_matrix, product_matrix, stoichiometric_matrix,
                                  reaction_names, parameter_names, species_names,
                                  N=20, alpha=2.6, beta=0.4, seed=None, verbose=True):
    """
    Randomly sample a subset of reactions from a full CRN and construct propensity functions.

    This function returns both the sampled CRN and a full-length "ground truth" parameter
    vector (`trueTheta`) for the entire CRN. Optionally prints a readable summary table
    of sampled reactions and parameter values.

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
    verbose : bool, optional (default=True)
        If True, prints a formatted summary of sampled reactions, their indices,
        parameter values, and the full trueTheta vector.

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
    trueTheta : np.ndarray
        Full-length vector of reaction parameters for the entire CRN.
        Sampled reactions have their gamma-sampled value; unsampled remain 0.
    parameter_values : dict
        Dictionary mapping sampled parameter names to their gamma-distributed values.
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
    
    # Set up trueTheta: full-length CRN vector, zeros initially
    trueTheta = np.zeros(stoichiometric_matrix.shape[1])

    CRN_propensities = []
    parameter_values = {}

    num_species = reactant_matrix.shape[0]

    for idx in sampled_indices:
        reactants = reactant_matrix[:, idx]
        par_name  = parameter_names[idx]

        # Sample parameter value
        pVal = np.random.gamma(alpha, beta)
        parameter_values[par_name] = pVal
        trueTheta[idx] = pVal

        # Build propensity lambda string depending on reactants
        nz = np.nonzero(reactants)[0]
        propensity_string = ""

        if len(nz) == 0:
            # Zero reactants
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
            raise ValueError("Only up to bi-molecular reactants supported here (<=2 distinct reactant species).")

        # Safety check on identifiers
        tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', propensity_string)
        for t in tokens:
            if t in {'lambda', 'return'}:
                continue
            if t not in parameter_names and t not in species_names:
                raise ValueError(f"Disallowed identifier in propensity: {t} (prop string: {propensity_string})")

        # Create function object
        propensity_func = eval(propensity_string)
        CRN_propensities.append(propensity_func)

    # Verbose output
    if verbose:
        print(f"\nSampling {N} reactions out of {stoichiometric_matrix.shape[1]} total reactions in the CRN.\n")
        print(f"{'Index':<5} {'Param':<6} {'Reaction Name':<25} {'Value':<8}")
        print("-"*50)
        for idx, pname, rname in zip(sampled_indices, CRN_parameter_names, CRN_reaction_names):
            print(f"{idx:<5} {pname:<6} {rname:<25} {parameter_values[pname]:<8.3f}")
        print("\nFull trueTheta vector (length {}):".format(len(trueTheta)))
        #print(trueTheta)

    return (CRN_stoichiometric_matrix, CRN_reaction_names, CRN_parameter_names,
            CRN_propensities, trueTheta, parameter_values, sampled_indices)

def build_CRN_bySamplingReactions_withConstraints(
        reactant_matrix, product_matrix, stoichiometric_matrix,
        reaction_names, parameter_names, species_names,
        unique_changes, compatible_reactions,
        N=20, n_ambiguous_changes=2, min_reactions_per_change=2,
        alpha=2.6, beta=0.4, seed=None, verbose=True):
    """
    Sample N reactions from a full CRN, with a structural constraint:
    at least `n_ambiguous_changes` stoichiometric changes must each be
    represented by at least `min_reactions_per_change` sampled reactions.

    This ensures the resulting CRN contains genuine stoichiometric ambiguity —
    i.e., the net change alone does not uniquely identify the reaction.

    Parameters
    ----------
    reactant_matrix, product_matrix, stoichiometric_matrix : np.ndarray
    reaction_names, parameter_names, species_names : lists
    unique_changes : list of tuples
        All unique stoichiometric change vectors (from generate_reactions).
    compatible_reactions : dict
        Maps stoichiometric change tuple -> list of reaction indices.
    N : int
        Total number of reactions to sample.
    n_ambiguous_changes : int
        Number of stoichiometric changes that must each have >=min_reactions_per_change
        reactions sampled. Default 2.
    min_reactions_per_change : int
        Minimum number of reactions sampled per guaranteed ambiguous change. Default 2.
    alpha, beta : float
        Gamma distribution parameters for rate constants.
    seed : int or None
    verbose : bool

    Returns
    -------
    Same 7-tuple as build_CRN_bySamplingReactions:
        CRN_stoichiometric_matrix, CRN_reaction_names, CRN_parameter_names,
        CRN_propensities, trueTheta, parameter_values, sampled_indices
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # --- Validate N is large enough ---
    min_required = n_ambiguous_changes * min_reactions_per_change
    if N < min_required:
        raise ValueError(
            f"N={N} is too small: need at least {n_ambiguous_changes} changes × "
            f"{min_reactions_per_change} reactions = {min_required} reactions."
        )

    # --- Phase 1: find stoichiometric changes with enough compatible reactions ---
    # Filter to only changes that have >= min_reactions_per_change options
    eligible_changes = [
        ch for ch in unique_changes
        if len(compatible_reactions[ch]) >= min_reactions_per_change
    ]
    if len(eligible_changes) < n_ambiguous_changes:
        raise ValueError(
            f"Only {len(eligible_changes)} stoichiometric change(s) have "
            f">= {min_reactions_per_change} compatible reactions, "
            f"but n_ambiguous_changes={n_ambiguous_changes} were requested."
        )

    # Randomly pick n_ambiguous_changes of those eligible changes
    chosen_changes = random.sample(eligible_changes, n_ambiguous_changes)

    if verbose:
        print(f"\n[Constraint] Guaranteeing {n_ambiguous_changes} ambiguous stoichiometric changes:")
        for ch in chosen_changes:
            print(f"  ΔX={ch}  →  {len(compatible_reactions[ch])} compatible reactions available")

    # For each chosen change, sample min_reactions_per_change reactions from its pool
    guaranteed_indices = []
    for ch in chosen_changes:
        pool = compatible_reactions[ch]
        chosen = random.sample(pool, min_reactions_per_change)
        guaranteed_indices.extend(chosen)

    # Deduplicate (in case two changes share a reaction index — rare but possible)
    guaranteed_indices = list(dict.fromkeys(guaranteed_indices))

    if verbose:
        print(f"  Guaranteed reaction indices ({len(guaranteed_indices)} total): {guaranteed_indices}")

    # --- Phase 2: fill remaining slots randomly ---
    n_remaining = N - len(guaranteed_indices)
    all_indices = list(range(len(reaction_names)))
    remaining_pool = [i for i in all_indices if i not in guaranteed_indices]

    if n_remaining > len(remaining_pool):
        raise ValueError(
            f"Cannot sample {n_remaining} additional reactions: "
            f"only {len(remaining_pool)} reactions remain after guaranteed set."
        )

    filler_indices = random.sample(remaining_pool, n_remaining)
    sampled_indices = guaranteed_indices + filler_indices

    if verbose:
        print(f"  Filler reactions sampled: {len(filler_indices)}")
        print(f"  Total sampled: {len(sampled_indices)}\n")

    # --- Build CRN (identical logic to build_CRN_bySamplingReactions) ---
    CRN_stoichiometric_matrix = stoichiometric_matrix[:, sampled_indices]
    CRN_reaction_names  = [reaction_names[i]  for i in sampled_indices]
    CRN_parameter_names = [parameter_names[i] for i in sampled_indices]

    trueTheta        = np.zeros(stoichiometric_matrix.shape[1])
    CRN_propensities = []
    parameter_values = {}
    num_species      = reactant_matrix.shape[0]

    for idx in sampled_indices:
        reactants = reactant_matrix[:, idx]
        par_name  = parameter_names[idx]

        pVal = np.random.gamma(alpha, beta)
        parameter_values[par_name] = pVal
        trueTheta[idx] = pVal

        nz = np.nonzero(reactants)[0]
        if len(nz) == 0:
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
                propensity_string = (
                    f"lambda {par_name}, {sname}: "
                    f"{par_name}*({terms})/{np.math.factorial(multiplicity)}"
                )
        elif len(nz) == 2:
            i, j = nz
            s1, s2 = species_names[i], species_names[j]
            propensity_string = f"lambda {par_name}, {s1}, {s2}: {par_name}*{s1}*{s2}"
        else:
            raise ValueError("Only up to bimolecular reactants supported (<=2 distinct species).")

        # Safety check
        tokens = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', propensity_string)
        for t in tokens:
            if t in {'lambda', 'return'}:
                continue
            if t not in parameter_names and t not in species_names:
                raise ValueError(f"Disallowed identifier in propensity: {t}")

        CRN_propensities.append(eval(propensity_string))

    # --- Verbose summary ---
    if verbose:
        print(f"Sampling {N} reactions out of {stoichiometric_matrix.shape[1]} total.\n")
        print(f"{'Index':<6} {'Param':<8} {'Reaction Name':<30} {'Value':<8} {'Guaranteed'}")
        print("-" * 65)
        for idx, pname, rname in zip(sampled_indices, CRN_parameter_names, CRN_reaction_names):
            tag = "✓" if idx in guaranteed_indices else ""
            print(f"{idx:<6} {pname:<8} {rname:<30} {parameter_values[pname]:<8.3f} {tag}")
        print(f"\nFull trueTheta vector (length {len(trueTheta)}).")

    return (CRN_stoichiometric_matrix, CRN_reaction_names, CRN_parameter_names,
            CRN_propensities, trueTheta, parameter_values, sampled_indices)

def build_CRN_byNameSelection(reactant_matrix, product_matrix, stoichiometric_matrix,
                              reaction_names, parameter_names, species_names,
                              selected_names, rates=None,
                              alpha=2.6, beta=0.4, seed=None, verbose=True):
    """
    Construct a CRN by selecting specific reactions by name, with optional rates.

    PARAMETERS
    ----------
    reactant_matrix : np.ndarray, shape (num_species, num_reactions)
        Stoichiometric coefficients of reactants for all reactions.
    product_matrix : np.ndarray, shape (num_species, num_reactions)
        Stoichiometric coefficients of products for all reactions.
    stoichiometric_matrix : np.ndarray, shape (num_species, num_reactions)
        Stoichiometric changes for all reactions (product - reactant).
    reaction_names : list of str, length num_reactions
        Names of all reactions in the full CRN.
    parameter_names : list of str, length num_reactions
        Names of the parameters corresponding to each reaction.
    species_names : list of str, length num_species
        Names of all species in the system.
    selected_names : list of str
        Names of reactions to select for the CRN.
    rates : list of float, optional
        If provided, exact values for the selected reactions; otherwise, sampled from Gamma(alpha, beta).
    alpha : float, optional (default=2.6)
        Shape parameter for gamma distribution used if rates=None.
    beta : float, optional (default=0.4)
        Scale parameter for gamma distribution used if rates=None.
    seed : int or None, optional
        Random seed for reproducibility.
    verbose : bool, optional (default=True)
        If True, print a table summarizing the selected reactions and their parameter values.

    RETURNS
    -------
    CRN_stoichiometric_matrix : np.ndarray
        Stoichiometric matrix for the selected reactions.
    CRN_reaction_names : list of str
        Names of the selected reactions.
    CRN_parameter_names : list of str
        Parameter names corresponding to the selected reactions.
    CRN_propensities : list of callable
        Propensity functions for each selected reaction.
    trueTheta : np.ndarray
        Full-length vector of reaction parameters for the entire CRN.
        - Length: number of reactions in the full CRN (stoichiometric_matrix.shape[1])
        - For selected reactions, trueTheta[idx] holds the provided or sampled value
        - For non-selected reactions, trueTheta[idx] = 0
    CRN_parameter_values : dict
        Dictionary mapping parameter names to their numeric values.
    CRN_indices : list of int
        Indices of selected reactions in the full CRN.
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Get exact indices of the selected reactions
    CRN_indices = get_reaction_indices(reaction_names, selected_names)

    # Extract submatrices and names
    CRN_stoichiometric_matrix = stoichiometric_matrix[:, CRN_indices]
    CRN_reaction_names = [reaction_names[i] for i in CRN_indices]
    CRN_parameter_names = [parameter_names[i] for i in CRN_indices]

    # Initialize outputs
    CRN_propensities = []
    CRN_parameter_values = {}
    trueTheta = np.zeros(stoichiometric_matrix.shape[1])  # full-length vector

    num_species = reactant_matrix.shape[0]

    for idx, par_name in zip(CRN_indices, CRN_parameter_names):
        reactants = reactant_matrix[:, idx]

        # Determine parameter value
        if rates is not None:
            rate_val = rates[CRN_indices.index(idx)]
        else:
            rate_val = np.random.gamma(alpha, beta)
        CRN_parameter_values[par_name] = rate_val
        trueTheta[idx] = rate_val  # store in full-length trueTheta

        # Build propensity function depending on reactants
        nz = np.nonzero(reactants)[0]
        propensity_string = ""

        if len(nz) == 0:
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

        CRN_propensities.append(eval(propensity_string))

    # Verbose output: table of selected reactions and parameters
    if verbose:
        print("\nSelected CRN Reactions:")
        print(f"{'Index':<6} {'Parameter':<8} {'Reaction Name':<30} {'Value':<10}")
        print("-" * 60)
        for idx, pname, rname in zip(CRN_indices, CRN_parameter_names, CRN_reaction_names):
            print(f"{idx:<6} {pname:<8} {rname:<30} {trueTheta[idx]:<10.4f}")
        print(f"\nFull trueTheta vector: {trueTheta}\n")

    return (CRN_stoichiometric_matrix,
            CRN_reaction_names,
            CRN_parameter_names,
            CRN_propensities,
            trueTheta,
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

def export_for_sparse_learning(
    trajectory_files,
    working_dir,
    T,
    poly_order=2,
    regular_lambda=0.0001,
    tot_step=20000,
    solver_id=1,
    flag_backtracking=1,
    cost_stop_tol=1e-7,
):
    """
    Convert JSON trajectory files to the format expected by sparse-learning-CRN
    (https://github.com/zwpku/sparse-learning-CRN) and write a ready-to-run
    working directory.

    After calling this function, run from the shell:

        cd <working_dir>
        /path/to/sparse-learning-CRN/src/prepare
        /path/to/sparse-learning-CRN/src/sparse_learning

    Parameters
    ----------
    trajectory_files : str or list of str
        Path(s) to JSON trajectory file(s) produced by save_trajectory.
        Each file becomes one trajectory (traj_0.txt, traj_1.txt, …).
    working_dir : str
        Directory to create. Will contain traj_data/, output/, log/,
        and sparse_learning.cfg.
    T : float
        Total observation time of each trajectory (e.g. 200 or 400).
        Used only in the config file; does not truncate the data.
    poly_order : int
        Maximum polynomial order for basis functions (1 or 2). Use 2 for
        systems with bimolecular reactions (default 2).
    regular_lambda : float
        L1 regularisation strength (default 0.0001).
    tot_step : int
        Maximum FISTA/ISTA iterations (default 20000).
    solver_id : int
        Solver: 1 = FISTA (default), 2 = ISTA.
    flag_backtracking : int
        1 = use backtracking line search (default), 0 = fixed step size.
    cost_stop_tol : float
        Convergence tolerance (default 1e-7).

    Example
    -------
    >>> export_for_sparse_learning(
    ...     trajectory_files=["../data/example5_T200_trajectory.json"],
    ...     working_dir="../sparse_run_T200",
    ...     T=200,
    ... )
    """
    import os

    if isinstance(trajectory_files, str):
        trajectory_files = [trajectory_files]

    # ------------------------------------------------------------------ #
    # 1. Create directory structure
    # ------------------------------------------------------------------ #
    traj_dir   = os.path.join(working_dir, "traj_data")
    output_dir = os.path.join(working_dir, "output")
    log_dir    = os.path.join(working_dir, "log")
    for d in [traj_dir, output_dir, log_dir]:
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 2. Convert each JSON trajectory → their plain-text format
    #
    #    Format (one file per trajectory):
    #      Line 1:  n   (number of species)
    #      Lines 2+: t  x1 x2 … xn  tau
    #    where tau = waiting time in that state = t[i+1] - t[i].
    #    For the last state tau = 0 (no next event observed).
    # ------------------------------------------------------------------ #
    n_species = None
    for traj_idx, json_path in enumerate(trajectory_files):
        time_list, state_list = load_trajectory(json_path)

        if n_species is None:
            n_species = len(state_list[0])

        # Drop trailing snapshot rows: if the last row has the same state as
        # the row before it, it is just a time-T snapshot (no reaction fired),
        # and would create a spurious [0,0,0,0] stoichiometric channel when
        # prepare computes consecutive-row differences.
        n_rows = len(time_list)
        while (n_rows > 1 and
               list(state_list[n_rows - 1]) == list(state_list[n_rows - 2])):
            n_rows -= 1

        out_path = os.path.join(traj_dir, f"traj_{traj_idx}.txt")
        with open(out_path, "w") as f:
            f.write(f"{n_species}\n")
            for i in range(n_rows):
                t     = time_list[i]
                state = state_list[i]
                tau   = float(time_list[i + 1]) - float(t) if i < n_rows - 1 else 0.0
                state_str = " ".join(str(int(x)) for x in state)
                f.write(f"{float(t):.6f}  {state_str}  {tau:.6f}\n")

        print(f"Written: {out_path}  ({n_rows} states, {n_species} species)")

    # ------------------------------------------------------------------ #
    # 3. Write sparse_learning.cfg
    # ------------------------------------------------------------------ #
    cfg_path = os.path.join(working_dir, "sparse_learning.cfg")
    cfg_content = f"""\
# Auto-generated by export_for_sparse_learning (BayesCRNInference)

# Total length of each trajectory
T = {float(T)} ;

# Number of trajectories
N_traj = {len(trajectory_files)} ;

# Maximum polynomial order for basis functions (1 or 2)
poly_order = {poly_order} ;

# L1 regularisation strength
regular_lambda = {regular_lambda} ;

# Solver: 1=FISTA, 2=ISTA
solver_id = {solver_id} ;

# Use backtracking line search (1) or fixed step size (0)
flag_backtracking = {flag_backtracking} ;

# Maximum iterations
tot_step = {tot_step} ;

# Convergence tolerance
cost_stop_tol = {cost_stop_tol} ;

# --- rarely need changing ---
epsL1_flag = 0 ;
l1_eps = 0.01 ;
xx_basis_flag = 1 ;
eps = 0.1 ;
g_cut = 35.0 ;
max_step_since_prev_min_cost = 1000 ;
num_record_tail_cost = 20 ;
output_interval = 100 ;
grad_dt = 0.01 ;
Lbar_fixed = 1e5 ;
L0 = 1e5 ;
"""
    with open(cfg_path, "w") as f:
        f.write(cfg_content)

    print(f"\nWorking directory ready: {working_dir}")
    print(f"  Trajectories : {len(trajectory_files)}  ({n_species} species each)")
    print(f"  Config       : {cfg_path}")
    print(f"\nNext steps (run from your shell):")
    print(f"  cd {working_dir}")
    print(f"  /path/to/sparse-learning-CRN/src/prepare")
    print(f"  /path/to/sparse-learning-CRN/src/sparse_learning")
    print(f"\nResults will appear in {output_dir}/")


def _build_propensities_from_reactant_matrix(reactant_matrix, species_names, parameter_names):
    """
    Reconstruct propensity lambda functions from a reactant matrix.

    Uses the same logic as build_CRN_bySamplingReactions. Each column of
    reactant_matrix corresponds to one reaction; the propensity form is
    determined solely by the reactant stoichiometry (unimolecular,
    bimolecular, or zeroth-order). Called internally by load_reaction_network.

    Parameters
    ----------
    reactant_matrix : np.ndarray, shape (num_species, num_reactions)
        Reactant stoichiometry for each reaction.
    species_names : list of str
    parameter_names : list of str
        One parameter name per reaction column.

    Returns
    -------
    propensities : list of callable
    """
    import math
    propensities = []
    for col_idx in range(reactant_matrix.shape[1]):
        reactants = reactant_matrix[:, col_idx]
        par_name  = parameter_names[col_idx]
        nz = np.nonzero(reactants)[0]

        if len(nz) == 0:
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
                fac   = math.factorial(multiplicity)
                propensity_string = f"lambda {par_name}, {sname}: {par_name}*({terms})/{fac}"
        elif len(nz) == 2:
            i, j   = nz
            s1, s2 = species_names[i], species_names[j]
            propensity_string = f"lambda {par_name}, {s1}, {s2}: {par_name}*{s1}*{s2}"
        else:
            raise ValueError(
                f"Only up to bimolecular reactions are supported "
                f"(reaction {col_idx} has {len(nz)} distinct reactant species)."
            )

        propensities.append(eval(propensity_string))

    return propensities


def save_reaction_network(
    species_names,
    reactant_matrix,
    CRN_stoichiometric_matrix,
    CRN_reaction_names,
    CRN_parameter_names,
    trueTheta,
    parameter_values,
    sampled_indices,
    unique_changes,
    compatible_reactions,
    filename,
):
    """
    Save a sampled reaction network to a JSON file.

    Stores everything needed to reconstruct:
      - the CRN object (for SSA simulation and trajectory plotting)
      - trueTheta, parameter_values, sampled_indices (for inference)
      - reactant_matrix, unique_changes, compatible_reactions (for parse_trajectory)

    Propensity lambda functions are *not* serialized directly; they are
    rebuilt automatically by load_reaction_network from the reactant matrix.

    Parameters
    ----------
    species_names : list of str
    reactant_matrix : np.ndarray, shape (num_species, num_reactions_full)
        Full reactant matrix from generate_reactions — needed for parse_trajectory
        and to rebuild propensity functions on load.
    CRN_stoichiometric_matrix : np.ndarray, shape (num_species, N_sampled)
    CRN_reaction_names : list of str, length N_sampled
    CRN_parameter_names : list of str, length N_sampled
    trueTheta : np.ndarray
        Full-length parameter vector (length = total reactions in full CRN).
    parameter_values : dict
        Maps sampled parameter names -> rate values.
    sampled_indices : list of int
        Indices of sampled reactions in the full CRN.
    unique_changes : list of tuples
        All unique stoichiometric change vectors from the full CRN.
    compatible_reactions : dict
        Maps stoichiometric change tuple -> list of reaction indices (full CRN).
    filename : str
        Output JSON file path.

    Example
    -------
    >>> save_reaction_network(
    ...     species_names, reactant_matrix,
    ...     CRN_stoichiometric_matrix, CRN_reaction_names, CRN_parameter_names,
    ...     trueTheta, parameter_values, sampled_indices,
    ...     unique_changes, compatible_reactions,
    ...     filename="../data/example5_network.json"
    ... )
    """
    data = {
        "species_names":             species_names,
        "CRN_reaction_names":        CRN_reaction_names,
        "CRN_parameter_names":       CRN_parameter_names,
        "CRN_stoichiometric_matrix": np.array(CRN_stoichiometric_matrix).tolist(),
        "reactant_matrix":           np.array(reactant_matrix).tolist(),
        "trueTheta":                 np.array(trueTheta).tolist(),
        "parameter_values":          {k: float(v) for k, v in parameter_values.items()},
        "sampled_indices":           [int(i) for i in sampled_indices],
        # unique_changes: list of tuples → list of lists
        "unique_changes": [
            [int(x) for x in uc] for uc in unique_changes
        ],
        # compatible_reactions: tuple keys → JSON string keys (parsed back on load)
        "compatible_reactions": {
            json.dumps([int(x) for x in k]): [int(i) for i in v]
            for k, v in compatible_reactions.items()
        },
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Reaction network saved to {filename}")
    print(f"  Species:   {species_names}")
    print(f"  Reactions: {len(CRN_reaction_names)} sampled  |  "
          f"{len(trueTheta)} total in full CRN")
    print(f"  Unique stoichiometric changes: {len(unique_changes)}")


def load_reaction_network(filename):
    """
    Load a saved reaction network from a JSON file and reconstruct all components.

    Propensity lambda functions are rebuilt from the stored reactant matrix —
    no external state or re-running of generate_reactions is required.

    Parameters
    ----------
    filename : str
        Path to the JSON file produced by save_reaction_network.

    Returns
    -------
    reactionNetwork : CRN
        Fully reconstructed CRN object (ready for SSA / plot_trajectories).
    CRN_stoichiometric_matrix : np.ndarray
    CRN_reaction_names : list of str
    CRN_parameter_names : list of str
    trueTheta : np.ndarray
    parameter_values : dict
    sampled_indices : list of int
    reactant_matrix : np.ndarray
        Full reactant matrix (pass directly to parse_trajectory).
    unique_changes : list of tuples
        Pass directly to parse_trajectory / extract_local_data.
    compatible_reactions : dict
        Tuple-keyed dict; pass directly to parse_trajectory / extract_local_data.
    species_names : list of str

    Example
    -------
    >>> (reactionNetwork,
    ...  CRN_stoichiometric_matrix, CRN_reaction_names, CRN_parameter_names,
    ...  trueTheta, parameter_values, sampled_indices,
    ...  reactant_matrix, unique_changes, compatible_reactions,
    ...  species_names) = load_reaction_network("../data/example5_network.json")
    """
    from CRN_Simulation.CRN import CRN

    with open(filename, "r") as f:
        data = json.load(f)

    species_names             = data["species_names"]
    CRN_reaction_names        = data["CRN_reaction_names"]
    CRN_parameter_names       = data["CRN_parameter_names"]
    CRN_stoichiometric_matrix = np.array(data["CRN_stoichiometric_matrix"])
    reactant_matrix           = np.array(data["reactant_matrix"])
    trueTheta                 = np.array(data["trueTheta"])
    parameter_values          = data["parameter_values"]
    sampled_indices           = data["sampled_indices"]
    unique_changes            = [tuple(uc) for uc in data["unique_changes"]]

    # Restore tuple keys for compatible_reactions
    compatible_reactions = {}
    for k_str, v in data["compatible_reactions"].items():
        k_tuple = tuple(int(x) for x in json.loads(k_str))
        compatible_reactions[k_tuple] = v

    # Rebuild propensity functions from the sampled columns of reactant_matrix
    CRN_reactant_matrix = reactant_matrix[:, sampled_indices]
    CRN_propensities = _build_propensities_from_reactant_matrix(
        CRN_reactant_matrix, species_names, CRN_parameter_names
    )

    reactionNetwork = CRN(
        species_names=species_names,
        stoichiometric_matrix=CRN_stoichiometric_matrix,
        parameters_names=CRN_parameter_names,
        reaction_names=CRN_reaction_names,
        propensities=CRN_propensities,
    )

    print(f"Reaction network loaded from {filename}")
    print(f"  Species:   {species_names}")
    print(f"  Reactions: {len(CRN_reaction_names)} sampled  |  "
          f"{len(trueTheta)} total in full CRN")
    print(f"  Unique stoichiometric changes: {len(unique_changes)}")

    return (
        reactionNetwork,
        CRN_stoichiometric_matrix,
        CRN_reaction_names,
        CRN_parameter_names,
        trueTheta,
        parameter_values,
        sampled_indices,
        reactant_matrix,
        unique_changes,
        compatible_reactions,
        species_names,
    )


def propensity_values(x, reactant_matrix, j_values):
    """
    Compute propensities for a vector of reactions (j_values) given the current state.

    Note: Assumes that the reactions are (at most) bimolecular.

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

def parse_trajectory(state_list, time_list, reactant_matrix, unique_changes, compatible_reactions, verbose=True):
    """
    Parse a single stochastic trajectory to compute, for each unique state:
      - counts of each unique stoichiometric change (jump_counts)
      - cumulative waiting times for each change (waiting_times)
      - propensity arrays for each change at that state (propensities)

    Args:
        state_list (list[np.ndarray] or np.ndarray): ordered sequence of state vectors.
        time_list  (list[float] or np.ndarray): ordered sequence of times corresponding to state_list.
        reactant_matrix (np.ndarray): reactant stoichiometry matrix (num_species x num_reactions).
        unique_changes (list[tuple]): list of unique stoichiometric change vectors (ΔX), in stable order.
        compatible_reactions (dict): mapping from stoichiometric change tuple -> list of reaction indices.
        verbose (bool): print progress messages (default True).

    Returns:
        unique_states  (list[tuple]): list of unique visited states
        jump_counts    (defaultdict): state_key -> np.array counts of length len(unique_changes)
        waiting_times  (defaultdict): state_key -> np.array cumulative times of length len(unique_changes)
        propensities   (defaultdict): state_key -> list of propensity arrays (aligned to unique_changes)
    """
    import numpy as np
    from collections import defaultdict

    # ---------- Input validation ----------
    if state_list is None or time_list is None:
        raise ValueError("state_list and time_list must be provided.")

    # Convert to lists of arrays if necessary
    state_list = [np.asarray(s) for s in state_list] if isinstance(state_list, (list, np.ndarray)) else list(state_list)
    time_list  = list(time_list) if isinstance(time_list, (list, np.ndarray)) else list(time_list)

    if len(state_list) != len(time_list):
        raise ValueError("state_list and time_list must have equal length.")
    if len(state_list) < 2:
        raise ValueError("Trajectory too short: need at least two timepoints.")

    num_unique_changes = len(unique_changes)
    if verbose:
        print(f"Tracking {num_unique_changes} unique stoichiometric changes.")
        print(f"Trajectory length: {len(state_list)} timepoints. Iterating to len-2 to avoid final non-jump.")

    # ---------- Precompute ΔX tuple → index lookup for speed ----------
    deltaX_to_idx = {dx: idx for idx, dx in enumerate(unique_changes)}

    # ---------- Initialize containers ----------
    jump_counts   = defaultdict(lambda: np.zeros(num_unique_changes, dtype=int))
    waiting_times = defaultdict(lambda: np.zeros(num_unique_changes, dtype=float))
    propensities  = defaultdict(lambda: [[] for _ in range(num_unique_changes)])

    # ---------- Main loop over states ----------
    for i in range(len(state_list) - 2):
        Xcurr = np.asarray(state_list[i])
        Xnext = np.asarray(state_list[i + 1])
        deltaX = Xnext - Xcurr
        deltaT = time_list[i + 1] - time_list[i]

        # normalize delta to ints for dictionary keys
        try:
            deltaX_tuple = tuple(deltaX.astype(int))
        except Exception:
            deltaX_tuple = tuple(int(x) for x in deltaX)

        if deltaX_tuple in compatible_reactions:
            state_key = tuple(int(x) for x in Xcurr)
            idx = deltaX_to_idx[deltaX_tuple]  # fast lookup instead of .index()
            jump_counts[state_key][idx] += 1
            waiting_times[state_key][idx] += deltaT
        else:
            if verbose:
                print(f"\tStep {i}: deltaX {deltaX_tuple} not in compatible_reactions (ignored)")

    # ---------- Compute propensities ----------
    for state_key in jump_counts.keys():
        for idx, deltaX_tuple in enumerate(unique_changes):
            reaction_indices = compatible_reactions[deltaX_tuple]
            propensities[state_key][idx] = propensity_values(state_key, reactant_matrix, reaction_indices)

    unique_states = list(jump_counts.keys())
    if verbose:
        print(f"Finished parsing trajectory. Observed {len(unique_states)} unique states.")

    return unique_states, jump_counts, waiting_times, propensities
