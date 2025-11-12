import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.inference import local_log_likelihood

from scipy.stats import gamma, norm, multivariate_normal,truncnorm
import os

from scipy import stats
from statsmodels.tsa.stattools import acf
from collections import Counter


# ==========================
# 1. Utility Functions
# ==========================

def transform_theta_star(theta_star):
    """Transform latent theta* to positive theta."""
    return np.maximum(theta_star, 0)

def effective_sample_size(x):
    """Estimate effective sample size (ESS) using autocorrelation."""
    n = len(x)
    acf = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    acf = acf[n-1:] / acf[n-1]
    ess = n / (1 + 2 * np.sum(acf[1:]))
    return ess

def geweke_diagnostic(x, first=0.1, last=0.5):
    """Compute simple Geweke z-score for convergence check."""
    n = len(x)
    a = x[:int(first*n)]
    b = x[int((1-last)*n):]
    z = (np.mean(a) - np.mean(b)) / np.sqrt(np.var(a)/len(a) + np.var(b)/len(b))
    return z

def compute_zero_probability(mcmc_samples, epsilon=1e-6):
    """Probability that parameter is essentially zero."""
    return np.mean(np.abs(mcmc_samples) < epsilon, axis=0)

# ==========================
# 2. Priors
# ==========================

def log_prior_gamma(theta, alpha, beta):
    """Standard Gamma prior."""
    if theta <= 0:
        return -np.inf
    return gamma.logpdf(theta, a=alpha, scale=1/beta)

def log_prior_spike_and_slab(theta, alpha, beta, pi):
    """Spike-and-slab prior: spike at 0, slab is Gamma."""
    if theta <= 0:
        return np.log(pi)
    else:
        return np.log(1 - pi) + gamma.logpdf(theta, a=alpha, scale=1/beta)


def log_prior_theta_star(theta_star, alpha=1.0, beta=1.0, pi=0.5, sigma_spike=10.0):
    """
    Compute the log prior over theta_star using spike-and-slab reparameterization.
    
    Args:
    - theta_star (float or np.array): Latent parameter(s).
    - alpha, beta: Parameters of the Gamma distribution (slab).
    - pi: Probability of the spike (at theta = 0, i.e., theta_star <= 0).
    - sigma_spike: Std deviation for the truncated normal (spike) on negative theta_star.
    
    Returns:
    - log prior value(s) for theta_star.
    """
    theta_star = np.atleast_1d(theta_star)
    log_prior = np.zeros_like(theta_star)

    # Slab part: theta_star > 0
    positive_mask = theta_star > 0
    log_prior[positive_mask] = np.log(1 - pi) + gamma.logpdf(theta_star[positive_mask], alpha, scale=1 / beta)

    # Spike part: theta_star <= 0
    negative_mask = ~positive_mask
    # Truncated normal: mean = 0, std = sigma_spike, truncated above at 0
    a, b = -np.inf, 0
    log_trunc_norm = truncnorm.logpdf(theta_star[negative_mask], a, b, loc=0, scale=sigma_spike)
    log_prior[negative_mask] = np.log(pi) + log_trunc_norm

    return np.sum(log_prior)

# ==========================
# 3. Proposal Distribution
# ==========================

def log_proposal_distribution(kappa, kappa_star, sigma):
    """Log-probability for asymmetric Gaussian proposal."""
    return norm.logpdf(kappa_star - kappa, 0, sigma)

# ==========================
# 4. Metropolis-Hastings MCMC
# ==========================

def metropolis_hastings_mcmc(theta_init, log_likelihood_fn, alpha, beta, pi,
                             sigma_prop=0.1, n_iter=10000):
    """
    Standard MH MCMC for spike-and-slab prior.
    Returns: chain of theta values.
    """
    n_params = len(theta_init)
    chain = np.zeros((n_iter, n_params))
    theta = np.array(theta_init)
    
    for t in range(n_iter):
        for i in range(n_params):
            theta_prop = theta.copy()
            theta_prop[i] += np.random.normal(0, sigma_prop)
            # Transform if using latent theta*
            theta_curr_val = theta[i]
            theta_prop_val = theta_prop[i]
            
            log_accept_ratio = (
                log_likelihood_fn(theta_prop) + log_prior_spike_and_slab(theta_prop_val, alpha, beta, pi)
                - log_likelihood_fn(theta) - log_prior_spike_and_slab(theta_curr_val, alpha, beta, pi)
            )
            
            if np.log(np.random.rand()) < log_accept_ratio:
                theta[i] = theta_prop[i]
        
        chain[t] = theta
    
    return chain

# ==========================
# 5. Adaptive MCMC
# ==========================

def adaptive_mcmc(theta_init, log_likelihood_fn, alpha, beta, pi,
                  n_iter=10000, burn_in=1000, cov_scale=0.01):
    """
    Adaptive multivariate Gaussian proposal MCMC.
    """
    n_params = len(theta_init)
    chain = np.zeros((n_iter, n_params))
    theta = np.array(theta_init)
    mean_est = theta.copy()
    cov_est = np.eye(n_params) * cov_scale
    accepted = 0

    for t in range(n_iter):
        theta_prop = multivariate_normal.rvs(mean=theta, cov=cov_est)
        log_accept_ratio = (
            log_likelihood_fn(theta_prop) + np.sum([log_prior_spike_and_slab(th, alpha, beta, pi) for th in theta_prop])
            - log_likelihood_fn(theta) - np.sum([log_prior_spike_and_slab(th, alpha, beta, pi) for th in theta])
        )
        
        if np.log(np.random.rand()) < log_accept_ratio:
            theta = theta_prop
            accepted += 1
        
        chain[t] = theta

        # Adapt covariance after burn-in
        if t >= burn_in:
            delta = theta - mean_est
            mean_est += delta / (t - burn_in + 1)
            cov_est = ((t - burn_in) * cov_est + np.outer(delta, delta)) / (t - burn_in + 1)
    
    acceptance_rate = accepted / n_iter
    return chain, acceptance_rate

def adaptive_mcmc_spike_slab(local_counts, local_waiting_times, local_propensities,
                             theta_init, trueTheta, num_iterations=1000,
                             alpha=1.0, beta=1.0, pi=0.5,
                             epsilon=1e-6, burn_in=100, adapt_every_n=1):
    """
    Adaptive Metropolis MCMC with Spike-and-Slab prior via transformation over theta*.
    
    Args:
    - local_counts, local_waiting_times, local_propensities: Observed data.
    - theta_init: Initial guess for theta (positive values).
    - trueTheta: True theta (for debugging/diagnostics).
    - num_iterations: Number of MCMC iterations.
    - alpha, beta: Gamma slab parameters.
    - pi: Spike probability.
    - epsilon: Small jitter to stabilize covariance.
    - burn_in: Number of initial samples before adaptation starts.
    - adapt_every_n: Frequency (in steps) to update covariance.
    
    Returns:
    - ThetaChain: List of theta (transformed from theta*) samples.
    """
    
    if len(local_counts) == 0:
        return np.zeros_like(theta_init), np.zeros_like(theta_init)
    
    d = len(theta_init)
    s_d = 2.38 ** 2 / d  # Scaling for adaptation
    sigma_spike = 10.0   # Std for truncated normal in spike prior
    
    theta_star = theta_init.copy()

    ThetaStarChain = [theta_star.copy()]
    ThetaChain = [transform_theta_star(theta_star)]  # Transformed values

    accept_count = 0
    mu = theta_star.copy()
    cov = np.eye(d) * 0.01  # small initial covariance

    printEveryNSteps = 10000
    
    for iteration in range(1, num_iterations + 1):
        # Propose new theta_star
        theta_star_prop = np.random.multivariate_normal(theta_star, s_d * cov)
        theta_prop = transform_theta_star(theta_star_prop)

        # Compute likelihoods
        log_like_current = local_log_likelihood(local_counts, local_waiting_times, local_propensities,
                                          transform_theta_star(theta_star))
        log_like_prop = local_log_likelihood(local_counts, local_waiting_times, local_propensities,
                                       theta_prop)

        # Compute priors
        log_prior_current = log_prior_theta_star(theta_star, alpha=alpha, beta=beta, pi=pi, sigma_spike=sigma_spike)
        log_prior_prop = log_prior_theta_star(theta_star_prop, alpha=alpha, beta=beta, pi=pi, sigma_spike=sigma_spike)

        # Compute acceptance probability
        log_accept_ratio = log_like_prop + log_prior_prop - log_like_current - log_prior_current
        accept_prob = np.exp(min(0, log_accept_ratio))

        # Accept or reject
        if np.random.rand() < accept_prob:
            theta_star = theta_star_prop
            accept_count += 1

        ThetaStarChain.append(theta_star.copy())
        ThetaChain.append(transform_theta_star(theta_star.copy()))

        # Adapt covariance
        if iteration > burn_in and iteration % adapt_every_n == 0:
            samples = np.array(ThetaStarChain[:iteration])
            mu = samples.mean(axis=0)
            cov = np.cov(samples.T) + epsilon * np.eye(d)

        if iteration % printEveryNSteps == 0:
            print(f"Iteration {iteration}: Accept Rate = {accept_count/iteration:.3f}, "
                  f"Theta = {transform_theta_star(theta_star)}")

    return ThetaChain

# ==========================
# 6. Plotting & Diagnostics
# ==========================

def plot_mcmc_samples(mcmc_samples, true_theta, epsilon=1e-5,burnin=1000, thinout=100, filename=None):
    """
    Plot the MCMC samples as probability density plots, with vertical lines showing:
      - True theta values (red dashed)
      - Posterior mean (green solid)
      - 95% credibility intervals (purple dotted)
    Also reports absolute error between the mean and the true value.
    """

    # Ensure mcmc_samples is a NumPy array
    mcmc_samples = np.array(mcmc_samples)

    # Discard burn-in samples
    mcmc_samples = mcmc_samples[burnin:]

    # Thinning: select every thinout-th sample
    mcmc_samples = mcmc_samples[::thinout]

    # Number of parameters
    num_params = len(true_theta)

    # Store the output
    est_theta = np.zeros(num_params)
    est_error = np.zeros(num_params)

    # Compute the probability that each parameter is zero
    zero_probabilities = compute_zero_probability(mcmc_samples)

    # Set up the plot
    fig, axes = plt.subplots(1, num_params, figsize=(5 * num_params, 4))

    # Ensure axes is iterable if there's only one parameter
    if num_params == 1:
        axes = [axes]

    # Loop over each parameter and create a histogram
    for i in range(num_params):
        samples_i = mcmc_samples[:, i]

        # Histogram
        axes[i].hist(samples_i, bins=30, color='blue', alpha=0.5, density=True)

        # True theta (red dashed)
        axes[i].axvline(true_theta[i], color='red', linestyle='--', label=f'True θ{i+1}: {true_theta[i]:.2f}')

        # Posterior mean (green)
        mean_theta = np.mean(samples_i)
        axes[i].axvline(mean_theta, color='green', linestyle='-', label=f'Mean θ{i+1}: {mean_theta:.2f}')
        est_theta[i] = mean_theta

        # Error
        error = np.abs(mean_theta - true_theta[i])
        est_error[i] = error

        # 95% credibility interval (purple dotted)
        lower_bound, upper_bound = np.percentile(samples_i, [2.5, 97.5])
        axes[i].axvline(lower_bound, color='purple', linestyle=':', label=f'2.5%: {lower_bound:.2f}')
        axes[i].axvline(upper_bound, color='purple', linestyle=':', label=f'97.5%: {upper_bound:.2f}')

        # Title and labels
        axes[i].set_title(
            f'Dimension {i+1} of θ\n|Error|: {error:.4f} | P(Zero): {zero_probabilities[i]:.4f}',
            fontsize=11
        )
        axes[i].set_xlabel(f'θ{i+1}')
        axes[i].set_ylabel('Density')
        axes[i].legend(fontsize=9)

    plt.tight_layout()

    # Output estimates
    print(f"Estimated Theta: {est_theta}")
    print(f"Estimated Error: {est_error}")

    # Save or show
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    else:
        plt.show()

    plt.close(fig)

def plot_mcmc_chain(ThetaChain, filename):
    ThetaChain = np.array(ThetaChain)
    n_samples, n_dims = ThetaChain.shape

    fig_width = 6 * n_dims
    fig, axs = plt.subplots(1, n_dims, figsize=(fig_width, 4), squeeze=False)

    for i in range(n_dims):
        ax = axs[0][i]
        ax.plot(ThetaChain[:, i])
        ax.set_title(f"ThetaChain Dimension {i}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Value")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


# ==========================
# 7. Summarization
# ==========================

def summarize_chains(
    chains,
    true_theta,
    labels,
    results_dir,
    filename="mcmc_summary.xlsx",
    burnin=0,
    thinout=1,
    alpha_ci=0.05,
    epsilon=1e-3,
    run_index=0,
    count = 0 # NEW argument to track the run!
):


    all_summaries = []

    for chain, label in zip(chains, labels):
        chain = np.array(chain)

        # Apply burnin and thinning
        post_burnin = chain[burnin:]
        thinned = post_burnin[::thinout]
        n_samples, n_params = thinned.shape

        means = np.mean(thinned, axis=0)
        stds = np.std(thinned, axis=0)
        ess = np.array([effective_sample_size(thinned[:, i]) for i in range(n_params)])
        geweke_z = np.array([geweke_diagnostic(thinned[:, i]).mean() for i in range(n_params)])
        autocorr_1 = np.array([acf(thinned[:, i], nlags=1)[1] for i in range(n_params)])

        # Empirical CI from quantiles
        lower_q = alpha_ci / 2
        upper_q = 1 - alpha_ci / 2
        ci_lower = np.quantile(thinned, lower_q, axis=0)
        ci_upper = np.quantile(thinned, upper_q, axis=0)

        # Probability param is OFF (within ±epsilon of zero)
        prob_off = np.mean((thinned >= -epsilon) & (thinned <= epsilon), axis=0)

        # L2 error between means and true_theta
        l2_error = np.linalg.norm(means - true_theta)

        for i in range(n_params):
            summary = {
                "Run_Index": run_index, #new
                "Count": count, # new!
                "Chain": label,
                "Param_Index": i,
                "True_Theta": true_theta[i],
                "Mean": means[i],
                "Std": stds[i],
                "ESS": ess[i],
                "Gweke_z": geweke_z[i],
                "Autocorr_1": autocorr_1[i],
                "CI_lower": ci_lower[i],
                "CI_upper": ci_upper[i],
                "Prob_Off": prob_off[i],
                "L2_Error": l2_error,
                "N_Samples": n_samples,
                "Burnin": burnin,
                "Thinout": thinout,
                "Epsilon": epsilon
            }
            all_summaries.append(summary)

        print(f"Run {run_index} | {label}: L2 error of mean from true theta = {l2_error:.5f}")

    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(all_summaries)

    full_path = os.path.join(results_dir, filename)
    print(f"Saving summary to: {full_path}")

    # If file exists, read old data and append new rows
    if os.path.exists(full_path):
        df_existing = pd.read_excel(full_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_combined = df

    df_combined.to_excel(full_path, index=False)
    print("Summary saved successfully.")

# ==========================
# 8. Network Posterior Analysis (Optional)
# ==========================

def calc_network_posteriors(mcmc_samples, epsilon=0.01, prob_cutoff=0.05,burnin=1000, thinout=100, labels=None):
    """
    Summarizes posterior over network structures based on MCMC samples,
    computes Bayes factors between all pairs of models above prob_cutoff,
    and prints the results without plotting.

    Args:
        mcmc_samples (array-like): MCMC samples of theta parameters.
        zero_threshold (float): Threshold for determining whether a reaction is "on".
        prob_cutoff (float): Minimum posterior probability to report a network.
        burnin (int): Number of burn-in samples to discard.
        thinout (int): Thinning factor for samples.
        labels (list): Optional list of reaction labels, e.g. ['R1', 'R2', 'R3']
    """
    mcmc_samples = np.array(mcmc_samples)
    d = mcmc_samples.shape[1]

    # Default reaction labels
    if labels is None:
        labels = [f'R{i+1}' for i in range(d)]

    # Process samples
    samples = mcmc_samples[burnin::thinout]

    # Convert each sample to a binary vector indicating presence/absence
    structure_bitstrings = []
    for theta in samples:
        bitstring = ''.join(['1' if val > epsilon else '0' for val in theta])
        structure_bitstrings.append(bitstring)

    # Count occurrences
    counts = Counter(structure_bitstrings)
    total = sum(counts.values())
    structure_probs = {k: v / total for k, v in counts.items()}

    # Filter based on posterior probability cutoff
    filtered_structures = {k: p for k, p in structure_probs.items() if p >= prob_cutoff}

    if not filtered_structures:
        print("No network structure exceeded the probability cutoff.")
        return filtered_structures, {}

    # Sort structures by descending probability
    sorted_structures = sorted(filtered_structures.items(), key=lambda x: -x[1])

    print("Posterior probabilities of network structures (above cutoff):\n")
    for bitstring, prob in sorted_structures:
        active_reactions = [labels[i] for i, bit in enumerate(bitstring) if bit == '1']
        print(f"{bitstring} ({' + '.join(active_reactions) if active_reactions else 'None'}) : {prob:.4f}")

    # Compute Bayes factors between all pairs of filtered models
    bayes_factors = {}
    keys = list(filtered_structures.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            model_a = keys[i]
            model_b = keys[j]
            p_a = filtered_structures[model_a]
            p_b = filtered_structures[model_b]
            bf = p_a / p_b if p_b > 0 else float('inf')
            bayes_factors[(model_a, model_b)] = bf
            bayes_factors[(model_b, model_a)] = 1 / bf if bf != 0 else float('inf')

    print("\nBayes Factors between model pairs:\n")
    for (m1, m2), bf in bayes_factors.items():
        print(f"BF({m1} / {m2}) = {bf:.2f}")

    return filtered_structures, bayes_factors

def plot_network_posteriors(mcmc_samples, epsilon=0.01, prob_cutoff=0.05, burnin=1000, thinout=100, labels=None, filename=None):
    """
    Summarizes posterior over network structures based on MCMC samples.

    Args:
        mcmc_samples (array-like): MCMC samples of theta parameters.
        zero_threshold (float): Threshold for determining whether a reaction is "on".
        prob_cutoff (float): Minimum posterior probability to report a network.
        burnin (int): Number of burn-in samples to discard.
        thinout (int): Thinning factor for samples.
        labels (list): Optional list of reaction labels, e.g. ['R1', 'R2', 'R3']
        filename (str): If provided, saves the plot to this file instead of showing it.
    """
    mcmc_samples = np.array(mcmc_samples)
    d = mcmc_samples.shape[1]

    # Default reaction labels
    if labels is None:
        labels = [f'R{i+1}' for i in range(d)]

    # Process samples
    samples = mcmc_samples[burnin::thinout]

    # Convert each sample to a binary vector indicating presence/absence
    structure_bitstrings = []
    for theta in samples:
        bitstring = ''.join(['1' if val > epsilon else '0' for val in theta])
        structure_bitstrings.append(bitstring)

    # Count occurrences
    counts = Counter(structure_bitstrings)
    total = sum(counts.values())
    structure_probs = {k: v / total for k, v in counts.items()}

    # Filter based on posterior probability cutoff
    filtered_structures = {k: p for k, p in structure_probs.items() if p >= prob_cutoff}

    if not filtered_structures:
        print("No network structure exceeded the probability cutoff.")
        return

    # Sort and display
    sorted_structures = sorted(filtered_structures.items(), key=lambda x: -x[1])
    print("Posterior probabilities of network structures (above cutoff):\n")
    for bitstring, prob in sorted_structures:
        active_reactions = [labels[i] for i, bit in enumerate(bitstring) if bit == '1']
        print(f"{bitstring} ({' + '.join(active_reactions) if active_reactions else 'None'}) : {prob:.3f}")

    # Extract sorted keys and probabilities
    sorted_keys = [k for k, _ in sorted_structures]
    sorted_probs = [p for _, p in sorted_structures]

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(sorted_keys, sorted_probs, color='skyblue')
    ax.set_ylabel("Posterior Probability")
    ax.set_xlabel("Network Structure (Bitstring)")
    ax.set_title("Posterior over Network Structures")
    ax.set_xticks(range(len(sorted_keys)))
    ax.set_xticklabels(sorted_keys, rotation=45)
    plt.tight_layout()

    # Save or show
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    else:
        plt.show()

    plt.close(fig)

def plot_network_and_parameter_posteriors(mcmc_samples, true_theta, epsilon=0.01, prob_cutoff=0.05,
                                          burnin=1000, thinout=100, labels=None, filename=None):
    """
    Plot posterior distributions of parameters for each network structure that exceeds a probability cutoff.
    Each row represents one network structure.
    """

    mcmc_samples = np.array(mcmc_samples)
    true_theta = np.array(true_theta)
    d = mcmc_samples.shape[1]

    if labels is None:
        labels = [f'θ{i+1}' for i in range(d)]

    samples = mcmc_samples[burnin::thinout]

    # Convert each sample to bitstring of active reactions
    structure_bitstrings = []
    for theta in samples:
        bitstring = ''.join(['1' if val > epsilon else '0' for val in theta])
        structure_bitstrings.append(bitstring)

    # Count and filter based on posterior probability
    counts = Counter(structure_bitstrings)
    total = sum(counts.values())
    structure_probs = {k: v / total for k, v in counts.items()}
    filtered_structures = {k: p for k, p in structure_probs.items() if p >= prob_cutoff}

    if not filtered_structures:
        print("No network structure exceeded the probability cutoff.")
        return

    sorted_structures = sorted(filtered_structures.items(), key=lambda x: -x[1])
    nrows = len(sorted_structures)

    fig, axes = plt.subplots(nrows=nrows, ncols=d + 1, figsize=(4 * (d + 1), 2.8 * nrows),
                             gridspec_kw={'width_ratios': [1.4] + [1] * d})

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (bitstring, prob) in enumerate(sorted_structures):
        mask = np.array([bit == '1' for bit in bitstring])
        conditional_samples = samples[np.array([''.join(['1' if val > epsilon else '0' for val in theta]) == bitstring
                                                for theta in samples])]

        summary_lines = [f'P(model) = {prob:.3f}']

        # Compute conditional stats for each active parameter
        for param_idx in np.where(mask)[0]:
            param_samples = conditional_samples[:, param_idx]
            mean = np.mean(param_samples)
            ci_lower, ci_upper = np.percentile(param_samples, [2.5, 97.5])
            summary_lines.append(f'{labels[param_idx]} = {mean:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]')

        # Display summary text in left column
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].text(0.05, 0.5, '\n'.join(summary_lines), fontsize=10, va='center', ha='left')

        # Plot parameter posteriors
        for param_idx in range(d):
            ax = axes[row_idx, param_idx + 1]
            if mask[param_idx]:
                param_samples = conditional_samples[:, param_idx]
                mean = np.mean(param_samples)
                ci_lower, ci_upper = np.percentile(param_samples, [2.5, 97.5])

                # Density plot
                ax.hist(param_samples, bins=30, density=True, color='skyblue', alpha=0.6)

                # Posterior mean
                ax.axvline(mean, color='green', linestyle='-', label='Mean')

                # 95% CI
                ax.axvline(ci_lower, color='purple', linestyle=':', label='2.5%')
                ax.axvline(ci_upper, color='purple', linestyle=':', label='97.5%')

                # True value
                ax.axvline(true_theta[param_idx], color='red', linestyle='--', label='True')

                ax.set_yticks([])
                ax.set_title(f'{labels[param_idx]}', fontsize=10)
            else:
                ax.axis('off')

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    else:
        plt.show()

    plt.close(fig)


