import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, multivariate_normal
import os

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

def log_prior_theta_star(theta_star, alpha, beta, pi, sigma_spike=0.01):
    """Spike-and-slab prior via latent theta_star."""
    theta = transform_theta_star(theta_star)
    if theta_star < 0:
        # Spike: truncated normal for negative theta_star
        return np.log(pi) + norm.logpdf(theta_star, 0, sigma_spike)
    else:
        # Slab: Gamma for positive theta_star
        return np.log(1 - pi) + gamma.logpdf(theta, a=alpha, scale=1/beta)

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

# ==========================
# 6. Plotting & Diagnostics
# ==========================

def plot_mcmc_samples(samples, true_values=None, param_names=None, filename=None):
    n_params = samples.shape[1]
    fig, axes = plt.subplots(n_params, 1, figsize=(6, 3*n_params))
    if n_params == 1:
        axes = [axes]
    for i in range(n_params):
        axes[i].hist(samples[:, i], bins=50, density=True, alpha=0.7)
        if true_values is not None:
            axes[i].axvline(true_values[i], color='r', linestyle='--', label='True')
        axes[i].set_title(f'Posterior: {param_names[i]}' if param_names else f'Parameter {i}')
        axes[i].legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def plot_mcmc_chain(chain, param_names=None, filename=None):
    n_iter, n_params = chain.shape
    fig, axes = plt.subplots(n_params, 1, figsize=(8, 2.5*n_params))
    if n_params == 1:
        axes = [axes]
    for i in range(n_params):
        axes[i].plot(chain[:, i])
        axes[i].set_title(f'Trace plot: {param_names[i]}' if param_names else f'Parameter {i}')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

# ==========================
# 7. Summarization
# ==========================

def summarize_chain(samples, param_names=None, save_excel=True, filename='../results/mcmc_summary.xlsx'):
    """Compute summary statistics and optionally save to Excel."""
    summary = []
    for i in range(samples.shape[1]):
        s = samples[:, i]
        mean = np.mean(s)
        std = np.std(s)
        ess = effective_sample_size(s)
        geweke_z = geweke_diagnostic(s)
        ci_lower, ci_upper = np.percentile(s, [2.5, 97.5])
        prob_zero = compute_zero_probability(s)
        summary.append([param_names[i] if param_names else f'param_{i}', mean, std, ess, geweke_z,
                        ci_lower, ci_upper, prob_zero])
    df_summary = pd.DataFrame(summary, columns=['Parameter', 'Mean', 'Std', 'ESS', 'Geweke_z',
                                                '2.5%', '97.5%', 'ProbZero'])
    if save_excel:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df_summary.to_excel(filename, index=False)
    return df_summary

# ==========================
# 8. Network Posterior Analysis (Optional)
# ==========================

def calc_network_posteriors(samples, threshold=1e-6):
    """Compute probability each reaction is active."""
    return np.mean(samples > threshold, axis=0)

def plot_network_posteriors(probabilities, reaction_names=None, filename=None):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(len(probabilities)), probabilities, tick_label=reaction_names)
    ax.set_ylabel('Posterior Probability')
    ax.set_ylim(0,1)
    if filename:
        plt.savefig(filename)
    plt.show()

