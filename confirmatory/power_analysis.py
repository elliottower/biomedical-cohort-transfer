"""
Power analysis for partial Spearman correlation.

Computes minimum detectable rho at 80% power (alpha=0.05, two-sided)
for each dataset's effective sample size (number of independent cohorts,
not number of pairs).

Also computes Monte Carlo verification.
"""

import numpy as np
from scipy import stats


def min_detectable_rho(n, alpha=0.05, power=0.80, n_covariates=1):
    """Asymptotic minimum detectable Spearman rho.

    For partial correlation with n_covariates, df = n - 2 - n_covariates.
    """
    df = n - 2 - n_covariates
    if df < 1:
        return float('nan')
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    for rho in np.arange(0.01, 1.0, 0.01):
        z = np.arctanh(rho)
        se = 1.0 / np.sqrt(n - 3 - n_covariates)
        ncp = z / se
        power_at_rho = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha/2) - ncp) + stats.norm.cdf(stats.norm.ppf(alpha/2) - ncp)
        if power_at_rho >= power:
            return rho
    return float('nan')


def monte_carlo_power(n_eff, true_rho, n_sims=10000, alpha=0.05):
    """Monte Carlo power estimate via Fisher z-transform."""
    rng = np.random.default_rng(42)
    z_true = np.arctanh(true_rho)
    se = 1.0 / np.sqrt(n_eff - 3)

    z_samples = rng.normal(z_true, se, n_sims)
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_samples) / se))

    return np.mean(p_values < alpha)


datasets = {
    'CRC microbiome': {'n_eff': 9, 'observed_partial_rho': 0.613},
    'QMDiab': {'n_eff': 9, 'observed_partial_rho': 0.404},
    'IBD': {'n_eff': 5, 'observed_partial_rho': -0.108},
    'SPIROMICS COPD': {'n_eff': 8, 'observed_partial_rho': 0.084},
    'Breast cancer GEO': {'n_eff': 7, 'observed_partial_rho': -0.004},
    'TCGA-BRCA': {'n_eff': 14, 'observed_partial_rho': -0.120},
    'MTBLS7260': {'n_eff': 15, 'observed_partial_rho': 0.040},
}

print("=" * 80)
print(f"{'Dataset':<22} {'n_eff':>5} {'Min rho':>8} {'Obs rho':>9} {'Power@obs':>10}")
print("=" * 80)

for name, d in datasets.items():
    mdr = min_detectable_rho(d['n_eff'])
    obs = d['observed_partial_rho']
    pwr = monte_carlo_power(d['n_eff'], abs(obs)) if abs(obs) > 0.01 else 0.0
    print(f"{name:<22} {d['n_eff']:>5} {mdr:>8.2f} {obs:>+9.3f} {pwr:>9.1%}")

print()
print("Power to detect CRC-level effect (rho=0.61) at each n_eff:")
for n in [5, 7, 8, 9, 11, 14, 15, 20, 30]:
    pwr = monte_carlo_power(n, 0.61)
    print(f"  n_eff = {n:>3}: power = {pwr:.1%}")

print()
print("How many studies needed to detect rho=0.61 at 80% power?")
for n in range(5, 30):
    pwr = monte_carlo_power(n, 0.61)
    if pwr >= 0.80:
        print(f"  n_eff = {n} gives power = {pwr:.1%} — FIRST >= 80%")
        break
    else:
        print(f"  n_eff = {n} gives power = {pwr:.1%}")

print()
print("How many studies needed to detect rho=0.40 at 80% power?")
for n in range(5, 60):
    pwr = monte_carlo_power(n, 0.40)
    if pwr >= 0.80:
        print(f"  n_eff = {n} gives power = {pwr:.1%} — FIRST >= 80%")
        break
