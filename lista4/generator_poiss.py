import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, gamma as gamma_dist
from math import floor, exp, log

# 1.
def poisson_small(lam):
    n = 0
    a = 1
    thr = exp(-lam)
    while True:
        a *= np.random.uniform()
        if a > thr:
            n += 1
        else:
            return n

# 2.
def poisson_gamma(lam):
    m = floor(7 / 8 * lam)
    Y = gamma_dist.rvs(a=m, scale=1)
    if Y <= lam:
        Z = np.random.poisson(lam - Y)
        return m + Z
    else:
        return np.random.binomial(m - 1, lam / Y)

# 3. i 4.
N = 10000
lambdas = [0.5, 2, 10, 50, 200]
for lam in lambdas:
    samp = []
    for i in range(N):
        samp.append(np.random.poisson(lam))
    samp = np.array(samp)
    emp_mean = samp.mean()
    emp_var = samp.var()
    print(f"\nλ = {lam}")
    print(f"  mean:  empirical={emp_mean:.4f}  theory={lam:.4f}")
    print(f"  var:   empirical={emp_var:.4f}  theory={lam:.4f}")

    k_min = max(0, int(lam - 4*lam**0.5))
    k_max = int(lam + 5*lam**0.5) + 1
    bins  = np.arange(k_min, k_max + 1)

    obs = []
    for k in bins:
        obs.append(np.sum(samp == k))
    obs = np.array(obs)

    log_probs = []
    for k in bins:
        if k == 0:
            log_p = -lam
        else:
            log_f = 0
            for i in range(1, k + 1):
                log_f += log(i)
            log_p = -lam + k * log(lam) - log_f
        log_probs.append(log_p)
    log_probs = np.array(log_probs)

    expected = N * np.exp(log_probs)

    while expected[0] < 5 and len(expected) > 1:
        obs[1] += obs[0];
        obs = obs[1:]
        expected[1] += expected[0];
        expected = expected[1:]
        bins = bins[1:]
    while expected[-1] < 5 and len(expected) > 1:
        obs[-2] += obs[-1];
        obs = obs[:-1]
        expected[-2] += expected[-1];
        expected = expected[:-1]
        bins = bins[:-1]

    log_probs = []
    for k in bins:
        if k == 0:
            log_p = -lam
        else:
            log_f = 0
            for i in range(1, k + 1):
                log_f += log(i)
            log_p = -lam + k * log(lam) - log_f
        log_probs.append(log_p)
    log_probs = np.array(log_probs)

    chi2_stat = np.sum((obs - expected) ** 2 / expected)
    df = len(obs) - 1 - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    print(f"  chi2:  stat={chi2_stat:.3f}  df={df}  p={p_value:.4f}  {'PASS' if p_value > 0.05 else 'FAIL'}")

    fig, ax = plt.subplots(figsize=(8, 3))
    width = 0.4
    ax.bar(bins - width/2, obs/N, width, label='empirical', alpha=0.85)
    ax.bar(bins + width/2, np.exp(log_probs), width, label='theoretical', alpha=0.85)
    ax.set_title(f'Poisson(λ={lam})  —  mean={emp_mean:.2f}, var={emp_var:.2f}, p={p_value:.3f}')
    ax.set_xlabel('k')
    ax.set_ylabel('P(X=k)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()