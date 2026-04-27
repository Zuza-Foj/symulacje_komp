import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

# def d_akcept_odrzu(n):
#     p = {1: 0.11, 2: 0.12, 3: 0.27, 4: 0.19, 5: 0.31}
#     vals = np.array(list(p.keys()))
#     probs = np.array(list(p.values()))
#     g_prob = 1 / 5
#     M = np.max(probs) / g_prob
#     samples = []
#     while len(samples) < n:
#         pass

n = 100000

# (a)
p = {1: 0.11, 2: 0.12, 3: 0.27, 4: 0.19, 5: 0.31}
vals = np.array(list(p.keys()))
probs = np.array(list(p.values()))
m = max(probs)
sample_a = []
for i in range(n):
    x = random.choice(vals)
    u = np.random.uniform(0, 1)
    a = probs[x - 1] / (len(vals) * m)
    if u <= a:
        sample_a.append(x)

counts = np.array([sample_a.count(v) for v in vals])
freqs = counts / len(sample_a)

plt.bar(vals - 0.2, freqs, width=0.4, alpha=0.6, label="Rejection samples")
plt.bar(vals + 0.2, probs, width=0.4, alpha=0.6, color='r', label="Target p(x)")
plt.xlabel("x")
plt.ylabel("Probability")
plt.title("Rejection Sampling: Discrete Distribution")
plt.xticks(vals)
plt.legend()
plt.show()


# (b)
sample_b = []
theta = 1.9
x_g = np.linspace(0.001, 50, n)
f_r = (1/16) * x_g**2 * np.exp(-x_g/2)
e_r = (1/theta) * np.exp(-x_g/theta)
k = np.max(f_r/e_r)
for i in range(n):
    x_e = np.random.exponential(theta)
    x_u = np.random.uniform(0, 1)
    f = (1/16) * x_e**2 * np.exp(-x_e/2)
    g = (1/theta) * np.exp(-x_e/theta)
    if x_u <= f / (k * g):
        sample_b.append(x_e)

plt.hist(sample_b, bins=40, density=True, alpha=0.6, label="Rejected samples")
x_plot = np.linspace(0, 20, 300)
plt.plot(x_plot, gamma.pdf(x_plot, a=3, scale=2), 'r-', label="Gamma(3,2) PDF")
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Rejection Sampling: Gamma(3,2) using Exponential proposal")
plt.legend()
plt.show()


# rysowanie dystrybuant i qq-plotów
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

t_cdf = np.cumsum(probs)
sample_sorted = np.sort(sample_a)
e_cdf = np.searchsorted(sample_sorted, vals, side='right') / len(sample_a)
axes[0, 0].step(vals, e_cdf, where='post', label="Empirical CDF", color='steelblue', linewidth=2)
axes[0, 0].step(vals, t_cdf, where='post', label="Theoretical CDF", color='red', linewidth=2, linestyle='--')
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("CDF")
axes[0, 0].set_title("CDF: Discrete Distribution")
axes[0, 0].set_xticks(vals)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

quantile_levels = (np.arange(1, len(sample_a) + 1) - 0.5) / len(sample_a)
theoretical_quantiles = np.array([
    vals[np.searchsorted(t_cdf, q)] for q in quantile_levels
])
empirical_quantiles = np.sort(sample_a)

axes[0, 1].scatter(theoretical_quantiles, empirical_quantiles, alpha=0.4, color='steelblue', s=15)
min_val, max_val = vals[0], vals[-1]
axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label="y = x")
axes[0, 1].set_xlabel("Theoretical Quantiles")
axes[0, 1].set_ylabel("Empirical Quantiles")
axes[0, 1].set_title("QQ-Plot: Discrete Distribution")
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

sample_b_sorted = np.sort(sample_b)
n_b = len(sample_b_sorted)
e_cdf_b = np.arange(1, n_b + 1) / n_b
x_plot = np.linspace(0, 20, 300)
t_cdf_b = gamma.cdf(x_plot, a=3, scale=2)

axes[1, 0].plot(sample_b_sorted, e_cdf_b, label="Empirical CDF", color='steelblue', linewidth=2)
axes[1, 0].plot(x_plot, t_cdf_b, label="Theoretical CDF", color='red', linewidth=2, linestyle='--')
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("CDF")
axes[1, 0].set_title("CDF: Gamma(3,2) via Rejection Sampling")
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

quantile_levels_b = (np.arange(1, n_b + 1) - 0.5) / n_b
theoretical_quantiles_b = gamma.ppf(quantile_levels_b, a=3, scale=2)
axes[1, 1].scatter(theoretical_quantiles_b, sample_b_sorted, alpha=0.4, color='steelblue', s=15)
min_q, max_q = theoretical_quantiles_b.min(), theoretical_quantiles_b.max()

axes[1, 1].plot([min_q, max_q], [min_q, max_q], 'r--', linewidth=2, label="y = x")
axes[1, 1].set_xlabel("Theoretical Quantiles")
axes[1, 1].set_ylabel("Empirical Quantiles")
axes[1, 1].set_title("QQ-Plot: Gamma(3,2)")
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# średnia i wariancja
theo_mean_1 = np.sum(vals * probs)
theo_var_1 = np.sum((vals**2) * probs) - theo_mean_1**2
emp_mean_1 = np.mean(sample_a)
emp_var_1 = np.var(sample_a)

theo_mean_2 = 3 * 2
theo_var_2 = 3 * 2**2
emp_mean_2 = np.mean(sample_b)
emp_var_2 = np.var(sample_b)

print("podpunkt (a):")
print(f"średnia teoretyczna:{theo_mean_1:.4f}")
print(f"średnia empiryczna:{emp_mean_1:.4f}")
print(f"wariancja teoretyczna:{theo_var_1:.4f}")
print(f"wariancja empiryczna:{emp_var_1:.4f}")

print("\npodpunkt (b):")
print(f"średnia teoretyczna: {theo_mean_2:.4f}")
print(f"średnia empiryczna: {emp_mean_2:.4f}")
print(f"wariancja teoretyczna: {theo_var_2:.4f}")
print(f"wariancja empiryczna: {emp_var_2:.4f}")