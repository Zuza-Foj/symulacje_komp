import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, kstest

T = 2.0
N = 2000

def m_func(t):
    return 0.5 * (t**2)

m_T = m_func(T)
NT_samples = []
all_jumps = []

for n in range(N):
    N_T = np.random.poisson(m_T)
    NT_samples.append(N_T)
    if N_T > 0:
        U = np.random.uniform(0, 1, N_T)
        jumps = T * np.sqrt(U)
        jumps_sorted = np.sort(jumps)
        all_jumps.extend(jumps_sorted)

NT_samples = np.array(NT_samples)
all_jumps = np.array(all_jumps)

# wizualizacja
emp_mean_NT = np.mean(NT_samples)
emp_var_NT = np.var(NT_samples)
print(f"Teoretyczna wartość oczekiwana m(T): {m_T}")
print(f"Empiryczna średnia z prób: {emp_mean_NT:.4f}")
print(f"Empiryczna wariancja z prób: {emp_var_NT:.4f}\n")

def theoretical_cdf(t):
    return (t**2) / (T**2)

ks_stat, p_value = kstest(all_jumps, theoretical_cdf)
print(f"Statystyka KS: {ks_stat:.4f}")
print(f"p-value: {p_value:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
max_NT = int(np.max(NT_samples))
bins = np.arange(0, max_NT + 2) - 0.5
ax1.hist(NT_samples, bins=bins, density=True, alpha=0.6, color='lightgreen', edgecolor='black', label='Empiryczny')
x = np.arange(0, max_NT + 1)
ax1.plot(x, poisson.pmf(x, m_T), 'ro-', lw=2, label=f'Teoretyczny Poiss({m_T})')
ax1.set_title("Rozkład łącznej liczby skoków $N_T$")
ax1.set_xlabel("Liczba skoków")
ax1.set_ylabel("Prawdopodobieństwo")
ax1.legend()

ax2.hist(all_jumps, bins=100, density=True, histtype='step', cumulative=True, color='blue', lw=2, label='Empiryczna dystrybuanta')
t_vals = np.linspace(0, T, 200)
ax2.plot(t_vals, theoretical_cdf(t_vals), 'r--', lw=2, label=r'Teoretyczna dystrybuanta $F(t) = t^2/T^2$')
ax2.set_title("Dystrybuanta momentów skoków")
ax2.set_xlabel("Czas $t$")
ax2.set_ylabel("Prawdopodobieństwo skumulowane")
ax2.legend()

plt.tight_layout()
plt.show()