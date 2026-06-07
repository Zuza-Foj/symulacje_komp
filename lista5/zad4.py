import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import poisson
from zad2 import thinning_method, lambda_1, lambda_2

T = 2.0
N = 5000
M1 = T + 0.1
M2 = 1.05

counts_p1 = []
counts_p2 = []
counts_combined = []
sample_jumps_1 = []
sample_jumps_2 = []
sample_jumps_combined = []

def lambda_sum(t):
    return lambda_1(t) + lambda_2(t)

for n in range(N):
    jumps_1 = thinning_method(lambda_1, M1, T)
    jumps_2 = thinning_method(lambda_2, M2, T)
    jumps_combined = np.sort(np.concatenate([jumps_1, jumps_2]))
    counts_p1.append(len(jumps_1))
    counts_p2.append(len(jumps_2))
    counts_combined.append(len(jumps_combined))
    if n == 0:
        sample_jumps_1 = jumps_1
        sample_jumps_2 = jumps_2
        sample_jumps_combined = jumps_combined

# wizualizacja
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

def plot_step_process(ax, jumps, label, color):
    times = np.concatenate(([0], jumps, [T]))
    counts = np.concatenate(([0], np.arange(1, len(jumps) + 1), [len(jumps)]))
    ax.step(times, counts, where='post', label=label, color=color, lw=2)

plot_step_process(ax1, sample_jumps_1, r'Proces 1: $\lambda_1(t)=t$', 'blue')
plot_step_process(ax1, sample_jumps_2, r'Proces 2: $\lambda_2(t)=\sin^2(t)$', 'orange')
plot_step_process(ax1, sample_jumps_combined, r'Suma procesów: $\lambda_1(t)+\lambda_2(t)$', 'green')
ax1.set_title("Przykładowa trajektoria procesów zliczających")
ax1.set_xlabel("Czas $t$")
ax1.set_ylabel("Liczba zdarzeń $N_t$")
ax1.grid(True, linestyle='--')
ax1.legend()

counts_combined = np.array(counts_combined)
max_count = int(np.max(counts_combined))
bins = np.arange(0, max_count + 2) - 0.5

theoretical_mean_sum, _ = quad(lambda_sum, 0, T)

ax2.hist(counts_combined, bins=bins, density=True, alpha=0.6, color='g', edgecolor='black', label='Suma (Empirycznie)')
x = np.arange(0, max_count + 1)
pmf_theoretical = poisson.pmf(x, theoretical_mean_sum)
ax2.plot(x, pmf_theoretical, 'ro-', lw=2, label=f'Teoretyczny Poiss({theoretical_mean_sum:.3f})')

ax2.set_title(f"Rozkład liczby skoków procesu sumarycznego w przedziale [0, {T}]")
ax2.set_xlabel("Łączna liczba skoków ($N_T$)")
ax2.set_ylabel("Prawdopodobieństwo")
ax2.legend()

# statystyki
emp_mean = np.mean(counts_combined)
emp_var = np.var(counts_combined)
print(f"Oczekiwana teoretyczna liczba skoków sumy: {theoretical_mean_sum:.4f}")
print(f"Średnia empiryczna sumy z symulacji:       {emp_mean:.4f}")
print(f"Wariancja empiryczna sumy z symulacji:     {emp_var:.4f}")

plt.tight_layout()
plt.show()