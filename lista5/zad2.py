import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import poisson


def thinning_method(lambda_func, M, T):
    t = 0
    S = []
    while True:
        U1 = np.random.uniform(0, 1)
        dt = -np.log(U1) / M
        t += dt
        if t > T:
            break

        U2 = np.random.uniform(0, 1)
        if U2 <= lambda_func(t) / M:
            S.append(t)
    return np.array(S)

# trzy wybrane funkcje
def lambda_1(t):
    return np.sin(t) ** 2

def lambda_2(t):
    return t ** 4

def lambda_4(t):
    return t


# parametry symulacji
T = 2.0
num_simulations = 5000
test_times = [1.0, 2.0]

functions = [{"name": r"$\lambda_1(t) = \sin^2(t)$", "func": lambda_1, "M": 1.05}, {"name": r"$\lambda_2(t) = t^4$", "func": lambda_2, "M": T ** 4 + 0.1}, {"name": r"$\lambda_4(t) = t$", "func": lambda_4, "M": T + 0.1}]

fig, axes = plt.subplots(len(functions), len(test_times), figsize=(12, 10))
for idx, fn_info in enumerate(functions):
    func = fn_info["func"]
    M = fn_info["M"]
    name = fn_info["name"]

    counts_at_times = {t: [] for t in test_times}

    for _ in range(num_simulations):
        jumps = thinning_method(func, M, T)
        for t in test_times:
            counts_at_times[t].append(np.sum(jumps <= t))

    for t_idx, t in enumerate(test_times):
        counts = np.array(counts_at_times[t])

        theoretical_mean, _ = quad(func, 0, t)
        emp_mean = np.mean(counts)
        emp_var = np.var(counts)

#empiryczy hist
        ax = axes[idx, t_idx]
        max_count = int(np.max(counts)) if len(counts) > 0 else 10
        bins = np.arange(0, max_count + 2) - 0.5
        ax.hist(counts, bins=bins, density=True, alpha=0.6, color='skyblue',
                edgecolor='black', label='Empiryczny')

#teoretyczby rozkład Poiss
        x = np.arange(0, max_count + 1)
        pmf = poisson.pmf(x, theoretical_mean)
        ax.plot(x, pmf, 'ro-', lw=2, ms=5, label='Teoretyczny (Poisson)')

        ax.set_title(
            f"{name}\nt = {t}, Teoret. $E[N_t]$ = {theoretical_mean:.3f}\nEmpir. średnia = {emp_mean:.3f}, wariancja = {emp_var:.3f}",
            fontsize=9)
        ax.set_xlabel("Liczba skoków ($N_t$)")
        ax.set_ylabel("Prawdopodobieństwo")
        ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
#plt.savefig("zadanie2_thinning_verification.png")