import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kstest
import time
from rozklady_6 import exp, poisson

def F_norm_erf(x):
    from scipy.special import erf
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def num_normal(n, dx=0.01):
    u_vals = np.random.uniform(0.01, 0.99, n)
    samples = []
    for u in u_vals:
        x = 0
        if 0.5 < u:
            while F_norm_erf(x) < u: x += dx
        else:
            while F_norm_erf(x) > u: x -= dx
        samples.append(x)
    return np.array(samples)

def test_generatora(data, dist_name, theory_pdf, theory_cdf, theory_params):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.hist(data, density=True, label='empiryczny')
    x = np.linspace(min(data), max(data), 100)
    ax1.plot(x, theory_pdf(x, **theory_params), 'r', label='teoretyczna')
    ax1.set_title(f'Histogram & Gęstość: {dist_name}')
    ax1.legend()

# dystrybuanty
    data_sorted = np.sort(data)
    y_emp = np.arange(1, len(data) + 1) / len(data)
    ax2.step(data_sorted, y_emp, label='empiryczna')
    ax2.plot(x, theory_cdf(x, **theory_params), label='teoretyczna')
    ax2.set_title('Dystrybuanty')
    ax2.legend()

# QQ-plot
    percs = np.linspace(1, 99, 50)
    q_emp = np.percentile(data, percs)
    q_theo = getattr(stats, dist_name).ppf(percs / 100, **theory_params)
    ax3.scatter(q_theo, q_emp, color='blue')
    ax3.plot([min(q_theo), max(q_theo)], [min(q_theo), max(q_theo)])
    ax3.set_title('QQ-Plot')

    plt.show()


data_exp = exp(1, 1000)
test_generatora(data_exp, 'expon', stats.expon.pdf, stats.expon.cdf, {'scale': 1})

# statystki dla r-du wykładniczego:
N = 1000
lam = 1
data_exp = np.random.exponential(scale=1/lam, size=N)
srednia_probkowa = np.mean(data_exp)
wariancja_probkowa = np.var(data_exp)
srednia_teoretyczna = 1 / lam
wariancja_teoretyczna = 1 / (lam**2)

print(f"ROZKŁAD WYKŁADNICZY (N={N})")
print(f"Średnia: Próbkowa = {srednia_probkowa:.4f}, Teoretyczna = {srednia_teoretyczna:.4f}")
print(f"Wariancja: Próbkowa = {wariancja_probkowa:.4f}, Teoretyczna = {wariancja_teoretyczna:.4f}")
print(f"Błąd średniej: {abs(srednia_probkowa - srednia_teoretyczna):.4f}")

# test zgodności:
data_norm = num_normal(500)
stat, p_val = kstest(data_norm, 'norm')
print(f"Test K-S dla N(0,1): statystyka={stat:.4f}, p-value={p_val:.4f}")

# porównanie wydajności (N=10^6)
N = 10**6

start_w = time.time()
np.random.exponential(1, N)
end_w = time.time()
res_w = end_w - start_w

start_inv = time.time()
exp(1, N)
end_inv = time.time()
res_inv = end_inv - start_inv

print(f"czas numpy:{res_w:.4f}s, mój czas:{res_inv:.4f}s")
# mój działa szybciej yay ;))