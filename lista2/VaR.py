import numpy as np
from scipy.stats import norm, cauchy, logistic
import matplotlib.pyplot as plt
import scipy.stats as stats

W0 = 5_000_000
mi = 0.0003
alphas = [0.99, 0.999]
p_vals = []
for a in alphas:
    p_vals.append(1 - a)
sigma_anna = 0.015
gamma_bartek = 0.008
s_celina = 0.008
res = []
for p in p_vals:
    q_anna = norm.ppf(p, loc=mi, scale=sigma_anna)
    q_bartek = cauchy.ppf(p, loc=mi, scale=gamma_bartek)
    q_celina = logistic.ppf(p, loc=mi, scale=s_celina)
    res.append({'alfa': 1 - p,
        'Anna (N)': W0 * abs(q_anna),
        'Bartek (C)': W0 * abs(q_bartek),
        'Celina (L)': W0 * abs(q_celina)})
for r in res:
    print(f"alfa = {r['alfa']}:")
    print(f"Ania: {r['Anna (N)']} pln")
    print(f"Bartek: {r['Bartek (C)']} pln")
    print(f"Celina: {r['Celina (L)']} pln")


N = 10**4
r_anna = np.random.normal(mi, sigma_anna, N)
r_bartek = stats.cauchy.rvs(mi, gamma_bartek, N)
r_celina = stats.logistic.rvs(mi, s_celina, N)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for data, ax, title in zip([r_anna, r_bartek, r_celina], axes, ["Anna (Normal)", "Bartek (Cauchy)", "Celina (Logistic)"]):
    stats.probplot(data, dist="norm", plot=ax)     # QQ-plot względem rozkładu normalnego
    ax.set_title(title)
    if title == "Bartek (Cauchy)":
        ax.set_ylim(-0.5, 0.5)
plt.tight_layout()
plt.show()

''' Najsilniej na ogonach odbiega Bartek(Cauchy): model normalny przewiduje, że pewne traty są niemożliwe,
natomiast Cauchy pokazuje, że są bardzo prawdopodobne.'''
