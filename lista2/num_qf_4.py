import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

def num_quantile_function(F, u, dx=0.1, x0=0):
    x = x0
    if F(x) < u:
        while F(x) < u:
            x += dx
    else:
        while F(x) > u:
            x -= dx
    return x

def F_exp(x, lam=1):
    return 1 - np.exp(-lam * x) if x >= 0 else 0

def F_norm(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

n = 100
lam = 1
dx = 0.001

u_t = np.linspace(0.001, 0.95, 500)
inv_f_t = -np.log(1 - u_t) / lam
u_s = np.sort(np.random.uniform(0.001, 0.95, n))
inv_f_e = []
for u in u_s:
    inv_f_e.append(num_quantile_function(F_exp, u, dx=dx))

plt.figure(figsize=(10, 6))
plt.plot(u_t, inv_f_t, 'r-', label='teoretyczna F^{-1}')
plt.step(u_s, inv_f_e, label='numeryczna F^{-1}')
plt.scatter(u_s, inv_f_e, s=15)

plt.title('Porównanie d. teoretycznej z numeryczną')
plt.xlabel('prawdopodobieństwo u')
plt.ylabel('wartość F^{-1}')
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()


n_samples = 1000
u_samples = np.random.uniform(0.01, 0.99, n_samples) # unikamy brzegów 0 i 1
normal_samples = []
for u in u_samples:
    num_quantile_function(F_norm, u, dx=0.01, x0=0)

plt.figure(figsize=(10, 6))
plt.hist(normal_samples, density=True, color='g', label='próba numeryczna')
x_range = np.linspace(-4, 4, 200)
plt.plot(x_range, 1/np.sqrt(2*np.pi) * np.exp(-x_range**2 / 2), 'r', label='gęstość teoretyczna')

plt.title("Histogram próby N(0,1) wygenerowanej numerycznie F^{-1}")
plt.xlabel("x")
plt.ylabel("gęstość")
plt.show()