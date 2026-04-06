import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def exp(lam, n):
    u = np.random.uniform(0, 1, n)
    return -np.log(u) / lam

def cauchy(x0, gamma, n):
    u = np.random.uniform(0, 1, n)
    return x0 + gamma * np.tan( np.pi * (u - 0.5) )

# za pomoca metody Boxa-Mullera
def normal(mi, sigma, n):
    u1 = np.random.uniform(0, 1, n)
    u2 = np.random.uniform(0, 1, n)
    z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return z0 * sigma + mi

n = 1000
data1 = exp(1, n)
data2 = cauchy(0.5, 2, n)
data3 = normal(0, 1, n)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')

ax1.hist(data1, density=True, color='steelblue', label='empiryczny')
x1 = np.linspace(0, 10, 200)
ax1.plot(x1, stats.expon.pdf(x1, 1), 'r-', label='teoretyczny')
ax1.set_title('Exp(1)')

ax2.hist(data2, density=True, color='salmon', label='empiryczny')
x2 = np.linspace(-15, 15, 500)
ax2.plot(x2, stats.cauchy.pdf(x2, 0.5, 2), 'r-', label='teoretyczny', lw=2)
ax2.set_title('Cauchy(0.5, 2)')

ax3.hist(data3, density=True, color='lightgreen', label='empiryczny')
x3 = np.linspace(-4, 4, 100)
ax3.plot(x3, stats.norm.pdf(x3, 0, 1), 'r-',label='teoretyczna')
ax3.set_title('N(0, 1)')

plt.show()

def geom(p, n):
    u = np.random.uniform(0, 1, n)
    return np.floor(np.log(u) / np.log(1 - p)).astype(int)

def poisson(lam, n):
    samples = []
    for i in range(n):
        u = np.random.random()
        k = 0
        p = np.exp(-lam)
        s = p
        while u > s:
            k += 1
            p *= lam / k
            s += p
        samples.append(k)
    return np.array(samples)

p_geom = 0.3
lam_poiss = 4

data_geom = np.random.geometric(p=p_geom, size=n) - 1
data_poiss = np.random.poisson(lam=lam_poiss, size=n)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')

bins_g = np.arange(0, data_geom.max() + 2) - 0.5
ax1.hist(data_geom, bins=bins_g, density=True, color='purple', label='empiryczny')
k_g = np.arange(0, data_geom.max() + 1)
ax1.scatter(k_g, stats.geom.pmf(k_g + 1, p_geom), color='r', label='teoretyczna')
ax1.vlines(k_g, 0, stats.geom.pmf(k_g + 1, p_geom), color='r', alpha=0.5)
ax1.set_title(f'Geometryczny p={p_geom}')


bins_p = np.arange(0, data_poiss.max() + 2) - 0.5
ax2.hist(data_poiss, bins=bins_p, density=True, alpha=0.4, color='green', label='Empiryczny')
k_p = np.arange(0, data_poiss.max() + 1)
ax2.scatter(k_p, stats.poisson.pmf(k_p, lam_poiss), color='r', label='teoretyczna')
ax2.vlines(k_p, 0, stats.poisson.pmf(k_p, lam_poiss))
ax2.set_title(f'Poisson lam={lam_poiss}')

plt.show()
