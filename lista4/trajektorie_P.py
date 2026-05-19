import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson


# 1.
def t_poiss_M1(lam, T):
    S_i = []
    t = 0
    I = 0
    while True:
        u = np.random.uniform(0, 1)
        d_t = -1 / lam * np.log(u)
        t = d_t + t
        if t > T:
           break
        I += 1
        S_i.append(t)
    return S_i

def t_poiss_M2(lam, T):
    n = np.random.poisson(lam*T)
    if n == 0:
        return []
    u = np.random.uniform(0, T, n)
    S_i = np.sort(u)
    return S_i

lam = 1
T_max = 10
N = 5

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
for i in range(N):
    a1 = t_poiss_M1(lam, T_max)
    times1 = np.concatenate([[0], np.repeat(a1, 2), [T_max]])
    counts1 = np.concatenate([[0], np.arange(1, len(a1) + 1).repeat(2), [len(a1)]])
    ax1.plot(times1, counts1, alpha=0.8, lw=1.8)

    a2 = t_poiss_M2(lam, T_max)
    times2 = np.concatenate([[0], np.repeat(a2, 2), [T_max]])
    counts2 = np.concatenate([[0], np.arange(1, len(a2) + 1).repeat(2), [len(a2)]])
    ax2.plot(times2, counts2, alpha=0.8, lw=1.8)

for ax, title in zip([ax1, ax2], ['Method 1 — inter-arrival times', 'Method 2 — order statistics']):
    ax.set_title(title)
    ax.set_ylabel('N(t)')
    ax.grid(True, alpha=0.3)

ax2.set_xlabel('time t')
plt.suptitle(f'Poisson process sample trajectories  (λ={lam}, T={T_max})', y=1.01)
plt.tight_layout()
plt.show()


# 2.
def N_t(S_i, t):
    return np.sum(np.array(S_i) <= t)

times = [T_max/4, T_max/2, T_max]
samples = {t: [] for t in times}
for i in range(N):
    S = t_poiss_M1(lam, T_max)
    for t in times:
        samples[t].append(N_t(S, t))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, t in zip(axes, times):
    data = np.array(samples[t])
    k_max = int(np.max(data)) + 1
    k_vals = np.arange(0, k_max)

    empirical = np.bincount(data, minlength=k_max) / N
    theoretical = poisson.pmf(k_vals, mu=lam * t)

    ax.bar(k_vals - 0.2, empirical,   width=0.38, label='empirical', alpha=0.8, color='#378ADD')
    ax.bar(k_vals + 0.2, theoretical, width=0.38, label=f'Poisson({lam*t:.1f})', alpha=0.8, color='#D85A30')
    ax.set_title(f't = {t}')
    ax.set_xlabel('k')
    ax.set_ylabel('P(N(t) = k)')
    ax.legend(fontsize=8)

plt.suptitle('Empirical vs theoretical distribution of N(t)', y=1.02)
plt.tight_layout()
plt.show()

# task 4: compare empirical mean and variance vs theoretical
print(f"{'t':>6}  {'E[N(t)] theory':>16}  {'E[N(t)] empirical':>18}  {'Var theory':>12}  {'Var empirical':>14}")
print("-" * 72)
for t in times:
    data = np.array(samples[t])
    print(f"{t:>6.1f}  {lam*t:>16.4f}  {np.mean(data):>18.4f}  {lam*t:>12.4f}  {np.var(data):>14.4f}")




