import numpy as np
import matplotlib.pyplot as plt


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





