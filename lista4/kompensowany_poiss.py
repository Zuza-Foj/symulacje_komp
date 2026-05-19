import numpy as np
import matplotlib.pyplot as plt
from trajektorie_P import t_poiss_M1, t_poiss_M2, N_t

# 1.
def compensated(S_i, t, lam):
    return N_t(S_i, t) - lam * t

lam = 1
T = 20
N = 5000
time_grid = np.linspace(0, T, 300)

fig, ax = plt.subplots(figsize=(10, 4))
for i in range(6):
    S = t_poiss_M1(lam, T)
    traj_1 = []
    for t in time_grid:
        traj_1.append(compensated(S, t, lam))
    ax.plot(time_grid, traj_1, lw=1.2, alpha=0.7)
ax.axhline(0, color='black', lw=0.8, linestyle='--')
ax.set_xlabel('t'); ax.set_ylabel('Ñ(t)')
ax.set_title(f'Compensated Poisson process  (λ={lam})')
ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# 2.
k = 10
grid = np.linspace(0, T, k + 1)
traj_2 = []
for i in range(N):
    S = t_poiss_M1(lam, T)
    row = []
    for t in grid:
        row.append(compensated(S, t, lam))
    traj_2.append(row)

traj_2 = np.array(traj_2)

emp_mean = traj_2.mean(axis=0)
emp_var  = traj_2.var(axis=0)

print(f"{'t':>6}  {'E[Ñ(t)] theory':>16}  {'E[Ñ(t)] empirical':>18}  {'Var theory':>12}  {'Var empirical':>14}")
print("-" * 72)
for i, t in enumerate(grid):
    print(f"{t:>6.1f}  {0.0:>16.4f}  {emp_mean[i]:>18.4f}  {lam*t:>12.4f}  {emp_var[i]:>14.4f}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(grid, emp_mean, 'o-', label='empirical mean')
ax1.axhline(0, color='gray', linestyle='--', label='theory (0)')
ax1.set_ylabel('E[Ñ(t)]'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(grid, emp_var, 'o-', color='#D85A30', label='empirical var')
ax2.plot(grid, lam * grid, '--', color='gray', label='theory (λt)')
ax2.set_ylabel('Var(Ñ(t))'); ax2.set_xlabel('t')
ax2.legend(); ax2.grid(True, alpha=0.3)
plt.suptitle('Empirical mean and variance of compensated Poisson process')
plt.tight_layout(); plt.show()

# 3.
s = T/4
t = T/2

Ns_vals = traj_2[:, int(s / T * k)]
Nt_vals = traj_2[:, int(t / T * k)]
bins = {}
for ns, nt in zip(Ns_vals, Nt_vals):
    key = round(ns)
    bins.setdefault(key, []).append(nt)

keys = []
for k, v in bins.items():
    if len(v) >= 30:
        keys.append(k)
keys = sorted(keys)

avg_Nt = []
for k in keys:
    avg_Nt.append(np.mean(bins[k]))

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(keys, avg_Nt, zorder=3, label='E[Ñ(t) | Ñ(s) ≈ k]')
ax.plot(keys, keys, '--', color='gray', label='identity (martingale)')
ax.set_xlabel('Ñ(s)'); ax.set_ylabel('E[Ñ(t) | Ñ(s)]')
ax.set_title(f'Martingale check: s={s}, t={t}')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()