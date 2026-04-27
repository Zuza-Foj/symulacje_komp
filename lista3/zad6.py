import numpy as np
import matplotlib.pyplot as plt

N = 10000
rho = 0.8
sigma_2 = np.array([[1, rho], [rho, 1]])
L_2 = np.linalg.cholesky(sigma_2)
Z_2 = np.random.standard_normal((2, N))
X_2 = L_2 @ Z_2

r12 = 0.7
r13 = -0.3
r23 = 0.4
sigma_3 = np.array([[1, r12, r13], [r12, 1, r23], [r13, r23, 1]])

L_3 = np.linalg.cholesky(sigma_3)
Z_3 = np.random.standard_normal((3, N))
X_3 = L_3 @ Z_3

print("── 2x2 ──")
print(f"teoretyczne ρ: {rho}")
print(f"empiryczna ρ: {np.corrcoef(X_2)[0,1]:.4f}")

print("\n── 3x3 ──")
emp = np.corrcoef(X_3)
print(f"teoretyczna ρ12={r12},  empiryczna: {emp[0,1]:.4f}")
print(f"teoretyczna ρ13={r13}, empiryczna: {emp[0,2]:.4f}")
print(f"teoretyczna ρ23={r23},  empiryczna: {emp[1,2]:.4f}")

#rysowanie
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Cholesky — Correlated Normal Vectors", fontsize=14)

# 2x2
axes[0].scatter(X_2[0], X_2[1], alpha=0.1, s=5, color='steelblue')
axes[0].set_title(f"2×2: X₁ vs X₂  (ρ={rho})")
axes[0].set_xlabel("X₁")
axes[0].set_ylabel("X₂")
axes[0].grid(alpha=0.3)

# 3x3 — all three pairs
pairs = [(0, 1, r12, "X₁ vs X₂"), (0, 2, r13, "X₁ vs X₃"), (1, 2, r23, "X₂ vs X₃")]
for ax, (i, j, rho_ij, label) in zip(axes[1:], pairs):
    ax.scatter(X_3[i], X_3[j], alpha=0.1, s=5, color='steelblue')
    ax.set_title(f"3×3: {label}  (ρ={rho_ij})")
    ax.set_xlabel(label.split(" vs ")[0])
    ax.set_ylabel(label.split(" vs ")[1])
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

