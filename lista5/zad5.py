import numpy as np
import matplotlib.pyplot as plt

def empirical_characteristic_function(u_vector, data_sample):
    ecf_values = []
    n = len(data_sample)
    for u in u_vector:
        cos_term = np.mean(np.cos(u * data_sample))
        sin_term = np.mean(np.sin(u * data_sample))
        ecf_values.append(complex(cos_term, sin_term))
    return np.array(ecf_values)

N = 2000
u_vals = np.linspace(-4, 4, 200)
normal_sample = np.random.normal(0, 1, N)
exponential_sample = np.random.exponential(scale=1.0, size=N)
ecf_normal = empirical_characteristic_function(u_vals, normal_sample)
ecf_exponential = empirical_characteristic_function(u_vals, exponential_sample)

theoretical_cf_normal = np.exp(-(u_vals**2) / 2)
theoretical_cf_exponential = 1 / (1 - 1j * u_vals)

# wizualizacja
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].plot(u_vals, ecf_normal.real, 'b-', lw=2, label='Empiryczna (Real)')
axes[0, 0].plot(u_vals, theoretical_cf_normal.real, 'r--', lw=2, label='Teoretyczna (Real)')
axes[0, 0].set_title("Rozkład Normalny $N(0,1)$ - Część Rzeczywista")
axes[0, 0].set_xlabel("$u$")
axes[0, 0].set_ylabel(r"$Re(\phi(u))$")
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle='--')

axes[0, 1].plot(u_vals, ecf_normal.imag, 'b-', lw=2, label='Empiryczna (Imag)')
axes[0, 1].plot(u_vals, theoretical_cf_normal.imag, 'r--', lw=2, label='Teoretyczna (Imag)')
axes[0, 1].set_title("Rozkład Normalny $N(0,1)$ - Część Urojona")
axes[0, 1].set_xlabel("$u$")
axes[0, 1].set_ylabel(r"$Im(\phi(u))$")
axes[0, 1].set_ylim(-0.2, 0.2)
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle='--')

axes[1, 0].plot(u_vals, ecf_exponential.real, 'g-', lw=2, label='Empiryczna (Real)')
axes[1, 0].plot(u_vals, theoretical_cf_exponential.real, 'r--', lw=2, label='Teoretyczna (Real)')
axes[1, 0].set_title("Rozkład Wykładniczy $Exp(1)$ - Część Rzeczywista")
axes[1, 0].set_xlabel("$u$")
axes[1, 0].set_ylabel(r"$Re(\phi(u))$")
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='--')

axes[1, 1].plot(u_vals, ecf_exponential.imag, 'g-', lw=2, label='Empiryczna (Imag)')
axes[1, 1].plot(u_vals, theoretical_cf_exponential.imag, 'r--', lw=2, label='Teoretyczna (Imag)')
axes[1, 1].set_title("Rozkład Wykładniczy $Exp(1)$ - Część Urojona")
axes[1, 1].set_xlabel("$u$")
axes[1, 1].set_ylabel(r"$Im(\phi(u))$")
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle='--')

plt.tight_layout()
plt.show()
