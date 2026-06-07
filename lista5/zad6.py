import numpy as np
import matplotlib.pyplot as plt

def empirical_cf(u_vector, sample_data):
    ecf_values = []
    for u in u_vector:
        cos_term = np.mean(np.cos(u * sample_data))
        sin_term = np.mean(np.sin(u * sample_data))
        ecf_values.append(complex(cos_term, sin_term))
    return np.array(ecf_values)

lambda_param = 2.0
T = 3.0
N = 5000
u_vals = np.linspace(-3, 3, 150)


def generate_compound_poisson_trajectory(dist_type, lambda_p, T_max):
    N = np.random.poisson(lambda_p * T_max)
    if N == 0:
        return np.array([0, T_max]), np.array([0, 0])

    jump_times = np.sort(np.random.uniform(0, T_max, N))
    if dist_type == 'normal':
        z_jumps = np.random.normal(0, 1, N)
    elif dist_type == 'cauchy':
        z_jumps = np.random.standard_cauchy(N)

    y_values = np.cumsum(z_jumps)
    times = np.concatenate(([0], jump_times, [T_max]))
    values = np.concatenate(([0], y_values, [y_values[-1]]))
    return times, values

YT_normal = []
YT_cauchy = []

for n in range(N):
    N_T = np.random.poisson(lambda_param * T)
    if N_T > 0:
        YT_normal.append(np.sum(np.random.normal(0, 1, N_T)))
        YT_cauchy.append(np.sum(np.random.standard_cauchy(N_T)))
    else:
        YT_normal.append(0.0)
        YT_cauchy.append(0.0)

YT_normal = np.array(YT_normal)
YT_cauchy = np.array(YT_cauchy)

ecf_normal_T = empirical_cf(u_vals, YT_normal)
ecf_cauchy_T = empirical_cf(u_vals, YT_cauchy)
theoretical_cf_normal_T = np.exp(lambda_param * T * (np.exp(-(u_vals ** 2) / 2) - 1))
theoretical_cf_cauchy_T = np.exp(lambda_param * T * (np.exp(-np.abs(u_vals)) - 1))

# wizualizacja
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
axes[0, 0].set_title("Przykładowe trajektorie $Y_t$ (Skoki: Rozkład Normalny)")
for i in range(4):
    t_plot, y_plot = generate_compound_poisson_trajectory('normal', lambda_param, T)
    axes[0, 0].step(t_plot, y_plot, where='post', alpha=0.8, label=f'Trajektoria {i+1}')
axes[0, 0].set_xlabel("Czas $t$")
axes[0, 0].set_ylabel("$Y_t$")
axes[0, 0].grid(True, linestyle='--')
axes[0, 0].legend()

axes[0, 1].plot(u_vals, ecf_normal_T.real, 'b-', lw=2, label='Empiryczna (Real)')
axes[0, 1].plot(u_vals, theoretical_cf_normal_T, 'r--', lw=2, label='Teoretyczna')
axes[0, 1].set_title(f"Funkcja charakterystyczna $Y_T$ dla $t={T}$ (Skoki: Normalne)")
axes[0, 1].set_xlabel("$u$")
axes[0, 1].set_ylabel(r"$\phi_{Y_T}(u)$")
axes[0, 1].grid(True, linestyle='--')
axes[0, 1].legend()

axes[1, 0].set_title("Przykładowe trajektorie $Y_t$ (Skoki: Rozkład Cauchy'ego)")
for i in range(4):
    t_plot, y_plot = generate_compound_poisson_trajectory('cauchy', lambda_param, T)
    axes[1, 0].step(t_plot, y_plot, where='post', alpha=0.8, label=f'Trajektoria {i+1}')
axes[1, 0].set_xlabel("Czas $t$")
axes[1, 0].set_ylabel("$Y_t$")
axes[1, 0].grid(True, linestyle='--')
axes[1, 0].legend()

axes[1, 1].plot(u_vals, ecf_cauchy_T.real, 'g-', lw=2, label='Empiryczna (Real)')
axes[1, 1].plot(u_vals, theoretical_cf_cauchy_T, 'r--', lw=2, label='Teoretyczna')
axes[1, 1].set_title(f"Funkcja charakterystyczna $Y_T$ dla $t={T}$ (Skoki: Cauchy)")
axes[1, 1].set_xlabel("$u$")
axes[1, 1].set_ylabel(r"$\phi_{Y_T}(u)$")
axes[1, 1].grid(True, linestyle='--')
axes[1, 1].legend()

plt.tight_layout()
plt.show()