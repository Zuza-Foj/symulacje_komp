import numpy as np
import matplotlib.pyplot as plt

r0 = 5               # Kapitał początkowy
c = 0.5              # Stały przychód ze składek
lambda_param = 1.0   # Intensywność procesu zgłoszeń szkód
num_simulations = 10000  # Liczba trajektorii do estymacji prawdopodobieństwa
T_max = 5.0          # Maksymalny horyzont czasowy

check_times = [1, 3, 5]
ruin_counts = {t: 0 for t in check_times}
sample_trajectories = []
for sim in range(num_simulations):
    inter_arrival_times = np.random.exponential(1.0 / lambda_param, 30)
    jump_times = np.cumsum(inter_arrival_times)
    jump_times = jump_times[jump_times <= T_max]
    n_jumps = len(jump_times)
    z_jumps = np.random.exponential(1.0, n_jumps) if n_jumps > 0 else np.array([])
    ruined_by_time = {t: False for t in check_times}
    for i in range(n_jumps):
        t_jump = jump_times[i]
        z_size = z_jumps[i]
        capital_before_jump = r0 + c * t_jump - np.sum(z_jumps[:i])
        capital_after_jump = capital_before_jump - z_size
        if capital_after_jump < 0:
            for t in check_times:
                if t_jump <= t:
                    ruined_by_time[t] = True

    for t in check_times:
        if ruined_by_time[t]:
            ruin_counts[t] += 1
    if sim < 5:
        sample_trajectories.append({'jump_times': jump_times, 'z_jumps': z_jumps})

# wizualizacja
for t in check_times:
    prob = ruin_counts[t] / num_simulations
    print(f"Dla t = {t}: {prob:.4f}  ({prob*100:.2f}%)")

plt.figure(figsize=(12, 6))
for idx, traj in enumerate(sample_trajectories):
    j_times = traj['jump_times']
    z_vals = traj['z_jumps']
    plot_t = [0]
    plot_R = [r0]
    current_R = r0
    last_t = 0
    for t_j, z in zip(j_times, z_vals):
        current_R += c * (t_j - last_t)
        plot_t.append(t_j)
        plot_R.append(current_R)
        current_R -= z
        plot_t.append(t_j)
        plot_R.append(current_R)
        last_t = t_j
    current_R += c * (T_max - last_t)
    plot_t.append(T_max)
    plot_R.append(current_R)

    plt.plot(plot_t, plot_R, lw=2, label=f'Trajektoria {idx + 1}')

plt.axhline(0, color='red', linestyle='--', lw=2, label='Linia ruiny (R(t) = 0)')
for t in check_times:
    plt.axvline(t, color='gray', linestyle=':', alpha=0.7)

plt.title("Symulacja procesu ryzyka $R(t) = 5 + 0.5t - \sum Z_i$")
plt.xlabel("Czas $t$")
plt.ylabel("Kapitał $R(t)$")
plt.xlim(0, T_max)
plt.grid(True, linestyle='--')
plt.legend()
plt.show()
