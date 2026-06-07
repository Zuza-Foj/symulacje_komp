import numpy as np
import matplotlib.pyplot as plt

def simulate_hawkes_process(mu, alpha, beta, T):
    t = 0
    jump_times = []
    def get_lambda(current_t, history):
        if not history:
            return mu
        history = np.array(history)
        return mu + np.sum(alpha * np.exp(-beta * (current_t - history)))
    M = mu
    while True:
        U1 = np.random.uniform(0, 1)
        dt = -np.log(U1) / M
        t += dt
        if t > T:
            break
        lambda_t = get_lambda(t, jump_times)
        U2 = np.random.uniform(0, 1)
        if U2 <= lambda_t / M:
            jump_times.append(t)
            M = lambda_t + alpha
        else:
            pass
    return np.array(jump_times)

T = 15.0
mu = 1.0
beta = 1.0
cases = [{"n_val": 0.4, "alpha": 0.4, "label": "Podkrytyczny (n < 1)", "color": "blue"}, {"n_val": 0.95, "alpha": 0.95, "label": "Krytyczny (n ≈ 1)", "color": "orange"}, {"n_val": 1.3, "alpha": 1.3, "label": "Nadkrytyczny (n > 1)", "color": "red"}]

# rysowanie trajektorii i intensywności
selected_alpha = 1.2
jumps = simulate_hawkes_process(mu, selected_alpha, beta, T)
t_grid = np.linspace(0, T, 1000)
lambda_grid = []
for tg in t_grid:
    past_jumps = jumps[jumps < tg]
    l_val = mu + np.sum(selected_alpha * np.exp(-beta * (tg - past_jumps)))
    lambda_grid.append(l_val)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

times_step = np.concatenate(([0], jumps, [T]))
counts_step = np.concatenate(([0], np.arange(1, len(jumps) + 1), [len(jumps)]))
ax1.step(times_step, counts_step, where='post', color='darkgreen', lw=2)
ax1.set_title(f"Trajektoria procesu zliczającego udostępnień (Hawkes, $\\alpha$={selected_alpha}, $\\beta$={beta})")
ax1.set_ylabel("Łączna liczba udostępnień")
ax1.grid(True, linestyle='--')

ax2.plot(t_grid, lambda_grid, color='crimson', lw=2, label='$\\lambda(t)$')

for j in jumps:
    ax2.axvline(j, color='gray', linestyle=':', alpha=0.5)
ax2.set_title("Przebieg intensywności $\\lambda(t)$ w czasie")
ax2.set_xlabel("Czas $t$")
ax2.set_ylabel("Intensywność $\\lambda(t)$")
ax2.grid(True, linestyle='--')
ax2.legend()

plt.tight_layout()
plt.show()

# analliza łącznej liczby udostępnień
num_sims = 200
for case in cases:
    total_counts = []
    for _ in range(num_sims):
        sim_jumps = simulate_hawkes_process(mu, case["alpha"], beta, T)
        total_counts.append(len(sim_jumps))

    print(f"{case['label']} dla n = {case['n_val']}:")
    print(f"   -> Średnia liczba udostępnień w [0, {T}]: {np.mean(total_counts):.2f}")
    print(f"   -> Maksymalna liczba udostępnień w próbie: {np.max(total_counts)}")