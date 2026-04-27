import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

N = 20000

# (a)
x_pi = np.random.uniform(-1, 1, N)
y_pi = np.random.uniform(-1, 1, N)
inside_circle = (x_pi**2 + y_pi**2 <= 1)
hits_cumulative = np.cumsum(inside_circle)
n = np.arange(1, N + 1)
pi_estimates = 4 * hits_cumulative / n

# (b)
x_int = np.random.uniform(0, 1, N)
y_int = np.random.uniform(0, 1, N)
under_curve = (y_int <= np.exp(-x_int**2))
integral_est = np.sum(under_curve) / N
integral_true, _ = quad(lambda x: np.exp(-x**2), 0, 1)

plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
plt.plot(n, pi_estimates, color='blue', label='Estymator $\pi$')
plt.axhline(y=np.pi, color='red', linestyle='--', label='Wartość rzeczywista $\pi$')
plt.xscale('log')
plt.title('Zbieżność estymatora $\pi$')
plt.xlabel('Liczba prób (skala log)')
plt.ylabel('Wartość $\pi$')
plt.legend()
plt.show()

print(f"Oszacowana całka: {integral_est:.6f}")
print(f"Scipy (quad): {integral_true:.6f}")
print(f"Różnica: {abs(integral_true - integral_est):.6f}")