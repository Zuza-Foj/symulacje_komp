import numpy as np
import matplotlib.pyplot as plt

def demp(X, x):
    suma = 0
    for elem in X:
        if elem <= x:
            suma += 1
    return suma / len(X)

zm_l = [1, 2, 3, 4, 5, 6]

def drawdemp(X):
    X_s = sorted(X)
    min = X_s[0] - 1
    max = X_s[-1] + 1
    x_vals = np.linspace(min, max, 1000)
    y_vals = []
    for x in x_vals:
        y_vals.append(demp(X, x))

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals)
    plt.xlabel('wartości zmiennej losowej X')
    plt.ylabel('prawdopodobieństwo')
    plt.show()

print(drawdemp(zm_l))


#3.
N = 1000
N_1 = 50
N_2 = 10000
# zmiana parametru N we wszystkich miejscach na N_1 lub N_2, to reszta podpunktu 3.
exp_data = np.random.exponential(1, N)

x_theory = np.linspace(0, np.max(exp_data), N)
tcdf = 1 - np.exp(-x_theory)

x_sorted = np.sort(exp_data)
ecdf = np.arange(1, N + 1) / N

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
ax.step(x_sorted, ecdf, label='empiryczna', color='steelblue')
ax.plot(x_theory, tcdf, label='teoretyczna', color='tomato')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.show()