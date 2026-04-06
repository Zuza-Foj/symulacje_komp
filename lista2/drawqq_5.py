import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from num_qf_4 import num_quantile_function

def drawqq(X, distr):
    X_s = sorted(X)
    N = len(X_s)
    Q = []
    for i in range(1, N + 1):
        u = (i - 0.5) / N
        q = num_quantile_function(distr, u, dx=0.01)
        Q.append(q)

    plt.figure(figsize=(10, 6))
    plt.scatter(Q, X_s, color='blue', label='punkty empiryczne')
    min_val = min(min(Q), min(X_s))
    max_val = max(max(Q), max(X_s))
    plt.plot([min_val, max_val], [min_val, max_val], label='y = x')

    plt.xlabel('kwantyle teoretyczne')
    plt.ylabel('kwantyle z próby')
    plt.title('Wykres Kwantylowy (QQ-Plot)')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()

data = np.random.exponential(scale=1, size=100)
def F_exp(x):
    return 1 - np.exp(-x) if x >= 0 else 0
drawqq(data, F_exp)

plt.figure(figsize=(8, 6))
stats.probplot(data, dist="expon", plot=plt)
plt.title("Wbudowany QQ-Plot (SciPy)")
plt.show()