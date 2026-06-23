import numpy as np
import matplotlib.pyplot as plt

#zad 4.
def zaawansowany_proces_ryzyka(r_0, c, k, lam, T, alpha, tau, L):
    t = 0
    R = r_0
    while L >= R >= 0 and t < T:
        d_t = np.random.exponential(1 / lam)
        t += d_t
        if T < t:
            break
        R += c * d_t
        Z = np.random.exponential(1 / k)
        Z_i = alpha * Z
        R -= Z_i
        if R > L:
            R -= (c * d_t) * tau
        elif R <= 0:
            return t  # mój czas ruiny
    return None

def rysuj_R(r_0, c, k, lam, T, alpha, tau, L, nazwa_zestawu):
    t = 0
    R = r_0
    czasy = [0]
    kapital = [r_0]
    while L >= R >= 0 and t < T:
        d_t = np.random.exponential(1 / lam)
        t_przed = t + d_t
        if t_przed > T:
            czasy.append(T)
            kapital.append(R + c * (T - t))
            break
        czasy.append(t_przed)
        kapital.append(R + c * d_t)
        t = t_przed
        R += c * d_t
        Z = np.random.exponential(1 / k)
        Z_i = alpha * Z
        R -= Z_i
        czasy.append(t)
        if R > L:
            kapital.append(R - (c * d_t) * tau)
        elif R <= L:
            kapital.append(R)
        if R <= 0:
            break
    plt.plot(czasy, kapital, label=nazwa_zestawu)

if __name__ == "__main__":
    N = 2000
    T = 100
    b_r = 50
    b_c = 1.2
    b_lam = 1.0
    b_k = 1

    alpha = []
    tau = []
