import numpy as np
import matplotlib.pyplot as plt

#zad 4.
def zaawansowany_proces_ryzyka(r_0, c, k, lam, T, alpha, tau, L):
    t = 0
    R = r_0
    while R >= 0 and t < T:
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
    while R >= 0 and t < T:
        d_t = np.random.exponential(1 / lam)
        t += d_t
        if t > T:
            # czasy.append(t_przed)
            # R += c * d_t
            # if R <= L:
            #     kapital.append(R)
            # elif R > L:
            #     kapital.append(R - (c * d_t * tau))
            break
        czasy.append(t)
        R += c * d_t
        Z = np.random.exponential(1 / k)
        Z_i = alpha * Z
        R -= Z_i
        if R <= L:
            kapital.append(R)
        elif R > L:
            R -= c * d_t * tau
            kapital.append(R)
        if R <= 0:
            break
    plt.plot(czasy, kapital, label=nazwa_zestawu)
    plt.hlines(L, 0, T, color="red", linestyle="--")

if __name__ == "__main__":
    N = 2000
    T = 100
    b_r = 50
    b_c = 1.2
    b_lam = 1.0
    b_k = 1
    b_tau = 0.5
    b_L = 100

    alpha = [0.3, 0.6, 0.9]
    tau = [0.3, 0.6, 0.9]
    L = [50, 100, 500, 1000]

    dane_a = {}

    for a in alpha:
        a_czasy = []
        # for n in range(N):
        t_ruiny = rysuj_R(b_r, b_c, b_k, b_lam, T, a, b_tau, b_L, "rozne alfy")
        if t_ruiny is not None:
                a_czasy.append(t_ruiny)
        dane_a[a] = a_czasy

    plt.show()
