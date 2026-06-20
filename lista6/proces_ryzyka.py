import numpy as np
import matplotlib.pyplot as plt

#zad 1. (analiza czułości czasu ruiny)
def czas_ruiny(r_0, c, lam, k, T):
    t = 0
    R = r_0 #kwota w czasie t=0 wynosi r_0
    while R >= 0 and t < T:
        d_t = np.random.exponential(1/lam)
        t += d_t
        if T < t:
            break
        R += c * d_t
        Z = np.random.exponential(1/k)
        R -= Z
        if R <= 0:
            return t #mój czas ruiny
    return None

def rysuj_sciezke_R(r_0, c, lam, k, T, nazwa_zestawu):
    t = 0
    R = r_0
    czasy = [0]
    kapital = [r_0]
    while R >= 0 and t < T:
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
        R -= Z
        czasy.append(t)
        kapital.append(R)
        if R <= 0:
            break
    plt.plot(czasy, kapital, label=nazwa_zestawu)

if __name__ == "__main__":
    N = 2000
    T = 100
    r_0 = [10, 50, 100]
    c = [1, 2, 3]
    lam = [0.5, 1.0, 2.0]
    k = [0.5, 1.0, 2.0]

    b_r = 15
    b_c = 1.2
    b_lam = 1.0
    b_k = 1

    #(a) i (c) histogram czasu ruiny i wnioski
    dane_r0 = {}
    dane_c = {}
    dane_lam = {}
    dane_k = {}

    for r in r_0:
        r_czasy = []
        for n in range(N):
            t_ruiny = czas_ruiny(r, 2, 1.0, 1, T)
            if t_ruiny is not None:
                r_czasy.append(t_ruiny)
        dane_r0[r] = r_czasy
        if len(r_czasy) > 0:
            srednia = np.mean(r_czasy)
            odchylenie = np.std(r_czasy)
            procent_ruin = len(r_czasy) / N * 100
            print(f"Dla r0={r}:")
            print(f"  -> Średni czas ruiny: {srednia:.2f}")
            print(f"  -> Odchylenie standardowe: {odchylenie:.2f}")
            print(f"  -> Prawdopodobieństwo ruiny: {procent_ruin:.1f}%")
        else:
            print(
                f"Dla r0={r}: brak przypadków ruiny w zadanym czasie T={T} (prawdopodobieństwo ruiny empirycznej = 0%)")

    for c_i in c:
        c_czasy = []
        for n in range(N):
            t_ruiny = czas_ruiny(b_r, c_i, b_lam, b_k, T)
            if t_ruiny is not None:
                c_czasy.append(t_ruiny)
        dane_c[c_i] = c_czasy
        if len(c_czasy) > 0:
            srednia = np.mean(c_czasy)
            odchylenie = np.std(c_czasy)
            procent_ruin = len(c_czasy) / N * 100
            print(f"Dla c={c_i}:")
            print(f"  -> Średni czas ruiny: {srednia:.2f}")
            print(f"  -> Odchylenie standardowe: {odchylenie:.2f}")
            print(f"  -> Prawdopodobieństwo ruiny: {procent_ruin:.1f}%")
        else:
            print(f"Dla c={c_i}: brak przypadków ruiny w symulacji")

    for lam_i in lam:
        lam_czasy = []
        for n in range(N):
            t_ruiny = czas_ruiny(b_r, b_c, lam_i, b_k, T)
            if t_ruiny is not None:
                lam_czasy.append(t_ruiny)
        dane_lam[lam_i] = lam_czasy
        if len(lam_czasy) > 0:
            srednia = np.mean(lam_czasy)
            odchylenie = np.std(lam_czasy)
            procent_ruin = len(lam_czasy) / N * 100
            print(f"Dla lambda={lam_i}:")
            print(f"  -> Średni czas ruiny: {srednia:.2f}")
            print(f"  -> Odchylenie standardowe: {odchylenie:.2f}")
            print(f"  -> Prawdopodobieństwo ruiny: {procent_ruin:.1f}%")
        else:
            print(f"Dla lambda={lam_i}: brak przypadków ruiny w symulacji")

    for k_i in k:
        k_czasy = []
        for n in range(N):
            t_ruiny = czas_ruiny(b_r, b_c, b_lam, k_i, T)
            if t_ruiny is not None:
                k_czasy.append(t_ruiny)
        dane_k[k_i] = k_czasy
        if len(k_czasy) > 0:
            srednia = np.mean(k_czasy)
            odchylenie = np.std(k_czasy)
            procent_ruin = len(k_czasy) / N * 100
            print(f"Dla k={k_i}:")
            print(f"  -> Średni czas ruiny: {srednia:.2f}")
            print(f"  -> Odchylenie standardowe: {odchylenie:.2f}")
            print(f"  -> Prawdopodobieństwo ruiny: {procent_ruin:.1f}%")
        else:
            print(f"Dla k={k_i}: brak przypadków ruiny w symulacji")

    fig, axs = plt.subplots(1, 4, figsize=(24, 6), sharey=True, tight_layout=True)

    for r in r_0:
        if len(dane_r0[r]) > 0:
            axs[0].hist(dane_r0[r], bins=30, alpha=0.5, label=f"r0 = {r}")
    axs[0].set_title("Wpływ kapitału początkowego (r0)")
    axs[0].set_xlabel("Czas ruiny")
    axs[0].set_ylabel("Liczba ruin")
    axs[0].legend()

    for c_val in c:
        if len(dane_c[c_val]) > 0:
            axs[1].hist(dane_c[c_val], bins=30, alpha=0.5, label=f"c = {c_val}")
    axs[1].set_title("Wpływ intensywności składek (c)")
    axs[1].set_xlabel("Czas ruiny")
    axs[1].legend()

    for l in lam:
        if len(dane_lam[l]) > 0:
            axs[2].hist(dane_lam[l], bins=30, alpha=0.5, label=f"c = {l}")
    axs[2].set_title("Wpływ częstości pojawiania się szkód (lam)")
    axs[2].set_xlabel("Czas ruiny")
    axs[2].legend()

    for k_i in k:
        if len(dane_k[k_i]) > 0:
            axs[3].hist(dane_k[k_i], bins=30, alpha=0.5, label=f"c = {k_i}")
    axs[3].set_title("Wpływ intensywności napływu roszczeń (k)")
    axs[3].set_xlabel("Czas ruiny")
    axs[3].legend()

    plt.show()

    #(d) Wnioski:
    # Im większe r_0, tym mniejsze prawdopodobieństwo ruiny, ale nie uratuje firmy w nieskończoność, jeśli składki są za niskie. Dlatego mamy histogram tylko
    # dla r_0 = 10, no dla następnych wartości ruina się nie zdarza, dla moich ustalonych default c, k, lambda, T.
    # Zbyt niska składka gwarantuje powolną ruinę.
    # Jeśli szkód jest zbyt wiele, trzeba podnieść składki, aby uniknąć ruiny.
    # Im większa średnia wartość szkody (k), tym większe prawdopodobiństwo ruiny (małe wahnięcia gwałtownie i poważnie szkodzą).

    #(b) wykresy przebiegu R(t)
    plt.figure(figsize=(10, 5))

    rysuj_sciezke_R(r_0=10, c=1.1, lam=1.5, k=1.0, T=T, nazwa_zestawu="Zestaw Ryzykowny (r0=10, c=1.1, lam=1.5)")
    rysuj_sciezke_R(r_0=50, c=3.0, lam=0.5, k=1.0, T=T, nazwa_zestawu="Zestaw Bezpieczny (r0=50, c=3.0, lam=0.5)")

    plt.axhline(0, color='red', linestyle='--', alpha=0.7, label="Poziom Ruiny (R=0)")
    plt.title("Przykładowe przebiegi procesu ryzyka R(t)")
    plt.xlabel("Czas (t)")
    plt.ylabel("Kapitał firmy R(t)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.show()




