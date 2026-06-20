import numpy as np
import matplotlib.pyplot as plt

def ruina_paryska(r_0, c, lam, k, T_max, T_paris):
    t = 0
    R = r_0
    czasy = [0]
    kapital = [r_0]
    t_bankructwa = None
    while t < T_max:
        d_t = np.random.exponential(1 / lam)
        t_nastepny = t + d_t
        if t_nastepny > T_max:
            d_t_k = T_max - t
            R_k = R + c * d_t_k
            if R < 0 and t_bankructwa is not None:
                if T_max - t_bankructwa > T_paris:
                    return t_bankructwa + T_paris, czasy, kapital
            czasy.append(T_max)
            kapital.append(R_k)
            break
        R_przed_szkoda = R + c * d_t

        if R < 0:
            if R_przed_szkoda >= 0:
                dt_do_zera = -R / c
                t_przeciecia = t + dt_do_zera
                if t_przeciecia - t_bankructwa > T_paris:
                    moment_ruiny = t_bankructwa + T_paris
                    czasy.append(moment_ruiny)
                    kapital.append(0)
                    return moment_ruiny, czasy, kapital