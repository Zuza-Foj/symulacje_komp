import numpy as np
import matplotlib.pyplot as plt

def ruina_paryska(r_0, c, lam, k, T_max, T_paris):
   t_miedzy = []
   t = 0
   while t < T_max:
        dt = np.random.exponential(1.0 / lam)
        t += dt
        t_miedzy.append(t)
   t_szkod = np.array(t_miedzy)
   t_szkod = t_szkod[t_szkod < T_max]
   rozm_szkod = np.random.exponential(1 / k, size=len(t_szkod))




