import numpy as np
import matplotlib.pyplot as plt
from zad4 import boxmuller, generator_bm
import time

def boxmuller_b(mi, sig):
    while True:
        u1 = np.random.uniform(-1, 1)
        u2 = np.random.uniform(-1, 1)
        rr = u1**2 + u2**2
        if  rr <= 1:
           break
    factor = np.sqrt(-2 * np.log(rr) / rr)
    z0 = u1 * factor
    z1 = u2 * factor
    return mi + sig * z0, mi + sig * z1

def generator_bmb(mi, sig, n):
    samples = []
    for i in range(0, n, 2):
        z0, z1 = boxmuller(mi, sig)
        samples.append(z0)
        if len(samples) < n:
            samples.append(z1)
    return samples

if __name__ == '__main__':
    n = 10000
    sizes = np.logspace(2, 6, 10, dtype=int)
    times_s = []
    times_b = []
    for i in sizes:
        start = time.time()
        boxmuller(0, 1)
        generator_bm(0, 1, n)
        times_s.append(time.time() - start)

        start = time.time()
        boxmuller_b(0, 1)
        times_b.append(time.time() - start)

    # rysowanie
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_s, 'o-', label='Klasyczny Box-Muller (cos/sin)')
    plt.plot(sizes, times_b, 's-', label='Biegunowy (Marsaglia - bez cos/sin)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Liczba generowanych zmiennych (n)')
    plt.ylabel('Czas wykonania (sekundy)')
    plt.title('Porównanie wydajności metod Boxa-Mullera')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()
