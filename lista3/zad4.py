import numpy as np
import math

def boxmuller(mi, sig):
    u1 = np.random.uniform()
    u2 = np.random.uniform()
    r = np.sqrt( -2 * np.log(u1))
    theta = 2 * np.pi * u2
    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)
    return mi + sig * z0, mi + sig * z1

def generator_bm(mi, sig, n):
    samples = []
    for i in range(0, n, 2):
        z0, z1 = boxmuller(mi, sig)
        samples.append(z0)
        if len(samples) < n:
            samples.append(z1)
    return samples

def opis(samples):
    n = len(samples)
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / n
    std = math.sqrt(var)
    print(f"liczba prób: {n}")
    print(f"średnia: {mean:.4f}")
    print(f"odchylenie standardowe: {std:.4f}")
    print(f"wartość minimalna: {min(samples):.4f}")
    print(f"wartość maksymalna: {max(samples):.4f}")


if __name__ == "__main__":
    print("Standardowy rozkład normalny N(0, 1) - n=10 000")
    s1 = generator_bm(mi=0, sig=1, n=10_000)
    print(opis(s1))
    print("Mój normalany N(5, 2²) - n=10 000")
    s2 = generator_bm(mi=5, sig=2, n=10_000)
    print(opis(s2))