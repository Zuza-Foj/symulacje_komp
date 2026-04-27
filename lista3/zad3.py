import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.integrate import quad

n = 10000

# (a)
sample_a = []
M_a = 3/2      #bierzemy maximum funkcji f, bo g jednostajengo to 1
while len(sample_a) < n:
    x = np.random.uniform(0, 1)
#   u = np.random.uniform(0, 1)
    f = 3/2 * (1 - x**2)    #nowa wartość na podstwie gęstości z (a)
    g = 1
    if x <= f / (M_a * g):
        sample_a.append(x)
sample_a = np.array(sample_a)

def pdf_a(x):
    return 1.5 * (1 - x**2)
def cdf_a(x):
    return 1.5 * x - 0.5 * x**3

class dist_a_gen(stats.rv_continuous):
    def _pdf(self, x): return pdf_a(x)
dist_a = dist_a_gen(a=0, b=1)


# (b)
sample_b = []
x_r = np.linspace(0, np.pi, n)
f_b = 3/2 * np.sin(x_r) * np.cos(x_r)**2
g_b = 1/np.pi
M_b = np.max(f_b / g_b)     # znalezione numerycznie, bo rozwiązanie równania mnie przerosło
while len(sample_b) < n:
    x = np.random.uniform(0, np.pi)
    u = np.random.uniform(0, 1)
    f = 3/2 * np.sin(x) * np.cos(x)**2
    g = 1/np.pi
    if u <= f / (M_b * g):
        sample_b.append(x)
sample_b = np.array(sample_b)

def pdf_b(x):
    return 1.5 * np.sin(x) * np.cos(x)**2
def cdf_b(x):
    return 0.5 * (1 - np.cos(x)**3)

class dist_b_gen(stats.rv_continuous):
    def _pdf(self, x): return pdf_b(x)
dist_b = dist_b_gen(a=0, b=np.pi)


# (c)
sample_c = []
lam = 1.0
x_r_c = np.linspace(0.001, 10, n)
f_c = 2 * np.sqrt(1/(2*np.pi)) * np.exp(-x_r_c**2 / 2)
g_c = lam * np.exp(-lam * x_r_c)
M_c = np.max(f_c / g_c)
while len(sample_c) < n:
    x = np.random.exponential(1/lam)
    u = np.random.uniform(0, 1)
    f = 2 * np.sqrt(1/(2*np.pi)) * np.exp(-x**2 / 2)
    g = lam * np.exp(-lam * x)
    if u <= f / (M_c * g):
        sample_c.append(x)
sample_c = np.array(sample_c)

def pdf_c(x):
    return np.sqrt(2/np.pi) * np.exp(-x**2/2)
def cdf_c(x):
    return 2 * stats.norm.cdf(x) - 1

# rysowanie:
def plot_analysis(sample, pdf_func, cdf_func, dist_obj, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=16)
    x_grid = np.linspace(min(sample), max(sample), 1000)

    # cdf
    axes[0].plot(x_grid, cdf_func(x_grid), 'r-', lw=2, label='Teoretyczna')
    sorted_sample = np.sort(sample)
    y_emp = np.arange(1, n + 1) / n
    axes[0].step(sorted_sample, y_emp, 'b--', label='Empiryczna', alpha=0.7)
    axes[0].set_title('Dystrybuanta')
    axes[0].legend()

    # pdf, gęstość
    axes[1].hist(sample, bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.6, label='Próba')
    axes[1].plot(x_grid, pdf_func(x_grid), 'r-', lw=2, label='Teoretyczna PDF')
    axes[1].set_title('Histogram vs Gęstość')
    axes[1].legend()

    # qq-plot
    stats.probplot(sample, dist=dist_obj, plot=axes[2])
    axes[2].set_title('Wykres QQ')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# wywołanie:
plot_analysis(sample_a, pdf_a, cdf_a, dist_a, "f(x) = 1.5(1 - x^2)")
plot_analysis(sample_b, pdf_b, cdf_b, dist_b, "f(x) = 1.5*sin(x)*cos^2(x)")
plot_analysis(sample_c, pdf_c, cdf_c, stats.halfnorm, "Rozkład półnormalny")


#statystyki:
def print_stats(name, sample, theory_pdf, a, b):
    emp_mean = np.mean(sample)
    emp_var = np.var(sample)
    mean_th, _ = quad(lambda x: x * theory_pdf(x), a, b)
    second_moment, _ = quad(lambda x: (x ** 2) * theory_pdf(x), a, b)
    var_th = second_moment - mean_th ** 2
    print(f"--- {name} ---")
    print(f"Średnia:   Empiryczna = {emp_mean:.5f} | Teoretyczna = {mean_th:.5f}")
    print(f"Wariancja: Empiryczna = {emp_var:.5f} | Teoretyczna = {var_th:.5f}\n")

#printy:
print_stats("Podpunkt (a)", sample_a, pdf_a, 0, 1)
print_stats("Podpunkt (b)", sample_b, pdf_b, 0, np.pi)
print_stats("Podpunkt (c)", sample_c, pdf_c, 0, 20)