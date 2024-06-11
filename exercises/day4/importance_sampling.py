from scipy import stats
import scipy
import numpy as np

def crude_MC_estimator(n: int, a: float) -> float:
    Z = stats.norm.rvs(size=n)
    return (Z > a).mean()

def IS_norm(n: int, a: float, var: float):
    def h(x: float):
        if x > a:
            return 1
        else:
            return 0
    
    f = stats.norm.pdf
    g = lambda x: stats.norm.pdf(x, a, var)

    g_samples = stats.norm.rvs(a, var, n)

    estimates = []
    for y in g_samples:
        estimates.append(h(y) * f(y) / g(y))
    
    return np.mean(estimates)

def IS_exp(n: int, l: float):
    g = lambda x: l*np.exp(-l*x)
    h = lambda x: np.exp(x)

    # since f is the uniform distribution, the density function for f is always 1 for 0 < x < 1
    estimates = []
    for i in range(n):
        u = np.random.rand()
        estimates.append(h(u)/g(u))
    return np.mean(estimates)

def find_lambda():
    def f(l):
        return 1/l**2 * (1/(1+l)*np.exp(2+2*l) - 1/(2+l)) - 1/(l+l**2)**2*(np.exp(1+l)+1)**2
    res = scipy.optimize.minimize_scalar(f)
    return res
    


if __name__ == "__main__":
    # Task 7
    a = 3
    print("Task 7")
    print("Crude MC Estimator")
    n = 10000
    p_MC = crude_MC_estimator(n, a)
    print(f"p(Z>a) = {p_MC:.3f}")
    print()

    print("IS")
    var = 1
    p_IS = IS_norm(n, a, var)
    print(f"p(Z>a) = {p_IS:.3f}")
    print()

    

    

