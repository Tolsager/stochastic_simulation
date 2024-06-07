import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, expon, pareto, chi2


def box_mueller(n: int) -> np.ndarray:
    u1 = np.random.random(n)
    u2 = np.random.random(n)
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return np.array([z1, z2])


def exponential(lamb: float, n: int) -> np.ndarray:
    unif_rvs = np.random.random(n)
    return -np.log(unif_rvs) / lamb


def _pareto(beta: float, k: float, n: int) -> np.ndarray:
    unif_rvs = np.random.random(n)
    return beta / (unif_rvs ** (1 / k))


if __name__ == '__main__':
    n: int = 10000

    l = 0.5
    exp_rvs = exponential(l, n)

    x = np.linspace(0, np.max(exp_rvs), 500)
    plt.hist(exp_rvs, bins=50, density=True, rwidth=0.8, color="dodgerblue", edgecolor="black")
    plt.plot(x, expon.pdf(x, scale=1/l), linewidth=2, color="orange")
    plt.show()

    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    ks = [2.05, 2.5, 3, 4]
    beta = 1
    for idx, k in enumerate(ks):
        pareto_rvs = _pareto(beta, k, n)

        print(f"Beta: {beta}, k: {k}")
        print(f"Mean: {np.mean(pareto_rvs):.3f}, Variance: {np.var(pareto_rvs):.3f}")
        print(f"Theoretical mean: {k*beta/(k-1):.3f}, Theoretical variance: {beta**2 * (k / ((k-1)**2 * (k-2))):.3f}")
        
        x = np.linspace(0, np.max(pareto_rvs), 500)
        ax[idx].hist(pareto_rvs, bins=100, density=True, rwidth=0.8, color="dodgerblue", edgecolor="black")
        ax[idx].plot(x, pareto.pdf(x, b=beta, scale=1/k), linewidth=2, color="orange")

    plt.show()

    z1, z2 = box_mueller(n//2)

    x = np.linspace(-4, 4, 100)
    plt.hist(np.concatenate((z1, z2)), bins=50, density=True, rwidth=0.8, color="dodgerblue", edgecolor="black")
    plt.plot(x, norm.pdf(x), linewidth=2, color="orange")
    plt.show()

    # 100 95% confidence intervals using 10 samples for the normal distribution
    mean_samples = np.zeros((100, 2))
    var_samples = np.zeros((100, 2))
    for i in range(100):
        z1, z2 = box_mueller(10 // 2)
        sample = np.concatenate((z1, z2))

        mean = np.mean(sample)
        var = np.var(sample)

        # Mean is normally distributed
        z = 1.96
        bounds = mean + np.array([-1, 1]) * z * np.sqrt(var / 10)

        mean_samples[i] = bounds

        # Variance is chi-squared distributed with n-1 degrees of freedom
        chi2_1 = chi2.ppf(0.975, 9)
        chi2_2 = chi2.ppf(0.025, 9)
        var_bounds = [9 * var / chi2_1, 9 * var / chi2_2]

        var_samples[i] = var_bounds

    print(f"Mean confidence intervals: {mean_samples.mean(axis=0)}")
    print(f"Variance confidence intervals: {var_samples.mean(axis=0)}")






