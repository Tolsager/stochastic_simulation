import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, expon, pareto, chi2, kstest


def box_mueller(n: int) -> np.ndarray:
    u1 = np.random.random(n)
    u2 = np.random.random(n)
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return np.array([z1, z2])


def transform_normal(z: np.ndarray, mu: float, sigma: float):
    # Transform the standard normal random variables to normal random variables
    # sigma: standard deviation, mu: mean
    return mu + sigma * z


def exponential(lamb: float, n: int) -> np.ndarray:
    unif_rvs = np.random.random(n)
    return -np.log(unif_rvs) / lamb


def _pareto(beta: float, k: float, n: int) -> np.ndarray:
    unif_rvs = np.random.random(n)
    return beta / (unif_rvs ** (1 / k))


if __name__ == '__main__':
    n: int = 10000

    lam = 0.5
    exp_rvs = exponential(lam, n)
    print("Exponential distribution")
    print(f"Sample mean: {np.mean(exp_rvs):.3f}, Sample variance {np.var(exp_rvs):.3f}")
    print(f"Theoretical mean: {1/lam:.3f}, Theoretical variance: {1/pow(lam, 2):.3f}")
    print(kstest(exp_rvs, expon(scale=1/lam).cdf))

    x = np.linspace(0, np.max(exp_rvs), 500)
    plt.hist(exp_rvs, bins=50, density=True, rwidth=0.8, color="dodgerblue", edgecolor="black")
    plt.plot(x, expon.pdf(x, scale=1/lam), linewidth=2, color="orange")
    plt.tight_layout()
    plt.show()

    print("\nPareto distributions")
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    ks = [2.05, 2.5, 3, 4]
    beta = 1
    for idx, k in enumerate(ks):
        pareto_rvs = _pareto(beta, k, n)

        print(f"Beta: {beta}, k: {k}")
        print(kstest(pareto_rvs, pareto(k, scale=beta).cdf))
        print(f"Mean: {np.mean(pareto_rvs):.3f}, Variance: {np.var(pareto_rvs):.3f}")
        print(f"Theoretical mean: {k*beta/(k-1):.3f}, Theoretical variance: {beta**2 * (k / ((k-1)**2 * (k-2))):.3f}")
        
        x = np.linspace(beta, np.max(pareto_rvs), 500)
        ax[idx].hist(pareto_rvs, bins=100, density=True, rwidth=0.8, color="dodgerblue", edgecolor="black")
        # pdf of the pareto distribution: f(x) = k * beta^k / x^(k+1)
        ax[idx].plot(x, (k * beta ** k) / (x ** (k+1)), linewidth=2, color="orange")
        ax[idx].set_title(f"Pareto distribution with k = {k}")

    plt.tight_layout()
    plt.show()

    z1, z2 = box_mueller(n//2)

    z = np.concatenate((z1, z2))
    print(f"Box Muller Normal distribution\nMean: {np.mean(z):.3f}, Variance: {np.var(z):.3f}")
    print(kstest(z, norm().cdf))

    x = np.linspace(-4, 4, 100)
    plt.hist(z, bins=50, density=True, rwidth=0.8, color="dodgerblue", edgecolor="black")
    plt.plot(x, norm.pdf(x), linewidth=2, color="orange")
    plt.tight_layout()
    plt.show()

    # 100 95% confidence intervals using 10 samples for the normal distribution
    mean_samples = np.zeros((100, 2))
    var_samples = np.zeros((100, 2))
    for i in range(100):
        z1, z2 = box_mueller(10 // 2) # Box Muller generates 2 samples at a time
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
    print(f"Variance of mean confidence interval bounds: {mean_samples.var(axis=0)}")
    print(f"Variance confidence intervals: {var_samples.mean(axis=0)}")
    print(f"Variance of variance confidence interval bounds: {var_samples.var(axis=0)}")

    

    # Use composition to sample from the Pareto distribution
    # U = F(x) = 1 - (1 + X / mu)^(-1) = 1 - mu / (mu + X)
    # U = 1 - mu / (mu + X) => mu + X = mu / (1 - U) => X = mu / (1 - U) - mu = mu / U - mu
    # Pareto distribution with support [0, inf]
    # k = 1
    # beta = mu

    mu = 1
    n = 1000

    y = exponential(mu, n)

    x = exponential(y, n)

    x = x[x < 20]

    plt.hist(x, bins=50, density=True, rwidth=0.8, color="dodgerblue", edgecolor="black")
    plt.plot(np.linspace(0, x.max(), 100), mu / (mu + np.linspace(0, x.max(), 100)) ** 2, linewidth=1, alpha=0.7, color="orange")
    plt.show()

