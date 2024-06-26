import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.stats import geom, chisquare
from discrete_sampling_functions import direct_sampling, rejection_sampling, alias_sampling


def kl_divergence(p: list, q: list) -> float:
    return np.sum([p[i]*np.log(p[i]/q[i]) for i in range(len(p))])


def time_function(function, *args, iterations: int = 1000):
    start = time.perf_counter()
    for _ in range(iterations):
        function(*args)
    return (time.perf_counter() - start) / iterations


if __name__ == '__main__':
    TIME = True
    timing_iterations = 100
    p = 0.3
    n = 10000

    unif_rvs = np.random.random(n)

    # Transform the uniforms random variables to geometric random variables
    geom_rvs = np.floor(np.log(unif_rvs)/np.log(1-p)) + 1
    scipy_geom_rvs = geom.rvs(p, size=n)

    fig, ax = plt.subplots(1, 1)

    ax.hist(geom_rvs, bins=range(1, int(max(geom_rvs))+1), density=True, rwidth=0.8, edgecolor="black")
    ax.plot(np.arange(1, int(max(geom_rvs))+1) + 0.5, geom.pmf(range(1, int(max(geom_rvs))+1), p), 'ro-')
    plt.show()


    # compare the two distributions
    print(f"Sample mean: {geom_rvs.mean()}, Sample variance: {geom_rvs.var()}")
    print(f"True mean: {1/p}, True variance: {(1-p)/(p**2)}")

    # The six-point distribution we want to generate samples from
    p = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]

    # Generate the six-point distribution
    fig, ax = plt.subplots(1, 3, figsize=(13, 5))

    if TIME:
        direct_sampling_time = time_function(direct_sampling, unif_rvs, p, iterations=timing_iterations)
        rejection_sampling_time = time_function(rejection_sampling, p, n, iterations=timing_iterations)
        alias_sampling_time = time_function(alias_sampling, p, n, iterations=timing_iterations)

        print(f"Direct sampling average time: {direct_sampling_time:.4f} seconds")
        print(f"Rejection sampling average time: {rejection_sampling_time:.4f} seconds")
        print(f"Alias sampling average time: {alias_sampling_time:.4f} seconds")

    direct_six_point_dist = direct_sampling(unif_rvs, p)
    rejection_six_point_dist = rejection_sampling(p, n)
    alias_six_point_dist = alias_sampling(p, n)

    ax[0].hist(direct_six_point_dist, bins=range(1, 8), density=True, width=0.9, align='mid', rwidth=0.9, edgecolor="black")
    ax[0].set_title("Direct Sampling")
    ax[1].hist(rejection_six_point_dist, bins=range(1, 8), density=True, width=0.9, align='mid', rwidth=0.9, edgecolor="black")
    ax[1].set_title("Rejection Sampling")
    ax[2].hist(alias_six_point_dist, bins=range(1, 8), density=True, width=0.9, align='mid', rwidth=0.9, edgecolor="black")
    ax[2].set_title("Alias Sampling")
    plt.tight_layout()
    plt.show()

    # Compute the KL divergence for the distributions generated by the three methods and the true distribution
    direct_p = [direct_six_point_dist.count(i)/n for i in range(1, 7)]
    rejection_p = [rejection_six_point_dist.count(i)/n for i in range(1, 7)]
    alias_p = [alias_six_point_dist.count(i)/n for i in range(1, 7)]

    print(f"KL-divergence between True and Direct: {kl_divergence(p, direct_p)}")
    print(f"KL-divergence between True and Rejection: {kl_divergence(p, rejection_p)}")
    print(f"KL-divergence between True and Alias: {kl_divergence(p, alias_p)}")


