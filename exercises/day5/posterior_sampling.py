from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from typing import *
import os
import random
import numpy as np


def seedBasic(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def sample_prior(n: int):
    cov = np.array([[1, 1 / 2], [1 / 2, 1]])
    samples = stats.multivariate_normal.rvs([0, 0], cov, n)
    return np.exp(samples)


def sample_likelihood(n: int, m: float, v: float):
    return stats.norm.rvs(m, np.sqrt(v), n)


class MetropolisHastingsPosterior:
    def __init__(
        self,
        x0: Iterable,
        n_iterations: int,
        v1: float,
        v2: float,
        sample_var: float,
        sample_mean: float,
    ):
        self.x0 = x0
        self.n_iterations = n_iterations
        self.v1 = v1
        self.v2 = v2
        self.sample_mean = sample_mean
        self.sample_var = sample_var

    def h(self, x: Iterable):
        theta = np.exp(stats.norm.rvs(np.log(x[0]), self.v1))
        psi = np.exp(stats.norm.rvs(np.log(x[1]), self.v2))
        return theta, psi

    def g(self, x: Iterable):
        loglik = -5 * np.log(2 * np.pi * x[1]) - 1 / (2 * x[1]) * (
            9 * self.sample_var + 10 * (self.sample_mean - x[0]) ** 2
        )
        logprior = -np.log(2 * np.pi * x[0] * x[1] * np.sqrt(3 / 4)) - 2 / 3 * (
            np.log(x[0]) ** 2 - np.log(x[0]) * np.log(x[1]) + np.log(x[1]) ** 2
        )
        return loglik + logprior

    def metropolis(self):
        x = self.x0
        samples = [x]
        for i in range(self.n_iterations - 1):
            proposal = self.h(x)

            prop1 = list(x)
            prop1[0] = proposal[0]

            U = np.random.rand()
            accept_p = np.minimum(1, np.exp(self.g(prop1)) / np.exp(self.g(x)))
            # p_prop = np.exp(self.g(proposal))
            # p_cur = np.exp(self.g(x))
            # accept_p = np.minimum(1, p_prop / p_cur)

            # if U < accept_p:
            #     x = proposal
            if U < accept_p:
                x = prop1
            
            prop2 = list(x)
            prop2[1] = proposal[1]
            accept_p = np.minimum(1, np.exp(self.g(prop2)) / np.exp(self.g(x)))
            if U < accept_p:
                x = prop2

            samples.append(x)
        return samples

def plot_mcmc_samples(true_mean: float, true_var: float, sample_mean: float, sample_var: float, samples: Iterable, ax, n: int):
    x = [x[0] for x in samples]
    y = [x[1] for x in samples]
    ax.scatter(x, y, label="Metropolis-Hasting samples", s=0.5, c="gray", alpha=0.7)
    ax.scatter(true_mean, true_var, label=r"$(\Theta, \Psi)$", c="r", marker="x", s=100)
    ax.scatter(sample_mean, sample_var, label=r"$(\bar{x},s^2)$", c="darkorchid", marker="+", s=100)
    ax.legend()
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\psi$")
    ax.set_title(f"$n={n}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

if __name__ == "__main__":
    seedBasic()
    ## a)
    prior = sample_prior(1)
    m = prior[0]
    v = prior[1]
    print(np.array2string(prior, precision=3))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    n_iterations = 10_000
    warmup_frac = 0.8
    for i, n in enumerate([10, 100, 1000]):
        ## b)
        # n_samples = 100
        likelihood = sample_likelihood(n, m, v)
        # print(np.array2string(likelihood, precision=3))

        ## d)
        sample_mean = likelihood.mean()
        sample_var = likelihood.var()
        print(sample_mean, sample_var)
        x0 = (sample_mean, sample_var)
        v1 = 1 / 4
        v2 = 1 / 4

        MCMC = MetropolisHastingsPosterior(
            x0=x0,
            sample_mean=sample_mean,
            sample_var=sample_var,
            n_iterations=n_iterations,
            v1=v1,
            v2=v2,
        )
        samples = MCMC.metropolis()
        # print(samples[-5:])
        warm_samples = samples[int(n_iterations*warmup_frac):]
        plot_mcmc_samples(m, v, sample_mean, sample_var, warm_samples, axes[i], n)
    plt.tight_layout()
    plt.savefig("post_mcmc.pdf")
    plt.show()
