from scipy import stats
import numpy as np


def sample_prior(n: int):
    cov = np.array([[1, 1 / 2], [1 / 2, 1]])
    samples = stats.multivariate_normal.rvs([0, 0], cov, n)
    return np.exp(samples)

def sample_likelihood(n: int, m: float, v: float):
    return stats.norm.rvs(m, np.sqrt(v), n)



if __name__ == "__main__":
    ## a)
    prior = sample_prior(1)
    m = prior[0]
    v = prior[1]
    print(np.array2string(prior, precision=3))

    ## b)
    likelihood = sample_likelihood(10, m, v)
    print(np.array2string(likelihood, precision=3))
