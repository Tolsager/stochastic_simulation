import numpy as np




def task1():
    X = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69])

    a, b = -5, 5
    n = 10
    r = 1000

    bootstrap_samples = np.random.choice(X, size=(r, n), replace=True)

    bootstrap_means = np.mean(bootstrap_samples, axis=1)

    mu = np.mean(bootstrap_means)

    return np.mean((a < bootstrap_means-mu) & (bootstrap_means-mu < b))


def task2():
    X = np.array([5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8])

    n = 15
    r = 1000

    bootstrap_samples = np.random.choice(X, size=(r, n), replace=True)

    bootstrap_variances = np.var(bootstrap_samples, axis=1)

    variance = np.var(bootstrap_variances)

    return variance


def bootstrap_estimation(data: np.ndarray, r: int):
    n = len(data)
    sample_mean = np.mean(data)
    sample_median = np.median(data)

    bootstrap_samples = np.random.choice(data, size=(r, n), replace=True)

    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    mean_variance = np.var(bootstrap_means)

    print("Variance of sample mean", mean_variance)

    bootstrap_medians = np.median(bootstrap_samples, axis=1)
    median_variance = np.var(bootstrap_medians)

    print("Variance of sample median", median_variance)

    print("Precision of the estimated mean", 1/mean_variance)
    print("Precision of the estimated median", 1/median_variance)



def _pareto(beta: float, k: float, n: int) -> np.ndarray:
    unif_rvs = np.random.random(n)
    return beta / (unif_rvs ** (1 / k))

if __name__ == '__main__':
    # print(task1())

    # print(task2())

    data = _pareto(beta=1, k=1.05, n=200)

    bootstrap_estimation(data, 100)
