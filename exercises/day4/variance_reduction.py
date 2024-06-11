import numpy as np
import numpy.typing as npt

from scipy import stats

def compute_ci(mean: float, var: float, n: int, confidence: float = 0.05) -> tuple[float, float]:
    lower_quantile = stats.norm.ppf(confidence/2)
    upper_quantile = stats.norm.ppf(1-confidence/2)
    return (mean + lower_quantile * np.sqrt(var/n), mean + upper_quantile * np.sqrt(var/n))


def crude_MC_estimator(n: int) -> tuple[npt.NDArray, float, tuple[float, float]]:
    """calculates crude mc estimate

    Args:
        n (int): number of samples

    Returns:
        tuple[npt.NDArray, float, tuple[float, float]]: samples, mean, confidence-itnerval
    """
    U = np.random.rand(n)
    X = np.exp(U)
    v = X.var()
    m = X.mean()

    ci = compute_ci(m, v, n)

    return X, m, ci


def antithetic_estimator(n: int) -> tuple[npt.NDArray, float, tuple[float, float]]:
    U = np.random.rand(n)
    X = np.exp(U)
    Y = (X + np.exp(1)/X)/2
    m = Y.mean()
    v = Y.var()
    ci = compute_ci(m, v, n)

    return Y, m, ci


def control_variable_estimator(n: int) -> tuple[npt.NDArray, float, tuple[float, float]]:
    U = np.random.rand(n)
    X = np.exp(U)
    cov = np.mean(U*X) - 0.5 * (np.exp(1) - 1)
    # cov = 0.14086
    Y_var = 1/12
    c = -cov / Y_var
    Y_mean = 1/2
    Z = X + c*(U - Y_mean)

    m = Z.mean()
    v = Z.var()
    ci = compute_ci(m, v, n)
    return Z, m, ci


def stratified_sampling_estimator(n: int, bins: int = 10) -> tuple[npt.NDArray, float, tuple[float, float]]:
    u = np.linspace(0, 1, bins+1)
    samples_per_bin = n//bins

    W_i = np.zeros(samples_per_bin)
    for i in range(1, len(u)):
        W_i += np.exp(u[i-1] + (np.random.rand(samples_per_bin)*(u[i]-u[i-1])+u[i-1]) / bins) / bins

    m = W_i.mean()
    v = W_i.var()

    ci = compute_ci(m, v, samples_per_bin)

    return W_i, m, ci

if __name__ == "__main__":
    # crude MC estimate
    print("Crude MC estimator")
    X, m, ci = crude_MC_estimator(100)
    print(f"Sample mean: {m:.3f}")
    print(f"CI: ({ci[0]:.3f}, {ci[1]:.3f})")
    print()

    # Antithetic variables
    print("Antithetic Variables")
    X, m, ci = antithetic_estimator(100)
    print(f"Sample mean: {m:.3f}")
    print(f"CI: ({ci[0]:.3f}, {ci[1]:.3f})")
    print()

    # Control variates
    print("Control Variates")
    X, m, ci = control_variable_estimator(100)
    print(f"Sample mean: {m:.3f}")
    print(f"CI: ({ci[0]:.3f}, {ci[1]:.3f})")
    print()

    # Stratified sampling
    print("Stratified sampling")
    X, m, ci = stratified_sampling_estimator(100)
    print(f"Sample mean: {m:.3f}")
    print(f"CI: ({ci[0]:.3f}, {ci[1]:.3f})")
    print()
