from typing import *
import scipy
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import chi2, norm


def chi_square_test(random_numbers: Iterable, n_bins: int) -> tuple[float, float]:
    # Chi-square test statistic for uniform distribution
    random_numbers = sorted(random_numbers)
    n = len(random_numbers)
    k = 0
    bins = [[] for _ in range(n_bins)]
    for j, i in enumerate(np.linspace(0.1, 1, n_bins)):
        while k < n and random_numbers[k] < i:
            bins[j].append(random_numbers[k])
            k += 1

    expected = n / n_bins
    test_statistic = sum([((len(bin) - expected) ** 2) / expected for bin in bins])

    p_val = 1 - chi2.cdf(test_statistic, n_bins - 1)

    return test_statistic, p_val


def kolmogorov_smirnov_test(samples: Iterable, plot: bool = False) -> tuple[float, float]:
    sorted_samples = sorted(samples)
    n = len(sorted_samples)
    xs = np.linspace(0, 1, 1000)
    i = 0
    Fs = []
    for x in xs:
        while (i < n) and (sorted_samples[i] <= x):
            i += 1
        Fs.append(i / n)

    if plot:
        plt.plot(xs, xs)
        plt.plot(xs, Fs)
        plt.tight_layout()
        plt.show()

    T = max(np.abs(np.array(Fs) - xs))
    p = scipy.special.kolmogorov(T * np.sqrt(n))
    return T, p


def above_below_test(samples: Iterable) -> tuple[float, float]:
    median = np.median(samples)
    above = samples > median
    n = len(samples)
    n1 = above.sum().astype(np.int64)
    below = samples < median
    n2 = below.sum().astype(np.int64)
    Ra = above[0]
    for i in range(1, n):
        if above[i] == 1 and above[i - 1] == 0:
            Ra += 1

    Rb = below[0]
    for i in range(1, n):
        if below[i] == 1 and below[i - 1] == 0:
            Rb += 1

    T = Ra + Rb
    mean = 2 * (n1 * n2) / (n1 + n2) + 1
    var = 2 * (n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    if T < mean:
        p_val = norm.cdf(T, mean, np.sqrt(var)) * 2
    else:
        p_val = (1 - norm.cdf(T, mean, np.sqrt(var))) * 2

    return T, p_val


def knuth_up_down_run_test(random_numbers: Iterable) -> tuple[float, float]:
    R = np.array([0, 0, 0, 0, 0, 0])
    run_length = 1
    n = len(random_numbers)
    for i in range(1, n):
        if random_numbers[i] > random_numbers[i - 1]:
            run_length += 1
        else:
            R[min(run_length, 6) - 1] += 1
            run_length = 1

    B = np.array([1 / 6, 5 / 24, 11 / 120, 19 / 720, 29 / 5040, 1 / 840])

    A = np.array([[4529.4, 9044.9, 13568, 18091, 22615, 27892],
                  [9044.9, 18097, 27139, 36187, 45234, 55789],
                  [13568, 27139, 40721, 54281, 67852, 83685],
                  [18091, 36187, 54281, 72414, 90470, 111580],
                  [22615, 45234, 67852, 90470, 113262, 139476],
                  [27892, 55789, 83685, 111580, 139476, 172860]])

    Z = 1 / (n - 6) * (R - n * B) @ A @ (R - n * B)

    p_val = 1 - chi2.cdf(Z, 6)

    return Z, p_val


def up_down_run_test(random_numbers: Iterable) -> tuple[float, float]:
    runs = []
    up_length = 1
    down_length = 1
    n = len(random_numbers)
    i = 1
    down = False
    up = False
    for i in range(1, n):
        if random_numbers[i] > random_numbers[i - 1]:  # Going up
            if down:
                runs.append(down_length)
                down_length = 1
                down = False
            up = True
            up_length += 1
        else:  # Going down
            if up:
                runs.append(up_length)
                up_length = 1
                up = False
            down = True
            down_length += 1
        if i == n - 1:
            if up:
                runs.append(up_length)
            if down:
                runs.append(down_length)

    num_runs = len(runs)
    test_stat = (num_runs - (2 * n - 1) / 3) / np.sqrt((16 * n - 29) / 90)

    if test_stat < 0:
        p_val = norm.cdf(test_stat) * 2
    else:
        p_val = (1 - norm.cdf(test_stat)) * 2

    return test_stat, p_val


def correlation_test(random_numbers: Iterable, h: int) -> tuple[float, float]:
    n = len(random_numbers)

    c = sum([random_numbers[i] * random_numbers[i + h] for i in range(n - h)]) / (n - h)

    if c < 0.25:
        p_val = 2 * norm.cdf(c, 0.25, np.sqrt(7 / (144 * n)))
    else:
        p_val = 2 * (1 - norm.cdf(c, 0.25, np.sqrt(7 / (144 * n))))

    return c, p_val
