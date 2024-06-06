from typing import *
import matplotlib.pyplot as plt
import numpy as np

def chi_square_test(n_bins, random_numbers):
    # Chi-square test statistic for uniform distribution
    random_numbers = sorted(random_numbers)
    n = len(random_numbers)
    bin_size = n // n_bins
    bins = [random_numbers[i*bin_size:(i+1)*bin_size] for i in range(n_bins)]

    expected = n / n_bins
    chi_square = sum([(len(bin) - expected)**2 / expected for bin in bins])

    return chi_square



def kolmogorov_smirnov_test(samples: Iterable, plot: bool = False) -> float:
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
        plt.show()
    return max(np.abs(np.array(Fs) - xs))


def up_down_run_test(random_numbers):
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

    return Z
