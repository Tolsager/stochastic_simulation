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
