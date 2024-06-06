from typing import *
import scipy
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
        plt.show()
    
    T =  max(np.abs(np.array(Fs) - xs))
    p = scipy.special.kolmogorov(T*np.sqrt(n))
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
        if above[i] == 1 and above[i-1] == 0:
            Ra += 1
    
    Rb = 1 - above[0]
    for i in range(1, n):
        if above[i] == 0 and above[i] == 1:
            Rb += 1

    T = Ra + Rb
    mean = 2 * (n1*n2) / (n1 + n2) + 1
    var = 2 * (n1*n2*(2*n1*n2-n1-n2)) / ((n1+n2)**2 * (n1+n2-1))
    if T < mean:
        p_val = scipy.stats.norm.cdf(T, mean, np.sqrt(var)) * 2
    else:
        p_val = (1 - scipy.stats.norm.cdf(T, mean, np.sqsrt(var))) * 2

    return T, p_val
