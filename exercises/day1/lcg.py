import random
import matplotlib.pyplot as plt

from scipy.stats import chi2


def chi_square_test(n_bins, random_numbers):
    random_numbers = sorted(random_numbers)
    n = len(random_numbers)
    bin_size = n // n_bins
    bins = []
    for i in range(n_bins):
        bins.append(random_numbers[i * bin_size:(i + 1) * bin_size])

    expected = n / n_bins
    chi_square = 0
    for i in range(n_bins):
        chi_square += (len(bins[i]) - expected)**2 / expected

    return chi_square



def kolmogorov_smirnov_test():
    pass


def lcg(seed, multiplier, shift, modulus, size):
    random_numbers = []
    for i in range(size):
        seed = (multiplier * seed + shift) % modulus
        random_numbers.append(seed / modulus)

    return random_numbers


if __name__ == '__main__':
    seed = 1
    multiplier = 1664525
    shift = 1013904223
    modulus = 2**32
    size = 10000
    random_numbers = lcg(seed, multiplier, shift, modulus, size)
    print(random_numbers)

    plt.hist(random_numbers, bins=10, edgecolor='black', width=0.09)
    plt.show()

    test_stat = chi_square_test(10, random_numbers)