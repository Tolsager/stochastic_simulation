import random
import matplotlib.pyplot as plt
from stat_tests import chi_square_test, kolmogorov_smirnov_test, above_below_test, knuth_up_down_run_test, up_down_run_test, correlation_test

from scipy.stats import chi2


def lcg(seed, multiplier, shift, modulus, size):
    random_numbers = [(seed := (multiplier * seed + shift) % modulus) / modulus for _ in range(size)]
    # random_numbers = []
    # for i in range(size):
    #     seed = (multiplier * seed + shift) % modulus
    #     random_numbers.append(seed / modulus)

    return random_numbers


if __name__ == '__main__':
    seed = 1
    multiplier = 1664525
    shift = 1013904223
    modulus = 2**32
    size = 10000
    random_numbers = lcg(seed, multiplier, shift, modulus, size)

    # Histogram
    # plt.hist(random_numbers, bins=10, edgecolor='black', width=0.09)
    # plt.show()

    # Scatter plot

    # Chi-square test
    chi2_test_stat = chi_square_test(random_numbers, 10)

    # Kolmogorov-Smirnov test
    ks_test_stat = kolmogorov_smirnov_test(random_numbers, plot=True)

    # Run tests
    # Run test 1
    print(above_below_test(random_numbers))
    # Run test 2
    print(knuth_up_down_run_test(random_numbers))
    # Run test 3
    print(up_down_run_test(random_numbers))

    # Correlation test
    print(correlation_test(random_numbers, h=1))