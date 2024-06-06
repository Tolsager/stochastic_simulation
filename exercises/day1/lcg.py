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
    plt.hist(random_numbers, bins=10, edgecolor='black', width=0.09)
    plt.show()

    # Scatter plot
    plt.scatter(random_numbers[:100], random_numbers[1:101])
    plt.show()

    # Chi-square test
    chi2_test_stat, p = chi_square_test(random_numbers, 10)
    print("Chi-square")
    print("T: ", chi2_test_stat)
    print("p: ", p)

    # Kolmogorov-Smirnov test
    ks_test_stat, p = kolmogorov_smirnov_test(random_numbers, plot=True)
    print("Kolmogorov Smirnov")
    print("T: ", ks_test_stat)
    print("p: ", p)

    # Run tests
    # Run test 1 - Above below test
    T_ab, p_ab = above_below_test(random_numbers)
    print("Above below")
    print("Test statistic: ", T_ab)
    print("P value: ", p_ab)

    # Run test 2
    T, p = knuth_up_down_run_test(random_numbers)
    print("Knuth up-down")
    print("Test statistic: ", T)
    print("P value: ", p)

    # Run test 3
    T, p = up_down_run_test(random_numbers)
    print("Up down test")
    print("Test statistic: ", T)
    print("P value: ", p)

    # Correlation test
    T, p = correlation_test(random_numbers, h=5)
    print("Correlation test")
    print("Test statistic: ", T)
    print("P value: ", p)

