import random
import matplotlib.pyplot as plt
from stat_tests import chi_square_test, kolmogorov_smirnov_test, knuth_up_down_run_test, up_down_run_test, correlation_test, above_below_test

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

    ### Uniformity tests ###
    # plt.hist(random_numbers, bins=10, edgecolor='black', width=0.09)
    # plt.show()
    # test_stat = chi_square_test(10, random_numbers)
    # print(test_stat)
    # ks_test_stat = kolmogorov_smirnov_test(random_numbers, plot=True)

    ### Correlation tests ###
    # plt.scatter(random_numbers[:100], random_numbers[1:101])
    # plt.show()

    # # Degrees of freedom = n_bins - 1
    # dof = 10 - 1
    # # Significance level
    # alpha = 0.05
    # critical_value = chi2.ppf(1 - alpha, dof)

    # print(critical_value > test_stat)

    # above belov test
    T_ab, p_ab = above_below_test(random_numbers)
    print("Test statistic: ", T_ab)
    print("P value: ", p_ab)

    # print(critical_value > test_stat)

    print(up_down_run_test(random_numbers))
    print(knuth_up_down_run_test(random_numbers))
    print(correlation_test(random_numbers, 5))
