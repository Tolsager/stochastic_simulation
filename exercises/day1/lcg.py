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


def main(random_nums, correlation_order=1):
    # Histogram
    plt.hist(random_nums, bins=10, edgecolor='black', width=0.09)
    plt.show()

    # Scatter plot
    plt.scatter(random_nums[:100], random_nums[1:101])
    plt.show()

    # Chi-square test
    chi2_test_stat, p = chi_square_test(random_nums, 10)
    print("Chi-square")
    print("T: ", chi2_test_stat)
    print("p: ", p)

    # Kolmogorov-Smirnov test
    ks_test_stat, p = kolmogorov_smirnov_test(random_nums, plot=True)
    print("Kolmogorov Smirnov")
    print("T: ", ks_test_stat)
    print("p: ", p)

    # Run tests
    # Run test 1 - Above below test
    T_ab, p_ab = above_below_test(random_nums)
    print("Above below")
    print("Test statistic: ", T_ab)
    print("P value: ", p_ab)

    # Run test 2
    T, p = knuth_up_down_run_test(random_nums)
    print("Knuth up-down")
    print("Test statistic: ", T)
    print("P value: ", p)

    # Run test 3
    T, p = up_down_run_test(random_nums)
    print("Up down test")
    print("Test statistic: ", T)
    print("P value: ", p)

    # Correlation test
    T, p = correlation_test(random_nums, h=correlation_order)
    print(f"Correlation test: h={correlation_order}")
    print("Test statistic: ", T)
    print("P value: ", p)


if __name__ == '__main__':
    size = 10000

    params = [(39, 2, 1, 5**21), (39, 1664525, 1013904223, 2**32+1), (39, 5, 1, 16)] # Mediocre, Good, Terrible
    for param_set in params:
        seed, multiplier, shift, modulus = param_set
        print(f"a={multiplier}, c={shift}, m={modulus}, seed={seed}")
        random_numbers = lcg(seed, multiplier, shift, modulus, size)

        main(random_numbers)

    python_random_numbers = [random.random() for _ in range(size)]
    print("Python random")
    main(python_random_numbers)
