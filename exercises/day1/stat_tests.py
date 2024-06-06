def chi_square_test(n_bins, random_numbers):
    # Chi-square test statistic for uniform distribution
    random_numbers = sorted(random_numbers)
    n = len(random_numbers)
    bin_size = n // n_bins
    bins = [random_numbers[i*bin_size:(i+1)*bin_size] for i in range(n_bins)]

    expected = n / n_bins
    chi_square = sum([(len(bin) - expected)**2 / expected for bin in bins])

    return chi_square



def kolmogorov_smirnov_test():
    pass