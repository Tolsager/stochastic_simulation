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