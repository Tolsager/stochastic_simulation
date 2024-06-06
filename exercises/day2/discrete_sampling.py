import matplotlib.pyplot as plt

from scipy.stats import geom

import numpy as np

if __name__ == '__main__':
    p = 0.3
    n = 10000

    rvs = np.random.random(n)

    # Transform the random variables to geometric
    rvs = np.floor(np.log(rvs)/np.log(1-p)) + 1


    plt.hist(rvs, bins=range(1, int(max(rvs))+1), density=True)
    plt.show()