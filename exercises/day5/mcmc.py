import numpy as np
import math
from scipy import stats

import matplotlib.pyplot as plt

class MetropolisHastings:
    def __init__(self, n_iterations: int, m: int, x0: int, A: int):
        self.n_iterations = n_iterations
        self.m = m
        self.x0 = x0
        self.A = A
        

    def h(self, x: int):
        # If we want to sample a binomial value it needs to be symmetric around an integer value.
        # n = self.m
        # p = 0.5
        # delta = stats.binom.rvs(n, p)
        # delta = delta - x
        # if delta >= 0:
        #     delta += 1
        # return delta
        return np.random.choice(11)
    
    def g(self, x: float):
        return pow(self.A, x) / math.factorial(x)
     
    def metropolis(self):
        x = self.x0
        samples = [x]
        for i in range(self.n_iterations-1):
            delta = self.h(x)
            proposal = delta

            U = np.random.rand()

            accept_p = np.minimum(1, self.g(proposal) / self.g(x))
            
            if U < accept_p:
                x = proposal
            
            samples.append(x)
        return samples
                
class MetropolisHastingsTwoQueues:
    def __init__(self, n_iterations: int, m: int, x0: int, y0: int, A1: int, A2: int):
        self.n_iterations = n_iterations
        self.m = m
        self.x0 = x0
        self.y0 = y0
        self.A1 = A1
        self.A2 = A2

    def h(self):
        x = np.random.choice(11)
        y = np.random.choice(11-x)
        return x, y
    
    def g(self, x: float, y: float):
        return pow(self.A1, x) / math.factorial(x) * pow(self.A2, y) / math.factorial(y)
    
    def metropolis(self):
        x, y = self.x0, self.y0
        samples = [(x,y)]
        for _ in range(self.n_iterations-1):
            delta_x, delta_y = self.h()   
            proposal_x, proposal_y = delta_x, delta_y

            U = np.random.rand()

            accept_p = np.minimum(1, self.g(proposal_x, proposal_y) / self.g(x, y))

            if U < accept_p:
                x, y = proposal_x, proposal_y

            samples.append((x, y))
        
        return samples
    
    def coordinate_wise_metropolis(self):
        x, y = self.x0, self.y0
        samples = [(x, y)]
        for _ in range(self.n_iterations-1):
            # First sample a proposal for x which satisfies x + y <= m
            proposal_x = np.random.choice(11-y)

            # Then reject or accept the proposal of x
            U = np.random.rand()

            accept_p = np.minimum(1, self.g(proposal_x, y) / self.g(x, y))

            if U < accept_p:
                x = proposal_x
            
            # Sample a proposal for y which satisfies x + y <= m
            proposal_y = np.random.choice(11-x)

            # Then reject or accept the proposal of y
            U = np.random.rand()

            accept_p = np.minimum(1, self.g(x, proposal_y) / self.g(x, y))

            if U < accept_p:
                y = proposal_y

            samples.append((x, y))
        
        return samples
    
    def gibbs_sampler(self):
        x, y = self.x0, self.y0
        samples = [(x,y)]

        for _ in range(self.n_iterations-1):
            ps = [pow(self.A1, i) / math.factorial(i) / np.sum([pow(self.A1, j) / math.factorial(j) for j in range(self.m+1-y)]) for i in range(self.m+1-y)]

            x = np.random.choice(self.m+1-y, p=ps)

            ps = [pow(self.A2, i) / math.factorial(i) / np.sum([pow(self.A2, j) / math.factorial(j) for j in range(self.m+1-x)]) for i in range(self.m+1-x)]

            y = np.random.choice(self.m+1-x, p=ps)

            samples.append((x, y))
        
        return samples


def task1():
    A = 8
    n_experiments = 100
    p_vals = []
    n = 10_000
    m = 10
    x0 = 5 
    warm_up_fraction = 0.2
    for _ in range(n_experiments):


        mc_hastings = MetropolisHastings(n, m, x0, A)

        samples = mc_hastings.metropolis()
        # plt.hist(samples, bins=range(12))

        samples_post_warmup = samples[int(warm_up_fraction*n):]

        unique, observed_freq = np.unique(samples_post_warmup, return_counts=True)

        observed = np.zeros(m+1)
        for i, uniq in enumerate(unique):
            observed[uniq] = observed_freq[i]

        c = 1 / np.sum([pow(A, j)/math.factorial(j) for j in range(m+1)])

        expected = [(n-n*warm_up_fraction)*(pow(A, i)/math.factorial(i))*c for i in range(m+1)]
        
        p_val = stats.chisquare(observed, expected).pvalue
        p_vals.append(p_val)
    plt.hist(p_vals)
    plt.xlabel("p value")
    plt.ylabel("count")
    plt.show()

def task2(subtask="a"):
    n = 10_000
    m = 10
    A1 = 4
    A2 = 4
    x0 = 0
    y0 = 0
    n = 10000
    warm_up_fraction = 0.2
    n_experiments = 100

    ## a)
    if subtask == "a":
        p_values = []
        for _ in range(n_experiments):
            mc_hastings_two_queues = MetropolisHastingsTwoQueues(n, m, x0, y0, A1, A2)

            samples = np.array(mc_hastings_two_queues.metropolis())

            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            # plt.hist2d(samples[:,0], samples[:,1], bins=[range(12), range(12)])
            # plt.colorbar()
            # plt.show()

            c = 1 / np.sum([pow(A1, i)/math.factorial(i) * pow(A2, j)/math.factorial(j) for i in range(m+1) for j in range(m+1) if i+j <= m])

            expected = {}
            for i in range(m+1):
                for j in range(m+1):
                    if i+j <= m:
                        expected[(i, j)] = (n-n*warm_up_fraction)*(pow(A1, i)/math.factorial(i) * pow(A2, j)/math.factorial(j))*c

            samples_post_warmup = samples[int(warm_up_fraction*n):]

            observed = {(i, j): 0 for i in range(m+1) for j in range(m+1) if i+j <= m}

            for sample in samples_post_warmup:
                i, j = sample
                observed[(i, j)] += 1

            observed_freqs = np.zeros(66)
            expected_freqs = np.zeros(66)
            for i, k in enumerate(expected.keys()):
                observed_freqs[i] = observed[k]
                expected_freqs[i] = expected[k]

            # plt.plot(expected_freqs, label='Expected')
            # plt.plot(observed_freqs, label='Observed')
            # plt.legend()
            # plt.show()
            
            p_val = stats.chisquare(observed_freqs, expected_freqs, ddof=0).pvalue
            p_values.append(p_val)
        plt.hist(p_values)
        plt.xlabel("p value")
        plt.ylabel("count")
        plt.show()
    

    elif subtask == "b":
        p_values = []
        for _ in range(n_experiments):
            mc_hastings_two_queues = MetropolisHastingsTwoQueues(n, m, x0, y0, A1, A2)
            samples = np.array(mc_hastings_two_queues.coordinate_wise_metropolis())

            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            # plt.hist2d(samples[:,0], samples[:,1], bins=(range(12), range(12)))
            # plt.colorbar()
            # plt.show()

            samples_post_warmup = samples[int(warm_up_fraction*n):]

            observed = {(i, j): 0 for i in range(m+1) for j in range(m+1) if i+j <= m}

            for sample in samples_post_warmup:
                i, j = sample
                observed[(i, j)] += 1

            c = 1 / np.sum([pow(A1, i)/math.factorial(i) * pow(A2, j)/math.factorial(j) for i in range(m+1) for j in range(m+1) if i+j <= m])
            expected = {}
            for i in range(m+1):
                for j in range(m+1):
                    if i+j <= m:
                        expected[(i, j)] = (n-n*warm_up_fraction)*(pow(A1, i)/math.factorial(i) * pow(A2, j)/math.factorial(j))*c

            observed_freqs = np.zeros(66)
            expected_freqs = np.zeros(66)
            for i, k in enumerate(expected.keys()):
                observed_freqs[i] = observed[k]
                expected_freqs[i] = expected[k]

            # plt.plot(expected_freqs, label='Expected')
            # plt.plot(observed_freqs, label='Observed')
            # plt.legend()
            # plt.show()
            
            p_val = stats.chisquare(observed_freqs, expected_freqs, ddof=0).pvalue
            p_values.append(p_val)

        plt.hist(p_values)
        plt.xlabel("p value")
        plt.ylabel("count")
        plt.show()

    elif subtask == "c":
        p_values = []
        for _ in range(n_experiments):
            mc_hastings_two_queues = MetropolisHastingsTwoQueues(n, m, x0, y0, A1, A2)
            samples = np.array(mc_hastings_two_queues.gibbs_sampler())

            # fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            # plt.hist2d(samples[:,0], samples[:,1], bins=(range(12), range(12)))
            # plt.colorbar()
            # plt.show()

            samples_post_warmup = samples[int(warm_up_fraction*n):]

            observed = {(i, j): 0 for i in range(m+1) for j in range(m+1) if i+j <= m}
            c = 1 / np.sum([pow(A1, i)/math.factorial(i) * pow(A2, j)/math.factorial(j) for i in range(m+1) for j in range(m+1) if i+j <= m])
            expected = {}
            for i in range(m+1):
                for j in range(m+1):
                    if i+j <= m:
                        expected[(i, j)] = (n-n*warm_up_fraction)*(pow(A1, i)/math.factorial(i) * pow(A2, j)/math.factorial(j))*c

            observed_freqs = np.zeros(66)
            expected_freqs = np.zeros(66)
            for i, k in enumerate(expected.keys()):
                observed_freqs[i] = observed[k]
                expected_freqs[i] = expected[k]
                
            for sample in samples_post_warmup:
                i, j = sample
                observed[(i, j)] += 1

            observed_freqs = np.zeros(66)
            expected_freqs = np.zeros(66)
            for i, k in enumerate(expected.keys()):
                observed_freqs[i] = observed[k]
                expected_freqs[i] = expected[k]

            # plt.plot(expected_freqs, label='Expected')
            # plt.plot(observed_freqs, label='Observed')
            # plt.legend()
            # plt.show()
            
            p_val = stats.chisquare(observed_freqs, expected_freqs, ddof=0).pvalue
            p_values.append(p_val)
        plt.hist(p_values)
        plt.xlabel("p value")
        plt.ylabel("count")
        plt.show()



    
if __name__ == '__main__':
    ## Task 1
    # task1()

    ## Task 2
    task2("c")

    ## Task 3
    
