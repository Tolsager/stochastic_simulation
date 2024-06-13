import numpy as np


class Annealing:
    def __init__(self, n_iterations: int, m: int, x0: int, n: int):
        self.n_iterations = n_iterations
        self.m = m
        self.x0 = x0
        self.n = n
        

    def h(self, x: int):
        to_swap = np.random.choice(np.arange(1, self.n), size=2, replace=False)
        temp = x[to_swap[0]]
        x[to_swap[0]] = x[to_swap[1]]
        x[to_swap[1]] = temp
        return x
    
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