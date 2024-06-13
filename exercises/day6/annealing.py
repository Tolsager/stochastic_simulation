import numpy as np
import numpy.typing as npt
from typing import *


class Annealing:
    def __init__(self, n_iterations: int, x0: int, n: int, T0: float, A: npt.NDArray):
        self.n_iterations = n_iterations
        self.x0 = x0
        self.n = n
        self.T0 = T0
        self.A = A

    def h(self, x: Iterable):
        to_swap = np.random.choice(np.arange(1, self.n), size=2, replace=False)
        temp = x[to_swap[0]]
        x[to_swap[0]] = x[to_swap[1]]
        x[to_swap[1]] = temp
        return x
    
    def f(self, x: Iterable):
        total_cost = 0
        for i in range(self.n):
            total_cost += self.A[x[i], x[i+1]]
        return total_cost
    
    def metropolis(self):
        x = self.x0
        samples = [x]
        T = self.T0
        for i in range(self.n_iterations-1):
            proposal = self.h(x)
            accept_p = min(1, np.exp(-(self.f(proposal) - self.f(x)))/T)

            U = np.random.rand()            
            if U < accept_p:
                x = proposal
            
            samples.append(x)
        
            T = 1 / np.sqrt(2 + i)
        return samples

if __name__ == "__main__":

