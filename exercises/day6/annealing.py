import numpy as np
import matplotlib.pyplot as plt
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
        to_swap = np.random.choice(self.n, size=2, replace=False)
        x_cpy = np.copy(x)
        temp = x[to_swap[0]]
        x_cpy[to_swap[0]] = x_cpy[to_swap[1]]
        x_cpy[to_swap[1]] = temp
        return x_cpy
    
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
    
class Annealing_EC(Annealing):
    def __init__(self, stations: Iterable, n_iterations: int, x0: Iterable, T0: float):
        super().__init__(n_iterations=n_iterations, x0=x0, T0=T0, A=None, n=len(stations))
        self.stations = stations
    
    def f(self, x: Iterable):
        total_cost = 0
        for i in range(self.n-1):
            total_cost += np.linalg.norm(stations[x[i+1]] - stations[x[i]])
        total_cost += np.linalg.norm(stations[x[0]] - stations[x[-1]])
        return total_cost



    


def get_circle_stations(n_stations: int, r: float):
    thetas = np.linspace(0, 2*np.pi, n_stations)
    coords = [np.array([np.cos(t)*r, np.sin(t)*r]) for t in thetas]
    return coords
    

if __name__ == "__main__":
    stations = get_circle_stations(20, 20)
    # x = [c[0] for c in stations]
    # y = [c[1] for c in stations]
    # plt.scatter(x, y)
    # plt.plot(x, y)
    # plt.show()
    x0 = np.arange(20)
    np.random.shuffle(x0)
    print(x0)
    n_iterations = 10_000
    an1 = Annealing_EC(stations, n_iterations, x0, 5)
    samples = an1.metropolis()
    finale = samples[-1]
    x = [stations[s][0] for s in finale]
    y = [stations[s][1] for s in finale]
    plt.plot(x, y, "-o")
    plt.show()
    print(samples[-5:])



