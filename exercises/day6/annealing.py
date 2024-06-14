import numpy as np
import random
import os
from python_tsp.heuristics import solve_tsp_simulated_annealing
import matplotlib.pyplot as plt
import numpy.typing as npt
from typing import *

all_costs = []


class Annealing:
    def __init__(self, n_iterations: int, x0: int, n: int, A: npt.NDArray):
        self.n_iterations = n_iterations
        self.x0 = x0
        self.n = n
        self.A = A

    def h(self, x: Iterable):
        to_swap = np.random.choice(self.n, size=2, replace=False)
        x_cpy = np.copy(x)
        temp = x[to_swap[0]]
        x_cpy[to_swap[0]] = x_cpy[to_swap[1]]
        x_cpy[to_swap[1]] = temp
        return x_cpy

    def f(self, x: Iterable):
        global all_costs
        total_cost = 0
        for i in range(self.n - 1):
            total_cost += self.A[x[i], x[i + 1]]
        total_cost += self.A[x[-1], x[0]]
        all_costs.append(total_cost)
        return total_cost

    def metropolis(self):
        x = self.x0
        samples = [x]
        for i in range(self.n_iterations - 1):
            T = 1 / np.sqrt(1 + i)
            # T = -np.log(i+1)
            proposal = self.h(x)
            accept_p = min(1, np.exp(-(self.f(proposal) - self.f(x))) / T)

            U = np.random.rand()
            if U < accept_p:
                x = proposal

            samples.append(x)

        return samples


class Annealing_EC(Annealing):
    def __init__(self, stations: Iterable, n_iterations: int, x0: Iterable, T0: float):
        super().__init__(
            n_iterations=n_iterations, x0=x0, T0=T0, A=None, n=len(stations)
        )
        self.stations = stations

    def f(self, x: Iterable):
        total_cost = 0
        for i in range(self.n - 1):
            total_cost += np.linalg.norm(self.stations[x[i + 1]] - self.stations[x[i]])
        total_cost += np.linalg.norm(self.stations[x[0]] - self.stations[x[-1]])
        return total_cost


def get_circle_stations(n_stations: int, r: float):
    thetas = np.linspace(0, 2 * np.pi, n_stations)
    coords = [np.array([np.cos(t) * r, np.sin(t) * r]) for t in thetas]
    return coords


def get_simplex_stations(n_stations: int, max_c: float):
    stations = []
    while len(stations) < n_stations:
        station = np.random.choice(max_c, 2, replace=False)
        stations.append(station)
    return stations


def seedBasic(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    n_iterations = 10_000

    # r = 10
    # n_stations = 10
    # stations = get_circle_stations(n_stations, r)
    # x = [c[0] for c in stations]
    # y = [c[1] for c in stations]
    # plt.scatter(x, y)
    # plt.plot(x, y)
    # plt.show()
    # x0 = np.arange(n_stations)
    # np.random.shuffle(x0)
    # print(x0)
    # an1 = Annealing_EC(stations, n_iterations, x0, 100)
    # samples = an1.metropolis()
    # finale = samples[-1]
    # finale = np.append(finale, finale[0])
    # x = [stations[s][0] for s in finale]
    # y = [stations[s][1] for s in finale]
    # plt.plot(x, y, "-o")
    # plt.show()
    # print(samples[-5:])

    # max_c = 20
    # stations = get_simplex_stations(n_stations, max_c)
    # x0 = np.arange(n_stations)
    # np.random.shuffle(x0)
    # print(x0)
    # T = 50
    # an2 = Annealing_EC(stations, n_iterations, x0, T)
    # samples = an2.metropolis()
    # finale = samples[-1]
    # finale = np.append(finale, finale[0])
    # x = [stations[s][0] for s in finale]
    # y = [stations[s][1] for s in finale]
    # plt.plot(x, y, "-o")
    # plt.show()

    seedBasic(5)

    A = np.loadtxt("exercises/day6/cost.csv", delimiter=",")
    n = len(A)
    x0 = np.random.choice(n, replace=False, size=n)
    ann = Annealing(n_iterations, x0, n, A)
    samples = ann.metropolis()
    print(samples[np.argmin(all_costs)])
    # print(samples[-5:])

    # plt.matshow(cost_matrix)
    # plt.show()

    # plt.plot(all_costs)
    # plt.show()
    print(min(all_costs))

    permutation, distance = solve_tsp_simulated_annealing(A)
    print(permutation)
    print(distance)
