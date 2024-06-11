import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from typing import Callable

from scipy.stats import norm
import scipy.stats as stats

class Server:
    def __init__(self, mean_service_time):
        self.occupied = False
        self.mean_service_time = mean_service_time

    def start_service(self, time):
        self.occupied = True
        service_time = -np.log(np.random.random()) * self.mean_service_time
        departure_time = time + service_time
        return service_time, departure_time

    def end_service(self):
        self.occupied = False

    def __bool__(self):
        return self.occupied

class ServerConstant(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def start_service(self, time):
        self.occupied = True
        service_time = self.mean_service_time
        departure_time = time + service_time
        return service_time, departure_time


class ServerPareto(Server):
    def __init__(self, k, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.beta = beta

    def start_service(self, time):
        self.occupied = True
        service_time = self.beta * (np.random.random() ** (-1 / self.k) - 1)
        departure_time = time + service_time
        return service_time, departure_time
    

class ServerHalfNormal(Server):
    def __init__(self, mu, sigma, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu
        self.sigma = sigma

    def start_service(self, time):
        self.occupied = True
        service_time = abs(np.random.normal(self.mu, self.sigma))
        departure_time = time + service_time
        return service_time, departure_time
        

def get_arrivals(n: int, mean_interarrival_time):
    time: float = 0.0
    return deque([(time := time - np.log(np.random.random()) * mean_interarrival_time) for _ in range(n)])

def get_arrivals_Erlang(n: int, k: int):
    time: float = 0.0
    return deque([(time := time - (1/k) * np.sum([np.log(np.random.random()) for i in range(k)])) for _ in range(n)])

def get_arrivals_hyperexp(n: int, ps: tuple[float, float], lambdas: tuple[float, float]):
    time: float = 0.0
    return deque([(time := time - np.log(np.random.random()) * 1/lambdas[0] if np.random.random() < ps[0] else time - np.log(np.random.random()) * 1/lambdas[1]) for _ in range(n)])
            

def simulate_blocking_system(n_servers: int, n_customers: int, server_class: Server, arrival_function: callable, server_args: tuple, arrival_args: tuple):
    t = 0
    n_served = 0
    n_blocked = 0
    n_in_system = []
    n_arrivals = 0
    times = []
    servers = [server_class(*server_args) for _ in range(n_servers)]

    arrivals = arrival_function(*arrival_args)
    departures = []
    service_times = []
    
    while n_arrivals < n_customers:
            T_A = arrivals[0]
            T_D = np.inf if len(departures) == 0 else departures[0][0]

            if T_A < T_D:
                n_arrivals += 1
                t = arrivals.popleft()
                if all(servers):
                    n_blocked += 1
                else:
                    for i, server in enumerate(servers):
                        if not server:
                            service_time, departure_time = server.start_service(t)
                            departures.append((departure_time, i))
                            service_times.append(service_time)
                            break
                    departures = sorted(departures, key=lambda x: x[0])
            else:
                T_D, server_idx = departures.pop(0)
                t = T_D
                servers[server_idx].end_service()

                n_served += 1
            
            times.append(t)
            n_in_system.append(sum([s.occupied for s in servers]))
    

    return n_served, n_blocked, service_times, (times, n_in_system)


if __name__ == '__main__':
    # Blocking system with 10 servers and no waiting room
    # Mean service time is 8 time units
    # Mean time between arrivals is 1 time unit
    # 10000 Customers
    n_servers = 10
    mean_service_time = 8
    mean_interarrival_time = 1
    n_customers = 10_000
    
    total_blocked = []
    blocked_probabilities = []
    service_time_averages = []

    # Initialize the simulation
    n_simulations = 10
    for j in range(n_simulations):
        # Servers
        # Server: args - (mean_service_time,)
        # ServerConstant: args - (mean_service_time,)
        # ServerPareto: args - (k, beta, mean_service_time)
        # ServerHalfNormal: args - (mu, sigma, mean_service_time)
        served, blocked, service_times, utilization_stats = simulate_blocking_system(n_servers=n_servers, 
                                                                      n_customers=n_customers, 
                                                                      server_class = Server, 
                                                                      arrival_function=get_arrivals, 
                                                                      server_args=(mean_service_time,), 
                                                                      arrival_args=(n_customers, mean_interarrival_time)
                                                                      )
        
        service_time_averages.append(np.mean(service_times))
        blocked_probabilities.append(blocked/n_customers)
        
        
        total_blocked.append(blocked)


        # arrivals = get_arrivals_hyperexp(10_000, (0.8, 0.2), (0.8333, 5))
        # arrivals = get_arrivals(n_customers, mean_interarrival_time)
    
    e_x = np.mean(blocked_probabilities)
    e_y = mean_service_time

    var_x = np.var(blocked_probabilities)
    var_y = np.var(service_time_averages)


    cov = np.multiply(blocked_probabilities, service_time_averages).mean() - e_x * e_y

    c = -cov/var_y

    Z = np.array(blocked_probabilities) - c*(np.array(service_time_averages) - e_y)

    print(Z.mean(), (Z.mean() -1.96 * np.sqrt(Z.var()/10), Z.mean() + 1.96 * np.sqrt(Z.var()/10)))

    # print("Customers served: ", n_served)
    # print("Arrivals: ", n_arrivals)
    print("Average percentage of customers blocked: ", np.mean(total_blocked) / 100)

    # plt.rcParams["figure.figsize"] = (16, 7)
    # print(np.unique(n_in_system, return_counts=True))

    # plt.hist(n_in_system, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], align='left', width=0.9, edgecolor="black")
    # plt.show()
    total_blocked = np.array(total_blocked)
    mean_hats = total_blocked / n_customers
    mean_bar = np.mean(mean_hats)
    var = 1/(n_simulations-1) * (np.sum([m**2 for m in mean_hats])-n_simulations*mean_bar**2)
    confidence_interval = (mean_bar + np.sqrt(var)/np.sqrt(n_simulations) * stats.t.ppf(0.025, n_simulations-1), mean_bar + np.sqrt(var)/np.sqrt(n_simulations)*stats.t.ppf(0.975, n_simulations-1))
    print(mean_bar)
    print(confidence_interval)





