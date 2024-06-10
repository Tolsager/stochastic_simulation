import numpy as np
import matplotlib.pyplot as plt



class Server:
    def __init__(self, mean_service_time):
        self.occupied = False
        self.mean_service_time = mean_service_time

    def start_service(self, time):
        self.occupied = True
        service_time = np.random.exponential(self.mean_service_time)
        departure_time = time + service_time
        return service_time, departure_time

    def end_service(self):
        self.occupied = False

    def __bool__(self):
        return self.occupied


if __name__ == '__main__':
    # Blocking system with 10 servers and no waiting room
    # Mean service time is 8 time units
    # Mean time between arrivals is 1 time unit
    # Simulate until 10000 customers have been served
    n_servers = 10
    mean_service_time = 8
    mean_interarrival_time = 1
    n_customers = 10000

    # Initialize the simulation
    t = 0
    n_served = 0
    n_blocked = 0
    n_in_system = 0
    n_arrivals = 0
    n_departures = 0

    servers = [Server(mean_service_time) for _ in range(n_servers)]

    event_list = []

    while n_arrivals < n_customers:
        ...

        if all(servers):
            n_blocked += 1
            continue