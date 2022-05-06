import random
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import math
import scipy.integrate as integrate
from scipy.integrate import odeint
import datetime as dt
from queue import PriorityQueue
from dataclasses import dataclass
from functools import total_ordering
import pandas as pd
from queue import PriorityQueue
from scipy.optimize import curve_fit

import model
from model import logn, vlogn


def select_contact(node_list):
    contact = random.randint(0, len(node_list) - 1)
    return contact


def infect(node_list, queue, event, beta, inf_function, max_inf, inf_time_max, suscept_function, time_max, recovery_delay):
    # first, let's see if we are unlucky to be infected by this event, per our susceptibility
    node = node_list[event.node]
    rec_time = event.time if node['rec_time'] == -1 else node['rec_time']
    delta_t = event.time - rec_time
    susceptibility = suscept_function(delta_t)
    random_uniform = random.uniform(0, 1)
    if random_uniform > susceptibility:
        # infection didn't pass through (we do not update rec_time then)
        return
    # otherwise, we are infected. Let's generate infection events and add them to the queue
    # susceptibility is calculated from the recovery time
    node_list.at[event.node, 'rec_time'] = event.time + recovery_delay
    inf_times = model.get_inf_times_mi(inf_time_max, beta, inf_function, max_inf)
    for inf_time in inf_times:
        # creating infection event, adding it to the queue
        t = event.time + inf_time
        if t > time_max:
            return
        target = select_contact(node_list)
        infection_event = model.Event(node=target, time=t, action='infect')
        queue.put(infection_event)


def simulate(node_list, initial_infected, t_max, beta, inf_function, susc_function, resolution, max_disease_length, recovery_delay):
    infected = random.sample(node_list, initial_infected)
    queue = PriorityQueue()

    # sampling the max value from infectiousness function
    ts = np.linspace(0, max_disease_length, resolution)
    vf = np.vectorize(inf_function, otypes=[float])
    inf_samples = vf(ts)
    max_inf = np.max(inf_samples)

    # initial exposure events
    for u in infected:
        event = model.Event(node=u, time=0, action='infect')
        node_list.at[event.node, 'rec_time'] = event.time
        queue.put(event)

    result = [{'time': 0, 'cases': 0}]
    last_day = 0
    cases_by_day = {}

    while not queue.empty():
        event = queue.get()
        if event.time >= t_max:
            return pd.DataFrame(result)
        if event.action == 'infect':
            day = int(event.time)
            cases_by_day[day] = cases_by_day.get(day, 0) + 1
            infect(node_list, queue, event, beta, inf_function, susc_function, max_inf, susc_function, t_max, recovery_delay)
            if day > last_day:
                result.append({'time': last_day + 1, 'cases': cases_by_day[last_day]})
                last_day = day

    return pd.DataFrame(result)

def main():
    print("Simulation started")
    resolution = 1000
    max_days = 20

    scale = 0.2577
    mean = 1.4915
    k = 0.293

    ts = np.linspace(0, max_days, resolution)
    res = vlogn(ts, scale, mean, k)

    def inf_function(t):
        return logn(t, scale, mean, k)

    max_inf = np.max(res)

    inf_nums = []
    events = []

    for i in range(0, 10000):
        inf_times = model.get_inf_times_mi(20, 1.0, inf_function, max_inf)
        inf_nums.append(len(inf_times))
        events.extend(inf_times)

    plt.figure()
    plt.hist(events, bins=100, range=[0, max_days])
    plt.savefig('inf_times.png')
    plt.figure()
    plt.hist(inf_nums, bins=100, range=[0, max_days])
    plt.savefig('inf_num.png')


main()
