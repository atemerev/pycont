import math
import random
from queue import PriorityQueue

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import model
from model import logn


def select_contact(node_list):
    contact = random.randint(0, len(node_list) - 1)
    return contact


def infect(node_list, queue, event, beta, inf_function, max_inf, inf_time_max, suscept_function, time_max, recovery_delay):
    # first, let's see if we are unlucky to be infected by this event, per our susceptibility
    # node = node_list.iloc[[event.node]]
    rec_time = event.time if node_list.at[event.node, 'event_type'] == 'init' else node_list.at[event.node, 'rec_time']
    delta_t = event.time - rec_time
    susceptibility = 1.0 if node_list.at[event.node, 'event_type'] == 'init' else suscept_function(delta_t)
    # assuming that people are not susceptible until recovered
    if node_list.at[event.node, 'event_type'] == 'infected' and delta_t < recovery_delay:
        susceptibility = 0.0
    random_uniform = random.uniform(0, 1)
    if random_uniform > susceptibility:
        # infection didn't pass through (we do not update rec_time then)
        return
    # otherwise, we are infected. Let's generate infection events and add them to the queue
    # susceptibility is calculated from the recovery time
    node_list.at[event.node, 'rec_time'] = event.time
    node_list.at[event.node, 'event_type'] = 'infected'
    inf_times = model.get_inf_times_mi(inf_time_max, beta, inf_function, max_inf)
    for inf_time in inf_times:
        # creating infection event, adding it to the queue
        t = event.time + inf_time
        if t > time_max:
            return
        target = select_contact(node_list)
        infection_event = model.Event(node=target, time=t, action='infect')
        queue.put(infection_event)
        # creating the "removal" event, to use in statistics (and deaths estimations)
        remove_event = model.Event(node=target, time=t + inf_time_max, action='remove')
        queue.put(remove_event)


def simulate(nodes_frame, initial_infected, t_max, beta, inf_function, susc_function, resolution, max_disease_length, recovery_delay, max_inf):
    infected = random.sample(nodes_frame.index.tolist(), initial_infected)
    queue = PriorityQueue()

    # sampling the max value from infectiousness function
    ts = np.linspace(0, max_disease_length, resolution)
    # vf = np.vectorize(inf_function, otypes=[float])
    # inf_samples = vf(ts)
    # max_inf = np.max(inf_samples)
    # print("!!!", max_inf)

    # initial exposure events
    for u in infected:
        event = model.Event(node=u, time=0, action='infect')
        queue.put(event)

    result = [{'time': 0, 'cases': 0, 'net_dose': 1}]
    last_day = 0
    cases_by_day = {}
    total_cases = initial_infected
    prev_total = initial_infected
    recoveries_by_day = {}


    while not queue.empty():
        event = queue.get()
        day = int(event.time)
        if event.time >= t_max:
            return pd.DataFrame(result)

        day = int(event.time)
        if event.action == 'infect':
            total_cases += 1
            cases_by_day[day] = cases_by_day.get(day, 0) + 1
            infect(nodes_frame, queue, event, beta, inf_function, max_inf, max_disease_length, susc_function, t_max, recovery_delay)
            recovery_event = model.Event(node=event.node, time=event.time + max_disease_length, action='recover')
            queue.put(recovery_event)
        if event.action == 'recover':
            recoveries_by_day[day] = recoveries_by_day.get(day, 0) + 1

        if day > last_day:
            d_total = float(total_cases) / prev_total
            prev_total = total_cases
            d_cases = cases_by_day.get(last_day, 0)

            # d_recoveries = recoveries_by_day.get(last_day, 0)
            # net_dose = d_cases - d_recoveries
            result.append({'time': last_day + 1, 'cases': cases_by_day.get(last_day, 0), 'net_dose': d_total})
            last_day = day

    return pd.DataFrame(result)

def main():
    plt.rcParams.update({'font.size': 20})
    print("Simulation started")
    people_count = 5000
    resolution = 1000
    max_disease_length = 20
    recovery_delay = 7
    max_inf = 0.10908088834560385

    scale = 0.2577
    mean = 1.4915
    k = 0.293
    decay_popt = [-9.77555285e-03, 1.03320181e+00, 1.95573648e+02]
    #decay_popt = [-1e-01, 0.5, 10]

    initial_infected = 2
    t_max = 365 * 2 # days
    beta = 1.0

    def inf_function(t):
        return logn(t, scale, mean, k)

    def decay_approx(x, k, l, x0):
        try:
            ex = math.exp(-k * (x - x0))
        except OverflowError:
            ex = float('inf')
        return l / (1 + ex)

    def susc_function(t):
        return 1 - decay_approx(t, *decay_popt)

    rows_list = []
    for i in range(0, people_count):
        # event types: 'init', 'infected', 'vaccinated'
        # rec_time: last event timestamp
        rows_list.append({'rec_time': -1, 'susceptibility': 1.0, 'event_type': 'init'})
    nodes = pd.DataFrame(rows_list)

    result = simulate(nodes, initial_infected, t_max, beta, inf_function, susc_function, resolution, max_disease_length, recovery_delay, max_inf)
    print(result)

    plt.figure(figsize=(20, 12))
    plt.title("Daily new infections (simulation, time-dependent infectiousness and immunity)")
    plt.xlabel("Time (days)")
    plt.ylabel("Daily new infections")
    plt.plot(result['time'], result['cases'])
    plt.grid()
    plt.savefig('cases_plot5.png')

    plt.figure(figsize=(20, 12))
    plt.title("Net dose (daily infections - daily recoveries)")
    plt.xlabel("Time (days)")
    plt.ylabel("Net dose")
    plt.plot(result['time'], result['net_dose'])
    plt.grid()
    plt.savefig('net_dose_plot.png')

    inf_nums = []
    events = []

    for i in range(0, 10000):
        inf_times = model.get_inf_times_mi(20, 1.0, inf_function, max_inf)
        inf_nums.append(len(inf_times))
        events.extend(inf_times)

    plt.figure()
    plt.hist(events, bins=100, range=[0, max_disease_length])
    plt.savefig('inf_times.png')
    plt.figure()
    plt.hist(inf_nums, bins=100, range=[0, max_disease_length])
    plt.savefig('inf_num.png')


main()
