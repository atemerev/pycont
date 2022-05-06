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
