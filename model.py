import math
import numpy as np
import random
from functools import wraps
from scipy.optimize import minimize_scalar

def logn(x, s, m, k):
    if x == 0:
        return 0
    m1 = 1.0 / (x * s * math.sqrt(2 * math.pi))
    exp = -(math.pow(math.log(x) - m, 2) / math.pow(2 * s, 2))
    return k * m1 * math.exp(exp)


vlogn = np.vectorize(logn, otypes=[float])

def get_inf_times_mi(t_max, beta, inf_function, max_inf):
    inf_times = []
    t = 0
    while t < t_max:
        u = random.uniform(0, 1)
        t = t - math.log(u) / beta
        if t < t_max:
            rate = inf_function(t)
            s = random.uniform(0, 1)
            if s < (rate / max_inf): # todo verify the max_inf, write down the reference
                inf_times.append(t)

    return inf_times
