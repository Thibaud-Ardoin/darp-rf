"""
Functions to generate random instances
"""
from .parse import RawDataCordeau
import numpy as np
from oru import frozendict

def generate_cordeau(num_loc, max_demand, vehicle_cap, tw_width, delivery_delay, seed) ->RawDataCordeau:
    """See Ropke 2009 and Cherkesly 2014"""
    rng = np.random.RandomState(seed)
    loc_x = rng.uniform(0, 50, 2 * num_loc + 1)
    loc_y = rng.uniform(0, 50, 2 * num_loc + 1)
    p_locs = range(1, num_loc + 1)
    d_locs = range(num_loc + 1, 2 * num_loc + 1)
    locs = range(2 * num_loc + 1)
    travel_time = {(i, j): ((loc_x[i] - loc_x[j]) ** 2 + (loc_y[i] - loc_y[j]) ** 2) ** 0.5
                   for i in locs for j in range(i + 1)
                   }
    travel_time.update({(j, i): d for (i, j), d in travel_time.items()})
    max_time = 600
    tmp = np.array([travel_time[i, i + num_loc] for i in p_locs])
    Ep = (max_time - tmp) * rng.rand(*tmp.shape)
    Lp = Ep + tw_width
    Ed = Ep + tmp + delivery_delay
    Ld = Ed + tw_width

    Dp = rng.randint(5, max_demand + 1, num_loc)
    D = dict(zip(p_locs, Dp))
    D.update({i + num_loc: -v for i, v in D.items()})
    E = dict(zip(p_locs, Ep))
    E.update(dict(zip(d_locs, Ed)))
    L = dict(zip(p_locs, Lp))
    L.update(dict(zip(d_locs, Ld)))

    return RawDataCordeau(
        num_requests = num_loc,
        vehicle_cap = vehicle_cap,
        pos_x = frozendict(enumerate(loc_x)),
        pos_y = frozendict(enumerate(loc_y)),
        tw_start = frozendict(E),
        tw_end = frozendict(L),
        service_time=frozendict((i,0) for i in locs),
        demand = D,
    )
