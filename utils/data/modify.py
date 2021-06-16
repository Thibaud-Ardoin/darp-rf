"""
This module contains functions which modify instances.
"""
import dataclasses
from oru import frozendict, map_values
from .types import MSPRP_Data,DARP_Data,PDPTWLH_Data

def service_time(data : MSPRP_Data, service_time_multiplier : float) -> MSPRP_Data:
    return dataclasses.replace(data,
                               service_time = map_values(lambda t : t*service_time_multiplier, data.service_time),
                               id = data.id + f'_s{int(service_time_multiplier*1000):04d}')


def delay_delivery_windows(data : PDPTWLH_Data, delay_amount) -> PDPTWLH_Data:
    return dataclasses.replace(data,
                               E = map_values(lambda t : t + delay_amount, data.E),
                               L = map_values(lambda t : t + delay_amount, data.L),
                               id = data.id+"_MODIFIED")


def gschwind_extend(data : DARP_Data, extend : int) -> DARP_Data:
    if extend ==0:
        return data
    extra_time = extend*5
    s = (3 + extend)/3
    o_depot = 0
    d_depot = data.n*2+1
    tw_end = { i : t if i in (o_depot,d_depot) else t + extra_time for i,t in data.tw_end.items()}

    return dataclasses.replace(data,
                               tw_end = frozendict(tw_end),
                               capacity = s*data.capacity,
                               id = data.id + f"_EX{extend:d}"
                               )

def ride_times(data : DARP_Data, factor: float):
    new_max_ride_time = dict(data.max_ride_time)
    for p in data.P:
        new_max_ride_time[p] = (data.max_ride_time[p]-data.service_time[p])*factor + data.service_time[p]

    return dataclasses.replace(data,
                               max_ride_time = frozendict(new_max_ride_time),
                               id = data.id + f'_R{factor:.04f}'.replace('.', '_'))
