from oru import frozendict
from .parse import RawDataCordeau
from .core import build_DARP_from_cordeau

TANG2010 = build_DARP_from_cordeau(RawDataCordeau(
    num_requests = 2,
    num_vehicles = 1,
    max_route_duration = 200,
    max_ride_time = 16,
    vehicle_cap = 20,
    pos_x = frozendict({
        0 : -4,
        1 : 0,
        2 : 4,
        3 : 8,
        4 : 4
    }),
    pos_y = frozendict({
        0 : 3,
        1 : 0,
        2 : 3,
        3 : 0,
        4 : -3
    }),
    tw_start = frozendict({
        0 : 0,
        1 : 305,
        2 : 320,
        3 : 310,
        4 : 335,
    }),
    tw_end = frozendict({
        0 : 1440,
        1 : 345,
        2 : 360,
        3 : 350,
        4 : 375
    }),
    demand = frozendict({
        1 : 5,
        2 : 5,
        3 : -5,
        4 : -5
    }),
    service_time = frozendict({
        1 : 0,
        2 : 0,
        3 : 0,
        4 : 0
    })
), 'tang2010')
