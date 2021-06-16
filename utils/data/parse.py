from oru import frozendict, LazyHashFrozenDataclass
import dataclasses
import itertools
import re
import h5py
from .types import *
import io
from typing import Callable
frozen_dataclass = dataclasses.dataclass(frozen=True)

def _read_until(fp : io.TextIOBase, func : Callable[[str], bool]) -> Union[None, str]:
    for l in fp:
        if func(l):
            return l
    return None


def parse_format_schrotenboer(depot_file, geninfo_file, jobs_file, maxtravel_file) -> RawDataSchrotenboer:
    with open(geninfo_file, 'r') as f:
        first_line, second_line = f.readline().strip().split(), f.readline().strip().split()
        problem_size_info = dict(zip(first_line, map(int, second_line)))
        f.readline()  # next line is a comment
        vehicle_info = dict()
        for _ in range(problem_size_info['nr_vehicles']):
            idx, servicemen_max, parts_cap, cost_parameter, travel_parameter = f.readline().strip().split()
            vehicle_info[int(idx)] = {
                'servicemen_capacity': int(servicemen_max),
                'parts_capacity': int(parts_cap),
                'cost_factor': float(cost_parameter),
                'speed': float(travel_parameter),
            }

    n = problem_size_info['nr_jobs']
    depot_origin_idx = 0
    depot_dest_idx = 2 * n + 1
    L = range(problem_size_info['nr_reusables'])
    T = range(problem_size_info['nr_periods'])
    Nd = range(1, n + 1)
    Np = range(n + 1, 2 * n + 1)
    K = range(problem_size_info['nr_vehicles'])

    with open(depot_file, 'r') as f:
        # header is useless, format seems to be Cap(1), ..., Cap(L), Cost(1), ..., Cost(L) where Cap( ) is the maximum
        # number of servicement of a type per time period and Cost( ) is the unit cost of a type of serviceman.
        # There is a line for each time period
        contents = list(map(lambda x: x.strip().split(), f.readlines()[1:]))

        servicemen_max = map(lambda line: tuple(int(x) for x in line[:problem_size_info['nr_reusables']]), contents)
        servicemen_cost = map(lambda line: tuple(float(x) for x in line[problem_size_info['nr_reusables']:]), contents)

        servicemen_max = {(l, t): v for t, vals in zip(T, servicemen_max) for l, v in zip(L, vals)}
        servicemen_cost = {(l, t): v for t, vals in zip(T, servicemen_cost) for l, v in zip(L, vals)}

        assert (len(servicemen_max) == problem_size_info['nr_periods'] * problem_size_info['nr_reusables'])
        assert (len(servicemen_cost) == problem_size_info['nr_periods'] * problem_size_info['nr_reusables'])

    servicemen_cost = {l: servicemen_cost[l, 0] for l in L}
    servicemen_max = {l: servicemen_max[l, 0] for l in L}

    with open(maxtravel_file, 'r') as f:
        contents = map(lambda line: line.strip().split(), f.readlines())
        travel_time_max = {(k, t): int(v) for t, row in zip(T, contents) for k, v in zip(K, row)}

    with open(jobs_file, 'r') as f:
        contents = list(map(lambda line: line.strip().split(), f.readlines()[1:]))
        num_cols = len(contents[0])
        assert all(map(lambda l: len(l) == num_cols, contents[1:])), 'number of columns is not consistent'
        columns = [[contents[r][c] for r in range(len(contents))] for c in range(num_cols)]
        index = list(map(int, columns[0]))
        loc = {int(idx) + 1: (float(x), float(y)) for idx, x, y in zip(index, columns[1], columns[2])}
        servicemen_demand = {(i, l): int(val) for l in L for i, val in zip(Nd, columns[3 + l])}
        servicemen_demand.update({(i + len(Nd), l): -val for (i, l), val in servicemen_demand.items()})
        col_offset = 3 + len(L)
        parts_demand = dict(zip(Nd, map(int, columns[col_offset])))
        col_offset += 1
        service_time = dict(zip(Nd, map(lambda x: float(x), columns[col_offset])))
        col_offset += 1
        periodic_costs = {(i, t): float(val) for t in T for i, val in zip(Nd, columns[col_offset + t])}

    periodic_costs.update({(i, t): 0 for i in itertools.chain(Np, [depot_origin_idx, depot_dest_idx]) for t in T})

    loc.update({i + len(Nd): v for i, v in loc.items()})
    loc[depot_origin_idx] = loc[depot_dest_idx] = (0, 30)

    return RawDataSchrotenboer(
        num_requests=n,
        num_time_periods=problem_size_info['nr_periods'],
        num_vehicles=problem_size_info['nr_vehicles'],
        num_servicemen_types=problem_size_info['nr_reusables'],
        servicemen_cost=frozendict(servicemen_cost),
        servicemen_avail=frozendict(servicemen_max),
        vehicle_servicemen_cap=frozendict({k: vehicle_info[k]['servicemen_capacity'] for k in K}),
        vehicle_parts_cap=frozendict({k: vehicle_info[k]['parts_capacity'] for k in K}),
        vehicle_speed=frozendict({k: vehicle_info[k]['speed'] for k in K}),
        vehicle_travel_cost=frozendict({k: vehicle_info[k]['cost_factor'] for k in K}),
        pos_x=frozendict({i: xy[0] for i, xy in loc.items()}),
        pos_y=frozendict({i: xy[1] for i, xy in loc.items()}),
        servicemen_demand=frozendict(servicemen_demand),
        parts_demand=frozendict(parts_demand),
        service_time=frozendict(service_time),
        periodic_costs=frozendict(periodic_costs),
        max_travel_time=frozendict(travel_time_max),
        loading_time=0.25
    )


def _cordeau_fmt_parse_body_line(s: str):
    node_id, x, y, st, d, tw_start, tw_end = s.strip().split()
    return (int(node_id), float(x), float(y), float(st), float(d), float(tw_start), float(tw_end))


def parse_format_cordeau(path) -> RawDataCordeau:
    with open(path, 'r') as raw:
        header = raw.readline().strip().split()
        num_vehicles = int(header[0])
        num_request = int(header[1])
        max_route_dur = int(header[2])
        vehicle_cap = int(header[3])
        max_ride_time = int(header[4])

        body = list(map(_cordeau_fmt_parse_body_line, raw.readlines()))
        if len(body) == 2 * num_request + 2:
            pass
        elif len(body) == num_request + 1:
            num_request //= 2
            d_depot = list(body[0])
            d_depot[0] = 2 * num_request + 1
            body.append(tuple(d_depot))
        else:
            raise Exception("bad data format (length mismatch)")

        num_nodes = 2 * num_request + 2
        pos_x = dict()
        pos_y = dict()
        service_time = dict()
        earliest_times = dict()
        latest_times = dict()
        demand = dict()

        for l in body:
            node_id, x, y, st, d, tw_start, tw_end = l
            pos_x[node_id] = x
            pos_y[node_id] = y
            service_time[node_id] = st
            demand[node_id] = d
            earliest_times[node_id] = tw_start
            latest_times[node_id] = tw_end

        assert set(pos_x.keys()) == set(range(num_nodes))

        return RawDataCordeau(
            num_requests=num_request,
            num_vehicles=num_vehicles,
            max_ride_time=max_ride_time,
            max_route_duration=max_route_dur,
            vehicle_cap=vehicle_cap,
            pos_x=frozendict(pos_x),
            pos_y=frozendict(pos_y),
            tw_start=frozendict(earliest_times),
            tw_end=frozendict(latest_times),
            demand=frozendict(demand),
            service_time=frozendict(service_time)
        )


def parse_format_hemmati(path):
    with open(path, 'r') as f:
        while True:
            l = f.readline().strip()
            if len(l) == 0 or l.startswith('#'):
                continue
            break

        num_ports = int(l)
        f.readline()  # number of vessels
        num_vessels = int(f.readline().strip())

        f.readline()  # for each vessel: vessel index, home port, starting time, capacity
        capacity = {}
        vehicle_depot = {}
        vehicle_time = {}
        for _ in range(num_vessels):
            l = list(map(int, f.readline().strip().split(',')))
            v, origin_port, time_avail, cap = l
            v -= 1
            origin_port -= 1
            vehicle_depot[v] = origin_port
            vehicle_time[v] = time_avail
            capacity[v] = cap

        f.readline()
        # number of cargoes

        num_cargo = int(f.readline().strip())
        f.readline()
        # for each vessel, vessel index, and then a list of cargoes that can be transported using that vessel
        vessel_compat = dict()
        for _ in range(num_vessels):
            l = list(map(lambda i: int(i)-1, f.readline().strip().split(',')))
            vessel_compat[l[0]] = frozenset(l[1:])

        f.readline()
        # cargo index, origin port, destination port, size, penalty, lb tw pu, ub tw pu, lb tw d, ub tw d
        cargo_penalty = {}
        cargo_origin = {}
        cargo_dest = {}
        cargo_size = {}
        cargo_origin_tw_start = {}
        cargo_origin_tw_end = {}
        cargo_dest_tw_start = {}
        cargo_dest_tw_end = {}

        for _ in range(num_cargo):
            l = map(int, f.readline().strip().split(','))
            c, origin_port, dest_port, sz, penalty, tw_p_start, tw_p_end, tw_d_start, tw_d_end = l
            c -= 1
            origin_port -= 1
            dest_port -= 1
            cargo_penalty[c] = penalty
            cargo_origin[c] = origin_port
            cargo_dest[c] = dest_port
            cargo_size[c] = sz
            cargo_origin_tw_start[c] = tw_p_start
            cargo_origin_tw_end[c] = tw_p_end
            cargo_dest_tw_start[c] = tw_d_start
            cargo_dest_tw_end[c] = tw_d_end

        ports = set(cargo_dest.values()) | set(cargo_origin.values()) | set(vehicle_depot.values())

        f.readline()
        # vessel, origin port, destination port, travel time, travel cost

        travel_time = {}
        travel_cost = {}

        for _ in range(num_vessels * num_ports * num_ports):
            v, src, dst, time, cost = map(int, f.readline().strip().split(','))
            if time < 0: # why is this not sparse
                continue
            src -= 1
            dst -= 1
            if src in ports and dst in ports:
                v -= 1
                travel_time[v, src, dst] = time
                travel_cost[v, src, dst] = cost

        f.readline()
        # vessel, cargo, origin port time, origin port costs, destination port time, destination port costs

        cargo_origin_port_cost = {}
        cargo_origin_port_time = {}
        cargo_dest_port_cost = {}
        cargo_dest_port_time = {}
        for _ in range(num_cargo * num_vessels):
            l = list(map(int, f.readline().strip().split(',')))
            v, c, p_time, p_cost, d_time, d_cost = l
            v -= 1
            c -= 1
            if p_time < 0:
                continue  # why is this not sparse i do not know
            cargo_origin_port_cost[v, c] = p_cost
            cargo_origin_port_time[v, c] = p_time
            cargo_dest_port_cost[v, c] = d_cost
            cargo_dest_port_time[v, c] = d_time

        return RawDataHemmati(
            num_ports=num_ports,
            num_vessels=num_vessels,
            num_cargos=num_cargo,
            vessel_start_time=frozendict(vehicle_time),
            vessel_capacity=frozendict(capacity),
            vessel_start_port=frozendict(vehicle_depot),
            vessel_compatible=frozendict(vessel_compat),
            cargo_penalty=frozendict(cargo_penalty),
            cargo_size=frozendict(cargo_size),
            cargo_origin=frozendict(cargo_origin),
            cargo_dest=frozendict(cargo_dest),
            cargo_origin_tw_start=frozendict(cargo_origin_tw_start),
            cargo_origin_tw_end=frozendict(cargo_origin_tw_end),
            cargo_dest_tw_start=frozendict(cargo_dest_tw_start),
            cargo_dest_tw_end=frozendict(cargo_dest_tw_end),
            cargo_origin_port_time=frozendict(cargo_origin_port_time),
            cargo_origin_port_cost=frozendict(cargo_origin_port_cost),
            cargo_dest_port_time=frozendict(cargo_dest_port_time),
            cargo_dest_port_cost=frozendict(cargo_dest_port_cost),
            travel_time=frozendict(travel_time),
            travel_cost=frozendict(travel_cost)
        )


def parse_format_hemmati_hdf5(filename):
    f = h5py.File(filename, 'r')

    num_vessels = int(f.attrs['num_vessels'])
    num_ports = int(f.attrs['num_ports'])
    num_cargos = int(f.attrs['num_cargos'])

    vessel_start_time = dict(enumerate(map(int, f['vessel_start_time'][:])))
    vessel_capacity = dict(enumerate(map(int, f['vessel_capacity'][:])))
    vessel_start_port = dict(enumerate(map(int, f['vessel_start_port'][:])))

    vessel_compatible = {}
    for v in range(num_vessels):
        vessel_compatible[v] = frozenset(c for c, compat in enumerate(f['vessel_cargo_compatibility'][v, :]) if compat)

    cargo_size = dict((c, int(sz)) for c, sz in enumerate(f['cargo_size'][:]))
    cargo_origin = dict((c, int(i)) for c, i in enumerate(f['cargo_origin'][:]))
    cargo_dest = dict((c, int(i)) for c, i in enumerate(f['cargo_dest'][:]))

    cargo_origin_tw_start = dict((c, int(t)) for c, t in enumerate(f['cargo_origin_tw_start'][:]))
    cargo_origin_tw_end = dict((c, int(t)) for c, t in enumerate(f['cargo_origin_tw_end'][:]))
    cargo_dest_tw_start = dict((c, int(t)) for c, t in enumerate(f['cargo_dest_tw_start'][:]))
    cargo_dest_tw_end = dict((c, int(t)) for c, t in enumerate(f['cargo_dest_tw_end'][:]))
    cargo_penalty = dict((c, int(t)) for c, t in enumerate(f['cargo_penalty'][:]))

    vc_info = f['vessel_cargo_info']
    cargo_origin_port_time = {}
    cargo_origin_port_cost = {}
    cargo_dest_port_time = {}
    cargo_dest_port_cost = {}
    k = 0
    for v, n in enumerate(vc_info['count_by_vessel'][:]):
        cargo = [int(c) for c in vc_info['cargo'][k:k+n]]
        cargo_origin_port_time.update({
            (v, c): int(x) for c, x in zip(cargo, vc_info['origin_port_time'][k:k + n])
        })
        cargo_origin_port_cost.update({
            (v, c): int(x) for c, x in zip(cargo, vc_info['origin_port_cost'][k:k + n])
        })
        cargo_dest_port_time.update({
            (v, c): int(x) for c, x in zip(cargo, vc_info['dest_port_time'][k:k + n])
        })
        cargo_dest_port_cost.update({
            (v, c): int(x) for c, x in zip(cargo, vc_info['dest_port_cost'][k:k + n])
        })
        k += n

    trvl_info = f['travel_info']

    travel_time = {}
    travel_cost = {}

    k = 0
    for v, n in enumerate(trvl_info['count_by_vessel'][:]):
        arcs = [(int(i), int(j)) for i,j in zip(trvl_info['src_port'][k:k+n], trvl_info['dest_port'][k:k+n])]
        travel_time.update({
            (v,i,j) : int(x) for (i,j), x in zip(arcs, trvl_info['time'][k:k+n])
        })
        travel_cost.update({
            (v,i,j) : int(x) for (i,j), x in zip(arcs, trvl_info['cost'][k:k+n])
        })
        k += n

    f.close()

    return RawDataHemmati(
        num_ports=num_ports,
        num_vessels=num_vessels,
        num_cargos=num_cargos,
        vessel_start_time=frozendict(vessel_start_time),
        vessel_capacity=frozendict(vessel_capacity),
        vessel_start_port=frozendict(vessel_start_port),
        vessel_compatible=frozendict(vessel_compatible),
        cargo_penalty=frozendict(cargo_penalty),
        cargo_size=frozendict(cargo_size),
        cargo_origin=frozendict(cargo_origin),
        cargo_dest=frozendict(cargo_dest),
        cargo_origin_tw_start=frozendict(cargo_origin_tw_start),
        cargo_origin_tw_end=frozendict(cargo_origin_tw_end),
        cargo_dest_tw_start=frozendict(cargo_dest_tw_start),
        cargo_dest_tw_end=frozendict(cargo_dest_tw_end),
        cargo_origin_port_time=frozendict(cargo_origin_port_time),
        cargo_origin_port_cost=frozendict(cargo_origin_port_cost),
        cargo_dest_port_time=frozendict(cargo_dest_port_time),
        cargo_dest_port_cost=frozendict(cargo_dest_port_cost),
        travel_time=frozendict(travel_time),
        travel_cost=frozendict(travel_cost)
    )

def parse_format_hemmati_hdf5_to_skeleton(path) -> ITSRSP_Skeleton_Data:
    h5 = h5py.File(path, 'r')
    n = h5.attrs['num_cargos']
    v = h5.attrs['num_vessels']
    h5.close()
    P = range(n)
    D = range(n, 2*n)
    V = range(v)
    o_depots = range(2*n, 2*n+v)
    d_depot = 2*n+v

    return ITSRSP_Skeleton_Data(id="", n=n, P=P, D=D, V=V, o_depots=o_depots, d_depot=d_depot)

def parse_format_riedler(path) -> RawDataCordeau:
    with open(path, 'r') as fp:
        num_requests = int(_read_until(fp, lambda x : x.startswith("|N|")).split()[-1])
        num_vehicles = int(_read_until(fp, lambda x : x.startswith("|K|")).split()[-1])
        max_ride_time = int(_read_until(fp, lambda x : x.startswith("L")).split()[-1])

        o_depot = 0
        d_depot = num_requests*2 + 1
        depot_info = list(map(float, _read_until(fp, lambda x : x.startswith("Depot:")).split()[1:]))
        pos_x = { d_depot : depot_info[0], o_depot: depot_info[0]}
        pos_y = { d_depot : depot_info[1], o_depot: depot_info[1]}
        tw_start = { d_depot : depot_info[2], o_depot: depot_info[2]}
        tw_end = { d_depot : depot_info[3], o_depot: depot_info[3]}
        service_time = {d_depot : 0, o_depot: 0}
        demand = {}

        _read_until(fp, lambda x : x.startswith("Vehicles"))
        v_info = fp.readline().split()
        vehicle_capacity = int(v_info[0])
        max_route_time = int(v_info[1])
        for i in range(num_vehicles-1): # check vehicles are identical
            _v_info = fp.readline().split()
            assert v_info == _v_info

        _read_until(fp, lambda x : x.startswith("Requests"))
        for p,line in zip(range(1, num_requests+1), fp):
            line = map(float, line.split())
            d = p + num_requests
            pos_x[p] = next(line)
            pos_y[p] = next(line)
            tw_start[p] = next(line)
            tw_end[p] = next(line)

            pos_x[d] = next(line)
            pos_y[d] = next(line)
            tw_start[d] = next(line)
            tw_end[d] = next(line)

            # NOTE: the README is wrong, the last thing on the line is the service time, not the demand (c.f. riedler2018)
            demand[p] = next(line)
            demand[d] = -demand[p]

            service_time[p] = next(line)
            service_time[d] = service_time[p]


        assert len(pos_x) == 2*num_requests + 2

        return RawDataCordeau(
            num_requests=num_requests,
            num_vehicles=num_vehicles,
            max_ride_time=max_ride_time,
            max_route_duration=max_route_time,
            vehicle_cap=vehicle_capacity,
            pos_x=frozendict(pos_x),
            pos_y=frozendict(pos_y),
            tw_start=frozendict(tw_start),
            tw_end=frozendict(tw_end),
            demand=frozendict(demand),
            service_time=frozendict(service_time)
        )

