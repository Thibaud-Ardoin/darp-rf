import glob
import tarfile
import itertools
from pathlib import Path
from typing import Tuple
from oru import *
from .parse import *
from ..constants import DATA_DIR, ITSRSP_TIME_INFINITY
from .types import *
from . import modify
from . import utils as _u


def data_directory(subdir) -> Path:
    path = DATA_DIR / subdir
    if not path.exists():
        archive = DATA_DIR / f"compressed/{subdir}.tar.xz"
        tar = tarfile.open(archive, 'r:xz')
        tar.extractall(DATA_DIR)
        tar.close()
        assert path.exists()
    return path


def _resolve_name(name: str, data_dir: str, suffix=None):
    datasubdir = data_directory(data_dir)
    suffix = suffix or ""
    path = datasubdir / f"{name}{suffix}"
    if path.exists():
        return path
    raise ValueError(f"{name} is not a valid {data_dir} name")


# name resolvers should be by dataset not by datatype
_schrotenboer_filepattern_index = None


def resolve_name_schrotenboer(name: str) -> Tuple[Path]:
    global _schrotenboer_filepattern_index
    if _schrotenboer_filepattern_index is None:
        root_dir = data_directory("MSPRP_schrotenboer")
        index_file = root_dir / 'INDEX.txt'
        _schrotenboer_filepattern_index = {}
        with open(index_file) as f:
            for line in f:
                k, v = line.strip().split()
                _schrotenboer_filepattern_index[k] = tuple(sorted(root_dir.glob(v)))
    if name not in _schrotenboer_filepattern_index:
        raise ValueError(f"{name} is not a valid Schrotenboer-instance name")

    return _schrotenboer_filepattern_index[name]


def resolve_name_cordeau_PDPTW(name: str) -> Path:
    return _resolve_name(name, "PDPTW_cordeau")


def resolve_name_cordeau_DARP(name: str) -> Path:
    return _resolve_name(name, "DARP_cordeau", ".txt")


def resolve_name_hemmati(name: str) -> Path:
    return _resolve_name(name, 'ITSRSP_hemmati', ".txt")


def resolve_name_homsi(name: str) -> Path:
    return _resolve_name(name, 'ITSRSP_homsi', ".txt")


def resolve_name_hemmati_hdf5(name: str) -> Path:
    return _resolve_name(name, 'ITSRSP_hdf5_ti', ".hdf5")


def resolve_name_riedler(name: str) -> Path:
    return _resolve_name(name, "SDARP_riedler", ".dat")


def build_PDPTWLH_from_cordeau(raw: RawDataCordeau, vehicle_cost: float, rehandling_cost: float,
                               id_str: str) -> PDPTWLH_Data:
    n = raw.num_requests
    loc = frozendict((i, (raw.pos_x, raw.pos_y)) for i in range(2 * n + 1))
    pickup_loc = range(1, n + 1)
    delivery_loc = range(n + 1, 2 * n + 1)

    T = {(i, j): ((raw.pos_y[i] - raw.pos_y[j]) ** 2 + (raw.pos_x[i] - raw.pos_x[j]) ** 2) ** 0.5 for i in loc for j in
         range(i + 1)}
    T.update({(j, i): v for (i, j), v in T.items()})
    T = frozendict(T)
    C = T.copy()

    return PDPTWLH_Data(
        T=T,
        C=C,
        n=n,
        E=raw.tw_start,
        L=raw.tw_end,
        D=raw.demand,
        Q=raw.vehicle_cap,
        loc=loc,
        pickup_loc=pickup_loc,
        delivery_loc=delivery_loc,
        vehicle_cost=vehicle_cost,
        rehandle_cost=rehandling_cost,
        id=id_str
    )


def build_MSPRP_from_schrotenboer(raw: RawDataSchrotenboer, id_str: str) -> MSPRP_Data:
    K = range(raw.num_vehicles)
    distance_matrix = _u.euclidean_distance_matrix(raw.pos_x, raw.pos_y)

    travel_times = frozendict(
        ((i, j, k), (distance_matrix[i, j] / raw.vehicle_speed[k] + raw.loading_time))
        for i, j in distance_matrix for k in K
    )

    travel_costs = frozendict(
        ((i, j, k), distance_matrix[i, j] * raw.vehicle_travel_cost[k])
        for i, j in distance_matrix for k in K
    )

    return MSPRP_Data(
        servicemen_demand=raw.servicemen_demand,
        servicemen_max=raw.servicemen_avail,
        servicemen_capacity=raw.vehicle_servicemen_cap,
        servicemen_cost=raw.servicemen_cost,
        service_time=raw.service_time,
        travel_time=travel_times,
        travel_cost=travel_costs,
        parts_demand=raw.parts_demand,
        parts_capacity=raw.vehicle_parts_cap,
        travel_time_max=raw.max_travel_time,
        nr=raw.num_requests,
        Nd=range(1, raw.num_requests + 1),
        Np=range(raw.num_requests + 1, 2 * raw.num_requests + 1),
        T=range(raw.num_time_periods),
        K=K,
        L=range(raw.num_servicemen_types),
        loc=frozendict((i, (raw.pos_x[i], raw.pos_y[i])) for i in range(2 * raw.num_requests + 2)),
        late_penalty=raw.periodic_costs,
        id=id_str
    )


def build_DARP_from_cordeau(raw: RawDataCordeau, id_str: str) -> DARP_Data:
    # Note - service time must be stored separately, since a client's service time does not contribute to their ride
    # time.  I handle this by storing adding the service time onto the travel time, adding each client's service times
    # (both at pickup and delivery) to their max ride time.
    distance = _u.euclidean_distance_matrix(raw.pos_x, raw.pos_y)
    P = range(1, raw.num_requests + 1)
    max_ride_times = {i: raw.max_ride_time + raw.service_time[i] for i in P}
    max_ride_times[0] = raw.max_route_duration
    return DARP_Data(
        travel_time=frozendict({(i, j): d + raw.service_time[i] for (i, j), d in distance.items()}),
        service_time=raw.service_time,
        travel_cost=frozendict(distance),
        tw_start=raw.tw_start,
        tw_end=raw.tw_end,
        demand=raw.demand,
        loc=frozendict((i, (raw.pos_x[i], raw.pos_y[i])) for i in raw.pos_x),
        max_ride_time=frozendict(max_ride_times),
        capacity=raw.vehicle_cap,
        n=raw.num_requests,
        P=P,
        D=range(raw.num_requests + 1, 2 * raw.num_requests + 1),
        K=range(raw.num_vehicles),
        N=range(0, 2 * raw.num_requests + 2),
        id=id_str
    )


def build_ITSRSP_from_hemmati(raw: RawDataHemmati, id_str: str) -> ITSRSP_Data:
    n = raw.num_cargos
    P = range(0, n)
    D = range(n, 2 * n)
    V = range(raw.num_vessels)
    o_depots = range(2 * n, 2 * n + raw.num_vessels)
    d_depot = 2 * n + raw.num_vessels

    tw_start = {}
    tw_end = {}

    for p in P:
        tw_start[p] = raw.cargo_origin_tw_start[p]
        tw_start[p + n] = raw.cargo_dest_tw_start[p]
        tw_end[p] = raw.cargo_origin_tw_end[p]
        tw_end[p + n] = raw.cargo_dest_tw_end[p]

    for v in V:
        tw_start[o_depots[v]] = raw.vessel_start_time[v]
        tw_end[o_depots[v]] = 2 ** 32 - 1

    tw_start[d_depot] = 0
    tw_end[d_depot] = 2 ** 32 - 1

    travel_time = {}
    travel_cost = {}
    port_group = defaultdict(list)

    for i in P:
        port_group[raw.cargo_origin[i]].append(i)
        port_group[raw.cargo_dest[i]].append(i + n)

    for v in V:
        travel_time[v] = {}
        travel_cost[v] = {}

        for i in P:
            if i not in raw.vessel_compatible[v]:
                continue
            o_port_i = raw.cargo_origin[i]
            d_port_i = raw.cargo_dest[i]

            for j in P:
                if j not in raw.vessel_compatible[v]:
                    continue

                o_port_j = raw.cargo_origin[j]
                d_port_j = raw.cargo_dest[j]

                if i != j:
                    travel_time[v][i, j] = raw.travel_time[v, o_port_i, o_port_j] + raw.cargo_origin_port_time[v, i]
                    travel_time[v][i + n, j] = raw.travel_time[v, d_port_i, o_port_j] + raw.cargo_dest_port_time[v, i]
                    travel_time[v][i + n, j + n] = raw.travel_time[v, d_port_i, d_port_j] + raw.cargo_dest_port_time[
                        v, i]
                    travel_cost[v][i, j] = raw.travel_cost[v, o_port_i, o_port_j] + raw.cargo_origin_port_cost[v, i]
                    travel_cost[v][i + n, j] = raw.travel_cost[v, d_port_i, o_port_j] + raw.cargo_dest_port_cost[v, i]
                    travel_cost[v][i + n, j + n] = raw.travel_cost[v, d_port_i, d_port_j] + raw.cargo_dest_port_cost[
                        v, i]

                travel_time[v][i, j + n] = raw.travel_time[v, o_port_i, d_port_j] + raw.cargo_origin_port_time[v, i]
                travel_cost[v][i, j + n] = raw.travel_cost[v, o_port_i, d_port_j] + raw.cargo_origin_port_cost[v, i]

            travel_time[v][i + n, d_depot] = raw.cargo_dest_port_time[v, i]
            travel_cost[v][i + n, d_depot] = raw.cargo_dest_port_cost[v, i]

    # These are allowed to be different for vehicles in the same group
    travel_time_vehicle_origin_depot = {v: {} for v in V}
    travel_cost_vehicle_origin_depot = {v: {} for v in V}

    for v in V:
        od_port = raw.vessel_start_port[v]
        for i in P:
            if i not in raw.vessel_compatible[v]:
                continue

            o_port_i = raw.cargo_origin[i]
            travel_time_vehicle_origin_depot[v][i] = raw.travel_time[v, od_port, o_port_i]
            travel_cost_vehicle_origin_depot[v][i] = raw.travel_cost[v, od_port, o_port_i]

    vehicle_groups = defaultdict(set)

    for v in V:
        key = (
            frozendict(travel_time[v]),
            frozendict(travel_cost[v]),
            raw.vessel_capacity[v],
            raw.vessel_compatible[v],
        )
        vehicle_groups[key].add(v)
    vehicle_groups = frozendict((g, frozenset(grp)) for g, grp in enumerate(vehicle_groups.values()))
    vehicle_groups_inv = frozendict((v, vg) for vg, Vg in vehicle_groups.items() for v in Vg)

    # Now, after grouping, we may merge the travel_times and travel_costs to/from depots
    travel_time_vg = {}
    travel_cost_vg = {}
    capacity_vg = {}
    P_compat_vg = {}
    char_v = {}
    for vg, Vg in vehicle_groups.items():
        v = min(Vg)
        travel_time_vg[vg] = travel_time[v]
        travel_cost_vg[vg] = travel_cost[v]
        capacity_vg[vg] = raw.vessel_capacity[v]
        P_compat_vg[vg] = raw.vessel_compatible[v]

        for v in Vg:
            del travel_time[v]
            del travel_cost[v]
            o = o_depots[v]
            travel_time_vg[vg].update({(o, i): t for i, t in travel_time_vehicle_origin_depot[v].items()})
            travel_cost_vg[vg].update({(o, i): t for i, t in travel_cost_vehicle_origin_depot[v].items()})

        for p in P_compat_vg[vg]:
            _, v_char = min(
                (max(tw_start[o_depots[v]] + travel_time_vehicle_origin_depot[v][p], tw_start[p]), v) for v in Vg
            )
            char_v[vg, p] = v_char

    demand = frozendict(itertools.chain(
        raw.cargo_size.items(),
        ((p + n, -sz) for p, sz in raw.cargo_size.items())
    ))

    return ITSRSP_Data(
        id=id_str,
        n=n,
        P=P,
        D=D,
        V=V,
        o_depots=o_depots,
        d_depot=d_depot,
        demand=demand,
        vehicle_capacity=frozendict(capacity_vg),
        P_compatible=frozendict(P_compat_vg),
        P_incompatible=None,
        customer_penalty=raw.cargo_penalty,
        tw_start=frozendict(tw_start),
        tw_end=frozendict(tw_end),
        travel_time=frozendict((v, frozendict(tt)) for v, tt in travel_time_vg.items()),
        travel_cost=frozendict((v, frozendict(tc)) for v, tc in travel_cost_vg.items()),
        port_groups=frozendict((i, frozenset(g)) for g in port_group.values() for i in g),
        vehicle_groups=vehicle_groups,
        group_by_vehicle=vehicle_groups_inv,
        char_vehicle=frozendict(char_v)
    )


def build_ITSRSP_from_hemmati_aggressive_grouping(raw: RawDataHemmati, id_str: str) -> ITSRSP_Data:
    n = raw.num_cargos
    P = range(0, n)
    D = range(n, 2 * n)
    V = range(raw.num_vessels)
    o_depots = range(2 * n, 2 * n + raw.num_vessels)
    d_depot = 2 * n + raw.num_vessels

    tw_start = {}
    tw_end = {}

    for p in P:
        tw_start[p] = raw.cargo_origin_tw_start[p]
        tw_start[p + n] = raw.cargo_dest_tw_start[p]
        tw_end[p] = raw.cargo_origin_tw_end[p]
        tw_end[p + n] = raw.cargo_dest_tw_end[p]

    for v in V:
        tw_start[o_depots[v]] = raw.vessel_start_time[v]
        tw_end[o_depots[v]] = 2 ** 32 - 1

    tw_start[d_depot] = 0
    tw_end[d_depot] = 2 ** 32 - 1

    travel_time = {}
    travel_cost = {}
    port_group = defaultdict(list)

    for i in P:
        port_group[raw.cargo_origin[i]].append(i)
        port_group[raw.cargo_dest[i]].append(i + n)

    for v in V:
        travel_time[v] = {}
        travel_cost[v] = {}

        for i in P:
            if i not in raw.vessel_compatible[v]:
                continue
            o_port_i = raw.cargo_origin[i]
            d_port_i = raw.cargo_dest[i]

            for j in P:
                if j not in raw.vessel_compatible[v]:
                    continue

                o_port_j = raw.cargo_origin[j]
                d_port_j = raw.cargo_dest[j]

                if i != j:
                    travel_time[v][i, j] = raw.travel_time[v, o_port_i, o_port_j] + raw.cargo_origin_port_time[v, i]
                    travel_time[v][i + n, j] = raw.travel_time[v, d_port_i, o_port_j] + raw.cargo_dest_port_time[v, i]
                    travel_time[v][i + n, j + n] = raw.travel_time[v, d_port_i, d_port_j] + raw.cargo_dest_port_time[
                        v, i]
                    travel_cost[v][i, j] = raw.travel_cost[v, o_port_i, o_port_j] + raw.cargo_origin_port_cost[v, i]
                    travel_cost[v][i + n, j] = raw.travel_cost[v, d_port_i, o_port_j] + raw.cargo_dest_port_cost[v, i]
                    travel_cost[v][i + n, j + n] = raw.travel_cost[v, d_port_i, d_port_j] + raw.cargo_dest_port_cost[
                        v, i]

                travel_time[v][i, j + n] = raw.travel_time[v, o_port_i, d_port_j] + raw.cargo_origin_port_time[v, i]
                travel_cost[v][i, j + n] = raw.travel_cost[v, o_port_i, d_port_j] + raw.cargo_origin_port_cost[v, i]

            travel_time[v][i + n, d_depot] = raw.cargo_dest_port_time[v, i]
            travel_cost[v][i + n, d_depot] = raw.cargo_dest_port_cost[v, i]

    # These are allowed to be different for vehicles in the same group
    travel_time_vehicle_origin_depot = {v: {} for v in V}
    travel_cost_vehicle_origin_depot = {v: {} for v in V}

    for v in V:
        od_port = raw.vessel_start_port[v]
        for i in P:
            if i not in raw.vessel_compatible[v]:
                continue

            o_port_i = raw.cargo_origin[i]
            travel_time_vehicle_origin_depot[v][i] = raw.travel_time[v, od_port, o_port_i]
            travel_cost_vehicle_origin_depot[v][i] = raw.travel_cost[v, od_port, o_port_i]

    vehicle_groups = {}
    travel_time_vg = {}
    travel_cost_vg = {}
    capacity_vg = {}
    P_compat_vg = {}
    for v in V:
        for vg in vehicle_groups:
            if capacity_vg[vg] != raw.vessel_capacity[v]:
                continue

            for (i, j), t in travel_time[v].items():
                c = travel_cost[v][i, j]

                if (i, j) in travel_time_vg[vg]:
                    if travel_time_vg[vg][i, j] != t or travel_cost_vg[vg][i, j] != c:
                        break
                else:
                    pi = i if i < n else i - n
                    pj = j if j < n else j - n
                    assert not any({pi, pj} <= raw.vessel_compatible[vd] for vd in vehicle_groups[vg])

            else:
                vehicle_groups[vg].add(v)
                travel_time_vg[vg].update(travel_time[v])
                travel_cost_vg[vg].update(travel_cost[v])
                P_compat_vg[vg] |= raw.vessel_compatible[v]
                break

        else:
            vg = len(vehicle_groups)
            vehicle_groups[vg] = {v}
            travel_time_vg[vg] = travel_time[v]
            travel_cost_vg[vg] = travel_cost[v]
            capacity_vg[vg] = raw.vessel_capacity[v]
            P_compat_vg[vg] = raw.vessel_compatible[v]

        del travel_cost[v]
        del travel_time[v]

    vehicle_groups = frozendict((g, frozenset(grp)) for g, grp in vehicle_groups.items())
    vehicle_groups_inv = frozendict((v, vg) for vg, Vg in vehicle_groups.items() for v in Vg)

    # Now, after grouping, we may merge the travel_times and travel_costs to/from depots
    char_v = {}
    P_incompat = {v: frozenset(P) - raw.vessel_compatible[v] for v in V}
    for vg, Vg in vehicle_groups.items():
        for v in Vg:
            o = o_depots[v]
            travel_time_vg[vg].update({(o, i): t for i, t in travel_time_vehicle_origin_depot[v].items()})
            travel_cost_vg[vg].update({(o, i): t for i, t in travel_cost_vehicle_origin_depot[v].items()})

        for p in P_compat_vg[vg]:
            _, v_char = min(
                (max(tw_start[o_depots[v]] + travel_time_vehicle_origin_depot[v][p], tw_start[p]), v) for v in Vg
                if p not in P_incompat[v]
            )
            char_v[vg, p] = v_char

    P_incompat = frozendict(P_incompat)

    demand = frozendict(itertools.chain(
        raw.cargo_size.items(),
        ((p + n, -sz) for p, sz in raw.cargo_size.items())
    ))

    return ITSRSP_Data(
        id=id_str,
        n=n,
        P=P,
        D=D,
        V=V,
        o_depots=o_depots,
        d_depot=d_depot,
        demand=demand,
        vehicle_capacity=frozendict(capacity_vg),
        P_compatible=frozendict(P_compat_vg),
        P_incompatible=P_incompat,
        customer_penalty=raw.cargo_penalty,
        tw_start=frozendict(tw_start),
        tw_end=frozendict(tw_end),
        travel_time=frozendict((v, frozendict(tt)) for v, tt in travel_time_vg.items()),
        travel_cost=frozendict((v, frozendict(tc)) for v, tc in travel_cost_vg.items()),
        port_groups=frozendict((i, frozenset(g)) for g in port_group.values() for i in g),
        vehicle_groups=vehicle_groups,
        group_by_vehicle=vehicle_groups_inv,
        char_vehicle=frozendict(char_v)
    )


SDARP_TIME_SCALE = 10 ** 5


def convert_darp_to_sdarp(data: DARP_Data) -> SDARP_Data:
    def round_demand(x):
        y = round(x)
        assert math.isclose(x, y)
        return y

    round_time = lambda t: round(t * SDARP_TIME_SCALE)

    return SDARP_Data(
        travel_time=map_values(round_time, data.travel_time),
        demand=map_values(round_demand, data.demand),
        tw_start=map_values(round_time, data.tw_start),
        tw_end=map_values(round_time, data.tw_end),
        service_time=map_values(round_time, data.service_time),
        max_ride_time=map_values(round_time, data.max_ride_time),
        capacity=round_demand(data.capacity),
        n=data.n,
        o_depot=0,
        d_depot=2 * data.n + 1,
        P=data.P,
        D=data.D,
        K=data.K,
        N=data.N,
        id=data.id,
    )


def get_named_instance_PDPTWLH(name, rehandling_cost=0) -> PDPTWLH_Data:
    problem_group = name[0]
    id_str = name
    if name.endswith('s'):
        name = name[:-1]
        problem_group += '*'

    if problem_group == 'A*':
        vehicle_cap = 26
        delay_amount = 60
    elif problem_group == "A":
        vehicle_cap = 22
        delay_amount = 45
    elif problem_group == 'B*':
        vehicle_cap = 35
        delay_amount = 60
    elif problem_group == 'B':
        vehicle_cap = 30
        delay_amount = 45
    elif problem_group == 'C':
        vehicle_cap = 18
        delay_amount = 15
    elif problem_group == 'D':
        vehicle_cap = 25
        delay_amount = 15
    else:
        raise Exception

    raw = parse_format_cordeau(resolve_name_cordeau_PDPTW(name))
    data = build_PDPTWLH_from_cordeau(raw, 10000, rehandling_cost, name)
    data = modify.delay_delivery_windows(data, delay_amount)
    return dataclasses.replace(data, id=id_str, Q=vehicle_cap)


def get_named_instance_MSPRP(name: str) -> MSPRP_Data:
    rawdata = parse_format_schrotenboer(*resolve_name_schrotenboer(name))
    return build_MSPRP_from_schrotenboer(rawdata, name)


def get_named_instance_DARP(name: str) -> DARP_Data:
    rielder = False
    try:
        filename = resolve_name_cordeau_DARP(name)
    except ValueError:
        rielder = True
    if not rielder:
        raw = parse_format_cordeau(filename)
    else:
        filename = resolve_name_riedler(name)
        raw = parse_format_riedler(filename)
    data = build_DARP_from_cordeau(raw, name)
    return data


def get_named_instance_SDARP(name: str) -> SDARP_Data:
    filename = resolve_name_riedler(name)
    raw = parse_format_riedler(filename)
    data = build_DARP_from_cordeau(raw, name)
    return convert_darp_to_sdarp(data)


def get_named_instance_skeleton_SDARP(name: str) -> SDARP_Skeleton_Data:
    match = re.match(r"(?P<n>\d+)N_(?P<k>\d+)K_[ABC]", name)
    assert match is not None
    match = match.groupdict()
    n = int(match['n'])
    k = int(match['k'])
    data = SDARP_Skeleton_Data(
        id=name,
        n=n,
        o_depot=0,
        d_depot=2 * n + 1,
        P=range(1, n + 1),
        D=range(n + 1, 2 * n + 1),
        N=range(2 * n + 2),
        K=range(k)
    )
    return data


def get_named_instance_ITSRSP(name: str, group_with_compat=False) -> ITSRSP_Data:
    raw = parse_format_hemmati_hdf5(resolve_name_hemmati_hdf5(name))
    if group_with_compat:
        data = build_ITSRSP_from_hemmati(raw, name)
    else:
        data = build_ITSRSP_from_hemmati_aggressive_grouping(raw, name)

    return data


def get_named_instance_skeleton_ITSRSP(name: str) -> ITSRSP_Skeleton_Data:
    raw = parse_format_hemmati_hdf5_to_skeleton(resolve_name_hemmati_hdf5(name))
    return dataclasses.replace(raw, id=name)


def get_index_file(dataset: str, **kwargs) -> Path:
    data_subdir = {
        'itsrsp': "ITSRSP_hdf5_ti",
        'darp': "DARP_cordeau",
        'sdarp': "SDARP_riedler",
    }

    if dataset not in data_subdir:
        raise ValueError(f"no known index file for `{dataset!s}`")

    indexfile = data_directory(data_subdir[dataset]) / "INDEX.txt"
    return indexfile


_NAME_TO_INDEX = {}
_INDEX_TO_NAME = {}


def _ensure_index_name_map(dataset):
    global _NAME_TO_INDEX, _INDEX_TO_NAME
    if dataset in _NAME_TO_INDEX:
        return
    with open(get_index_file(dataset), 'r') as f:
        m = dict(enumerate(map(lambda s: s.strip(), f)))
    if dataset == "sdarp":
        _ensure_index_name_map("sdarp")
        offset = len(m)
        m.update({i + offset: n for i, n in _INDEX_TO_NAME['sdarp'].items()})
    m_inv = {v: k for k, v in m.items()}
    _NAME_TO_INDEX[dataset] = m_inv
    _INDEX_TO_NAME[dataset] = m


def get_name_by_index(dataset, idx: int):
    _ensure_index_name_map(dataset)
    return _INDEX_TO_NAME[dataset][idx]


def get_index_by_name(dataset, name: str):
    _ensure_index_name_map(dataset)
    return _NAME_TO_INDEX[dataset][name]
