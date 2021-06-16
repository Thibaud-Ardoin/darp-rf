from oru import frozendict, LazyHashFrozenDataclass, map_values
import dataclasses
from typing import Union, Sequence, FrozenSet, Tuple
from . import utils as _u
from .ipc import MsgPackSerialisableDataclass
import math
frozen_dataclass = dataclasses.dataclass(frozen=True)


@frozen_dataclass
class ProblemDataBase(LazyHashFrozenDataclass):
    id: str


@frozen_dataclass
class DARP_Data(ProblemDataBase):
    travel_time: frozendict  # +idx:IJ
    # service_time : frozendict #+idx:I
    travel_cost: frozendict  # +idx:IJ
    demand: frozendict  # +idx:IJ
    loc: frozendict  # +idx:I
    tw_start: frozendict  # +idx:I
    tw_end: frozendict  # +idx:I

    # Note: the max_ride_time includes the service time, but service_time is needed for some modify.* functionality
    service_time: frozendict  # +idx:I
    max_ride_time: frozendict  # +idx:I
    capacity: int

    n: int
    P: range
    D: range
    K: range
    N: range


@frozen_dataclass
class SDARP_Data(ProblemDataBase):
    travel_time: frozendict  # +idx:IJ
    demand: frozendict  # +idx:IJ
    tw_start: frozendict  # +idx:I
    tw_end: frozendict  # +idx:I

    # Note: the max_ride_time includes the service time, but service_time is needed for some modify.* functionality
    # service_time: frozendict  # +idx:I
    max_ride_time: frozendict  # +idx:I
    capacity: int

    n: int
    o_depot : int
    d_depot : int
    P: range
    D: range
    K: range
    N: range


@frozen_dataclass
class SDARP_Skeleton_Data(ProblemDataBase):
    n: int
    o_depot : int
    d_depot : int
    P: range
    D: range
    K: range
    N: range

@frozen_dataclass
class PDPTW_Data(ProblemDataBase):
    T: frozendict
    C: frozendict
    n: int
    Q: int
    E: frozendict
    L: frozendict
    D: frozendict
    loc: frozendict
    pickup_loc: range
    delivery_loc: range
    vehicle_cost: float


@frozen_dataclass
class PDPTWLH_Data(PDPTW_Data):
    rehandle_cost: float


@frozen_dataclass
class GenMSPRP_Data(ProblemDataBase):
    # indexed by i,l
    servicemen_demand: frozendict
    # indexed by l,t
    servicemen_max: frozendict
    # indexed by k,t
    servicemen_capacity: frozendict
    # indexed by l,t
    servicemen_cost: frozendict
    # indexed by i
    service_time: frozendict
    # indexed by i,j,k,t
    travel_time: frozendict
    # indexed by i,j,k,t
    travel_cost: frozendict
    # indexed by i
    parts_demand: frozendict
    # indexed by k
    parts_capacity: frozendict
    # indexed by k,t
    travel_time_max: frozendict

    nr: int
    Nd: range
    Np: range
    T: range
    L: range
    K: range
    loc: frozendict


@frozen_dataclass
class MSPRP_Data(GenMSPRP_Data):
    # note that travel_cost, travel_time are only indexed by (i,j,k) not (i,j,k,t)
    # likewise servicemen_cost,servicemen_max are indexed only by l, not (l,t)

    # indexed by i,t
    late_penalty: frozendict

    def reduce_size(self, req_keep: Union[int, Sequence]):
        node_map, num_req_keep = _u.get_node_map(req_keep, self.nr)


@frozen_dataclass
class AFVSP_Data(ProblemDataBase):
    T: range
    S: range
    D: range

    num_buses: frozendict

    fuel_capacity: float

    start_time_trip: frozendict
    end_time_trip: frozendict

    cost_trip: frozendict
    fuel_trip: frozendict

    cost_depot_station: frozendict
    cost_depot_trip: frozendict
    cost_station_depot: frozendict
    cost_station_trip: frozendict
    cost_trip_depot: frozendict
    cost_trip_station: frozendict
    cost_trip_trip: frozendict

    fuel_depot_station: frozendict
    fuel_depot_trip: frozendict
    fuel_station_depot: frozendict
    fuel_station_trip: frozendict
    fuel_trip_depot: frozendict
    fuel_trip_station: frozendict
    fuel_trip_trip: frozendict

    time_depot_station: frozendict
    time_depot_trip: frozendict
    time_station_depot: frozendict
    time_station_trip: frozendict
    time_trip_depot: frozendict
    time_trip_station: frozendict
    time_trip_trip: frozendict


@frozen_dataclass
class ITSRSP_Skeleton_Data(ProblemDataBase):
    n: int
    V: range
    P: range
    D: range
    V: range

    o_depots: range  # 2n,...,2n+v-1
    d_depot : int # 2n+v

VehicleGroup=int
Vehicle=int
Loc = int
Time = int
Cost = int
Demand = int

@frozen_dataclass
class ITSRSP_Data(ProblemDataBase, MsgPackSerialisableDataclass):
    n: int
    # num_v : int

    P: range  # 0,...,n-1
    D: range  # n,...,2n-1
    V: range  # 0,...,v-1

    # Note: a pretty fundamental assumption is that the vehicles have no defined end location.  This means that all
    # vehicles can share a single (fake) end loc, which is useful for flow consistency.  Adding backarcs will not solve
    # this problem.
    o_depots: range  # 2n,...,2n+v-1
    d_depot : int # 2n+v

    # vg -> FrozenSet[p]
    P_compatible: frozendict[VehicleGroup, FrozenSet[Loc]]

    # If we group vehicles without keying on their compatibilities, this attribute will contain the customers which are
    # not compatible for a given vehicle.  In this case, P_compatible should be treated has 'maybe compatible', in the
    # sense that at least one vehicle in the group can service any given customer.
    P_incompatible: Union[None, frozendict[Vehicle, FrozenSet[Loc]]]

    # p -> float
    customer_penalty: frozendict[Loc, Cost]

    # vg -> float
    vehicle_capacity: frozendict[VehicleGroup, Demand]

    # i -> int
    demand: frozendict[Loc, Demand]
    tw_start: frozendict[Loc, Time]
    tw_end: frozendict[Loc, Time]

    # vg -> ((i,j) -> int)
    travel_time: frozendict[VehicleGroup,frozendict[Tuple[Loc,Loc], Time]]
    travel_cost: frozendict[VehicleGroup,frozendict[Tuple[Loc,Loc], Cost]]

    # Locations within a port group do not have any travel time or travel cost (not counting service time/cost) between
    # one another.
    # i -> FrozenSet[i]
    port_groups: frozendict[Loc, FrozenSet[Loc]]

    # vg -> FrozenSet[v]
    vehicle_groups: frozendict[VehicleGroup, FrozenSet[Vehicle]]
    # v -> vg
    group_by_vehicle: frozendict[Vehicle, VehicleGroup]

    # vg,p -> v
    char_vehicle : frozendict[Tuple[VehicleGroup, Loc], Vehicle]

    @classmethod
    def from_msgpack(cls, data):
        for attribute in ('P', 'D', 'V'):
            data[attribute] = range(*data[attribute])
        for attribute in ('o_depots', 'd_depots'):
            data[attribute] = range(*data[attribute], -1)
        data['P_compatible'] = frozendict({k: frozenset(v) for k, v in data['P_compatible'].items()})

        return super().from_msgpack(data)


@frozen_dataclass
class RawDataBase(LazyHashFrozenDataclass):
    pass


@frozen_dataclass
class RawDataCordeau(RawDataBase):
    num_requests: int
    num_vehicles: int
    max_route_duration: int
    max_ride_time: int
    vehicle_cap: int
    pos_x: frozendict
    pos_y: frozendict
    tw_start: frozendict
    tw_end: frozendict
    demand: frozendict
    service_time: frozendict


@frozen_dataclass
class RawDataSchrotenboer(RawDataBase):
    num_requests: int
    num_time_periods: int
    num_vehicles: int
    num_servicemen_types: int
    servicemen_cost: frozendict
    servicemen_avail: frozendict
    vehicle_servicemen_cap: frozendict
    vehicle_parts_cap: frozendict
    vehicle_speed: frozendict
    vehicle_travel_cost: frozendict
    pos_x: frozendict
    pos_y: frozendict
    servicemen_demand: frozendict
    parts_demand: frozendict
    service_time: frozendict
    periodic_costs: frozendict
    max_travel_time: frozendict
    loading_time: float


@frozen_dataclass
class RawDataHemmati(RawDataBase):
    num_ports: int
    num_vessels: int
    num_cargos: int
    # cargo indexing starts at 0
    # vessel indexing starts at 0
    # port indexing starts at 0

    # vessel -> int
    vessel_start_time: frozendict
    vessel_capacity: frozendict

    # vessel -> port
    vessel_start_port: frozendict

    # vehicle -> Set[cargo]
    vessel_compatible: frozendict

    # cargo -> int
    cargo_penalty: frozendict
    cargo_size: frozendict

    # cargo -> port
    cargo_origin: frozendict
    cargo_dest: frozendict

    # cargo -> int
    cargo_origin_tw_start: frozendict
    cargo_origin_tw_end: frozendict
    cargo_dest_tw_start: frozendict
    cargo_dest_tw_end: frozendict

    # vessel,cargo -> int
    cargo_origin_port_cost: frozendict
    cargo_origin_port_time: frozendict
    cargo_dest_port_cost: frozendict
    cargo_dest_port_time: frozendict

    # vessel,i,j -> int
    travel_time: frozendict
    travel_cost: frozendict
