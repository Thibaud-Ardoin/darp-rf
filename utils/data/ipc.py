"""
Binary serialisation/deserialisation functions for interprocess communation
"""
import msgpack
import dataclasses
from oru import frozendict
import dacite

def _dict_to_frozendict_recurse(d):
    for k,v in d.items():
        if isinstance(v, dict):
            d[k] = _dict_to_frozendict_recurse(v)
    return frozendict(d)

@dataclasses.dataclass(frozen=True)
class MsgPackSerialisableDataclass:
    __slots__ = ()
    def to_msgpack(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_msgpack(cls, data):
        config = dacite.Config(type_hooks={
            frozendict : _dict_to_frozendict_recurse
        })
        return dacite.from_dict(cls, data, config=config)

def _msgpack_convert_types(x):
    if isinstance(x, MsgPackSerialisableDataclass):
        return x.to_msgpack()
    if isinstance(x, frozenset):
        return list(x)
    elif isinstance(x, frozendict):
        return x._dict
    elif isinstance(x, range):
        assert abs(x.step) == 1
        return (x.start, x.stop)
    raise TypeError(f"unsupported type conversion: {type(x)}")

MSGPACK_SERIALISATION_OPTS = {
    'default' : _msgpack_convert_types
}

MSGPACK_DERIALISATION_OPTS = {
    'use_list' : False,
    'strict_map_key' : False
}
