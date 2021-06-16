from . import constants as _C
from typing import Dict, Iterable, Callable, Any, Union, Mapping, TypeVar
import time
import dataclasses
from collections import defaultdict, OrderedDict
import os
import functools

memoise = functools.lru_cache(maxsize=None)

def take(iterable : Iterable):
    """Pick an arbitrary element of ``iterable``."""
    for val in iterable:
        return val
    return None

def get_keys_with_nonzero_val(d : Dict, X_attr=False, eps=_C.EPS):
    if X_attr:
        getval = lambda x: x.X
    else:
        getval = lambda x: x
    return list(sorted(k for k, v in d.items() if abs(getval(v)) > eps))


_K = TypeVar('_K')
_V = TypeVar('_V')
class frozendict(Mapping[_K,_V]):
    """
    An immutable wrapper around dictionaries that implements the complete :py:class:`collections.Mapping`
    interface. It can be used as a drop-in replacement for dictionaries where immutability is desired.
    """
    def __init__(self, *args, **kwargs):
        self._dict = dict(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def copy(self, **add_or_replace):
        return self.__class__(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return f'{self.__class__.__name__}({self._dict!r})'

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self._dict.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash

    # def to_json_dict(self, has_complex_keys=False):
    #     for k in self.keys():
    #         if not (isinstance(k, (str, int, bool, float)) or k is None):
    #             break
    #     else:
    #         return self._dict.copy()
    #
    #     return JSONDictWrapper(self._dict.copy())
    #
    #
    # @classmethod
    # def from_json_dict(cls, data):
    #     if isinstance(data, dict):
    #         new = cls()
    #         new._dict = data
    #         return new
    #     else:
    #         return cls((d['key'], d['val']) for d in data)



class Stopwatch:
    def __init__(self):
        self._start_time = 0
        self._stop_time = 0
        self._total_time = 0
        self._lap_time = 0
        self._active = False
        self._times = OrderedDict()

    @property
    def active(self):
        return self._active

    def start(self):
        if not self._active:
            self._active = True
            self._start_time = time.time()
            self._lap_time = 0
        return self

    def stop(self, label=None):
        if self._active:
            self._active = False
            self._stop_time = time.time()
            self._lap_time = self._stop_time - self._start_time
            self._total_time += self._lap_time
            if label is not None:
                self._times[label] = self._lap_time

        return self

    def lap(self, label):
        if self.active:
            self._stop_time = time.time()
            self._lap_time = self._stop_time - self._start_time
            self._total_time += self._lap_time
            self._times[label] = self._lap_time
            self._start_time = self._stop_time
        return self

    @property
    def time(self):
        if self.active:
            return self._total_time + self.lap_time
        else:
            return self._total_time

    @property
    def lap_time(self):
        if self._active:
            return time.time() - self._start_time
        else:
            return self._lap_time

    @property
    def times(self):
        return self._times

def dict_diff(a: Mapping, b: Mapping, a_id='A', b_id='B'):
    """
    Returns a nested dictionary representing the recursive difference of ``a`` and ``b``.
    """
    d = {}
    for k, va in a.items():
        if k not in b:
            d[k] = f'missing from {b_id}'
        else:
            vb = b[k]
            if type(va) == type(vb):
                if va != vb:
                    if isinstance(va, Mapping):
                        d[k] = dict_diff(va, vb, a_id=a_id, b_id=b_id)
                    else:
                        d[k] = f"value mismatch: ({a_id}) {va!s} != {vb!s} ({b_id})"
            else:
                d[k] = f"value type mismatch: ({a_id}) {type(va).__name__} != {type(vb).__name__!s} ({b_id})"

    for k in b:
        if k not in a:
            d[k] = f'missing from {a_id}'

    return d



@dataclasses.dataclass(eq=False, frozen=True)
class LazyHashFrozenDataclass:
    def __post_init__(self):
        hashval = 0
        for f in dataclasses.fields(self):
            hashval ^= hash((f.name,getattr(self, f.name)))
        self.__dict__['_hash'] = hashval # ugly hack to work around setting a frozen internal variable.

    def __hash__(self):
        return self._hash


@dataclasses.dataclass(frozen=True)
class SerialisableFrozenSlottedDataclass:
    """
    This is a workaround for allow frozen dataclasses with the __slots__ attribute to be pickled.
    See `https://stackoverflow.com/questions/55307017/pickle-a-frozen-dataclass-that-has-slots`_.
    """

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        for slot, value in state.items():
            object.__setattr__(self, slot, value)

def onerange(start,stop=None):
    if stop is None:
        return range(1,start+1)
    return range(start, stop+1)

def group_keys_by_values(d : Dict):
    gd = defaultdict(list)
    for k,v in d.items():
        gd[v].append(k)
    for val, key_group in gd.items():
        yield key_group, val

def tuple_select_items(selection, d : Dict):
    for key,val in d.items():
        if len(key) != len(selection):
            raise ValueError(f"`selection` (len={len(selection)}) does not match length of `key`={str(key)} "
                             f"(len={len(key)})")
        for x, y in zip(selection, key):
            if x != '*' and x != y:
                break
        else:
            yield key, val

def map_keys(func : Callable[[Any], Any], d : Mapping, drop_none = True) -> Dict:
    """
    Return a new dictionary from `d` by applying `func` to all keys.  If `drop_none` is True, then any keys that map to
    None are ignored.
    """
    if drop_none:
        return dict(filter(lambda kv : kv[0] is not None, zip(map(func, d.keys()), d.values())))
    else:
        return dict(zip(map(func, d.keys()), d.values()))


def map_values(func : Callable[[Any], Any], mapping : Mapping):
    """
    Return `mapping` where its func is applied to its values.
    """
    return mapping.__class__(zip(mapping.keys(), map(func, mapping.values())))


def rev_enumerate(iterable, length=None):
    try:
        length = len(iterable)
    except AttributeError:
        if length is None:
            raise Exception("Either the iterable must have __len__, or length must be given.")

    idx = length-1
    for item in reversed(iterable):
        yield idx, item
        idx -= 1



def rec_items(mapping : Mapping, _prefix=()):
    """Returns an iterator to yield key-value pairs from nested dictionaries recursively.  The nested keys from nested
    dictionaries will be given as a tuple.  Eg, if d[0]['a'] = 'x', then the iterator will contain ((0,'a'), 'x').
    Items are returned in depth-first order."""
    for k,v in mapping.items():
        k = _prefix + (k,)
        if isinstance(v, Mapping):
            for item in rec_items(v, _prefix=k):
                yield item
        else:
            yield (k,v)

def rec_keys(mapping : Mapping):
    for k in rec_items(mapping):
        yield k

def rec_values(mapping : Mapping):
    for v in mapping.values():
        if isinstance(v, Mapping):
            for vd in rec_values(v):
                yield vd
        else:
            yield v


def expand_sparse_dict(d : Dict, keyfunc : Callable):
    for kp in list(d.keys()):
        for k in keyfunc(kp):
            d[k] = d[kp]


def filterl(func,iterable):
    return list(filter(func, iterable))



class SurjectiveDict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mapped_keys = dict()  # ptr_key -> key
        self._mapped_keys_inv = defaultdict(set)

    # TODO add tests
    def share_value(self, new_key, existing_key):
        assert existing_key in self
        existing_key = self._mapped_keys.get(existing_key, existing_key)
        if new_key in self._mapped_keys:
            self._mapped_keys_inv[self._mapped_keys[new_key]].remove(new_key)
        elif super().__contains__(new_key):
            super().__delitem__(new_key)

        self._mapped_keys[new_key] = existing_key
        self._mapped_keys_inv[existing_key].add(new_key)

    @property
    def keymap(self):
        return self._mapped_keys.copy()

    @property
    def inverse_keymap(self):
        return dict((k,v) for k,v in self._mapped_keys_inv.items() if len(v) > 0)

    def get_one_to_one(self):
        return super().copy()

    def unique_items(self):
        return super().items()

    def unique_keys(self):
        return super().keys()

    def keys(self):
        for key in super().keys():
            yield key
            for ptr_key in self._mapped_keys_inv[key]:
                yield ptr_key

    def items(self):
        for key, val in super().items():
            yield key,val
            for ptr_key in self._mapped_keys_inv[key]:
                yield ptr_key, val

    def copy(self):
        new = SurjectiveDict(super().copy())
        new._mapped_keys = self._mapped_keys.copy()
        new._mapped_keys_inv = self._mapped_keys_inv.copy()
        return new

    def __len__(self):
        return len(self._mapped_keys) + super().__len__()

    def __getitem__(self, item):
        return super().__getitem__(self._mapped_keys.get(item, item))

    def __contains__(self, item):
        return super().__contains__(self._mapped_keys.get(item, item))

    def __setitem__(self, item, val):
        return super().__setitem__(self._mapped_keys.get(item, item), val)

    def __delitem__(self, key):
        if key in self._mapped_keys:
            del self._mapped_keys[key]
        else:
            if len(self._mapped_keys_inv[key]) > 0:
                # a new key will take over as the parent key
                new_key = self._mapped_keys_inv[key].pop()
                # update mappings
                for k in self._mapped_keys_inv[key]:
                    self._mapped_keys[k] = new_key
                self._mapped_keys_inv[new_key] = self._mapped_keys_inv[key]
                del self._mapped_keys_inv[key]
                del self._mapped_keys[new_key]
                # copy data to new parent
                super().__setitem__(new_key, self[key])
            super().__delitem__(key)

    def __repr__(self):
        return '{ ' + ', '.join(f'{repr(k)} : {repr(v)}' for k,v in self.items()) + ' }'

    def __str__(self):
        return self.__repr__()


class FrozenOrderedSet(frozenset):
    def __init__(self, seq=()):
        super().__init__(seq)
        self._order = tuple(seq)
        self._hash = None

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self._order)
        return self._hash

    def __iter__(self):
        return iter(self._order)

    def __getitem__(self, item):
        return self._order[item]


class FileNameManager:
    def __init__(self, directory='./', prefix='', create_dirs=False):
        if create_dirs:
            os.makedirs(directory, exist_ok=True)
        self.directory = directory
        self.prefix = prefix
        self.paths = set()

    def __call__(self, suffix, prefix=True, ignore_duplicates=False):
        if prefix:
            path = os.path.join(self.directory, self.prefix + suffix)
        else:
            path = os.path.join(self.directory, suffix)
        if not ignore_duplicates and path in self.paths:
            raise Exception(f"{path} may be duplicated.")
        self.paths.add(path)
        return path
