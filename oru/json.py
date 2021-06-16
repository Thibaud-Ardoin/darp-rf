import json
import dataclasses
import lzma
from typing import TextIO, Dict
from pathlib import Path
import re


@dataclasses.dataclass
class JSONSerialisableDataclass:
    def to_json_file(self, filename, compress=False):
        if compress:
            fp = lzma.open(filename, 'wt')
        else:
            fp = open(filename, 'w')

        json.dump(dataclasses.asdict(self), fp)
        fp.close()

    @classmethod
    def from_json_file(cls, filename):
        with open(filename, 'r') as f:
            d = json.load(f)

        return cls(**d)


def _try_load_json_file(p : Path):
    try:
        p = p.resolve()
    except ValueError as e:
        return (None, f"`{p}` doesn't look like a valid filepath: {e!s}")

    try:
        with open(p, 'r') as fp:
            return (json.load(fp), None)
    except Exception as e:
        return (None, f"Unable to read `{p}`: {e!s}")

DEFAULT_RESOLVE_PATTERN = re.compile('\.json$')

def resolve_files(val, rootpath : Path, regexp : re.Pattern = DEFAULT_RESOLVE_PATTERN, logger=None) -> dict:
    if isinstance(val, dict):
        for k,v in val.items():
            val[k] = resolve_files(v, rootpath, regexp, logger)

    elif isinstance(val, list):
        val = [resolve_files(v, rootpath, regexp, logger) for v in val]

    elif isinstance(val, str) and regexp.search(val) is not None:
        newval, reason = _try_load_json_file(rootpath / val)

        if newval is not None:
            val = resolve_files(newval, rootpath, regexp, logger)
        elif logger is not None:
            logger(reason)

    return val
