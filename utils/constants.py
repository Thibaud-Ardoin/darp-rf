import os as _os
from gurobi import GRB
import colorama
import re as _re
from warnings import warn
from pathlib import Path
import yaml
import sys
import cerberus
import copy

def _parse_slurm_time(string):
    match = _re.match(r'((?P<days>\d+)-)?(?P<hours>\d+):(?P<mins>\d+):(?P<secs>\d+)', string)
    if match is None:
        warn(f'Environment variable SLURM_TIME_LIMIT is invalid (`{string}`); ignoring')
        return GRB.INFINITY
    match = match.groupdict(0)
    match = dict(zip(match.keys(), map(int, match.values())))
    return ((int(match['days'])*24 + match['hours'])*60 + match['mins'])*60 + match['secs']

def _get_cpus() -> int:
    if 'SLURM_CPUS_PER_TASK' in _os.environ:
        return int(_os.environ['SLURM_CPUS_PER_TASK'])
    else:
        import multiprocessing as _mp
        return _mp.cpu_count()

def _load_config():
    schema = {
        "data-root" : {"type" : "string", "required": True},
        "log-root" : {"type" : "string", "required": True},
    }

    config_path = Path(__file__).parent.parent/"config.yaml"
    try:
        with open(config_path, "r") as fp:
            config = yaml.load(fp, yaml.CSafeLoader)
        validator = cerberus.Validator(schema, allow_unknown=False)
        if not validator.validate(config):
            msg = "config errors:\n", "\n".join(f"{k}: {v} " for k, v in validator.errors.items())
            raise ValueError(msg)

        config = copy.deepcopy(validator.document)

        for path_key in ("data-root", "log-root"):
            path = config[path_key]
            if path.startswith("/"):
                config[path_key] = Path(path)
            else:
                config[path_key] = (config_path.parent / Path(path)).resolve()

    except FileNotFoundError:
        print(f"you need to create {config_path.resolve()!s} first", file=sys.stderr)
        raise
    except yaml.YAMLError:
        raise Exception("Invalid YAML format")

    return config

_CONFIG = _load_config()

TTYCOLORS = colorama.Fore
EPS = 1e-5
ITSRSP_TIME_INFINITY = (2 ** 32) - 1
DATA_DIR = _CONFIG["data-root"]
LOGS_DIR = _CONFIG["log-root"]
N_CPUS = _get_cpus()
