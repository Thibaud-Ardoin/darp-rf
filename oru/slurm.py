import argparse
import json
import re
import sys
import cerberus
import hashlib
from pathlib import Path
from typing import Dict, Tuple
from .logging import TablePrinter
import copy
from functools import lru_cache
import dataclasses

class SLURM_INFO:
    TIME = 'time'
    JOB_NAME = 'job-name'
    MAIL_USER = 'mail-user'
    MAIL_TYPE = 'mail-type'
    NODES = 'nodes'
    MEMORY = 'mem'
    CPUS_PER_TASK = 'cpus-per-task'
    CONSTRAINT = 'constraint'
    SCRIPT = 'script'
    LOG_OUT = 'out'
    LOG_ERR = 'err'

SLURM_INFO_OPTIONAL_FIELDS = (
    SLURM_INFO.JOB_NAME,
    SLURM_INFO.TIME,
    SLURM_INFO.MAIL_USER,
    SLURM_INFO.MAIL_TYPE,
    SLURM_INFO.NODES,
    SLURM_INFO.MEMORY,
    SLURM_INFO.CPUS_PER_TASK,
    SLURM_INFO.CONSTRAINT,
)
SLURM_INFO_REQUIRED_FIELDS = (
    SLURM_INFO.SCRIPT,
    SLURM_INFO.LOG_OUT,
    SLURM_INFO.LOG_ERR, # TODO: make optional (need to adjust database)
)

ARGPARSE_SLURMINFO_ARGS = {
    "slurminfo": (("--slurminfo",),
                  {'action': 'store_true',
                   'help': "print SLURM information as a JSON string and exit."}),
    "p_slurminfo": (("--p-slurminfo",),
                    {'type': int,
                     'metavar': 'FD',
                     'nargs': 2,
                     'default': None,
                     'help': "Start pipe slurm-info server, reading input from the pipe with file descriptor FD"
                             "and writing to STDOUT",
                     }),
}


def slurm_format_time(seconds: int) -> str:
    seconds = int(seconds)
    minutes = seconds // 60
    seconds -= minutes * 60
    hours = minutes // 60
    minutes -= hours * 60
    days = hours // 24
    hours -= days * 24
    return f'{days:d}-{hours:02d}:{minutes:02d}:{seconds:02d}'

def slurm_parse_time(s : str) -> int:
    m = re.fullmatch(r'((?P<d>\d+)-)?(?P<h>\d\d):(?P<m>\d\d):(?P<s>\d\d)', s)
    if m is None:
        raise ValueError("Bad time format: should be [d-]hh:mm:ss")
    m = {k : int(v) for k,v in m.groupdict(0).items()}
    return 60*(60*(24*m['d'] + m['h']) + m['m']) + m['s']


def parse_slurm_info(jsonstr):
    info = json.loads(jsonstr)
    info_fields = set(info.keys())
    missing_req_fields= set(SLURM_INFO_REQUIRED_FIELDS) - info_fields
    if len(missing_req_fields) > 0:
        raise KeyError(f"Missing required JSON fields:\n\t" + "\n\t".join(missing_req_fields))
    unknown_fields = info_fields - set(SLURM_INFO_REQUIRED_FIELDS + SLURM_INFO_OPTIONAL_FIELDS)
    if len(unknown_fields) > 0:
        raise KeyError(f"Unknown JSON fields:\n\t" + "\n\t".join(unknown_fields))
    # non_slurm_info = {key : info.pop(key) for key in ["script"]}
    return info


def array_range(str):
    tokens = list(filter(lambda x : len(x) > 0, str.split(",")))
    indices = set()
    for t in tokens:
        m =re.fullmatch(r"(?P<start>\d+)-(?P<stop>\d+)(:(?P<step>\d+))?", t)
        if m is None:
            indices.add(int(t))
        else:
            groups = m.groupdict(default=1)
            indices.update(range(int(groups['start']), int(groups['stop'])+1, int(groups['step'])))
    return sorted(indices)


def _validate_and_raise(validator, document, schema=None):
    valid = validator.validate(document, schema)
    if not valid:
        raise ValueError(validator.errors)
    return validator.document


def _strip_custom_keywords(d, keep_derived_fields = True):
    custom_keywords = ('derived', 'help')
    s = copy.deepcopy(d)
    for key in list(s.keys()):
        if not keep_derived_fields and s[key].get('derived', False):
            del s[key]
        else:
            for k in custom_keywords:
                try:
                    del s[key][k]
                except KeyError:
                    pass
    return s

_CERBERBUS_TYPE_TO_PYTHON_TYPE = {
 "integer" : int,
 "float" : float,
 "boolean" : bool,
 "string" : str,
 "number" : float
}

class ErrorCatchingArgumentParser(argparse.ArgumentParser):
    def exit(self, status=0, message=None):
        if status:
            raise Exception(message)
        exit(status)

@lru_cache(None)
def _python_interpreter_arg_parser():
    flags = list('bBdEhiIOqsSuvVx')
    flags.append('OO')
    p = ErrorCatchingArgumentParser(add_help=False)
    for f in flags:
        p.add_argument(f'-{f}', action='store_true')
    p.add_argument('-W', nargs=1)
    p.add_argument('-X', nargs=1)
    p.add_argument('--check-hash-based-pycs', choices=('always', 'default', 'never'))
    p.add_argument('python')
    p.add_argument('file')
    p.add_argument('args', nargs=argparse.REMAINDER)
    return p

def strip_python_args(argv):
    p = _python_interpreter_arg_parser()
    try:
        return p.parse_args(argv).args
    except Exception as e:
        raise ValueError(f"unable to strip python args: {e!s}")

def build_help_message(name, rules, arg_class : str):
    arg_class = arg_class.upper()

    if 'min' in rules and 'max' in rules:
        domain_help =   "Must be between {} and {}, inclusive".format(str(rules['min']), str(rules['max']))
    elif 'min' in rules:
        domain_help =   "Must be at least {}".format(str(rules['min']))
    elif 'max' in rules:
        domain_help =   "Must be at most {}".format(str(rules['max']))
    else:
        domain_help = ""

    if rules.get('type') == 'boolean':
        if rules.get('default', False):
            arg_class_description = f'{arg_class}: Disable {name}'
        else:
            arg_class_description = f'{arg_class}: Enable {name}'
    else:
        if 'default' in rules:
            if len(domain_help) > 0:
                domain_help += " (default is %(default)s)"
            else:
                domain_help = "Default is %(default)s"

        arg_class_description = f'{arg_class}: {name}'


    help_msg = [arg_class_description, rules.get('help', ''), domain_help]
    help_msg = '. '.join(filter(lambda s : len(s) > 0, help_msg)) + '.'

    return help_msg

class ExperimentValidator(cerberus.Validator):
    types_mapping = cerberus.Validator.types_mapping.copy()
    types_mapping['path'] = cerberus.TypeDefinition('path', (Path,), ())


class Experiment:
    INPUTS = {"index" : {"type" : "integer", "min" : 0}}
    OUTPUTS = None
    PARAMETERS = {}
    ROOT_PATH = Path.cwd()
    PATH_SEP = "_"

    def __init__(self, inputs, outputs, parameters=None):
        if parameters is None:
            parameters = dict()

        v = ExperimentValidator(require_all=True)

        self.inputs = _validate_and_raise(v, inputs, _strip_custom_keywords(self.INPUTS, False))
        self.outputs = _validate_and_raise(v, outputs, _strip_custom_keywords(self.OUTPUTS, False))
        self.parameters = _validate_and_raise(v, parameters, _strip_custom_keywords(self.PARAMETERS, False))
        self._directory = None
        self._parameter_string = None

        self.define_derived()

        self.inputs = _validate_and_raise(v, self.inputs, _strip_custom_keywords(self.INPUTS, True))
        self.outputs = _validate_and_raise(v, self.outputs,  _strip_custom_keywords(self.OUTPUTS, True))
        self.parameters = _validate_and_raise(v, self.parameters,  _strip_custom_keywords(self.PARAMETERS, True))


    @classmethod
    def from_cl_args(cls, args=None):
        """Alternate constructor."""
        p = argparse.ArgumentParser()
        cl_args = cls.get_parser_arguments()
        for pargs, pkwargs in cl_args.values():
            p.add_argument(*pargs, **pkwargs)

        if args is None:
            args = sys.argv[1:]

        inputs = vars(p.parse_args(args))
        paramsfile = inputs.pop("load_params")
        slurminfo = inputs.pop("slurminfo")
        pipe_slurminfo = inputs.pop('p_slurminfo')

        def convert_types_json(x):
            if isinstance(x, Path):
                return str(x.absolute())
            return x

        if pipe_slurminfo is not None:
            pr, pw = pipe_slurminfo
            with open(pr, 'r') as fp:
                args_list = json.load(fp)

            slurm_info_list = []
            for argv in args_list:
                argv = strip_python_args(argv)
                exp = cls.from_cl_args(argv)
                slurm_info_list.append(exp.get_slurminfo())

            with open(pw, 'w') as fp:
                json.dump(slurm_info_list, fp, default=convert_types_json)

            sys.exit(0)

        parameters = {pname : inputs.pop(pname) for pname in cls.PARAMETERS}
        if paramsfile is not None:
            with open(paramsfile, 'r') as fp:
                parameters = json.load(fp)
        experiment = cls(inputs, {}, parameters)

        if slurminfo:
            print(json.dumps(experiment.get_slurminfo(), default=convert_types_json, indent='\t'))
            sys.exit(0)

        return experiment

    def define_derived(self):
        """
        Any inputs/outputs/parameters with the `derived` rule set must be defined here.
        """
        pass

    @property
    def parameter_string(self):
        """A string which should be filesystem friendly and unique for unique parameters. Note this is
        technically not unique, but the chances of a hash collision is about 10**10 times smaller than me winning lotto"""
        if self._parameter_string is None:
            s =  json.dumps(self.parameters, sort_keys=True)
            self._parameter_string = hashlib.blake2b(s.encode(), digest_size=8).hexdigest()
        return self._parameter_string

    @property
    def input_string(self):
        """A string which should be filesystem friendly and unique for unique inputs."""
        return self.PATH_SEP.join(str(self.inputs[iname]) for iname in sorted(self.inputs.keys()))

    @property
    def directory(self) -> Path:
        """Directory in which experiment results shall be placed."""
        if self._directory is None:
            self._directory = self.ROOT_PATH / self.parameter_string
            self._directory.mkdir(parents=True, exist_ok=True)
            with open(self._directory/ "parameters.json", 'w') as fp:
                json.dump(self.parameters, fp, indent='\t')
        return self._directory

    def get_output_path(self, suffix):
        """Output files should be created using a file path obtained from this method."""
        if not suffix.startswith('.'):
            suffix = self.PATH_SEP + suffix
        return self.directory/(self.input_string + suffix)

    @classmethod
    def get_parser_arguments(cls) -> Dict[str, Tuple[Tuple, Dict]]:
        """Using the non-derived inputs and parameters, create an ArgumentParser to parse these from the command line.
        This should return a dictionary mapping an input/parameter name to args and kwargs, which will be
         to ArgumentParser.add_argument (see argparse docs)"""
        cl_arguments =  {
            "load_params" : (("--load-params",),
                             {'type' : str,
                              'help' : "Load parameters from a JSON file.  All parameter-related arguments are ignored.",
                              "default" : None,
                              'metavar' : "JSONFILE"
                            }),
            **ARGPARSE_SLURMINFO_ARGS
        }

        def create_argp_args(name, rulesdict, arg_class):
            assert rulesdict.get("type", None) != "boolean" or "default" in rulesdict, "Boolean types must have a default"
            kwargs = {}
            if arg_class == "parameter" or "default" in rulesdict:
                argname = "--" + name.replace("_", "-")
                if "default" in rulesdict:
                    kwargs['default'] = rulesdict['default']
                    kwargs['required'] = False
                else:
                    kwargs['required'] = True
                kwargs["dest"] = name
            elif arg_class == "input":
                argname = name
            else:
                raise ValueError("arg_class must be `input` or `parameter")

            argtype = rules.get('type', None)
            if argtype == "boolean":
                if kwargs.pop('default', False):
                    argname = "--no-" + argname.lstrip('-')
                    kwargs = {"dest": name}
                    kwargs['action'] = 'store_false'
                else:
                    kwargs['action'] = 'store_true'

            elif argtype in _CERBERBUS_TYPE_TO_PYTHON_TYPE:
                kwargs['type'] = _CERBERBUS_TYPE_TO_PYTHON_TYPE[argtype]

            kwargs['help'] = build_help_message(name, rules, arg_class)

            return ((argname,), kwargs)


        for name, rules in cls.INPUTS.items():
            if rules.get('derived', False):
                continue
            cl_arguments[name] = create_argp_args(name, rules, "input")

        for name, rules in cls.PARAMETERS.items():
            if rules.get('derived', False):
                continue
            cl_arguments[name] = create_argp_args(name, rules, "parameter")

        return cl_arguments

    @property
    def resource_nodes(self) -> str:
        return "1"

    @property
    def resource_mail_user(self) -> str:
        return None

    @property
    def resource_mail_type(self) -> str:
        return "FAIL"

    @property
    def resource_memory(self) -> str:
        """Should return memory to be allocated in MB"""
        return None

    @property
    def resource_time(self) -> str:
        """Should return time to be allocated in seconds"""
        return None

    @property
    def resource_cpus(self) -> str:
        """Should return time to be allocated in seconds"""
        return None

    @property
    def resource_stdout_log(self) -> str:
        return self.get_output_path('.out')

    @property
    def resource_stderr_log(self) -> str:
        return self.get_output_path('.err')

    @property
    def resource_name(self) -> str:
        return '%j-{index:d}'.format_map(self.inputs)

    @property
    def resource_constraints(self) -> str:
        return None

    @property
    def resource_slurm_script(self):
        raise NotImplementedError

    def get_slurminfo(self):
        cl_opts = {
            "time" : self.resource_time,
            "job-name" : self.resource_name,
            "out" : self.resource_stdout_log,
            "err" : self.resource_stderr_log,
            "mem" : self.resource_memory,
            "cpus-per-task" : self.resource_cpus,
            "nodes" : self.resource_nodes,
            "mail-user" : self.resource_mail_user,
            "mail-type" : self.resource_mail_type,
            "constraint" : self.resource_constraints,
            "script" : self.resource_slurm_script,
        }
        assert set(cl_opts.keys()) == set(SLURM_INFO_OPTIONAL_FIELDS + SLURM_INFO_REQUIRED_FIELDS)
        cl_opts = dict(filter(lambda kv : kv[1] is not None, cl_opts.items()))
        if "mail-user" not in cl_opts and "mail-type" in cl_opts:
            del cl_opts['mail-type']

        return cl_opts

    @staticmethod
    def format_time(seconds : int) -> str:
        import warnings
        warnings.warn("format_time() method is deprecated, use the slurm_format_time() function instead.",
                      DeprecationWarning, stacklevel=2)
        return slurm_format_time(seconds)

    def print_summary_table(self,col_widths=(30, 50), justify=(">", "<")):
        output = TablePrinter(["", ""], print_header=False, col_widths=col_widths, justify=justify)
        output.print_hline()
        output.print_line("Input", "Value")
        output.print_hline()
        for k in sorted(self.inputs):
            output.print_line(k, self.inputs[k])
        output.print_hline()
        output.print_line("Parameter", "Value")
        output.print_hline()
        for k in sorted(self.parameters):
            output.print_line(k, self.parameters[k])
        output.print_hline()
        output.print_line("Output", "Value")
        output.print_hline()
        for k in sorted(self.outputs):
            output.print_line(k, self.outputs[k])
        output.print_hline()

class PipeSlurmInfoAction(argparse.Action):
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        super().__init__(option_strings, dest, nargs, const, default, type, choices, required, help, metavar)

    def __call__(self, parser, ns, args, _=None):
        pr, pw = args
        pr = int(pr)
        pw = int(pw)
        with open(pr, 'r') as fp:
            args_list = json.load(fp)

        slurm_info_list = []
        for argv in args_list:
            argv = strip_python_args(argv)
            slurm_info = ns.get_slurm_info(parser.parse_args(argv)).to_json_dict()
            slurm_info_list.append(slurm_info)

        with open(pw, 'w') as fp:
            json.dump(slurm_info_list, fp)

        parser.exit(0)


# Simpler, functional API
@dataclasses.dataclass(frozen=True)
class SlurmInfo:
    script : str
    log_err : str
    log_out : str
    job_name : str = None
    time : str = None
    mail_user : str = None
    mail_type : str= None
    nodes : int = None
    memory : str = None
    cpus : int = None
    constraint : str = None

    def to_json_dict(self):
        d = dataclasses.asdict(self)
        name_map = {'log_out' : 'out', 'log_err' : 'err', 'cpus' : 'cpus-per-task', 'memory' : 'mem'}
        d['time'] = slurm_format_time(d['time'])
        return {name_map.get(k, k.replace('_', '-')) : str(v) for k,v in d.items() if v is not None}


def create_parser(parser : argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser()
    for argname, (args, kwargs) in ARGPARSE_SLURMINFO_ARGS.items():
        if argname == 'p_slurminfo':
            kwargs['action'] = PipeSlurmInfoAction
        parser.add_argument(*args, dest=argname, **kwargs)
    return parser

def parse_args(parser : argparse.ArgumentParser, get_slurm_info, argv=None):
    ns = argparse.Namespace(get_slurm_info=get_slurm_info)
    args = parser.parse_args(argv, namespace=ns)
    delattr(ns, 'get_slurm_info')
    return args
