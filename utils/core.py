from .constants import *
import pickle
import colorama
import oru.grb
import os
import contextlib

class NullFile:
    def write(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass


class BaseModel(oru.grb.BaseGurobiModel):
    cpus : None

    def __init__(self, *args,cpus=N_CPUS, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpus = cpus
        with contextlib.redirect_stdout(NullFile()):
            self.setParam("Threads", self.cpus)


def colortext(string, color):
    if color is None:
        return string
    else:
        return color + string + colorama.Fore.RESET


def pickle_to_file(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def unpickle_from_file(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj
