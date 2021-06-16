import sys
import io

def setup_sigpipe():
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

def eprint(*values, sep=' ', end='\n'):
    print(*values, sep=sep, end=end, file=sys.stderr)


class _StdIOContext:
    def __init__(self, io):
        self.io = io

    def __enter__(self):
        return self.io

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def open_default_stdin(filename='-', mode='r'):
    if filename == '-':
        if 'b' in mode:
            stdin =  sys.stdin.buffer
        else:
            stdin =  sys.stdin
        return _StdIOContext(stdin)
    return open(filename, mode=mode)

def open_default_stdout(filename='-', mode='w'):
    if filename == '-':
        if 'b' in mode:
            stdout = sys.stdout.buffer
        else:
            stdout = sys.stdout
        return _StdIOContext(stdout)
    return open(filename, mode=mode)
