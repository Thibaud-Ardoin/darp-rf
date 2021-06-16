import subprocess
from pathlib import Path

def unpack_file(file : Path):
    if not file.exists():
        subprocess.run(['xz', '-kd', file.parent/(file.name + '.xz')], check=True)
    assert file.exists()
    return file

def unpack_dir(dirname : Path):
    if not dirname.exists():
        subprocess.run(['tar', 'Jxf', dirname.parent/(dirname.name + '.tar.xz')], check=True)
    assert dirname.exists()
    return dirname
