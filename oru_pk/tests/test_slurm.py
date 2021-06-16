from oru.slurm import *

def test_array_range():
    assert array_range("0-4") == [0,1,2,3,4]
    assert array_range("1,2,1-5:3") == [1,2,4]
    assert array_range("1,,1-21:10") == [1,11,21]

