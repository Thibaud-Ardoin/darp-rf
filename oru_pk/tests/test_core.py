from oru.core import *
from pytest import approx

def test_take():
    assert take({1}) == 1
    assert take([]) == None
    assert take([0,1,2]) == 0


def test_nonzero_key_get():
    d = {'a' : 0, 'b' : 1, 'c' : 1e-6}
    assert get_keys_with_nonzero_val(d,eps=1e-9) == ['b','c']
    assert get_keys_with_nonzero_val(d,eps=1e-4) == ['b']

def tapprox(val, **kwargs):
    kwargs.setdefault("abs", 0.0005)
    return approx(val, **kwargs)

def test_stopwatch():
    sw = Stopwatch()
    time.sleep(.01)
    assert sw.time == 0
    sw.start()
    assert sw.active
    time.sleep(.01)
    a = sw.stop().time
    assert a == tapprox(0.01)
    time.sleep(.01)
    b = sw.stop().time
    assert not sw.active
    assert a == b
    sw.start()
    time.sleep(.01)
    c = sw.stop().time
    assert c >= b
    assert c == tapprox(0.02)

def test_stopwatch_named_laps():
    sw = Stopwatch().start()
    time.sleep(.01) # ACTIVE
    sw.stop('egg-time')
    sw.start()
    time.sleep(.01)  # ACTIVE
    sw.stop('bacon-time')
    sw.stop('bacon-time2')
    sw.start()
    time.sleep(.02)  # ACTIVE
    sw.stop('egg-time')
    sw.start()
    time.sleep(.01)  # ACTIVE
    sw.lap('dog')
    time.sleep(.01)  # ACTIVE
    sw.lap('lemon')
    sw.stop()
    sw.start()  # ACTIVE
    time.sleep(.03)
    sw.stop('cat')

    t = sw.time
    times = sw.times
    assert set(times.keys()) == {"egg-time", "bacon-time", "dog", "cat", "lemon"}
    assert t == approx(.09, abs=0.0005)
    assert times['egg-time'] == tapprox(.02)
    assert times['bacon-time'] == tapprox(.01)
    assert times['dog'] == tapprox(.01)
    assert times['lemon'] == tapprox(.01)
    assert times['cat'] == tapprox(.03)
