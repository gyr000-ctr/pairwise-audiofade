import math

import pytest

import audiofade

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

S = [0.5] * 5


def _check_fade_start():
    return audiofade.Fader().fade_in(S)[0] == 0.0


def _check_fade_end():
    return audiofade.Fader().fade_in(S)[-1] == 0.5


def _check_fade_out_start():
    return audiofade.Fader().fade_out(S)[0] == 0.5


def _check_fade_out_end():
    return audiofade.Fader().fade_out(S)[-1] == 0.0


def _check_fade_len():
    return len(audiofade.Fader().fade_in(S)) == 5


def _check_fade_monotonic():
    out = audiofade.Fader().fade_in(S)
    return all(out[i] <= out[i + 1] + 1e-9 for i in range(4))


def _check_cross_endpoints():
    out = audiofade.Fader().crossfade(S, S)
    return abs(out[0] - 0.5) < 1e-9 and abs(out[-1] - 0.5) < 1e-9


def _check_cross_len():
    return len(audiofade.Fader().crossfade(S, S)) == 5


def _check_empty():
    return audiofade.Fader().fade_in([]) == []


def _check_bad_len():
    return _raises(lambda: audiofade.Fader().crossfade(S, S[:2]))


def _check_det():
    return audiofade.Fader().fade_in(S) == audiofade.Fader().fade_in(S)


def _check_type():
    return isinstance(audiofade.Fader().fade_in(S), list)


def _check_range():
    return all(0.0 <= v <= 0.5 for v in audiofade.Fader().fade_in(S))


def _check_single():
    return len(audiofade.Fader().fade_in([0.3])) == 1


def _check_no_mutation():
    src = list(S)
    audiofade.Fader().fade_in(S)
    return S == src

CASES = [("fade_start", "_check_fade_start"), ("fade_end", "_check_fade_end"), ("fade_out_start", "_check_fade_out_start"), ("fade_out_end", "_check_fade_out_end"), ("fade_len", "_check_fade_len"), ("fade_monotonic", "_check_fade_monotonic"), ("cross_endpoints", "_check_cross_endpoints"), ("cross_len", "_check_cross_len"), ("empty", "_check_empty"), ("bad_len", "_check_bad_len"), ("det", "_check_det"), ("type", "_check_type"), ("range", "_check_range"), ("single", "_check_single"), ("no_mutation", "_check_no_mutation")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
