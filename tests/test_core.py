import math

import numpy as np
import pytest

from amp_benchkit.dsp import find_knees, thd_fft
from amp_benchkit.fy import FY_MAX_VPP, FYError, build_fy_cmds
from unified_gui_layout import _decode_ieee_block


def test_build_fy_cmds_basic():
    cmds = build_fy_cmds(1000, 1.0, 0.0, "Sine", duty=12.3, ch=1)
    assert any(c.startswith("bw") for c in cmds)
    assert any(c.startswith("bd") and c.endswith("123") for c in cmds)
    assert cmds[1].endswith("000100000")  # 1000 Hz -> 100000 centi-Hz padded


def test_build_fy_cmds_rejects_above_limit():
    with pytest.raises(FYError):
        build_fy_cmds(1000, FY_MAX_VPP + 0.1, 0.0, "Sine", duty=50.0, ch=1)


def test_ieee_block_decode():
    raw = b"#3100" + bytes(range(100)) + b"extra"
    dec = _decode_ieee_block(raw)
    assert len(dec) == 100 and dec[0] == 0 and dec[-1] == 99


def test_thd_fft():
    fs = 50000.0
    f0 = 1000.0
    N = 4096
    t = np.arange(N) / fs
    sig = np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
    thd, f_est, _ = thd_fft(t, sig, f0=f0, nharm=5, window="hann")
    assert abs(thd - 0.1) < 0.03


def test_find_knees():
    freqs = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], dtype=float)
    # Flat then roll off after 1k
    amps = np.array([1, 1, 1, 1, 1, 1, 1, 0.7, 0.4, 0.2], dtype=float)
    f_lo, f_hi, ref_amp, ref_db = find_knees(freqs, amps, ref_mode="max", drop_db=3.0)
    assert math.isfinite(f_hi) and f_hi > 1000
