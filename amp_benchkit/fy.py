"""FY3200S function generator helpers.

Contains command building logic and convenience wrappers for applying settings
and configuring sweeps. Hardware access requires pyserial.
"""

from __future__ import annotations

import logging
import math
import time

from .deps import (
    HAVE_SERIAL,
    INSTALL_HINTS,
    _serial,
    find_fy_port,  # reuse existing port finder
)

FY_BAUD_EOLS = [(9600, "\n"), (115200, "\r\n")]
FY_PROTOCOLS = ["FY ASCII 9600", "Auto (115200/CRLF→9600/LF)"]
WAVE_CODE = {"Sine": "0", "Square": "1", "Pulse": "2", "Triangle": "3"}
SWEEP_MODE = {"Linear": "0", "Log": "1"}
FY_MAX_VPP = 1.35


class FYError(Exception):
    """Base exception for FY generator operations."""


class FYTimeoutError(FYError):
    """Raised when communication with FY times out."""


log = logging.getLogger("amp_benchkit.fy")

__all__ = [
    "FY_BAUD_EOLS",
    "FY_PROTOCOLS",
    "WAVE_CODE",
    "SWEEP_MODE",
    "FY_MAX_VPP",
    "fy_apply",
    "fy_sweep",
    "build_fy_cmds",
    "check_amp_vpp",
    "FYError",
    "FYTimeoutError",
]


def check_amp_vpp(amp_vpp: float, *, allow_zero: bool = True) -> float:
    """Ensure requested FY amplitude stays within safety limits."""

    amp = float(amp_vpp)
    if not math.isfinite(amp):
        raise FYError("FY amplitude must be a finite number.")
    if amp < 0:
        raise FYError("FY amplitude must be >= 0 Vpp.")
    if not allow_zero and amp <= 0:
        raise FYError("FY amplitude must be > 0 Vpp.")
    if amp > FY_MAX_VPP + 1e-9:
        raise FYError(f"FY amplitude {amp:.3f} Vpp exceeds safety limit of {FY_MAX_VPP:.2f} Vpp.")
    return amp


def build_fy_cmds(freq_hz, amp_vpp, off_v, wave, duty=None, ch=1):
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def step(value: float, step_size: float) -> float:
        if step_size <= 0:
            return value
        return round(round(value / step_size) * step_size, 10)

    pref = "b" if ch == 1 else "d"
    cmds = [
        f"{pref}w{WAVE_CODE.get(wave, '0')}",
        f"{pref}f{int(round(float(freq_hz) * 100)):09d}",
        f"{pref}o{step(float(off_v), 0.01):0.2f}",
    ]
    if duty is not None:
        dp = int(round(clamp(step(float(duty), 0.1), 0.0, 99.9) * 10))
        cmds.append(f"{pref}d{dp:03d}")
    target_amp = check_amp_vpp(float(amp_vpp))
    cmds.append(f"{pref}a{step(clamp(target_amp, 0.0, FY_MAX_VPP), 0.01):0.2f}")
    for c in cmds:
        if len(c) + 1 > 15:
            raise ValueError("FY command too long: " + c)
    return cmds


def fy_apply(
    freq_hz=1000,
    amp_vpp=0.25,
    wave="Sine",
    off_v=0.0,
    duty=None,
    ch=1,
    port=None,
    proto="FY ASCII 9600",
):
    if not HAVE_SERIAL:
        raise ImportError(f"pyserial not available. {INSTALL_HINTS['pyserial']}")
    port = port or find_fy_port()
    if not port:
        raise RuntimeError("No serial ports found.")
    baud, eol = (9600, "\n") if proto == "FY ASCII 9600" else (115200, "\r\n")
    log.debug(
        "fy_apply(ch=%s, port=%s, proto=%s, freq=%s, amp=%s, off=%s, duty=%s)",
        ch,
        port,
        proto,
        freq_hz,
        amp_vpp,
        off_v,
        duty,
    )
    sent = []
    try:
        with _serial.Serial(port, baudrate=baud, timeout=1) as s:
            for cmd in build_fy_cmds(freq_hz, amp_vpp, off_v, wave, duty, ch):
                log.debug("write %s", cmd)
                sent.append(cmd)
                s.write((cmd + eol).encode())
                time.sleep(0.02)
    except Exception as e:
        last_err = e
        # Try alternate baud/EOL pairs automatically
        for b, e2 in [(115200, "\r\n"), (9600, "\n")]:
            try:
                with _serial.Serial(port, baudrate=b, timeout=1) as s:
                    for cmd in build_fy_cmds(freq_hz, amp_vpp, off_v, wave, duty, ch):
                        log.debug("retry write %s", cmd)
                        sent.append(cmd)
                        s.write((cmd + e2).encode())
                        time.sleep(0.02)
                return sent
            except Exception as e_alt:
                last_err = e_alt
        msg = f"FY write failed on {port}: {last_err}"
        if "timeout" in str(last_err).lower():
            raise FYTimeoutError(msg) from last_err
        raise FYError(msg) from last_err
    return sent


def fy_sweep(port, ch, proto, start=None, end=None, t_s=None, mode=None, run=None, cycles=None):
    if ch not in (1, 2):
        raise ValueError(f"Unsupported FY channel: {ch}")
    baud, eol = (9600, "\n") if proto == "FY ASCII 9600" else (115200, "\r\n")
    log.debug(
        "fy_sweep(ch=%s, port=%s, proto=%s, start=%s, end=%s, t_s=%s, mode=%s, run=%s)",
        ch,
        port,
        proto,
        start,
        end,
        t_s,
        mode,
        run,
    )
    commands = []
    try:
        with _serial.Serial(port, baudrate=baud, timeout=1) as s:
            if ch == 2:
                wants_handshake = any(val is not None for val in (start, end, t_s, mode)) or (
                    run is None or bool(run)
                )
                if wants_handshake:
                    cyc = 1000000 if cycles is None else int(cycles)
                    cmd = f"tn{cyc:07d}"
                    log.debug("write %s", cmd)
                    s.write((cmd + eol).encode())
                    time.sleep(0.02)
                    commands.append(cmd)
                    cmd = "tt2"
                    log.debug("write %s", cmd)
                    s.write((cmd + eol).encode())
                    time.sleep(0.02)
                    commands.append(cmd)
            pref = "b" if ch == 1 else "d"
            if start is not None:
                cmd = f"{pref}b{int(start * 100):09d}"
                log.debug("write %s", cmd)
                s.write((cmd + eol).encode())
                time.sleep(0.02)
                commands.append(cmd)
            if end is not None:
                cmd = f"{pref}e{int(end * 100):09d}"
                log.debug("write %s", cmd)
                s.write((cmd + eol).encode())
                time.sleep(0.02)
                commands.append(cmd)
            if t_s is not None:
                cmd = f"{pref}t{int(t_s):02d}"
                log.debug("write %s", cmd)
                s.write((cmd + eol).encode())
                time.sleep(0.02)
                commands.append(cmd)
            if mode is not None:
                cmd = f"{pref}m{SWEEP_MODE.get(mode, '0')}"
                log.debug("write %s", cmd)
                s.write((cmd + eol).encode())
                time.sleep(0.02)
                commands.append(cmd)
            if run is not None:
                cmd = f"{pref}r{1 if run else 0}"
                log.debug("write %s", cmd)
                s.write((cmd + eol).encode())
                time.sleep(0.02)
                commands.append(cmd)
    except Exception as e:
        msg = f"FY sweep command failed on {port}: {e}"
        if "timeout" in str(e).lower():
            raise FYTimeoutError(msg) from e
        raise FYError(msg) from e
    return commands
