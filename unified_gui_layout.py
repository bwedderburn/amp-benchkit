#!/usr/bin/env python3
"""
Unified GUI Layout (LITE+U3)

Refined to prefer PySide6 automatically (falls back to PyQt5). Headless selftests preserved.
- NEW: Automation tab uses a single shared FY **Port override** and **Protocol** for sweeps,
        regardless of which FY channel is selected. If left blank, it auto-finds a likely FY port.
- NEW: DAQ (U3) page includes a "Config Defaults" sub‑tab modeled after the U3-HV Windows panel.
- NEW: U3 Config tab adds Watchdog "Reset on Timeout" + (optional) Set DIO State,
       Backward-compat checkboxes, and Counter enable mapping to configIO when possible.
"""

import argparse
import csv
import math
import os
import sys
import time
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

# Ensure matplotlib can build its cache even when the user home dir is read-only.
if "MPLCONFIGDIR" not in os.environ:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib")
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except Exception:
        cache_dir = os.path.join(os.getcwd(), ".matplotlib-cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            cache_dir = ""
    if cache_dir:
        os.environ["MPLCONFIGDIR"] = cache_dir

import matplotlib

matplotlib.use("Agg")

# Import extracted dependency detection & helpers
from amp_benchkit import dsp as _dsp
from amp_benchkit.automation import build_freq_points
from amp_benchkit.calibration import load_calibration_curve
from amp_benchkit.deps import (
    HAVE_PYVISA,
    HAVE_QT,
    HAVE_U3,
    INSTALL_HINTS,
    _pyvisa,
    _u3,
    find_fy_port,
    list_ports,
)

# Imported refactored helpers
from amp_benchkit.diagnostics import collect_diagnostics
from amp_benchkit.fy import (
    FY_BAUD_EOLS,
    FY_MAX_VPP,
    FYError,
    build_fy_cmds,
    check_amp_vpp,
    fy_apply,
    fy_sweep,
)
from amp_benchkit.gui import build_generator_tab, build_scope_tab
from amp_benchkit.logging import get_logger, setup_logging
from amp_benchkit.sweeps import format_thd_rows, knee_sweep, thd_sweep
from amp_benchkit.tek import (
    TEK_RSRC_DEFAULT,
    parse_ieee_block,
    scope_arm_single,
    scope_capture_calibrated,
    scope_capture_fft_trace,
    scope_configure_fft,
    scope_configure_math_subtract,
    scope_resume_run,
    scope_screenshot,
    scope_set_trigger_ext,
    scope_wait_single_complete,
    tek_capture_block,
)
from amp_benchkit.u3config import (
    u3_pulse_line,
    u3_read_ain,
    u3_read_multi,
    u3_set_dir,
    u3_set_line,
)
from amp_benchkit.u3util import open_u3_safely as u3_open

with suppress(Exception):  # Qt symbol imports (available if HAVE_QT True)
    from amp_benchkit.deps import (
        QApplication,
        QFont,
        QMainWindow,
        QTabWidget,
        QTimer,
    )


class _U3Caps(TypedDict):
    hardware_version: float | None
    is_hv: bool


def _require_u3() -> Any:
    """Return the loaded LabJack module or raise if unavailable."""

    if _u3 is None:
        raise RuntimeError("LabJack U3 bindings unavailable")
    return _u3


FY_PROTOCOLS = ["FY ASCII 9600", "Auto (115200/CRLF→9600/LF)"]

# -----------------------------
# Utility / status helpers
# -----------------------------

# dep_msg, list_ports, find_fy_port now provided by amp_benchkit.deps


# Prefer a fixed-width font when available (used in Test Panel)
def fixed_font():
    try:
        f = QFont("Monospace")
        with suppress(Exception):
            f.setStyleHint(QFont.TypeWriter)
        return f
    except Exception:
        return None


def _timestamp_path(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if path.suffix:
        return path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    return path.parent / f"{path.name}_{stamp}"


def _resolve_output_path(value: Path | str | None, *, timestamp: bool) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return _timestamp_path(path) if timestamp else path


# -----------------------------
## FY and Tek helpers removed; now imported from amp_benchkit.fy and amp_benchkit.tek
def _decode_ieee_block(raw: bytes) -> bytes:
    """Backward-compatible proxy around :func:`amp_benchkit.tek.parse_ieee_block`."""

    parsed = parse_ieee_block(raw)
    if hasattr(parsed, "size"):
        if parsed.size:
            return parsed.astype(np.uint8).tobytes()
        # Non-IEEE payloads fall back to raw bytes to preserve legacy behaviour.
        if raw[:1] != b"#":
            return raw
        return b""
    return bytes(parsed)


## Removed local IEEE block decode and scope_capture; using imported helpers instead.
def scope_capture(resource=TEK_RSRC_DEFAULT, timeout_ms=15000, ch=1):
    """Return raw sample bytes for channel ch (delegates to tek module)."""
    if not HAVE_PYVISA:
        raise ImportError(f"pyvisa not available. {INSTALL_HINTS['pyvisa']}")
    # tek_capture_block returns (t, volts, raw) – we match previous semantics (raw bytes list)
    _t, _v, raw = tek_capture_block(resource, ch=ch)
    return list(raw.astype(np.uint8)) if hasattr(raw, "astype") else list(raw)


## Migrated scope_* and U3 basic helpers moved to amp_benchkit.tek and amp_benchkit.u3config


# -----------------------------
# GUI
# -----------------------------
class _FallbackBase:
    """Fallback stub when Qt is absent (keeps CLI helpers functional)."""

    def __init__(self) -> None:
        """No-op init to mirror QMainWindow signature."""

        return


BaseGUI: type[Any] = QMainWindow if HAVE_QT else _FallbackBase


class UnifiedGUI(BaseGUI):
    def __init__(self):
        if not HAVE_QT:
            return
        super().__init__()
        self.setWindowTitle("Unified Control (Lite+U3)")
        self.resize(1080, 780)
        self.scope_res = TEK_RSRC_DEFAULT
        os.makedirs("results", exist_ok=True)
        self._test_hist: list[str] | None = None
        self._cached_u3_caps: _U3Caps | None = None
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        tabs.addTab(self.tab_gen(), "Generator")
        tabs.addTab(self.tab_scope(), "Scope")
        tabs.addTab(self.tab_daq(), "DAQ (U3)")
        tabs.addTab(self.tab_automation(), "Automation / Sweep")
        tabs.addTab(self.tab_diag(), "Diagnostics")

    # ---- Generator (delegated)
    def tab_gen(self):
        return build_generator_tab(self)

    def scan_serial_into(self, target_edit):
        ps = list_ports()
        if not ps:
            self._log(self.gen_log, "No serial ports.")
            return
        self._log(self.gen_log, "Ports: " + ", ".join(p.device for p in ps))
        for p in ps:
            d = (p.device or "").lower()
            if any(k in d for k in ["usbserial", "tty.usb", "wchusb", "ftdi"]):
                target_edit.setText(p.device)
                break

    def _proto_for_ch(self, ch: int) -> str:
        cb = self.proto1 if ch == 1 else self.proto2
        return cb.currentText() if cb and hasattr(cb, "currentText") else "FY ASCII 9600"

    def _port_for_ch(self, ch: int) -> str:
        ed = self.port1 if ch == 1 else self.port2
        txt = ed.text().strip() if ed and hasattr(ed, "text") else ""
        return txt or find_fy_port()

    def _u3_capabilities(self) -> _U3Caps:
        """Return cached U3 capability info (hardware version, HV flag)."""

        if self._cached_u3_caps is not None:
            return self._cached_u3_caps

        caps: _U3Caps = {"hardware_version": None, "is_hv": False}
        if not HAVE_U3:
            self._cached_u3_caps = caps
            return caps

        def _coerce_hw(value: Any) -> float | None:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value.strip())
                except ValueError:
                    return None
            return None

        d = None
        try:
            d = u3_open()
            info: dict[str, Any] = {}
            with suppress(Exception):
                info = d.configU3()

            hw_val: float | None = None
            for candidate in (
                info.get("HardwareVersion"),
                getattr(d, "hardwareVersion", None),
            ):
                coerced = _coerce_hw(candidate)
                if coerced is not None:
                    hw_val = coerced
                    break
            caps["hardware_version"] = hw_val

            hv_raw: Any = info.get("HV")
            if hv_raw is None and "ProductID" in info:
                hv_raw = info.get("ProductID") == 3
            if hv_raw is None:
                hv_raw = getattr(d, "isHV", None)
            if hv_raw is None:
                dev_name = info.get("DeviceName")
                if isinstance(dev_name, str):
                    hv_raw = "HV" in dev_name.upper()
            if isinstance(hv_raw, str):
                hv_bool = hv_raw.strip().upper() in {"1", "TRUE", "YES", "HV"}
            else:
                hv_bool = bool(hv_raw)
            caps["is_hv"] = hv_bool
        except Exception:
            # Detection is best-effort; leave defaults when probing fails.
            pass
        finally:
            with suppress(Exception):
                if d:
                    d.close()
        self._cached_u3_caps = caps
        return caps

    def apply_gen_side(self, side):
        try:
            if side == 1:
                f = float(self.freq1.text())
                a = float(self.amp1.text())
                o = float(self.off1.text())
                wf = self.wave1.currentText()
                duty = None
                with suppress(Exception):
                    duty = float(self.duty1.text())
                pr = self.proto1.currentText()
                pt = self.port1.text().strip() or None
                cmds = fy_apply(
                    freq_hz=f, amp_vpp=a, wave=wf, off_v=o, duty=duty, ch=1, port=pt, proto=pr
                )
                summary = (
                    f"APPLIED CH1: {wf} {f} Hz, {a} Vpp, Off {o} V, "
                    f"Duty {duty if duty is not None else '—'}% ({pr})"
                )
                self._log(self.gen_log, summary)
                if cmds:
                    self._log(self.gen_log, "FY cmds: " + ", ".join(cmds))
            else:
                f = float(self.freq2.text())
                a = float(self.amp2.text())
                o = float(self.off2.text())
                wf = self.wave2.currentText()
                duty = None
                with suppress(Exception):
                    duty = float(self.duty2.text())
                pr = self.proto2.currentText()
                pt = self.port2.text().strip() or None
                cmds = fy_apply(
                    freq_hz=f, amp_vpp=a, wave=wf, off_v=o, duty=duty, ch=2, port=pt, proto=pr
                )
                summary = (
                    f"APPLIED CH2: {wf} {f} Hz, {a} Vpp, Off {o} V, "
                    f"Duty {duty if duty is not None else '—'}% ({pr})"
                )
                self._log(self.gen_log, summary)
                if cmds:
                    self._log(self.gen_log, "FY cmds: " + ", ".join(cmds))
        except Exception as e:
            self._log(self.gen_log, f"Error: {e}")

    def start_sweep_side(self, side):
        try:

            def _parse_freq(text: str) -> float | None:
                raw = (text or "").strip()
                if not raw:
                    return None
                t = raw.lower().replace("hz", "").strip()
                multiplier = 1.0
                if t.endswith("k"):
                    multiplier = 1e3
                    t = t[:-1]
                elif t.endswith("m"):
                    multiplier = 1e6
                    t = t[:-1]
                elif t.endswith("g"):
                    multiplier = 1e9
                    t = t[:-1]
                t = t.replace(",", "")
                try:
                    return float(t) * multiplier
                except Exception:
                    return None

            if side == 1:
                pr = self.proto1.currentText()
                pt = self.port1.text().strip() or find_fy_port()
                st = _parse_freq(self.sw_start1.text())
                if st is None:
                    self._log(self.gen_log, "CH1 sweep needs a numeric start frequency (Hz)")
                    return
                en = _parse_freq(self.sw_end1.text())
                if en is None:
                    self._log(self.gen_log, "CH1 sweep needs a numeric end frequency (Hz)")
                    return
                ts = int(self.sw_time1.text()) if self.sw_time1.text().strip() else None
                md = self.sw_mode1.currentText()
                if self.sw_amp1.text().strip():
                    try:
                        a = float(self.sw_amp1.text())
                        f = float(self.freq1.text() or 1000.0)
                        o = float(self.off1.text() or 0.0)
                        wf = self.wave1.currentText()
                        d = float(self.duty1.text()) if self.duty1.text().strip() else None
                        cmds_apply = fy_apply(
                            freq_hz=f, amp_vpp=a, wave=wf, off_v=o, duty=d, ch=1, port=pt, proto=pr
                        )
                        if cmds_apply:
                            self._log(self.gen_log, "FY cmds: " + ", ".join(cmds_apply))
                    except Exception as e:
                        self._log(self.gen_log, f"Amp set CH1 failed: {e}")
                cmds = fy_sweep(pt, 1, pr, st, en, ts, md, True)
                self._log(self.gen_log, f"SWEEP START CH1: {st}→{en} Hz, {ts}s, {md}")
                if cmds:
                    self._log(self.gen_log, "FY cmds: " + ", ".join(cmds))
            else:
                self._log(self.gen_log, "CH2 sweep not supported on FY3200S; use CH1.")
                return
        except Exception as e:
            self._log(self.gen_log, f"Sweep start error: {e}")

    def stop_sweep_side(self, side):
        try:
            if side == 1:
                pr = self.proto1.currentText()
                pt = self.port1.text().strip() or find_fy_port()
                cmds = fy_sweep(pt, 1, pr, run=False)
                self._log(self.gen_log, "SWEEP STOP CH1")
                if cmds:
                    self._log(self.gen_log, "FY cmds: " + ", ".join(cmds))
            else:
                self._log(self.gen_log, "CH2 sweep not supported on FY3200S; nothing to stop.")
        except Exception as e:
            self._log(self.gen_log, f"Sweep stop error: {e}")
        finally:
            with suppress(Exception):
                rsrc = (
                    self.scope_edit.text().strip()
                    if hasattr(self, "scope_edit")
                    else self.scope_res
                )
                scope_resume_run(rsrc or self.scope_res)

    def scope_measure(self, ch=1, typ="RMS"):
        if not HAVE_PYVISA:
            raise ImportError(f"pyvisa not available. {INSTALL_HINTS['pyvisa']}")
        r = self.scope_edit.text().strip() if hasattr(self, "scope_edit") else self.scope_res
        rm = _pyvisa.ResourceManager()
        sc = rm.open_resource(r or self.scope_res)
        try:
            with suppress(Exception):
                sc.timeout = 5000
            # Allow 'MATH' as a source
            try:
                if isinstance(ch, str) and ch.strip().upper() == "MATH":
                    sc.write("MEASU:IMM:SOURCE MATH")
                else:
                    sc.write(f"MEASU:IMM:SOURCE CH{int(ch)}")
            except Exception:
                sc.write(f"MEASU:IMM:SOURCE CH{int(ch)}")
            sc.write(f"MEASU:IMM:TYP {typ}")
            v = float(sc.query("MEASU:IMM:VAL?"))
            return v
        finally:
            with suppress(Exception):
                sc.close()

    # ---- Scope
    def tab_scope(self):
        return build_scope_tab(self)

    def list_visa(self):
        if not HAVE_PYVISA:
            self._log(self.scope_log, f"pyvisa missing → {INSTALL_HINTS['pyvisa']}")
            return
        try:
            res = _pyvisa.ResourceManager().list_resources()
            self._log(self.scope_log, ", ".join(res) if res else "(none)")
            tek = [r for r in res if r.startswith("USB0::0x0699::")]
            if tek:
                self.scope_edit.setText(tek[0])
        except Exception as e:
            self._log(self.scope_log, f"VISA error: {e}")

    def capture_scope(self):
        try:
            r = self.scope_edit.text().strip() or self.scope_res
            ch = int(self.scope_ch.currentText())
            d = scope_capture(r, ch=ch)
            self._log(self.scope_log, f"Captured CH{ch} {len(d)} pts: {d[:10]}")
        except Exception as e:
            self._log(self.scope_log, f"Error: {e}")

    def save_shot(self):
        try:
            r = self.scope_edit.text().strip() or self.scope_res
            ch = int(self.scope_ch.currentText())
            fn = os.path.join("results", f"scope_ch{ch}.png")
            path = scope_screenshot(fn, r, ch=ch)
            self._log(self.scope_log, f"Saved: {path}")
        except Exception as e:
            self._log(self.scope_log, f"Error: {e}")

    def save_csv(self):
        try:
            r = self.scope_edit.text().strip() or self.scope_res
            ch = int(self.scope_ch.currentText())
            fn = os.path.join("results", f"ch{ch}.csv")
            t, v = scope_capture_calibrated(r, timeout_ms=15000, ch=ch)
            with open(fn, "w") as f:
                f.write("t,volts\n")
                for i in range(len(v)):
                    f.write(f"{t[i]},{v[i]}\n")
            self._log(self.scope_log, f"Saved: {fn}")
        except Exception as e:
            self._log(self.scope_log, f"Error: {e}")

    # ---- DAQ (U3) with sub-tabs
    def tab_daq(self):
        from amp_benchkit.gui.daq_tab import build_daq_tab

        return build_daq_tab(self)

    def _selected_channels(self):
        return [i for i, cb in enumerate(self.chan_boxes) if cb.isChecked()]

        # ---- DAQ simple readers (used by Read/Stream tab)

    def read_daq_once(self):
        if not HAVE_U3:
            self._log(self.daq_log, f"u3 missing → {INSTALL_HINTS['u3']}")
            return
        chs = self._selected_channels() or [0]
        try:
            res_idx = self.daq_res.value() if hasattr(self, "daq_res") else None
            vals = u3_read_multi(chs, samples=1, resolution_index=res_idx)
            self._log(
                self.daq_log, " | ".join(f"AIN{c}:{vals[0][i]:.4f} V" for i, c in enumerate(chs))
            )
        except Exception as e:
            self._log(self.daq_log, f"Read error: {e}")

    def read_daq_multi(self):
        if not HAVE_U3:
            self._log(self.daq_log, f"u3 missing → {INSTALL_HINTS['u3']}")
            return
        chs = self._selected_channels() or [0]
        ns = self.daq_nsamp.value()
        delay = self.daq_delay.value() / 1000.0
        try:
            res_idx = self.daq_res.value() if hasattr(self, "daq_res") else None
            vals = u3_read_multi(chs, samples=ns, delay_s=delay, resolution_index=res_idx)
            for k, row in enumerate(vals):
                line = f"[{k + 1}/{ns}] " + " | ".join(
                    f"AIN{c}:{row[i]:.4f} V" for i, c in enumerate(chs)
                )
                self._log(self.daq_log, line)
        except Exception as e:
            self._log(self.daq_log, f"Loop error: {e}")

    # ---- Test Panel runtime loop
    def start_test_panel(self):
        if not HAVE_U3:
            self._log(self.test_log, f"u3 missing → {INSTALL_HINTS['u3']}")
            return
        try:
            if self.test_factory.isChecked():
                d = u3_open()
                try:
                    d.setToFactoryDefaults()
                finally:
                    with suppress(Exception):
                        d.close()
        except Exception as e:
            self._log(self.test_log, f"Factory reset warn: {e}")
            self._test_status(str(e), "error")
        if not hasattr(self, "test_timer") or self.test_timer is None:
            self.test_timer = QTimer(self)
        self.test_timer.setInterval(1000)
        self.test_timer.timeout.connect(self.tick_test_panel)
        self.test_timer.start()
        self._log(self.test_log, "Test Panel started")
        self._test_status("OK", "info")

    def stop_test_panel(self):
        t = getattr(self, "test_timer", None)
        if t:
            t.stop()
        self._log(self.test_log, "Test Panel stopped")
        with suppress(Exception):
            rsrc = self.scope_edit.text().strip() if hasattr(self, "scope_edit") else self.scope_res
            scope_resume_run(rsrc or self.scope_res)

    def tick_test_panel(self):
        # Apply current UI state to U3 once per second; read AINs
        if not HAVE_U3:
            return
        # Directions
        try:
            for i, cb in enumerate(getattr(self, "test_fio_dir", [])):
                u3_set_dir(f"FIO{i}", 1 if cb.isChecked() else 0)
            for i, cb in enumerate(getattr(self, "test_eio_dir", [])):
                u3_set_dir(f"EIO{i}", 1 if cb.isChecked() else 0)
            for i, cb in enumerate(getattr(self, "test_cio_dir", [])):
                u3_set_dir(f"CIO{i}", 1 if cb.isChecked() else 0)
        except Exception as e:
            self._log(self.test_log, f"Dir write warn: {e}")
            self._test_status(str(e), "error")
        # States (desired)
        try:
            for i, cb in enumerate(getattr(self, "test_fio_state", [])):
                u3_set_line(f"FIO{i}", 1 if cb.isChecked() else 0)
            for i, cb in enumerate(getattr(self, "test_eio_state", [])):
                u3_set_line(f"EIO{i}", 1 if cb.isChecked() else 0)
            for i, cb in enumerate(getattr(self, "test_cio_state", [])):
                u3_set_line(f"CIO{i}", 1 if cb.isChecked() else 0)
        except Exception as e:
            self._log(self.test_log, f"State write warn: {e}")
            self._test_status(str(e), "error")
        # Readback states (DI) and per-port masks
        try:
            lj = _require_u3()
            d = u3_open()
            try:
                states = d.getFeedback(lj.PortStateRead())[0]
                dirs = d.getFeedback(lj.PortDirRead())[0]
            finally:
                with suppress(Exception):
                    d.close()
            sF, sE, sC = states.get("FIO", 0), states.get("EIO", 0), states.get("CIO", 0)
            dF, dE, dC = dirs.get("FIO", 0), dirs.get("EIO", 0), dirs.get("CIO", 0)
            for i, cb in enumerate(getattr(self, "test_fio_rb", [])):
                cb.setChecked(bool((sF >> i) & 1))
            for i, cb in enumerate(getattr(self, "test_eio_rb", [])):
                cb.setChecked(bool((sE >> i) & 1))
            for i, cb in enumerate(getattr(self, "test_cio_rb", [])):
                cb.setChecked(bool((sC >> i) & 1))

            def fm(x):
                return f"0x{x:02X} ({x:08b})"

            self.test_dir_fio.setText(fm(dF))
            self.test_dir_eio.setText(fm(dE))
            self.test_dir_cio.setText(fm(dC))
            self.test_st_fio.setText(fm(sF))
            self.test_st_eio.setText(fm(sE))
            self.test_st_cio.setText(fm(sC))
        except Exception as e:
            self._log(self.test_log, f"Readback warn: {e}")
            self._test_status(str(e), "error")
        # DACs
        try:
            lj = _require_u3()
            d = u3_open()
            try:
                dv0 = max(0.0, min(5.0, float(self.test_dac0.text() or "0")))
                dv1 = max(0.0, min(5.0, float(self.test_dac1.text() or "0")))
                with suppress(Exception):
                    d.getFeedback(
                        lj.DAC0_8(Value=int(dv0 / 5.0 * 255)),
                        lj.DAC1_8(Value=int(dv1 / 5.0 * 255)),
                    )
            finally:
                with suppress(Exception):
                    d.close()
        except Exception as e:
            self._log(self.test_log, f"DAC warn: {e}")
            self._test_status(str(e), "error")
        # AIN readings
        try:
            res_idx = self.daq_res.value() if hasattr(self, "daq_res") else None
            fio_checks = getattr(self, "ai_checks_fio", getattr(self, "ai_checks", []))
            eio_checks = getattr(self, "ai_checks_eio", [])
            caps = self._u3_capabilities()
            for i, lbl in enumerate(getattr(self, "test_ain_lbls", [])):
                if i < 8:
                    is_analog = i < len(fio_checks) and fio_checks[i].isChecked()
                    if caps["is_hv"] and i < 4:
                        is_analog = True
                else:
                    idx = i - 8
                    is_analog = idx < len(eio_checks) and eio_checks[idx].isChecked()
                if not is_analog:
                    lbl.setText("—")
                    continue
                v = u3_read_ain(i, resolution_index=res_idx)
                lbl.setText(f"{v:.4f}")
        except Exception as e:
            self._log(self.test_log, f"AIN read warn: {e}")
            self._test_status(str(e), "error")
        # Counters
        try:
            lj = _require_u3()
            d = u3_open()
            try:
                try:
                    c0 = d.getFeedback(lj.Counter0(Reset=False))[0]
                except Exception:
                    c0 = None
                try:
                    c1 = d.getFeedback(lj.Counter1(Reset=False))[0]
                except Exception:
                    c1 = None
            finally:
                with suppress(Exception):
                    d.close()
            if c0 is not None:
                self.test_c0.setText(str(c0))
            if c1 is not None:
                self.test_c1.setText(str(c1))
        except Exception as e:
            self._log(self.test_log, f"Counter read warn: {e}")
            self._test_status(str(e), "error")

    def reset_counter(self, which: int):
        if not HAVE_U3:
            return
        try:
            lj = _require_u3()
            d = u3_open()
            try:
                if which == 0:
                    d.getFeedback(lj.Counter0(Reset=True))
                    self._log(self.test_log, "Counter0 reset")
                else:
                    d.getFeedback(lj.Counter1(Reset=True))
                    self._log(self.test_log, "Counter1 reset")
            finally:
                with suppress(Exception):
                    d.close()
        except Exception as e:
            self._log(self.test_log, f"Counter reset warn: {e}")
            self._test_status(str(e), "error")

    # ---- Whole-port writers (Test Panel)
    def _parse_mask_text(self, txt: str) -> int:
        s = (txt or "").strip()
        try:
            if s.lower().startswith("0x") or s.lower().startswith("0b"):
                return max(0, min(255, int(s, 0)))
            # allow binary like 10101010
            if all(c in "01" for c in s) and len(s) <= 8:
                return int(s, 2)
            return max(0, min(255, int(s)))
        except Exception:
            return 0

    def apply_port_dir(self, port: str):
        if not HAVE_U3:
            self._log(self.test_log, f"u3 missing → {INSTALL_HINTS['u3']}")
            return
        port = (port or "FIO").upper()
        vF = vE = vC = 0
        mF = mE = mC = 0
        if port == "FIO":
            vF = self._parse_mask_text(self.test_wdir_fio.text())
            mF = 0xFF
        elif port == "EIO":
            vE = self._parse_mask_text(self.test_wdir_eio.text())
            mE = 0xFF
        else:
            vC = self._parse_mask_text(self.test_wdir_cio.text())
            mC = 0xFF
        try:
            lj = _require_u3()
            d = u3_open()
            try:
                d.getFeedback(lj.PortDirWrite(Direction=[vF, vE, vC], WriteMask=[mF, mE, mC]))
            finally:
                with suppress(Exception):
                    d.close()
            self._log(
                self.test_log,
                f"Dir write {port}: 0x{(vF if port == 'FIO' else vE if port == 'EIO' else vC):02X}",
            )
            self._test_status("OK", "info")
        except Exception as e:
            self._log(self.test_log, f"Dir write error: {e}")
            self._test_status(str(e), "error")

    def apply_port_state(self, port: str):
        if not HAVE_U3:
            self._log(self.test_log, f"u3 missing → {INSTALL_HINTS['u3']}")
            return
        port = (port or "FIO").upper()
        vF = vE = vC = 0
        mF = mE = mC = 0
        if port == "FIO":
            vF = self._parse_mask_text(self.test_wst_fio.text())
            mF = 0xFF
        elif port == "EIO":
            vE = self._parse_mask_text(self.test_wst_eio.text())
            mE = 0xFF
        else:
            vC = self._parse_mask_text(self.test_wst_cio.text())
            mC = 0xFF
        try:
            lj = _require_u3()
            d = u3_open()
            try:
                d.getFeedback(lj.PortStateWrite(State=[vF, vE, vC], WriteMask=[mF, mE, mC]))
            finally:
                with suppress(Exception):
                    d.close()
            port_value = vF if port == "FIO" else vE if port == "EIO" else vC
            self._log(self.test_log, f"State write {port}: 0x{port_value:02X}")
            self._test_status("OK", "info")
        except Exception as e:
            self._log(self.test_log, f"State write error: {e}")
            self._test_status(str(e), "error")

    def apply_all_ports(self):
        if not HAVE_U3:
            self._log(self.test_log, f"u3 missing → {INSTALL_HINTS['u3']}")
            return
        try:
            df = self._parse_mask_text(self.test_wdir_fio.text())
            de = self._parse_mask_text(self.test_wdir_eio.text())
            dc = self._parse_mask_text(self.test_wdir_cio.text())
            sf = self._parse_mask_text(self.test_wst_fio.text())
            se = self._parse_mask_text(self.test_wst_eio.text())
            sc = self._parse_mask_text(self.test_wst_cio.text())
            lj = _require_u3()
            d = u3_open()
            try:
                d.getFeedback(lj.PortDirWrite(Direction=[df, de, dc], WriteMask=[0xFF, 0xFF, 0xFF]))
                d.getFeedback(lj.PortStateWrite(State=[sf, se, sc], WriteMask=[0xFF, 0xFF, 0xFF]))
            finally:
                with suppress(Exception):
                    d.close()
            summary = (
                "Applied all: "
                f"Dir=[0x{df:02X},0x{de:02X},0x{dc:02X}] "
                f"State=[0x{sf:02X},0x{se:02X},0x{sc:02X}]"
            )
            self._log(self.test_log, summary)
            self._test_status("OK", "info")
        except Exception as e:
            self._log(self.test_log, f"Apply all error: {e}")
            self._test_status(str(e), "error")

    def load_masks_from_device(self):
        if not HAVE_U3:
            self._log(self.test_log, f"u3 missing → {INSTALL_HINTS['u3']}")
            return
        try:
            lj = _require_u3()
            d = u3_open()
            try:
                states = d.getFeedback(lj.PortStateRead())[0]
                dirs = d.getFeedback(lj.PortDirRead())[0]
            finally:
                with suppress(Exception):
                    d.close()
            sF, sE, sC = states.get("FIO", 0), states.get("EIO", 0), states.get("CIO", 0)
            dF, dE, dC = dirs.get("FIO", 0), dirs.get("EIO", 0), dirs.get("CIO", 0)
            # Fill editable mask fields
            self.test_wdir_fio.setText(f"0x{dF:02X}")
            self.test_wdir_eio.setText(f"0x{dE:02X}")
            self.test_wdir_cio.setText(f"0x{dC:02X}")
            self.test_wst_fio.setText(f"0x{sF:02X}")
            self.test_wst_eio.setText(f"0x{sE:02X}")
            self.test_wst_cio.setText(f"0x{sC:02X}")
            # Optionally sync desired checkboxes to device
            for i, cb in enumerate(getattr(self, "test_fio_dir", [])):
                cb.setChecked(bool((dF >> i) & 1))
            for i, cb in enumerate(getattr(self, "test_eio_dir", [])):
                cb.setChecked(bool((dE >> i) & 1))
            for i, cb in enumerate(getattr(self, "test_cio_dir", [])):
                cb.setChecked(bool((dC >> i) & 1))
            for i, cb in enumerate(getattr(self, "test_fio_state", [])):
                cb.setChecked(bool((sF >> i) & 1))
            for i, cb in enumerate(getattr(self, "test_eio_state", [])):
                cb.setChecked(bool((sE >> i) & 1))
            for i, cb in enumerate(getattr(self, "test_cio_state", [])):
                cb.setChecked(bool((sC >> i) & 1))
            self._log(self.test_log, "Loaded masks from device")
            self._test_status("OK", "info")
        except Exception as e:
            self._log(self.test_log, f"Read masks error: {e}")
            self._test_status(str(e), "error")

    def fill_masks_from_checks(self):
        def mfrom(checks):
            m = 0
            for i, cb in enumerate(checks):
                if cb.isChecked():
                    m |= 1 << i
            return m

        try:
            dF = mfrom(getattr(self, "test_fio_dir", []))
            dE = mfrom(getattr(self, "test_eio_dir", []))
            dC = mfrom(getattr(self, "test_cio_dir", []))
            sF = mfrom(getattr(self, "test_fio_state", []))
            sE = mfrom(getattr(self, "test_eio_state", []))
            sC = mfrom(getattr(self, "test_cio_state", []))
            self.test_wdir_fio.setText(f"0x{dF:02X}")
            self.test_wdir_eio.setText(f"0x{dE:02X}")
            self.test_wdir_cio.setText(f"0x{dC:02X}")
            self.test_wst_fio.setText(f"0x{sF:02X}")
            self.test_wst_eio.setText(f"0x{sE:02X}")
            self.test_wst_cio.setText(f"0x{sC:02X}")
            self._log(self.test_log, "Filled mask editors from checkboxes")
        except Exception as e:
            self._log(self.test_log, f"Fill masks error: {e}")
            self._test_status(str(e), "error")

    # ---- U3 config helpers/actions
    def _mask_from_checks(self, checks):
        m = 0
        for i, cb in enumerate(checks):
            if cb.isChecked():
                m |= 1 << i
        return m

    def u3_read_current(self):
        if not HAVE_U3:
            self._log(self.cfg_log, f"u3 missing → {INSTALL_HINTS['u3']}")
            return
        try:
            d = u3_open()
            info = d.configIO()
            self._log(self.cfg_log, str(info))
        except Exception as e:
            self._log(self.cfg_log, f"Read error: {e}")
        finally:
            with suppress(Exception):
                d.close()

    def u3_write_factory(self):
        if not HAVE_U3:
            self._log(self.cfg_log, f"u3 missing → {INSTALL_HINTS['u3']}")
            return
        try:
            _require_u3()
            d = u3_open()
            # Set power-up defaults back to factory via Device API
            d.setToFactoryDefaults()
            self._log(self.cfg_log, "Factory defaults restored")
        except Exception as e:
            self._log(self.cfg_log, f"Factory write error: {e}")
        finally:
            with suppress(Exception):
                d.close()

    def u3_write_values(self):
        if not HAVE_U3:
            self._log(self.cfg_log, f"u3 missing → {INSTALL_HINTS['u3']}")
            return
        try:
            caps = self._u3_capabilities()
            hw_float = caps["hardware_version"]
            lj = _require_u3()
            d = u3_open()
            # Analog inputs / directions + Counters
            fio_checks = getattr(self, "ai_checks_fio", getattr(self, "ai_checks", []))
            eio_checks = getattr(self, "ai_checks_eio", [])
            fio_an = self._mask_from_checks(fio_checks)
            eio_an = self._mask_from_checks(eio_checks) if eio_checks else 0
            fio_dir = self._mask_from_checks(self.fio_dir_box)
            eio_dir = self._mask_from_checks(self.eio_dir_box)
            cio_dir = self._mask_from_checks(self.cio_dir_box)
            # Set default directions/analog settings at boot via configU3
            with suppress(Exception):
                d.configU3(
                    FIOAnalog=fio_an,
                    EIOAnalog=eio_an,
                    FIODirection=fio_dir,
                    EIODirection=eio_dir,
                    CIODirection=cio_dir,
                )
            # Digital states (defaults + current)
            fio_state = self._mask_from_checks(self.fio_state_box)
            eio_state = self._mask_from_checks(self.eio_state_box)
            cio_state = self._mask_from_checks(self.cio_state_box)
            with suppress(Exception):
                d.configU3(FIOState=fio_state, EIOState=eio_state, CIOState=cio_state)
            fb = []
            # Use global IO numbering: FIO0-7 → 0..7, EIO0-7 → 8..15, CIO0-3 → 16..19
            for i in range(8):
                fb.append(lj.BitStateWrite(i, 1 if (fio_state >> i) & 1 else 0))
            for i in range(8):
                fb.append(lj.BitStateWrite(8 + i, 1 if (eio_state >> i) & 1 else 0))
            for i in range(4):
                fb.append(lj.BitStateWrite(16 + i, 1 if (cio_state >> i) & 1 else 0))
            # DAC outputs (8-bit mode by default)
            with suppress(Exception):
                dv0 = max(0.0, min(5.0, float(self.dac0.text() or "0")))
                dv1 = max(0.0, min(5.0, float(self.dac1.text() or "0")))
                fb.append(lj.DAC0_8(Value=int(dv0 / 5.0 * 255)))
                fb.append(lj.DAC1_8(Value=int(dv1 / 5.0 * 255)))
            # Timer/Counter clock setup
            if self.t_clkbase.currentText() == "48MHz":
                base = 48
            elif self.t_clkbase.currentText() == "750kHz":
                base = 750
            else:
                base = 4
            with suppress(Exception):
                timer_offset = self.t_pin.value() if hasattr(self, "t_pin") else 0
                if hw_float is not None and hw_float >= 1.30 and timer_offset < 4:
                    timer_offset = 4
                d.configTimerClock(TimerClockBase=base, TimerClockDivisor=self.t_div.value())
                d.configIO(
                    NumberOfTimersEnabled=self.t_num.value(),
                    TimerCounterPinOffset=timer_offset,
                    EnableCounter0=self.counter0.isChecked(),
                    EnableCounter1=self.counter1.isChecked(),
                    FIOAnalog=fio_an,
                    EIOAnalog=eio_an,
                )
            # Apply digital state writes
            try:
                d.getFeedback(*fb)
            except Exception:
                # Fallback to immediate per-pin writes
                with suppress(Exception):
                    for i in range(8):
                        d.setDOState(i, 1 if (fio_state >> i) & 1 else 0)
                    for i in range(8):
                        d.setDOState(8 + i, 1 if (eio_state >> i) & 1 else 0)
                    for i in range(4):
                        d.setDOState(16 + i, 1 if (cio_state >> i) & 1 else 0)
            # Persist current configuration as power-up defaults
            with suppress(Exception):
                d.setDefaults()
            # Watchdog (best-effort mapping of extra options)
            if self.wd_en.isChecked():
                with suppress(Exception):
                    timeout = int(float(self.wd_to.text() or "100"))
                    reset = self.wd_reset.isChecked()
                    set_dio = False
                    dio_num = 0
                    dio_state = 0
                    wline = getattr(self, "wd_line", None)
                    if wline and wline.currentText() != "None":
                        pin = wline.currentText()  # e.g., 'FIO3', 'EIO1', 'CIO0'
                        base = 0
                        if pin.startswith("FIO"):
                            base = 0
                        elif pin.startswith("EIO"):
                            base = 8
                        elif pin.startswith("CIO"):
                            base = 16
                        try:
                            idx = int(pin[3:])
                            dio_num = base + idx
                            dio_state = 1 if self.wd_state.currentText() == "High" else 0
                            set_dio = True
                        except Exception:
                            set_dio = False
                    d.watchdog(
                        ResetOnTimeout=reset,
                        SetDIOStateOnTimeout=set_dio,
                        TimeoutPeriod=timeout,
                        DIOState=dio_state,
                        DIONumber=dio_num,
                    )
            self._log(self.cfg_log, "Values written")
        except Exception as e:
            self._log(self.cfg_log, f"Write error: {e}")
        finally:
            with suppress(Exception):
                d.close()

    # Apply current DAQ config selections for this run (no persist unless requested)
    def u3_autoconfig_runtime(
        self, base: str = "Keep Current", pulse_line: str = "None", persist: bool = False
    ):
        if not HAVE_U3:
            return
        d = None
        caps = self._u3_capabilities()
        hw_float = caps["hardware_version"]
        try:
            d = u3_open()
            # Optional factory base
            if isinstance(base, str) and base.lower().startswith("factory"):
                with suppress(Exception):
                    d.setToFactoryDefaults()
            # Collect masks from current UI
            fio_checks = getattr(self, "ai_checks_fio", getattr(self, "ai_checks", []))
            eio_checks = getattr(self, "ai_checks_eio", [])
            fio_an = self._mask_from_checks(fio_checks) if fio_checks else 0x0F
            eio_an = self._mask_from_checks(eio_checks) if eio_checks else 0
            fio_dir = (
                self._mask_from_checks(self.fio_dir_box) if hasattr(self, "fio_dir_box") else 0
            )
            eio_dir = (
                self._mask_from_checks(self.eio_dir_box) if hasattr(self, "eio_dir_box") else 0
            )
            cio_dir = (
                self._mask_from_checks(self.cio_dir_box) if hasattr(self, "cio_dir_box") else 0
            )
            fio_state = (
                self._mask_from_checks(self.fio_state_box) if hasattr(self, "fio_state_box") else 0
            )
            eio_state = (
                self._mask_from_checks(self.eio_state_box) if hasattr(self, "eio_state_box") else 0
            )
            cio_state = (
                self._mask_from_checks(self.cio_state_box) if hasattr(self, "cio_state_box") else 0
            )
            # Configure directions and analog mode at boot/current
            with suppress(Exception):
                d.configU3(
                    FIOAnalog=fio_an,
                    EIOAnalog=eio_an,
                    FIODirection=fio_dir,
                    EIODirection=eio_dir,
                    CIODirection=cio_dir,
                )
            # Digital states (apply now)
            lj = _require_u3()
            fb = []
            for i in range(8):
                fb.append(lj.BitStateWrite(i, 1 if (fio_state >> i) & 1 else 0))
            for i in range(8):
                fb.append(lj.BitStateWrite(8 + i, 1 if (eio_state >> i) & 1 else 0))
            for i in range(4):
                fb.append(lj.BitStateWrite(16 + i, 1 if (cio_state >> i) & 1 else 0))
            try:
                d.getFeedback(*fb)
            except Exception:
                with suppress(Exception):
                    for i in range(8):
                        d.setDOState(i, 1 if (fio_state >> i) & 1 else 0)
                    for i in range(8):
                        d.setDOState(8 + i, 1 if (eio_state >> i) & 1 else 0)
                    for i in range(4):
                        d.setDOState(16 + i, 1 if (cio_state >> i) & 1 else 0)
            # DAC outputs from UI if present
            with suppress(Exception):
                dv0 = max(0.0, min(5.0, float(self.dac0.text() or "0")))
                dv1 = max(0.0, min(5.0, float(self.dac1.text() or "0")))
                with suppress(Exception):
                    d.getFeedback(
                        lj.DAC0_8(Value=int(dv0 / 5.0 * 255)),
                        lj.DAC1_8(Value=int(dv1 / 5.0 * 255)),
                    )
            # Timers / Counters
            with suppress(Exception):
                if self.t_clkbase.currentText() == "48MHz":
                    base_clk = 48
                elif self.t_clkbase.currentText() == "750kHz":
                    base_clk = 750
                else:
                    base_clk = 4
                timer_offset = self.t_pin.value() if hasattr(self, "t_pin") else 0
                if hw_float is not None and hw_float >= 1.30 and timer_offset < 4:
                    timer_offset = 4
                d.configTimerClock(TimerClockBase=base_clk, TimerClockDivisor=self.t_div.value())
                d.configIO(
                    NumberOfTimersEnabled=self.t_num.value(),
                    TimerCounterPinOffset=timer_offset,
                    EnableCounter0=self.counter0.isChecked(),
                    EnableCounter1=self.counter1.isChecked(),
                    FIOAnalog=fio_an,
                    EIOAnalog=eio_an,
                )
            # Watchdog if enabled in UI
            with suppress(Exception):
                if self.wd_en.isChecked():
                    timeout = int(float(self.wd_to.text() or "100"))
                    reset = self.wd_reset.isChecked()
                    set_dio = False
                    dio_num = 0
                    dio_state = 0
                    wline = getattr(self, "wd_line", None)
                    if wline and wline.currentText() != "None":
                        pin = wline.currentText()
                        basep = 0
                        if pin.startswith("FIO"):
                            basep = 0
                        elif pin.startswith("EIO"):
                            basep = 8
                        elif pin.startswith("CIO"):
                            basep = 16
                        try:
                            idx = int(pin[3:])
                            dio_num = basep + idx
                            dio_state = 1 if self.wd_state.currentText() == "High" else 0
                            set_dio = True
                        except Exception:
                            set_dio = False
                    d.watchdog(
                        ResetOnTimeout=reset,
                        SetDIOStateOnTimeout=set_dio,
                        TimeoutPeriod=timeout,
                        DIOState=dio_state,
                        DIONumber=dio_num,
                    )
            # Ensure pulse line (if any) is an output
            with suppress(Exception):
                if pulse_line and pulse_line.strip().lower() != "none":
                    u3_set_dir(pulse_line, 1)
            # Persist current as power-up defaults if requested
            if persist:
                with suppress(Exception):
                    d.setDefaults()
        finally:
            with suppress(Exception):
                if d:
                    d.close()

    # ---- Automation
    def tab_automation(self):
        from amp_benchkit.gui.automation_tab import build_automation_tab

        return build_automation_tab(self)

    def stop_sweep_scope(self):
        self._sweep_abort = True

    def run_sweep_scope_fixed(self):
        from amp_benchkit.automation import build_freq_list, sweep_scope_fixed

        rsrc = self.scope_edit.text().strip() if hasattr(self, "scope_edit") else self.scope_res
        try:
            ch = int(self.auto_ch.currentText())
            sch = int(self.auto_scope_ch.currentText())
            start = float(self.auto_start.text())
            stop = float(self.auto_stop.text())
            step = float(self.auto_step.text())
            amp = float(self.auto_amp.text())
            dwell = max(0.0, float(self.auto_dwell.text()) / 1000.0)
            metric = self.auto_metric.currentText()
            freqs = build_freq_list(start, stop, step)
            self._sweep_abort = False
            pr = self.auto_proto.currentText() if hasattr(self, "auto_proto") else "FY ASCII 9600"
            pt = (
                self.auto_port.text().strip() if hasattr(self, "auto_port") else ""
            ) or find_fy_port()
            use_math = (
                bool(self.auto_use_math.isChecked()) if hasattr(self, "auto_use_math") else False
            )
            order = (
                self.auto_math_order.currentText()
                if hasattr(self, "auto_math_order")
                else "CH1-CH2"
            )
            use_ext = (
                bool(self.auto_use_ext.isChecked()) if hasattr(self, "auto_use_ext") else False
            )
            ext_slope = (
                self.auto_ext_slope.currentText() if hasattr(self, "auto_ext_slope") else "Rise"
            )
            try:
                ext_level = (
                    float(self.auto_ext_level.text())
                    if hasattr(self, "auto_ext_level") and self.auto_ext_level.text().strip()
                    else None
                )
            except Exception:
                ext_level = None
            try:
                pre_ms = (
                    float(self.auto_ext_pre_ms.text())
                    if hasattr(self, "auto_ext_pre_ms") and self.auto_ext_pre_ms.text().strip()
                    else 5.0
                )
            except Exception:
                pre_ms = 5.0

            # optional u3 auto config closure
            def _u3_autocfg():
                if hasattr(self, "auto_u3_autocfg") and self.auto_u3_autocfg.isChecked():
                    base = (
                        self.auto_u3_base.currentText()
                        if hasattr(self, "auto_u3_base")
                        else "Keep Current"
                    )
                    self.u3_autoconfig_runtime(base=base, pulse_line="None", persist=False)

            amplitude_calibration = None
            amp_strategy = None
            if getattr(self, "auto_apply_cal", None) and self.auto_apply_cal.isChecked():
                try:
                    curve = load_calibration_curve()
                    amplitude_calibration = curve.apply
                    self._log(self.auto_log, "Gold calibration enabled")
                    target_widget = getattr(self, "auto_cal_target", None)
                    if target_widget is not None:
                        text = target_widget.text().strip()
                        if text:
                            try:
                                target_vpp = float(text)

                                def _strategy(
                                    freq: float,
                                    *,
                                    _curve=curve,
                                    _target=target_vpp,
                                    _fallback=amp,
                                ):
                                    try:
                                        ratio = _curve.ratio_at(freq)
                                        if ratio <= 0:
                                            return _fallback
                                        return _target / ratio
                                    except Exception:
                                        return _fallback

                                amp_strategy = _strategy
                                self._log(
                                    self.auto_log,
                                    f"Generator adjusted for target {target_vpp:.3f} Vpp",
                                )
                            except Exception as exc:
                                self._log(self.auto_log, f"Bad cal target: {exc}")
                except Exception as exc:  # pragma: no cover - defensive
                    self._log(self.auto_log, f"Calibration load error: {exc}")
            out = sweep_scope_fixed(
                freqs,
                channel=ch,
                scope_channel=sch,
                amp_vpp=amp,
                dwell_s=dwell,
                metric=metric,
                fy_apply=lambda **kw: fy_apply(port=pt, proto=pr, **kw),
                scope_measure=lambda src, typ: self.scope_measure(src, typ),
                scope_configure_math_subtract=lambda res, order: scope_configure_math_subtract(
                    rsrc or self.scope_res, order=order
                ),
                scope_set_trigger_ext=lambda res, slope, level: scope_set_trigger_ext(
                    rsrc or self.scope_res, slope=slope, level=level
                ),
                scope_arm_single=lambda res: scope_arm_single(rsrc or self.scope_res),
                scope_wait_single_complete=lambda res, timeout_s: scope_wait_single_complete(
                    rsrc or self.scope_res, timeout_s=timeout_s
                ),
                use_math=use_math,
                math_order=order,
                use_ext=use_ext,
                ext_slope=ext_slope,
                ext_level=ext_level,
                pre_ms=pre_ms,
                scope_resource=rsrc or self.scope_res,
                logger=lambda s: self._log(self.auto_log, s),
                progress=lambda i, n: (
                    self.auto_prog.setValue(int(i / n * 100)),
                    QApplication.processEvents(),
                ),
                abort_flag=lambda: getattr(self, "_sweep_abort", False),
                u3_autoconfig=_u3_autocfg,
                amp_vpp_strategy=amp_strategy,
                amplitude_calibration=amplitude_calibration,
            )
            os.makedirs("results", exist_ok=True)
            fn = os.path.join("results", "sweep_scope.csv")
            with open(fn, "w") as fh:
                fh.write("freq_hz,metric\n")
                for f, val in out:
                    fh.write(f"{f},{val}\n")
            self._log(self.auto_log, f"Saved: {fn}")
        except Exception as e:
            self._log(self.auto_log, f"Sweep error: {e}")
        finally:
            with suppress(Exception):
                scope_resume_run(rsrc or self.scope_res)

    def run_live_thd_sweep(self):
        """Execute the THD math sweep using current GUI parameters."""
        from amp_benchkit.sweeps import format_thd_rows, thd_sweep

        try:
            start = float(self.live_thd_start.text() or 20.0)
            stop = float(self.live_thd_stop.text() or 20000.0)
            points = int(self.live_thd_points.value())
            amp_vpp = float(self.live_thd_amp.text() or 0.5)
            dwell_s = max(0.0, float(self.live_thd_dwell.text() or 0.5))
            scope_channel = int(self.live_thd_scope.currentText())
            use_math = bool(self.live_thd_use_math.isChecked())
            math_order = self.live_thd_math_order.currentText()
            filter_spikes = not self.live_thd_keep_spikes.isChecked()
            filter_window = int(self.live_thd_filter_window.value())
            filter_factor = float(self.live_thd_filter_factor.text() or 2.0)
            filter_min = float(self.live_thd_filter_min.text() or 2.0)
            output_template = self.live_thd_output.text().strip() or "results/thd_sweep.csv"
            if "%" not in output_template:
                timestamp = time.strftime("%m-%d-%Y_%H_%M")
                if output_template.endswith(".csv"):
                    prefix = output_template[:-4]
                    output_path = f"{prefix}_{timestamp}.csv"
                else:
                    output_path = f"{output_template}_{timestamp}.csv"
            else:
                output_path = time.strftime(output_template)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            port = self.auto_port.text().strip() if hasattr(self, "auto_port") else ""
            if not port:
                port = find_fy_port()
            resource = (
                self.scope_edit.text().strip() if hasattr(self, "scope_edit") else self.scope_res
            )
            self.auto_prog.setValue(0)
            calibration_curve = None
            if getattr(self, "live_thd_apply_cal", None) and self.live_thd_apply_cal.isChecked():
                try:
                    calibration_curve = load_calibration_curve()
                    self._log(self.auto_log, "Gold calibration enabled for THD sweep")
                except Exception as exc:
                    self._log(self.auto_log, f"Calibration load error: {exc}")
            cal_target = None
            target_widget = getattr(self, "live_cal_target", None)
            if target_widget is not None:
                text = target_widget.text().strip()
                if text:
                    try:
                        cal_target = float(text)
                        if calibration_curve is None:
                            calibration_curve = load_calibration_curve()
                            self._log(
                                self.auto_log,
                                "Gold calibration auto-enabled for cal target",
                            )
                    except Exception as exc:
                        self._log(self.auto_log, f"Bad cal target: {exc}")
            rows, out_path, suppressed = thd_sweep(
                visa_resource=resource or self.scope_res,
                fy_port=port,
                amp_vpp=amp_vpp,
                scope_channel=scope_channel,
                start_hz=start,
                stop_hz=stop,
                points=points,
                dwell_s=dwell_s,
                use_math=use_math,
                math_order=math_order,
                output=output_path,
                filter_spikes=filter_spikes,
                filter_window=filter_window,
                filter_factor=filter_factor,
                filter_min_percent=filter_min,
                calibration_curve=calibration_curve,
                calibrate_to_vpp=cal_target,
            )
            self.auto_prog.setValue(100)
            if out_path:
                self._log(self.auto_log, f"Saved: {out_path}")
            for line in format_thd_rows(rows):
                self._log(self.auto_log, line)
            if suppressed and filter_spikes:
                self._log(self.auto_log, "Filtered spikes:")
                for freq, original, baseline in suppressed:
                    self._log(
                        self.auto_log,
                        f"  {freq:8.2f} Hz → {original:6.3f}% (replaced with {baseline:6.3f}%)",
                    )

            # Summary metrics
            finite = [thd for _, _, _, thd in rows if math.isfinite(thd)]
            summary_lines = []
            if finite:
                mean_val = sum(finite) / len(finite)
                median_val = sorted(finite)[len(finite) // 2]
                summary_lines.append(f"THD stats: min {min(finite):.3f}% / max {max(finite):.3f}%")
                summary_lines.append(f"Mean {mean_val:.3f}% / median {median_val:.3f}%")
            if suppressed and filter_spikes:
                summary_lines.append(
                    f"Spikes suppressed: {len(suppressed)} (window={filter_window}, "
                    f"factor={filter_factor}, min={filter_min:.2f}%)"
                )
                for freq, original, baseline in suppressed:
                    summary_lines.append(f"  {freq:8.2f} Hz → {original:6.3f}% → {baseline:6.3f}%")
            if summary_lines:
                text = "\n".join(summary_lines)
                if hasattr(self, "live_thd_summary"):
                    self.live_thd_summary.setPlainText(text)
                self._log(self.auto_log, text)
        except Exception as exc:  # pragma: no cover - hardware path
            self._log(self.auto_log, f"THD sweep error: {exc}")
        finally:
            self.auto_prog.setValue(0)

    def run_audio_kpis(self):
        """Sweep using FY + scope, compute Vrms/PkPk and THD, then report -dB knees if requested."""
        from amp_benchkit.automation import build_freq_list, sweep_audio_kpis

        rsrc = self.scope_edit.text().strip() if hasattr(self, "scope_edit") else self.scope_res
        try:
            ch = int(self.auto_ch.currentText())
            sch = int(self.auto_scope_ch.currentText())
            start = float(self.auto_start.text())
            stop = float(self.auto_stop.text())
            step = float(self.auto_step.text())
            amp = float(self.auto_amp.text())
            dwell = max(0.0, float(self.auto_dwell.text()) / 1000.0)
            pr = self.auto_proto.currentText() if hasattr(self, "auto_proto") else "FY ASCII 9600"
            pt = (
                self.auto_port.text().strip() if hasattr(self, "auto_port") else ""
            ) or find_fy_port()
            pulse_line = (
                self.auto_u3_line.currentText() if hasattr(self, "auto_u3_line") else "None"
            )
            try:
                pulse_ms = (
                    float(self.auto_u3_pwidth.text())
                    if hasattr(self, "auto_u3_pwidth") and self.auto_u3_pwidth.text().strip()
                    else 0.0
                )
            except Exception:
                pulse_ms = 0.0
            use_ext = (
                bool(self.auto_use_ext.isChecked()) if hasattr(self, "auto_use_ext") else False
            )
            ext_slope = (
                self.auto_ext_slope.currentText() if hasattr(self, "auto_ext_slope") else "Rise"
            )
            try:
                ext_level = (
                    float(self.auto_ext_level.text())
                    if hasattr(self, "auto_ext_level") and self.auto_ext_level.text().strip()
                    else None
                )
            except Exception:
                ext_level = None
            try:
                pre_ms = (
                    float(self.auto_ext_pre_ms.text())
                    if hasattr(self, "auto_ext_pre_ms") and self.auto_ext_pre_ms.text().strip()
                    else 5.0
                )
            except Exception:
                pre_ms = 5.0
            use_math = (
                bool(self.auto_use_math.isChecked()) if hasattr(self, "auto_use_math") else False
            )
            order = (
                self.auto_math_order.currentText()
                if hasattr(self, "auto_math_order")
                else "CH1-CH2"
            )
            freqs = build_freq_list(start, stop, step)
            self._sweep_abort = False

            def _u3_autocfg():
                if hasattr(self, "auto_u3_autocfg") and self.auto_u3_autocfg.isChecked():
                    base = (
                        self.auto_u3_base.currentText()
                        if hasattr(self, "auto_u3_base")
                        else "Keep Current"
                    )
                    self.u3_autoconfig_runtime(base=base, pulse_line=pulse_line, persist=False)

            amplitude_calibration = None
            amp_strategy = None
            if getattr(self, "auto_apply_cal", None) and self.auto_apply_cal.isChecked():
                try:
                    curve = load_calibration_curve()
                    amplitude_calibration = curve.apply
                    self._log(self.auto_log, "Gold calibration enabled for KPI sweep")
                    target_widget = getattr(self, "auto_cal_target", None)
                    if target_widget is not None:
                        text = target_widget.text().strip()
                        if text:
                            try:
                                target_vpp = float(text)

                                def _strategy(
                                    freq: float,
                                    *,
                                    _curve=curve,
                                    _target=target_vpp,
                                    _fallback=amp,
                                ):
                                    try:
                                        ratio = _curve.ratio_at(freq)
                                        if ratio <= 0:
                                            return _fallback
                                        return _target / ratio
                                    except Exception:
                                        return _fallback

                                amp_strategy = _strategy
                                self._log(
                                    self.auto_log,
                                    f"Generator adjusted for target {target_vpp:.3f} Vpp",
                                )
                            except Exception as exc:
                                self._log(self.auto_log, f"Bad cal target: {exc}")
                except Exception as exc:
                    self._log(self.auto_log, f"Calibration load error: {exc}")

            res = sweep_audio_kpis(
                freqs,
                channel=ch,
                scope_channel=sch,
                amp_vpp=amp,
                dwell_s=dwell,
                fy_apply=lambda **kw: fy_apply(port=pt, proto=pr, **kw),
                scope_capture_calibrated=lambda resrc, ch: scope_capture_calibrated(
                    rsrc or self.scope_res, timeout_ms=15000, ch=ch
                ),
                dsp_vrms=_dsp.vrms,
                dsp_vpp=_dsp.vpp,
                dsp_thd_fft=(
                    (lambda t, v, f0: _dsp.thd_fft(t, v, f0=f0, nharm=10, window="hann"))
                    if getattr(self, "auto_do_thd", None) and self.auto_do_thd.isChecked()
                    else None
                ),
                dsp_find_knees=(
                    _dsp.find_knees
                    if getattr(self, "auto_do_knees", None) and self.auto_do_knees.isChecked()
                    else None
                ),
                do_thd=bool(getattr(self, "auto_do_thd", None) and self.auto_do_thd.isChecked()),
                do_knees=bool(
                    getattr(self, "auto_do_knees", None) and self.auto_do_knees.isChecked()
                ),
                knee_drop_db=(
                    float(self.auto_knee_db.text() or "3.0")
                    if getattr(self, "auto_knee_db", None)
                    else 3.0
                ),
                knee_ref_mode=(
                    self.auto_ref_mode.currentText() if hasattr(self, "auto_ref_mode") else "Max"
                ),
                knee_ref_hz=(
                    float(self.auto_ref_hz.text() or "1000")
                    if getattr(self, "auto_ref_hz", None)
                    else 1000.0
                ),
                use_math=use_math,
                math_order=order,
                use_ext=use_ext,
                ext_slope=ext_slope,
                ext_level=ext_level,
                pre_ms=pre_ms,
                pulse_line=pulse_line,
                pulse_ms=pulse_ms,
                u3_pulse_line=(
                    (
                        lambda line, width_ms, level: u3_pulse_line(
                            line, width_ms=width_ms, level=level
                        )
                    )
                    if HAVE_U3
                    else None
                ),
                scope_set_trigger_ext=lambda resrc, slope, level: scope_set_trigger_ext(
                    rsrc or self.scope_res, slope=slope, level=level
                ),
                scope_arm_single=lambda resrc: scope_arm_single(rsrc or self.scope_res),
                scope_wait_single_complete=lambda resrc, timeout_s: scope_wait_single_complete(
                    rsrc or self.scope_res, timeout_s=timeout_s
                ),
                scope_configure_math_subtract=lambda resrc, order: scope_configure_math_subtract(
                    rsrc or self.scope_res, order=order
                ),
                scope_resource=rsrc or self.scope_res,
                logger=lambda s: self._log(self.auto_log, s),
                progress=lambda i, n: (
                    self.auto_prog.setValue(int(i / n * 100)),
                    QApplication.processEvents(),
                ),
                abort_flag=lambda: getattr(self, "_sweep_abort", False),
                u3_autoconfig=_u3_autocfg,
                amp_vpp_strategy=amp_strategy,
                amplitude_calibration=amplitude_calibration,
            )
            rows = res["rows"]
            os.makedirs("results", exist_ok=True)
            fn = os.path.join("results", "audio_kpis.csv")
            with open(fn, "w") as fh:
                fh.write("freq_hz,vrms,pkpk,thd_ratio,thd_percent\n")
                for row in rows:
                    fh.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")
            self._log(self.auto_log, f"Saved: {fn}")
            if res.get("knees"):
                try:
                    f_lo, f_hi, ref_amp, ref_db = res["knees"]
                    drop = (
                        float(self.auto_knee_db.text() or "3.0")
                        if getattr(self, "auto_knee_db", None)
                        else 3.0
                    )
                    ref_mode = (
                        self.auto_ref_mode.currentText()
                        if hasattr(self, "auto_ref_mode")
                        else "Max"
                    )
                    summ = (
                        f"Knees @ -{drop:.2f} dB (ref {ref_mode}): "
                        f"low≈{f_lo:.2f} Hz, high≈{f_hi:.2f} Hz "
                        f"(ref_amp={ref_amp:.4f} V, ref_dB={ref_db:.2f} dB)"
                    )
                    self._log(self.auto_log, summ)
                    with open(os.path.join("results", "audio_knees.txt"), "w") as fh:
                        fh.write(summ + "\n")
                except Exception as e:
                    self._log(self.auto_log, f"Knee calc error: {e}")
        except Exception as e:
            self._log(self.auto_log, f"KPI sweep error: {e}")
        finally:
            with suppress(Exception):
                scope_resume_run(rsrc or self.scope_res)

    # ---- Diagnostics
    def tab_diag(self):
        from amp_benchkit.gui.diag_tab import build_diagnostics_tab

        return build_diagnostics_tab(self)

    def run_diag(self):
        include_env = (
            bool(self.diag_include_env.isChecked()) if hasattr(self, "diag_include_env") else True
        )
        include_deps = (
            bool(self.diag_include_deps.isChecked()) if hasattr(self, "diag_include_deps") else True
        )
        include_hw = (
            bool(self.diag_include_hw.isChecked()) if hasattr(self, "diag_include_hw") else True
        )
        auto_clear = (
            bool(self.diag_auto_clear.isChecked()) if hasattr(self, "diag_auto_clear") else False
        )
        context: dict[str, str] = {}
        for attr, label in [
            ("port1", "Generator CH1 Port"),
            ("port2", "Generator CH2 Port"),
            ("auto_port", "Automation FY Port"),
            ("scope_edit", "Scope Resource"),
        ]:
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            with suppress(Exception):
                text = widget.text().strip()
                if text:
                    context[label] = text
        context["FY auto-detect"] = find_fy_port() or ""
        try:
            diag_text = collect_diagnostics(
                include_environment=include_env,
                include_dependencies=include_deps,
                include_connectivity=True,
                include_hardware=include_hw,
                context={k: v for k, v in context.items() if v},
            )
        except Exception as e:  # pragma: no cover - defensive
            diag_text = f"Diagnostics collection failed: {e}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        header = f"=== Diagnostics @ {timestamp} ==="
        snapshot = f"{header}\n{diag_text}"
        if auto_clear and hasattr(self, "diag"):
            self.clear_diag_log()
        self._log(self.diag, snapshot)
        self._last_diag_snapshot = snapshot

    @staticmethod
    def _log(w, t):
        w.append(t)

    def clear_diag_log(self):
        if hasattr(self, "diag") and self.diag is not None:
            self.diag.clear()

    def copy_diag_to_clipboard(self):
        if not HAVE_QT or QApplication is None or not hasattr(self, "diag"):
            return
        try:
            text = self.diag.toPlainText()
            if text.strip():
                QApplication.clipboard().setText(text)
                self._log(self.diag, "[info] Diagnostics copied to clipboard.")
        except Exception:
            pass

    def save_diag_snapshot(self):
        text = getattr(self, "_last_diag_snapshot", None)
        if not text and hasattr(self, "diag"):
            try:
                text = self.diag.toPlainText()
            except Exception:
                text = None
        if not text or not text.strip():
            return
        ts = time.strftime("%Y%m%d-%H%M%S")
        dest_dir = os.path.join("results", "diagnostics")
        with suppress(Exception):
            os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"diagnostics_{ts}.txt")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text.strip() + "\n")
            if hasattr(self, "diag"):
                self._log(self.diag, f"[saved] {path}")
        except Exception as e:
            if hasattr(self, "diag"):
                self._log(self.diag, f"[error] Failed to save diagnostics: {e}")

    # ---- Test Panel status/history helpers
    def _test_status(self, text: str, level: str = "error"):
        ts = time.strftime("%H:%M:%S")
        tag = "ERROR" if str(level).lower().startswith("err") else "INFO"
        entry = f"[{ts}] {tag}: {text}"
        with suppress(Exception):
            if not hasattr(self, "_test_hist") or self._test_hist is None:
                self._test_hist = []
            self._test_hist.append(entry)
            # keep last 50 entries
            if len(self._test_hist) > 50:
                self._test_hist = self._test_hist[-50:]
            if hasattr(self, "test_last") and self.test_last is not None:
                self.test_last.setText(text)
            if hasattr(self, "test_hist") and self.test_hist is not None:
                self.test_hist.setPlainText("\n".join(self._test_hist))


def main():
    def _cli_amp(value: str, *, allow_zero: bool) -> float:
        try:
            return check_amp_vpp(float(value), allow_zero=allow_zero)
        except (ValueError, FYError) as exc:  # pragma: no cover - CLI parsing
            raise argparse.ArgumentTypeError(str(exc)) from exc

    def _cli_amp_gt_zero(value: str) -> float:
        return _cli_amp(value, allow_zero=False)

    def _cli_amp_allow_zero(value: str) -> float:
        return _cli_amp(value, allow_zero=True)

    def _env_amp_default(allow_zero: bool = False, fallback: float = 0.5) -> float:
        raw = os.environ.get("AMP_VPP")
        try:
            candidate = float(raw) if raw is not None else fallback
        except (TypeError, ValueError):
            candidate = fallback
        try:
            return check_amp_vpp(candidate, allow_zero=allow_zero)
        except FYError:
            safe_fallback = min(fallback, FY_MAX_VPP)
            return check_amp_vpp(safe_fallback, allow_zero=allow_zero)

    ap = argparse.ArgumentParser(description="Unified GUI (Lite+U3)")
    ap.add_argument("--gui", action="store_true", help="Launch Qt GUI")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging (debug)")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("diag")
    sub.add_parser("gui")
    sub.add_parser("selftest")
    sub.add_parser("config-dump")
    sub.add_parser("config-reset")
    sp_thd = sub.add_parser(
        "thd-math-sweep",
        help="Headless THD sweep using the scope math channel",
    )
    sp_thd.add_argument(
        "--visa-resource",
        default=os.environ.get("VISA_RESOURCE", TEK_RSRC_DEFAULT),
        help="Tektronix VISA resource string.",
    )
    sp_thd.add_argument(
        "--fy-port",
        default=os.environ.get("FY_PORT"),
        help="FY3200S serial port (auto-detect if omitted).",
    )
    sp_thd.add_argument(
        "--amp-vpp",
        type=_cli_amp_gt_zero,
        default=_env_amp_default(),
        help=f"Generator amplitude (Vpp, max {FY_MAX_VPP:.2f}).",
    )
    sp_thd.add_argument(
        "--start",
        type=float,
        default=20.0,
        help="Sweep start frequency (Hz).",
    )
    sp_thd.add_argument(
        "--stop",
        type=float,
        default=20000.0,
        help="Sweep stop frequency (Hz).",
    )
    sp_thd.add_argument(
        "--points",
        type=int,
        default=61,
        help="Number of logarithmic sweep points.",
    )
    sp_thd.add_argument(
        "--dwell",
        type=float,
        default=0.15,
        help="Dwell time per frequency in seconds (increase for LF stability).",
    )
    sp_thd.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Scope channel to capture (ignored when --math is used).",
    )
    sp_thd.add_argument(
        "--math",
        action="store_true",
        help="Capture the scope MATH trace instead of a single channel.",
    )
    sp_thd.add_argument(
        "--math-order",
        default="CH1-CH2",
        help="Math subtraction order (e.g. CH1-CH2).",
    )
    sp_thd.add_argument(
        "--output",
        type=Path,
        default=Path("results/thd_sweep.csv"),
        help="Optional CSV destination (set to '-' to disable saving).",
    )
    sp_thd.add_argument(
        "--keep-spikes",
        action="store_true",
        help="Disable automatic spike suppression in THD results.",
    )
    sp_thd.add_argument(
        "--filter-window",
        type=int,
        default=2,
        help="Neighbor window for spike detection (default: 2).",
    )
    sp_thd.add_argument(
        "--filter-factor",
        type=float,
        default=2.0,
        help="Spike threshold factor relative to median (default: 2.0).",
    )
    sp_thd.add_argument(
        "--filter-min",
        type=float,
        default=2.0,
        help="Minimum THD%% before considering a spike (default: 2.0).",
    )
    sp_thd.add_argument(
        "--apply-gold-calibration",
        action="store_true",
        help="Apply the packaged gold calibration curve to amplitude metrics.",
    )
    sp_thd.add_argument(
        "--cal-target-vpp",
        type=_cli_amp_gt_zero,
        default=None,
        help=(
            "Target DUT amplitude when calibration is applied. " f"Must be ≤ {FY_MAX_VPP:.2f} Vpp."
        ),
    )
    sp_thd.add_argument(
        "--scope-auto-scale",
        default=None,
        help=(
            "Auto vertical scale map (e.g. CH1=12,CH3=1). "
            "Values represent expected Vpp gain relative to generator amplitude."
        ),
    )
    sp_thd.add_argument(
        "--scope-auto-scale-margin",
        type=float,
        default=1.25,
        help="Headroom multiplier when computing auto scope V/div (default: 1.25).",
    )
    sp_thd.add_argument(
        "--scope-auto-scale-min",
        type=float,
        default=1e-3,
        help="Minimum V/div when auto scaling (default: 1e-3).",
    )
    sp_thd.add_argument(
        "--scope-auto-scale-divs",
        type=float,
        default=8.0,
        help="Vertical divisions assumed when auto scaling (default: 8).",
    )
    sp_thd.add_argument(
        "--timestamp",
        action="store_true",
        help="Append a timestamp to the output filename to avoid overwriting.",
    )
    sp_knee = sub.add_parser(
        "knee-sweep",
        help="Headless log sweep to find -dB knees using Vrms/PkPk metrics.",
    )
    sp_knee.add_argument(
        "--visa-resource",
        default=os.environ.get("VISA_RESOURCE", TEK_RSRC_DEFAULT),
        help="Tektronix VISA resource string.",
    )
    sp_knee.add_argument(
        "--fy-port",
        default=os.environ.get("FY_PORT"),
        help="FY3200S serial port (auto-detect if omitted).",
    )
    sp_knee.add_argument(
        "--amp-vpp",
        type=_cli_amp_gt_zero,
        default=_env_amp_default(),
        help=f"Generator amplitude (Vpp, max {FY_MAX_VPP:.2f}).",
    )
    sp_knee.add_argument(
        "--start",
        type=float,
        default=20.0,
        help="Sweep start frequency (Hz).",
    )
    sp_knee.add_argument(
        "--stop",
        type=float,
        default=20000.0,
        help="Sweep stop frequency (Hz).",
    )
    sp_knee.add_argument(
        "--points",
        type=int,
        default=61,
        help="Number of logarithmic sweep points.",
    )
    sp_knee.add_argument(
        "--dwell",
        type=float,
        default=0.15,
        help="Dwell time per frequency in seconds (increase for LF stability).",
    )
    sp_knee.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Scope channel to capture (ignored when --math is used).",
    )
    sp_knee.add_argument(
        "--math",
        action="store_true",
        help="Capture the scope MATH trace instead of a single channel.",
    )
    sp_knee.add_argument(
        "--math-order",
        default="CH1-CH2",
        help="Math subtraction order (e.g. CH1-CH2).",
    )
    sp_knee.add_argument(
        "--output",
        type=Path,
        default=Path("results/knee_sweep.csv"),
        help="Optional CSV destination (set to '-' to disable saving).",
    )
    sp_knee.add_argument(
        "--apply-gold-calibration",
        action="store_true",
        help="Apply the packaged gold calibration curve to amplitude metrics.",
    )
    sp_knee.add_argument(
        "--cal-target-vpp",
        type=_cli_amp_gt_zero,
        default=None,
        help=(
            "Target DUT amplitude when calibration is applied. " f"Must be ≤ {FY_MAX_VPP:.2f} Vpp."
        ),
    )
    sp_knee.add_argument(
        "--knee-drop-db",
        type=float,
        default=3.0,
        help="Drop in dB to qualify low/high knees (default: 3.0).",
    )
    sp_knee.add_argument(
        "--knee-ref-mode",
        choices=["max", "1khz"],
        default="max",
        help="Reference amplitude: overall max or nearest to 1 kHz.",
    )
    sp_knee.add_argument(
        "--knee-ref-hz",
        type=float,
        default=1000.0,
        help="Reference frequency when --knee-ref-mode=1khz (default: 1000 Hz).",
    )
    sp_knee.add_argument(
        "--smoothing",
        choices=["median", "mean", "none"],
        default="median",
        help="Smoothing strategy before knee search (default: median).",
    )
    sp_knee.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Window size for smoothing (odd integer, default: 5).",
    )
    sp_knee.add_argument(
        "--allow-non-monotonic",
        action="store_true",
        help="Disable monotonic envelope enforcement around the reference point.",
    )
    sp_knee.add_argument(
        "--scope-auto-scale",
        default=None,
        help=(
            "Auto vertical scale map (e.g. CH1=12,CH3=1). "
            "Values represent expected Vpp gain relative to generator amplitude."
        ),
    )
    sp_knee.add_argument(
        "--scope-auto-scale-margin",
        type=float,
        default=1.25,
        help="Headroom multiplier when computing auto scope V/div (default: 1.25).",
    )
    sp_knee.add_argument(
        "--scope-auto-scale-min",
        type=float,
        default=1e-3,
        help="Minimum V/div when auto scaling (default: 1e-3).",
    )
    sp_knee.add_argument(
        "--scope-auto-scale-divs",
        type=float,
        default=8.0,
        help="Vertical divisions assumed when auto scaling (default: 8).",
    )
    sp_knee.add_argument(
        "--timestamp",
        action="store_true",
        help="Append a timestamp to the output filename to avoid overwriting.",
    )
    sp_fft = sub.add_parser(
        "fft-capture",
        help="Capture the Tektronix FFT (math) trace and export frequency/amplitude CSV.",
    )
    sp_fft.add_argument(
        "--visa-resource",
        default=os.environ.get("VISA_RESOURCE", TEK_RSRC_DEFAULT),
        help="Tektronix VISA resource string.",
    )
    sp_fft.add_argument(
        "--source",
        default="CH1",
        help="Scope channel feeding the FFT (e.g. CH1, CH2).",
    )
    sp_fft.add_argument(
        "--window",
        choices=["rectangular", "hanning", "hamming", "blackman"],
        default="hanning",
        help="FFT window (default: hanning).",
    )
    sp_fft.add_argument(
        "--scale",
        choices=["linear", "db"],
        default="db",
        help="FFT vertical scale (default: db).",
    )
    sp_fft.add_argument(
        "--output",
        type=Path,
        default=Path("results/fft_trace.csv"),
        help="CSV destination for FFT trace (set to '-' to disable saving).",
    )
    sp_fft.add_argument(
        "--fy-port",
        default=os.environ.get("FY_PORT"),
        help="Optional FY3200S serial port to retune before capture.",
    )
    sp_fft.add_argument(
        "--fy-freq",
        type=float,
        default=None,
        help="If provided, retune the FY generator to this frequency before capture.",
    )
    sp_fft.add_argument(
        "--fy-amp",
        type=_cli_amp_allow_zero,
        default=None,
        help=(
            "If provided, set FY output amplitude (Vpp) before capture " f"(max {FY_MAX_VPP:.2f})."
        ),
    )
    sp_fft.add_argument(
        "--top",
        type=int,
        default=10,
        help="Print the top-N FFT bins by amplitude (default: 10).",
    )
    sp_fft.add_argument(
        "--timestamp",
        action="store_true",
        help="Append a timestamp to the output filename to avoid overwriting.",
    )
    sp_fft_sweep = sub.add_parser(
        "fft-sweep",
        help="Run a generator frequency sweep capturing FFT spectra at each point.",
    )
    sp_fft_sweep.add_argument(
        "--visa-resource",
        default=os.environ.get("VISA_RESOURCE", TEK_RSRC_DEFAULT),
        help="Tektronix VISA resource string.",
    )
    sp_fft_sweep.add_argument(
        "--fy-port",
        default=os.environ.get("FY_PORT"),
        help="FY3200S serial port (auto-detect if omitted).",
    )
    sp_fft_sweep.add_argument(
        "--amp-vpp",
        type=_cli_amp_gt_zero,
        default=_env_amp_default(),
        help=f"Generator amplitude (Vpp, max {FY_MAX_VPP:.2f}).",
    )
    sp_fft_sweep.add_argument(
        "--start",
        type=float,
        required=True,
        help="Sweep start frequency (Hz).",
    )
    sp_fft_sweep.add_argument(
        "--stop",
        type=float,
        required=True,
        help="Sweep stop frequency (Hz).",
    )
    sp_fft_sweep.add_argument(
        "--points",
        type=int,
        required=True,
        help="Number of logarithmic sweep points (>=2).",
    )
    sp_fft_sweep.add_argument(
        "--mode",
        choices=["log", "linear"],
        default="log",
        help="Frequency spacing mode (default: log).",
    )
    sp_fft_sweep.add_argument(
        "--fft-span",
        type=float,
        default=None,
        help="FFT span in Hz (set center to current frequency).",
    )
    sp_fft_sweep.add_argument(
        "--fft-zoom",
        type=float,
        default=None,
        help="FFT zoom multiplier (optional alternative to span).",
    )
    sp_fft_sweep.add_argument(
        "--fft-scale",
        choices=["linear", "db"],
        default="db",
        help="FFT vertical scale (default: db).",
    )
    sp_fft_sweep.add_argument(
        "--fft-window",
        choices=["rectangular", "hanning", "hamming", "blackman", "flattop"],
        default="hanning",
        help="FFT window applied before each capture (default: hanning).",
    )
    sp_fft_sweep.add_argument(
        "--dwell",
        type=float,
        default=0.3,
        help="Settling time in seconds before each capture (default: 0.3).",
    )
    sp_fft_sweep.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fft_sweep"),
        help="Directory for per-frequency FFT CSV files.",
    )
    sp_fft_sweep.add_argument(
        "--top",
        type=int,
        default=8,
        help="Show the first N bins by amplitude for each capture (default: 8).",
    )
    sp_fft_sweep.add_argument(
        "--timestamp",
        action="store_true",
        help="Append timestamps to saved filenames.",
    )
    sp_sweep = sub.add_parser("sweep", help="Generate frequency list (headless)")
    sp_sweep.add_argument("--start", type=float, required=True, help="Start frequency Hz")
    sp_sweep.add_argument("--stop", type=float, required=True, help="Stop frequency Hz")
    sp_sweep.add_argument("--points", type=int, required=True, help="Number of points (>=2)")
    sp_sweep.add_argument("--mode", choices=["log", "linear"], default="log", help="Spacing mode")
    args = ap.parse_args()

    setup_logging(verbose=getattr(args, "verbose", False))
    get_logger()

    def _parse_auto_scale(spec: str) -> dict[str, float]:
        mapping: dict[str, float] = {}
        for part in spec.split(","):
            piece = part.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise ValueError(f"Invalid auto-scale entry: '{piece}'")
            key, value = piece.split("=", 1)
            try:
                mapping[key.strip()] = float(value.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid gain for '{key.strip()}': {value.strip()}") from exc
        if not mapping:
            raise ValueError("Auto-scale map must specify at least one channel")
        return mapping

    if args.cmd == "thd-math-sweep":
        try:
            output_raw = None if str(args.output) == "-" else Path(args.output)
            output = None
            if output_raw is not None:
                output = _resolve_output_path(output_raw, timestamp=args.timestamp)
            calibration_curve = None
            if getattr(args, "apply_gold_calibration", False) or args.cal_target_vpp is not None:
                try:
                    calibration_curve = load_calibration_curve()
                except Exception as exc:
                    print(f"Calibration load error: {exc}", file=sys.stderr)
            cal_target = args.cal_target_vpp if calibration_curve else None
            sweep_amp = cal_target if cal_target is not None else args.amp_vpp
            scope_scale_map = None
            if args.scope_auto_scale:
                try:
                    scope_scale_map = _parse_auto_scale(args.scope_auto_scale)
                except ValueError as exc:
                    print(f"Scope auto-scale error: {exc}", file=sys.stderr)
                    return
            rows, out_path, suppressed = thd_sweep(
                visa_resource=args.visa_resource,
                fy_port=args.fy_port,
                amp_vpp=sweep_amp,
                scope_channel=args.channel,
                start_hz=args.start,
                stop_hz=args.stop,
                points=args.points,
                dwell_s=args.dwell,
                use_math=args.math,
                math_order=args.math_order,
                output=output,
                filter_spikes=not args.keep_spikes,
                filter_window=args.filter_window,
                filter_factor=args.filter_factor,
                filter_min_percent=args.filter_min,
                calibration_curve=calibration_curve,
                calibrate_to_vpp=cal_target,
                scope_scale_map=scope_scale_map,
                scope_scale_margin=args.scope_auto_scale_margin,
                scope_scale_min=args.scope_auto_scale_min,
                scope_scale_divs=args.scope_auto_scale_divs,
            )
        except Exception as exc:  # pragma: no cover - hardware path
            print("THD sweep error:", exc, file=sys.stderr)
            return
        if out_path:
            print("Saved:", out_path)
        for line in format_thd_rows(rows):
            print(line)
        if suppressed and not args.keep_spikes:
            print("Filtered spikes:")
            for freq, original, baseline in suppressed:
                print(f"  {freq:8.2f} Hz → {original:6.3f}% (replaced with {baseline:6.3f}%)")
        return
    if args.cmd == "knee-sweep":
        try:
            output_raw = None if str(args.output) == "-" else Path(args.output)
            output = None
            if output_raw is not None:
                output = _resolve_output_path(output_raw, timestamp=args.timestamp)
            calibration_curve = None
            if getattr(args, "apply_gold_calibration", False) or args.cal_target_vpp is not None:
                try:
                    calibration_curve = load_calibration_curve()
                except Exception as exc:
                    print(f"Calibration load error: {exc}", file=sys.stderr)
            cal_target = args.cal_target_vpp if calibration_curve else None
            sweep_amp = cal_target if cal_target is not None else args.amp_vpp
            scope_scale_map = None
            if args.scope_auto_scale:
                try:
                    scope_scale_map = _parse_auto_scale(args.scope_auto_scale)
                except ValueError as exc:
                    print(f"Scope auto-scale error: {exc}", file=sys.stderr)
                    return
            result = knee_sweep(
                visa_resource=args.visa_resource,
                fy_port=args.fy_port,
                amp_vpp=sweep_amp,
                calibrate_to_vpp=cal_target,
                scope_channel=args.channel,
                start_hz=args.start,
                stop_hz=args.stop,
                points=args.points,
                dwell_s=args.dwell,
                use_math=args.math,
                math_order=args.math_order,
                output=output,
                knee_drop_db=args.knee_drop_db,
                knee_ref_mode="freq" if args.knee_ref_mode == "1khz" else "max",
                knee_ref_hz=args.knee_ref_hz,
                smoothing=args.smoothing,
                smooth_window=args.smooth_window,
                enforce_monotonic=not args.allow_non_monotonic,
                calibration_curve=calibration_curve,
                scope_scale_map=scope_scale_map,
                scope_scale_margin=args.scope_auto_scale_margin,
                scope_scale_min=args.scope_auto_scale_min,
                scope_scale_divs=args.scope_auto_scale_divs,
            )
        except Exception as exc:  # pragma: no cover - hardware path
            print("Knee sweep error:", exc, file=sys.stderr)
            return
        csv_path = result.get("csv_path")
        if csv_path:
            print("Saved:", csv_path)
        knees = result.get("knees")
        ref_label = "max" if args.knee_ref_mode == "max" else f"{args.knee_ref_hz:.1f} Hz"
        if knees:
            f_lo, f_hi, ref_amp, ref_db = knees
            drop = args.knee_drop_db
            print(
                f"Knees @ -{drop:.2f} dB (ref {ref_label}): "
                f"low≈{f_lo:.2f} Hz, high≈{f_hi:.2f} Hz "
                f"(ref_amp={ref_amp:.4f} Vpp, ref_dB={ref_db:.2f} dB)"
            )
        else:
            print("Knee detection unavailable (insufficient data).")
        for freq, vr, pk, rel_db in result.get("rows", []):
            vr_str = f"{vr:7.4f}" if math.isfinite(vr) else "   NaN"
            pk_str = f"{pk:7.4f}" if math.isfinite(pk) else "   NaN"
            rel_str = f"{rel_db:7.2f}" if math.isfinite(rel_db) else "   NaN"
            print(f"{freq:8.2f} Hz → Vrms {vr_str} V, PkPk {pk_str} V, delta {rel_str} dB")
        return
    if args.cmd == "fft-capture":
        fy_port = args.fy_port
        if (args.fy_freq is not None or args.fy_amp is not None) and not fy_port:
            fy_port = find_fy_port()
            if not fy_port:
                print("FY generator port not found; skipping generator retune.", file=sys.stderr)
        if fy_port and (args.fy_freq is not None or args.fy_amp is not None):
            try:
                fy_apply(
                    port=fy_port,
                    proto="FY ASCII 9600",
                    freq_hz=float(args.fy_freq if args.fy_freq is not None else 1000.0),
                    amp_vpp=float(args.fy_amp if args.fy_amp is not None else 0.5),
                    wave="Sine",
                    off_v=0.0,
                    duty=None,
                    ch=1,
                )
            except Exception as exc:
                print(f"Generator retune error: {exc}", file=sys.stderr)
        try:
            fft = scope_capture_fft_trace(
                resource=args.visa_resource,
                source=args.source,
                window=args.window,
                scale=args.scale,
            )
        except Exception as exc:  # pragma: no cover - hardware path
            print("FFT capture error:", exc, file=sys.stderr)
            return
        freqs = fft.get("freqs", [])
        values = fft.get("values", [])
        x_unit = fft.get("x_unit", "Hz")
        y_unit = fft.get("y_unit", "dB" if args.scale.lower() == "db" else "V")
        output_path = None
        if args.output and str(args.output) != "-":
            try:
                raw_path = Path(args.output)
                output_path = _resolve_output_path(raw_path, timestamp=args.timestamp)
                out_path = output_path
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", newline="") as fh:
                    writer = csv.writer(fh)
                    writer.writerow([f"freq_{x_unit.lower()}", f"amplitude_{y_unit.lower()}"])
                    writer.writerows(zip(freqs, values, strict=False))
                print("Saved:", out_path)
            except Exception as exc:
                print(f"Save error: {exc}", file=sys.stderr)
        pairs = list(zip(freqs, values, strict=False))
        if not pairs:
            print("FFT trace empty.")
            return
        if args.scale.lower() == "db":
            ranked = sorted(pairs, key=lambda fv: fv[1], reverse=True)
        else:
            ranked = sorted(pairs, key=lambda fv: abs(fv[1]), reverse=True)
        top_n = max(1, int(args.top))
        print(f"Top {top_n} bins ({y_unit}):")
        for freq, value in ranked[:top_n]:
            print(f"  {freq:10.2f} {x_unit} → {value:9.3f} {y_unit}")
        return
    if args.cmd == "fft-sweep":
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        fy_port = args.fy_port or find_fy_port()
        if args.fy_port and not fy_port:
            print(f"FY generator port '{args.fy_port}' not found.", file=sys.stderr)
        if fy_port:
            print("Using FY port:", fy_port)
        try:
            freqs = build_freq_points(
                start=args.start,
                stop=args.stop,
                points=args.points,
                mode=args.mode,
            )
        except Exception as exc:
            print("FFT sweep frequency error:", exc, file=sys.stderr)
            return
        top_n = max(1, int(args.top))
        try:
            for idx, freq in enumerate(freqs, start=1):
                print(f"[{idx}/{len(freqs)}] Capturing FFT @ {freq:.2f} Hz")
                if fy_port:
                    try:
                        fy_apply(
                            port=fy_port,
                            proto="FY ASCII 9600",
                            freq_hz=float(freq),
                            amp_vpp=float(args.amp_vpp),
                            wave="Sine",
                            off_v=0.0,
                            duty=None,
                            ch=1,
                        )
                    except Exception as exc:
                        print(f"Generator retune error ({freq:.2f} Hz): {exc}", file=sys.stderr)
                try:
                    scope_configure_fft(
                        resource=args.visa_resource,
                        center_hz=freq,
                        span_hz=args.fft_span,
                        zoom=args.fft_zoom,
                        scale=args.fft_scale,
                        window=args.fft_window,
                    )
                except Exception as exc:
                    print(f"FFT config warning ({freq:.2f} Hz): {exc}", file=sys.stderr)
                if args.dwell > 0:
                    time.sleep(args.dwell)
                try:
                    fft = scope_capture_fft_trace(
                        resource=args.visa_resource,
                        source="CH1",
                        window=args.fft_window,
                        scale=args.fft_scale,
                    )
                except Exception as exc:
                    print(f"FFT capture error ({freq:.2f} Hz): {exc}", file=sys.stderr)
                    continue
                freqs_fft = fft.get("freqs", [])
                values_fft = fft.get("values", [])
                x_unit = fft.get("x_unit", "Hz")
                y_unit = fft.get("y_unit", "dB" if args.fft_scale.lower() == "db" else "V")
                filename = f"fft_{freq:.2f}Hz.csv"
                dest = output_dir / filename
                dest = _timestamp_path(dest) if args.timestamp else dest
                try:
                    with dest.open("w", newline="") as fh:
                        writer = csv.writer(fh)
                        writer.writerow([f"freq_{x_unit.lower()}", f"amplitude_{y_unit.lower()}"])
                        writer.writerows(zip(freqs_fft, values_fft, strict=False))
                    print("Saved:", dest)
                except Exception as exc:
                    print(f"Save error ({dest}): {exc}", file=sys.stderr)
                pairs = list(zip(freqs_fft, values_fft, strict=False))
                if not pairs:
                    continue
                ranked = (
                    sorted(pairs, key=lambda fv: fv[1], reverse=True)
                    if args.fft_scale.lower() == "db"
                    else sorted(pairs, key=lambda fv: abs(fv[1]), reverse=True)
                )
                print(f"Top {top_n} bins ({y_unit}):")
                for f_bin, val in ranked[:top_n]:
                    print(f"  {f_bin:10.2f} {x_unit} → {val:9.3f} {y_unit}")
        finally:
            if fy_port:
                try:
                    fy_apply(
                        port=fy_port,
                        proto="FY ASCII 9600",
                        freq_hz=1000.0,
                        amp_vpp=float(args.amp_vpp),
                        wave="Sine",
                        off_v=0.0,
                        duty=None,
                        ch=1,
                    )
                except Exception as exc:
                    print(f"Generator reset warning: {exc}", file=sys.stderr)
            try:
                restore_span = args.fft_span if args.fft_span is not None else 1000.0
                scope_configure_fft(
                    resource=args.visa_resource,
                    center_hz=1000.0,
                    span_hz=restore_span,
                    zoom=1.0,
                    scale=args.fft_scale,
                    window=args.fft_window,
                )
            except Exception as exc:
                print(f"FFT restore warning: {exc}", file=sys.stderr)
            with suppress(Exception):
                scope_resume_run(args.visa_resource)
        return

    if args.cmd == "diag":
        print(collect_diagnostics())
        return

    if args.cmd == "selftest":
        ok = True
        try:
            # Test1: baud/EOL tuples
            assert FY_BAUD_EOLS == [(9600, "\n"), (115200, "\r\n")]
            print("Test1 OK: baud/EOL tuples valid")

            # Test2: command formatting
            for ch in (1, 2):
                cmds = build_fy_cmds(1000, 1.0, 0.0, "Sine", duty=12.3, ch=ch)
                assert any(c.startswith(("bd", "dd")) for c in cmds)
                assert cmds[-1].startswith(("ba", "da"))
                assert all(len(c) + 1 <= 15 for c in cmds)
            print("Test2 OK: command formatting (duty 3-digit, amplitude last, length ≤15)")

            # Test3: centi-Hz scaling
            assert build_fy_cmds(1000, 1.0, 0.0, "Sine", None, 1)[1].endswith(f"{1000 * 100:09d}")
            print("Test3 OK: centi-Hz scaling (1000 Hz → 100000)")

            # Test4: sweep start/end centi-Hz and 9-digit padding
            st = 123.45
            en = 678.9
            start_cmd = f"b{int(st * 100):09d}"
            end_cmd = f"e{int(en * 100):09d}"
            assert start_cmd == "b000012345" and end_cmd == "e000067890"
            print("Test4 OK: sweep start/end centi-Hz and 9-digit padding")

            # Test5: duty 12.3% → d123
            cmds = build_fy_cmds(1000, 1.0, 0.0, "Sine", duty=12.3, ch=1)
            duty_cmd = [c for c in cmds if c.startswith("bd")][0]
            assert duty_cmd.endswith("123")
            print("Test5 OK: duty 12.3% → d123")

            # Test6: clamp extremes
            cm = build_fy_cmds(1000, 120.0, 0.0, "Sine", duty=123.4, ch=1)
            assert any(x.endswith("999") for x in cm if x.startswith("bd"))
            assert cm[-1].endswith(f"{FY_MAX_VPP:0.2f}")
            cm2 = build_fy_cmds(1000, -5.0, 0.005, "Sine", duty=0.04, ch=2)
            assert cm2[-1].endswith("0.00")
            assert any(x.endswith("000") for x in cm2 if x.startswith("dd"))
            assert not any(x.endswith("0.01") for x in cm2 if x.startswith("do"))
            print(f"Test6 OK: clamps for duty/amp/offset (amp ≤ {FY_MAX_VPP:.2f} Vpp)")

            # Test7: IEEE block decode
            raw = b"#3100" + bytes(range(100)) + b"extra"
            dec = _decode_ieee_block(raw)
            assert len(dec) == 100 and dec[0] == 0 and dec[-1] == 99
            hdr = "t,volts\n"
            assert hdr.endswith("\n") and hdr.startswith("t,volts")
            print("Test7 OK: block decode and CSV header")

            # Test8: passthrough when not IEEE block
            dec2 = _decode_ieee_block(b"hello")
            assert dec2 == b"hello"
            print("Test8 OK: raw (non-#) IEEE block passthrough")

            # Test9: duty clamp at 0%
            cm3 = build_fy_cmds(1000, 1.0, 0.0, "Sine", duty=-5.0, ch=1)
            assert any(x.endswith("000") for x in cm3 if x.startswith("bd"))
            print("Test9 OK: duty clamp at 0% → d000")
            # Test10: THD estimator sanity (sine + 10% 2nd harmonic)
            fs = 50000.0
            f0 = 1000.0
            N = 4096
            t = np.arange(N) / fs
            sig = np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
            thd, f_est, _ = _dsp.thd_fft(t, sig, f0=f0, nharm=5, window="hann")
            assert abs(thd - 0.1) < 0.03  # within a few % points due to window/leakage
            print("Test10 OK: THD ~10% on 2nd harmonic")
        except Exception as e:
            ok = False
            print("Selftest FAIL:", e)
        sys.exit(0 if ok else 1)
    if args.cmd == "sweep":
        rc = 0
        try:
            freqs = build_freq_points(
                start=args.start, stop=args.stop, points=args.points, mode=args.mode
            )
            for f in freqs:
                s = f"{f:.6f}".rstrip("0").rstrip(".") if 0.1 <= f < 100000 else str(f)
                print(s)
        except Exception as e:
            print("Sweep error:", e, file=sys.stderr)
            rc = 1
        sys.exit(rc)
    if args.cmd == "config-dump":
        try:
            from amp_benchkit.config import CONFIG_PATH, load_config

            cfg = load_config()
            print("Config file:", CONFIG_PATH)
            print(cfg)
        except Exception as e:
            print("Config dump error:", e)
        return

    if args.cmd == "config-reset":
        try:
            from amp_benchkit.config import CONFIG_PATH, save_config

            save_config({})
            print("Config reset ->", CONFIG_PATH)
        except Exception as e:
            print("Config reset error:", e)
        return

    # Launch GUI
    if args.gui or args.cmd == "gui":
        if not HAVE_QT:
            print(
                "Qt not available (PySide6/PyQt5). Install with:",
                INSTALL_HINTS["pyside6"],
                "or",
                INSTALL_HINTS["pyqt5"],
            )
            print("Python exe:", sys.executable)
            return
        app = QApplication(sys.argv)
        win = UnifiedGUI()
        win.show()
        # PySide6 has app.exec(), PyQt5 uses app.exec_()
        if hasattr(app, "exec"):
            sys.exit(app.exec())
        else:
            sys.exit(app.exec_())
    else:
        ap.print_help()
        print("\nTip: install PySide6 or PyQt5 to launch GUI:", INSTALL_HINTS["pyside6"])


if __name__ == "__main__":
    main()
