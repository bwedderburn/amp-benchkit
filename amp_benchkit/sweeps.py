"""Reusable sweep routines for headless instrumentation runs."""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Iterable, Sequence
from contextlib import suppress
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .automation import build_freq_points, sweep_audio_kpis
from .calibration import CalibrationCurve
from .dsp import find_knees, thd_fft, vpp, vrms
from .fy import FYError, check_amp_vpp, fy_apply
from .tek import (
    scope_arm_single,
    scope_capture_calibrated,
    scope_configure_math_subtract,
    scope_configure_timebase,
    scope_read_vertical_scale,
    scope_resume_run,
    scope_set_vertical_scale,
    scope_wait_single_complete,
)


def _filter_spikes(
    rows: list[tuple[float, float, float, float]],
    *,
    window: int,
    factor: float,
    min_percent: float,
):
    suppressed: list[tuple[float, float, float]] = []
    filtered: list[tuple[float, float, float, float]] = []
    total = len(rows)
    for idx, (freq, vr, pk, thd_percent) in enumerate(rows):
        neighbors = [
            rows[j][3]
            for j in range(max(0, idx - window), min(total, idx + window + 1))
            if j != idx and math.isfinite(rows[j][3])
        ]
        if not neighbors or not math.isfinite(thd_percent):
            filtered.append((freq, vr, pk, thd_percent))
            continue
        baseline = median(neighbors)
        threshold = max(min_percent, baseline * factor)
        if thd_percent > threshold:
            suppressed.append((freq, thd_percent, baseline))
            filtered.append((freq, vr, pk, float(baseline)))
        else:
            filtered.append((freq, vr, pk, thd_percent))
    return filtered, suppressed


def _smooth_series(values: Sequence[float], window: int, mode: str) -> list[float]:
    """Return a smoothed copy of ``values`` using a centered rolling reducer."""

    vals = list(values)
    n = len(vals)
    if n == 0 or window <= 1:
        return vals.copy()
    # enforce odd positive window
    win = max(1, int(window))
    if win % 2 == 0:
        win += 1
    half = win // 2
    padded = [vals[0]] * half + vals + [vals[-1]] * half
    out: list[float] = []
    mode = mode.lower()
    for i in range(n):
        segment = padded[i : i + win]
        finite = [abs(x) for x in segment if math.isfinite(x)]
        if not finite:
            out.append(float("nan"))
            continue
        if mode == "mean":
            out.append(float(fmean(finite)))
        else:  # default median
            out.append(float(median(finite)))
    return out


def _monotonic_envelope(values: Sequence[float], ref_index: int) -> list[float]:
    """Return values constrained to be monotonic around ``ref_index``.

    Low-frequency side (≤ ref) becomes non-decreasing, high-frequency side non-increasing.
    """

    vals = list(values)
    n = len(vals)
    if n == 0:
        return []
    idx = max(0, min(int(ref_index), n - 1))
    lo = vals[: idx + 1]
    hi = vals[idx:]

    lo_env: list[float] = []
    current = float("-inf")
    for val in lo:
        v = float(val) if math.isfinite(val) else current
        if not math.isfinite(v):
            v = current if math.isfinite(current) else 0.0
        if not math.isfinite(current) or v > current:
            current = v
        lo_env.append(current)

    hi_env_rev: list[float] = []
    current = float("-inf")
    for val in reversed(hi):
        v = float(val) if math.isfinite(val) else current
        if not math.isfinite(v):
            v = current if math.isfinite(current) else 0.0
        if not math.isfinite(current) or v > current:
            current = v
        hi_env_rev.append(current)
    hi_env = list(reversed(hi_env_rev))

    # Merge while avoiding duplicating the reference point
    return lo_env[:-1] + hi_env


def _clean_amplitudes(amps: Sequence[float]) -> list[float]:
    """Return amplitudes forced positive and finite for knee detection."""

    cleaned: list[float] = []
    last_finite = 1e-12
    for amp in amps:
        if math.isfinite(amp) and amp > 0:
            last_finite = float(abs(amp))
            cleaned.append(last_finite)
        else:
            cleaned.append(last_finite)
    return cleaned


def _reference_index(
    freqs: Sequence[float], amps: Sequence[float], ref_mode: str, ref_hz: float
) -> int:
    """Return index used as the reference amplitude for knee detection."""

    n = len(freqs)
    if n == 0:
        return 0
    finite_indices = [i for i, amp in enumerate(amps) if math.isfinite(amp) and amp > 0]
    if not finite_indices:
        return 0
    mode = ref_mode.lower()
    if mode.startswith("freq"):
        target = float(ref_hz)
        return min(finite_indices, key=lambda i: abs(freqs[i] - target))
    return max(finite_indices, key=lambda i: amps[i])


def thd_sweep(
    *,
    visa_resource: str,
    fy_port: str | None,
    amp_vpp: float = 0.5,
    calibrate_to_vpp: float | None = None,
    fy_proto: str = "FY ASCII 9600",
    scope_channel: int = 1,
    start_hz: float = 20.0,
    stop_hz: float = 20000.0,
    points: int = 61,
    dwell_s: float = 0.15,
    use_math: bool = False,
    math_order: str = "CH1-CH2",
    output: Path | None = None,
    post_freq_hz: float = 1000.0,
    post_seconds_per_div: float | None = 1e-4,
    filter_spikes: bool = True,
    filter_window: int = 2,
    filter_factor: float = 2.0,
    filter_min_percent: float = 2.0,
    calibration_curve: CalibrationCurve | None = None,
    scope_scale_map: dict[str, float] | None = None,
    scope_scale_margin: float = 1.25,
    scope_scale_min: float = 1e-3,
    scope_scale_divs: float = 8.0,
) -> tuple[
    list[tuple[float, float, float, float]],
    Path | None,
    list[tuple[float, float, float]],
]:
    """Run a THD sweep capturing either a single scope channel or the math trace.

    Returns the processed rows, optional CSV path, and a list describing any
    spike suppressions ``[(freq_hz, original_thd, replacement_thd), ...]`` when
    filtering is enabled.
    """
    if points < 2:
        raise ValueError("points must be >= 2")
    try:
        amp_vpp = check_amp_vpp(float(amp_vpp), allow_zero=False)
    except FYError as exc:
        raise ValueError(str(exc)) from exc
    if not math.isfinite(dwell_s) or dwell_s < 0:
        raise ValueError("dwell_s must be >= 0")

    log = logging.getLogger("amp_benchkit.sweeps")
    freqs = build_freq_points(start=start_hz, stop=stop_hz, points=points, mode="log")

    def _capture(_resource, ch):
        target = math_order if use_math else ch
        return scope_capture_calibrated(visa_resource, timeout_ms=15000, ch=target)

    if calibrate_to_vpp is not None and calibration_curve is None:
        raise ValueError("calibrate_to_vpp requires calibration_curve")

    amplitude_calibration = calibration_curve.apply if calibration_curve else None

    target_amp: float | None = None
    if calibrate_to_vpp is not None:
        try:
            target_amp = check_amp_vpp(float(calibrate_to_vpp), allow_zero=False)
        except FYError as exc:
            raise ValueError(str(exc)) from exc

    amp_vpp_strategy = None
    if target_amp is not None and calibration_curve is not None:

        def _amp_strategy(freq: float) -> float:
            ratio = calibration_curve.ratio_at(freq)
            candidate = target_amp if ratio <= 0 else target_amp / ratio
            return check_amp_vpp(candidate, allow_zero=False)

        amp_vpp_strategy = _amp_strategy

    sweep_amp = target_amp if target_amp is not None else amp_vpp

    def _fy_apply(**kw):
        return fy_apply(port=fy_port, proto=fy_proto, **kw)

    vertical_scale_map: dict[str, float] | None = None
    if scope_scale_map:
        normalized: dict[str, float] = {}
        for key, gain in scope_scale_map.items():
            if isinstance(key, int):
                label = f"CH{key}"
            else:
                label = str(key).strip().upper()
                if label and label[0].isdigit():
                    label = f"CH{label}"
            try:
                normalized[label or "CH1"] = float(gain)
            except Exception:
                log.warning("Invalid scope scale gain %r for %r", gain, key)
        vertical_scale_map = normalized or None

    def _scope_set_vertical_scale(_resource, channel, volts_per_div):
        return scope_set_vertical_scale(visa_resource, channel, volts_per_div)

    def _scope_read_vertical_scale(_resource, channel):
        return scope_read_vertical_scale(visa_resource, channel)

    result = sweep_audio_kpis(
        freqs,
        channel=1,
        scope_channel=scope_channel,
        amp_vpp=sweep_amp,
        dwell_s=dwell_s,
        fy_apply=_fy_apply,
        scope_capture_calibrated=_capture,
        dsp_vrms=vrms,
        dsp_vpp=vpp,
        dsp_thd_fft=lambda t, v, f0: thd_fft(t, v, f0=f0, nharm=10, window="hann"),
        do_thd=True,
        use_math=use_math,
        math_order=math_order,
        cycles_per_capture=10.0,
        scope_configure_math_subtract=(
            (lambda res, order: scope_configure_math_subtract(visa_resource, order))
            if use_math
            else None
        ),
        scope_arm_single=lambda res: scope_arm_single(visa_resource),
        scope_wait_single_complete=lambda res, timeout_s: scope_wait_single_complete(
            visa_resource, timeout_s=timeout_s
        ),
        scope_resource=visa_resource,
        scope_set_vertical_scale=_scope_set_vertical_scale if vertical_scale_map else None,
        scope_read_vertical_scale=_scope_read_vertical_scale if vertical_scale_map else None,
        vertical_scale_map=vertical_scale_map,
        vertical_scale_margin=scope_scale_margin,
        vertical_scale_min=scope_scale_min,
        vertical_scale_divs=scope_scale_divs,
        amplitude_calibration=amplitude_calibration,
        amp_vpp_strategy=amp_vpp_strategy,
    )

    rows: list[tuple[float, float, float, float]] = []
    corrected_full_rows: list[tuple[float, float, float, float, float]] = []
    for freq, vr, pk, thd_ratio, thd_percent in result["rows"]:
        if calibration_curve:
            if math.isfinite(vr):
                vr = calibration_curve.apply(freq, vr)
            if math.isfinite(pk):
                pk = calibration_curve.apply(freq, pk)
        rows.append((freq, vr, pk, thd_percent))
        corrected_full_rows.append((freq, vr, pk, thd_ratio, thd_percent))
    if calibration_curve:
        result["rows"] = corrected_full_rows

    suppressed: list[tuple[float, float, float]] = []
    if filter_spikes and rows:
        rows, suppressed = _filter_spikes(
            rows,
            window=filter_window,
            factor=filter_factor,
            min_percent=filter_min_percent,
        )

    out_path: Path | None = None
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["freq_hz", "vrms", "pkpk", "thd_percent"])
            writer.writerows(rows)

    scope_resume_run(visa_resource)
    # Restore predictable idle state (1 kHz tone, fast timebase) for quick follow-up work.
    try:  # pragma: no cover - hardware-specific
        _fy_apply(
            freq_hz=post_freq_hz,
            amp_vpp=amp_vpp,
            wave="Sine",
            off_v=0.0,
            duty=None,
            ch=1,
        )
    except Exception as exc:  # pragma: no cover - hardware-specific
        log.warning(
            "FY reset to %.1f Hz failed (amp %.2f Vpp, port=%s, proto=%s): %s",
            post_freq_hz,
            amp_vpp,
            fy_port or "<auto>",
            fy_proto,
            exc,
        )
    if post_seconds_per_div is not None:
        with suppress(Exception):  # pragma: no cover - hardware-specific
            scope_configure_timebase(visa_resource, post_seconds_per_div)
    return rows, out_path, suppressed


def format_thd_rows(rows: Iterable[tuple[float, float, float, float]]) -> list[str]:
    """Return human readable strings for THD sweep rows."""
    formatted: list[str] = []
    for freq, _vr, _pk, thd_percent in rows:
        if math.isnan(thd_percent):
            formatted.append(f"{freq:8.2f} Hz → THD NaN")
        else:
            formatted.append(f"{freq:8.2f} Hz → THD {thd_percent:6.3f}%")
    return formatted


def knee_sweep(
    *,
    visa_resource: str,
    fy_port: str | None,
    amp_vpp: float = 0.5,
    calibrate_to_vpp: float | None = None,
    fy_proto: str = "FY ASCII 9600",
    scope_channel: int = 1,
    start_hz: float = 20.0,
    stop_hz: float = 20000.0,
    points: int = 61,
    dwell_s: float = 0.15,
    use_math: bool = False,
    math_order: str = "CH1-CH2",
    output: Path | None = None,
    knee_drop_db: float = 3.0,
    knee_ref_mode: str = "max",
    knee_ref_hz: float = 1000.0,
    smoothing: str = "median",
    smooth_window: int = 5,
    enforce_monotonic: bool = True,
    calibration_curve: CalibrationCurve | None = None,
    scope_scale_map: dict[str, float] | None = None,
    scope_scale_margin: float = 1.25,
    scope_scale_min: float = 1e-3,
    scope_scale_divs: float = 8.0,
) -> dict[str, Any]:
    """Run an amplitude sweep and estimate the -dB bandwidth knees.

    Returns a dictionary with keys:
      rows: List[(freq_hz, vrms, pkpk, relative_db)]
      knees: Optional[(f_lo, f_hi, ref_amp, ref_db)]
      ref_db: float (reference amplitude in dB)
      target_db: float (reference minus knee_drop_db)
      csv_path: Optional[Path]
    """

    if points < 2:
        raise ValueError("points must be >= 2")
    try:
        amp_vpp = check_amp_vpp(float(amp_vpp), allow_zero=False)
    except FYError as exc:
        raise ValueError(str(exc)) from exc
    if not math.isfinite(dwell_s) or dwell_s < 0:
        raise ValueError("dwell_s must be >= 0")
    if not math.isfinite(knee_drop_db) or knee_drop_db <= 0:
        raise ValueError("knee_drop_db must be > 0")

    log = logging.getLogger("amp_benchkit.sweeps")
    freqs = build_freq_points(start=start_hz, stop=stop_hz, points=points, mode="log")

    def _capture(_resource, ch):
        target = math_order if use_math else ch
        return scope_capture_calibrated(visa_resource, timeout_ms=15000, ch=target)

    if calibrate_to_vpp is not None and calibration_curve is None:
        raise ValueError("calibrate_to_vpp requires calibration_curve")

    amplitude_calibration = calibration_curve.apply if calibration_curve else None

    target_amp: float | None = None
    if calibrate_to_vpp is not None:
        try:
            target_amp = check_amp_vpp(float(calibrate_to_vpp), allow_zero=False)
        except FYError as exc:
            raise ValueError(str(exc)) from exc

    amp_vpp_strategy = None
    if target_amp is not None and calibration_curve is not None:

        def _amp_strategy(freq: float) -> float:
            ratio = calibration_curve.ratio_at(freq)
            candidate = target_amp if ratio <= 0 else target_amp / ratio
            return check_amp_vpp(candidate, allow_zero=False)

        amp_vpp_strategy = _amp_strategy

    sweep_amp = target_amp if target_amp is not None else amp_vpp

    def _fy_apply(**kw):
        return fy_apply(port=fy_port, proto=fy_proto, **kw)

    vertical_scale_map: dict[str, float] | None = None
    if scope_scale_map:
        normalized: dict[str, float] = {}
        for key, gain in scope_scale_map.items():
            if isinstance(key, int):
                label = f"CH{key}"
            else:
                label = str(key).strip().upper()
                if label and label[0].isdigit():
                    label = f"CH{label}"
            try:
                normalized[label or "CH1"] = float(gain)
            except Exception:
                log.warning("Invalid scope scale gain %r for %r", gain, key)
        vertical_scale_map = normalized or None

    def _scope_set_vertical_scale(_resource, channel, volts_per_div):
        return scope_set_vertical_scale(visa_resource, channel, volts_per_div)

    def _scope_read_vertical_scale(_resource, channel):
        return scope_read_vertical_scale(visa_resource, channel)

    smoothing_mode = (smoothing or "none").lower()
    if smoothing_mode not in {"median", "mean", "none"}:
        raise ValueError("smoothing must be one of 'median', 'mean', or 'none'")
    window = max(1, int(smooth_window))

    def _knee_detector(
        freq_list: Sequence[float],
        amps: Sequence[float],
        ref_mode: str,
        ref_hz: float,
        drop_db: float,
    ) -> tuple[float, float, float, float]:
        cleaned = _clean_amplitudes(amps)
        if smoothing_mode != "none" and len(cleaned) > 1 and window > 1:
            cleaned = _smooth_series(cleaned, window, smoothing_mode)
        if enforce_monotonic and len(cleaned) > 2:
            ref_idx = _reference_index(freq_list, cleaned, ref_mode, ref_hz)
            cleaned = _monotonic_envelope(cleaned, ref_idx)
        return find_knees(freq_list, cleaned, ref_mode, ref_hz, drop_db)

    result = sweep_audio_kpis(
        freqs,
        channel=1,
        scope_channel=scope_channel,
        amp_vpp=sweep_amp,
        dwell_s=dwell_s,
        fy_apply=_fy_apply,
        scope_capture_calibrated=_capture,
        dsp_vrms=vrms,
        dsp_vpp=vpp,
        dsp_find_knees=_knee_detector,
        do_knees=True,
        use_math=use_math,
        math_order=math_order,
        cycles_per_capture=10.0,
        scope_configure_math_subtract=(
            (lambda res, order: scope_configure_math_subtract(visa_resource, order))
            if use_math
            else None
        ),
        scope_arm_single=lambda res: scope_arm_single(visa_resource),
        scope_wait_single_complete=lambda res, timeout_s: scope_wait_single_complete(
            visa_resource, timeout_s=timeout_s
        ),
        scope_resource=visa_resource,
        scope_set_vertical_scale=_scope_set_vertical_scale if vertical_scale_map else None,
        scope_read_vertical_scale=_scope_read_vertical_scale if vertical_scale_map else None,
        vertical_scale_map=vertical_scale_map,
        vertical_scale_margin=scope_scale_margin,
        vertical_scale_min=scope_scale_min,
        vertical_scale_divs=scope_scale_divs,
        amplitude_calibration=amplitude_calibration,
        amp_vpp_strategy=amp_vpp_strategy,
    )

    rows_raw: list[tuple[float, float, float, float, float]] = result["rows"]
    knees = result.get("knees")

    ref_db_val = float("nan")
    target_db = float("nan")
    if knees:
        _f_lo, _f_hi, _ref_amp, ref_db = knees
        ref_db_val = ref_db
        target_db = ref_db - knee_drop_db

    rows: list[tuple[float, float, float, float]] = []
    for freq, vr, pk, _thd_ratio, _thd_percent in rows_raw:
        if calibration_curve:
            if math.isfinite(vr):
                vr = calibration_curve.apply(freq, vr)
            if math.isfinite(pk):
                pk = calibration_curve.apply(freq, pk)
        rel_db = float("-inf")
        if knees and math.isfinite(pk) and pk > 0 and math.isfinite(ref_db_val):
            rel_db = 20.0 * math.log10(pk) - ref_db_val
        rows.append((freq, vr, pk, rel_db))

    csv_path: Path | None = None
    if output:
        csv_path = Path(output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["freq_hz", "vrms", "pkpk", "rel_db"])
            writer.writerows(rows)

    scope_resume_run(visa_resource)
    try:  # pragma: no cover - hardware-specific reset best effort
        _fy_apply(
            freq_hz=1000.0,
            amp_vpp=amp_vpp,
            wave="Sine",
            off_v=0.0,
            duty=None,
            ch=1,
        )
    except Exception as exc:  # pragma: no cover - hardware-specific
        log.warning(
            "FY reset to 1000.0 Hz failed (amp %.2f Vpp, port=%s, proto=%s): %s",
            amp_vpp,
            fy_port or "<auto>",
            fy_proto,
            exc,
        )

    return {
        "rows": rows,
        "knees": knees,
        "ref_db": ref_db_val,
        "target_db": target_db,
        "csv_path": csv_path,
    }
