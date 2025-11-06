#!/usr/bin/env python3
"""Live FFT sweep helper for Tektronix TDS2024B + FY3200S.

This script automates a frequency sweep while the scope is in FFT (math) mode.
It retunes the FY3200S generator, adjusts the FFT span/zoom, enforces vertical
scale/position, captures the FFT trace, and stores the results with timestamps.

Example (defaults match the user's current setup):

    python scripts/live_fft_sweep.py \
        --visa-resource USB0::0x0699::0x036A::C100563::INSTR \
        --start 20 --stop 20000 --points 25 \
        --amp-vpp 0.5 --fft-zoom 2 \
        --vertical-scale 10 --vertical-position 0 \
        --low-span 100 --low-stop 500 --low-timebase 0.02 \
        --timestamp --restore-freq 1000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure repository root is on PYTHONPATH when executed from checkout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amp_benchkit.automation import build_freq_points
from amp_benchkit.deps import find_fy_port
from amp_benchkit.fy import FY_MAX_VPP, FYError, check_amp_vpp, fy_apply
from amp_benchkit.tek import (
    TekError,
    scope_capture_fft_trace,
    scope_configure_fft,
    scope_configure_timebase,
    scope_read_fft_vertical_params,
    scope_resume_run,
)


def format_freq(freq: float) -> str:
    """Return sanitized frequency label for filenames."""
    if freq >= 1000:
        return f"{freq/1000:.3f}kHz".replace(".", "p")
    return f"{freq:.2f}Hz".replace(".", "p")


def ensure_output_dir(base: Path, *, timestamp: bool) -> Path:
    """Create output directory, optionally nested by timestamp."""
    if timestamp:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = base / f"fft_live_{stamp}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def pick_generator_port(user_port: str | None) -> str | None:
    """Resolve FY3200S port, honoring explicit override."""
    if user_port:
        return user_port
    return find_fy_port()


def describe_fft_baseline(resource: str | None) -> None:
    """Log current FFT scale/position for operator awareness."""
    if not resource:
        return
    params = scope_read_fft_vertical_params(resource)
    if not params:
        print("FFT vertical parameters: unable to query (non-fatal)")
        return
    scale = params.get("scale")
    position = params.get("position")
    scale_str = f"{scale:.3f}" if isinstance(scale, (int, float)) else "?"
    pos_str = f"{position:.3f}" if isinstance(position, (int, float)) else "?"
    print(f"FFT vertical (current): scale={scale_str} units/div, position={pos_str} div")


def compute_fft_span(
    freq_hz: float,
    *,
    explicit_span: float | None,
    min_span: float | None,
    factor: float | None,
) -> float:
    """Determine FFT span (Hz) for the current target frequency."""
    if explicit_span is not None and explicit_span > 0:
        return explicit_span
    min_value = min_span if (min_span is not None and min_span > 0) else 0.0
    mult = factor if (factor is not None and factor > 0) else 1.3
    return max(min_value, freq_hz * mult)


def rank_bins(
    freqs: Iterable[float],
    values: Iterable[float],
    *,
    scale: str,
    top: int,
) -> list[tuple[float, float]]:
    """Return top bins sorted by amplitude for reporting."""
    pairs = list(zip(freqs, values, strict=False))
    if not pairs:
        return []
    if scale.upper() == "DB":
        pairs.sort(key=lambda fv: fv[1], reverse=True)
    else:
        pairs.sort(key=lambda fv: abs(fv[1]), reverse=True)
    return pairs[:top]


def _coerce_float_list(data: Any) -> list[float]:
    """Best-effort conversion of arbitrary iterable/scalar data to float list."""
    if data is None:
        return []
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        out: list[float] = []
        for item in data:
            try:
                out.append(float(item))
            except (TypeError, ValueError):
                continue
        return out
    try:
        return [float(data)]
    except (TypeError, ValueError):
        return []


def capture_fft_point(
    *,
    resource: str,
    freq_hz: float,
    source: int | str,
    window: str,
    scale: str,
    vertical_scale: float | None,
    vertical_position: float | None,
) -> dict[str, list[float] | str]:
    """Configure and capture a single FFT trace."""
    return scope_capture_fft_trace(
        resource=resource,
        source=source,
        window=window,
        scale=scale,
        vertical_scale=vertical_scale,
        vertical_position=vertical_position,
    )


def interpolate_amplitude(
    freqs: Sequence[float],
    values: Sequence[float],
    target_hz: float,
    *,
    default: float = float("nan"),
) -> float:
    if not freqs or not values:
        return default
    if len(freqs) != len(values):
        raise ValueError("freq and value arrays must be same length")
    # assume freqs monotonically increasing
    last_freq = freqs[0]
    last_val = values[0]
    for f, v in zip(freqs[1:], values[1:], strict=False):
        if (last_freq <= target_hz <= f) or (last_freq >= target_hz >= f):
            if f == last_freq:
                return v
            weight = (target_hz - last_freq) / (f - last_freq)
            return last_val + weight * (v - last_val)
        last_freq, last_val = f, v
    # fallback to nearest
    nearest = min(zip(freqs, values, strict=False), key=lambda item: abs(item[0] - target_hz))
    return nearest[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live FFT sweep with timestamped capture.")
    parser.add_argument(
        "--visa-resource",
        default=os.environ.get("VISA_RESOURCE", "USB0::0x0699::0x036A::C100563::INSTR"),
        help="Tektronix VISA resource identifier.",
    )
    parser.add_argument(
        "--fy-port",
        default=os.environ.get("FY_PORT"),
        help="Optional FY3200S serial port override.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=20.0,
        help="Sweep start frequency in Hz.",
    )
    parser.add_argument(
        "--stop",
        type=float,
        default=20000.0,
        help="Sweep stop frequency in Hz.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=25,
        help="Number of sweep points (>=2).",
    )
    parser.add_argument(
        "--mode",
        choices=["log", "linear"],
        default="log",
        help="Frequency spacing mode.",
    )
    parser.add_argument(
        "--amp-vpp",
        type=_amp_type,
        default=0.5,
        help=f"Generator amplitude (Vpp) applied at each point (max {FY_MAX_VPP:.2f}).",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=0.4,
        help="Settle time (seconds) after generator retune before capture.",
    )
    parser.add_argument(
        "--fft-span",
        type=float,
        default=None,
        help="FFT span in Hz. If omitted, auto span uses --auto-span-factor.",
    )
    parser.add_argument(
        "--auto-span-factor",
        type=float,
        default=1.3,
        help="Multiplier for target frequency when auto span is active.",
    )
    parser.add_argument(
        "--min-span",
        type=float,
        default=500.0,
        help="Minimum span in Hz when auto span is active.",
    )
    parser.add_argument(
        "--low-stop",
        type=float,
        default=500.0,
        help="Upper frequency boundary (Hz) for the low-band pass (≤0 to disable).",
    )
    parser.add_argument(
        "--low-span",
        type=float,
        default=100.0,
        help="FFT span (Hz) used for the low-band pass (≤0 enables auto span).",
    )
    parser.add_argument(
        "--low-zoom",
        type=float,
        default=1.0,
        help="FFT zoom factor for the low-band pass.",
    )
    parser.add_argument(
        "--low-auto-span-factor",
        type=float,
        default=1.05,
        help="Auto-span multiplier for the low-band when --low-span ≤ 0.",
    )
    parser.add_argument(
        "--low-min-span",
        type=float,
        default=100.0,
        help="Minimum span for the low-band when auto span is active.",
    )
    parser.add_argument(
        "--timebase",
        type=float,
        default=None,
        help="Horizontal scale (seconds/div) applied before each high-band capture.",
    )
    parser.add_argument(
        "--low-timebase",
        type=float,
        default=0.02,
        help="Horizontal scale (seconds/div) applied before each low-band capture.",
    )
    parser.add_argument(
        "--fft-zoom",
        type=float,
        default=2.0,
        help="FFT zoom factor (Tektronix optional).",
    )
    parser.add_argument(
        "--fft-window",
        default="HANNING",
        choices=["RECTANGULAR", "HANNING", "HAMMING", "BLACKMAN", "FLATTOP"],
        help="FFT window function.",
    )
    parser.add_argument(
        "--fft-scale",
        default="DB",
        choices=["LINEAR", "DB"],
        help="FFT vertical scale mode.",
    )
    parser.add_argument(
        "--vertical-scale",
        type=float,
        default=10.0,
        help="FFT vertical scale (units/div).",
    )
    parser.add_argument(
        "--vertical-position",
        type=float,
        default=0.0,
        help="FFT vertical position in divisions.",
    )
    parser.add_argument(
        "--scope-source",
        default="CH1",
        help="Scope channel feeding FFT (e.g. CH1, 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fft_live"),
        help="Directory to store captured CSV files.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="Number of dominant bins to log for each capture.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Append timestamped subdirectory & filenames.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional description stored alongside summary CSV.",
    )
    parser.add_argument(
        "--restore-freq",
        type=float,
        default=1000.0,
        help="Generator frequency (Hz) to restore after sweep.",
    )
    parser.add_argument(
        "--restore-amp",
        type=_amp_type_allow_zero,
        default=None,
        help=(
            "Generator amplitude (Vpp) to restore after sweep "
            "(defaults to sweep amplitude, max "
            f"{FY_MAX_VPP:.2f})."
        ),
    )
    parser.add_argument(
        "--skip-restore",
        action="store_true",
        help="Skip generator reset and scope run-state restore.",
    )
    args = parser.parse_args()

    restore_amp = args.restore_amp if args.restore_amp is not None else args.amp_vpp
    last_span_used: float | None = None
    last_center_used: float | None = None
    last_zoom_used: float | None = None
    last_timebase_used: float | None = None

    output_dir = ensure_output_dir(args.output_dir, timestamp=args.timestamp)
    print(f"Output directory: {output_dir}")

    describe_fft_baseline(args.visa_resource)

    fy_port = pick_generator_port(args.fy_port)
    if args.fy_port and fy_port is None:
        print(f"Warning: FY port '{args.fy_port}' not found. Sweep will run without retuning.")
    if fy_port:
        print(f"Using FY3200S on: {fy_port}")
    else:
        print("FY3200S port not detected; generator retune skipped.")

    try:
        freqs = build_freq_points(
            start=args.start,
            stop=args.stop,
            points=args.points,
            mode=args.mode,
        )
    except Exception as exc:
        print(f"Frequency setup error: {exc}")
        return 1

    scope_source = args.scope_source
    if scope_source.isdigit():
        scope_source = int(scope_source)

    low_freqs: list[float] = []
    high_freqs: list[float] = freqs
    if args.low_stop and args.low_stop > 0:
        low_freqs = [f for f in freqs if f <= args.low_stop]
        high_freqs = [f for f in freqs if f > args.low_stop]
    passes = []
    if low_freqs:
        passes.append(
            {
                "label": "low",
                "freqs": low_freqs,
                "span": args.low_span,
                "zoom": args.low_zoom,
                "auto_factor": args.low_auto_span_factor,
                "min_span": args.low_min_span,
                "timebase": (
                    args.low_timebase if args.low_timebase and args.low_timebase > 0 else None
                ),
            }
        )
    if high_freqs:
        passes.append(
            {
                "label": "high" if low_freqs else "full",
                "freqs": high_freqs,
                "span": args.fft_span,
                "zoom": args.fft_zoom,
                "auto_factor": args.auto_span_factor,
                "min_span": args.min_span,
                "timebase": (args.timebase if args.timebase and args.timebase > 0 else None),
            }
        )
    total_points = sum(len(p["freqs"]) for p in passes)

    summary_rows: list[list[str | float]] = [
        [
            "test_freq_hz",
            "drive_amp_db",
            "bin_freq_hz",
            "bin_value_db",
            "bin_width_hz",
            "csv_path",
        ]
    ]

    try:
        overall_idx = 0
        for pass_cfg in passes:
            zoom_value = pass_cfg["zoom"]
            span_hint = pass_cfg["span"]
            min_span_hint = pass_cfg["min_span"]
            auto_factor_hint = pass_cfg["auto_factor"]
            timebase_value = pass_cfg.get("timebase")
            for freq in pass_cfg["freqs"]:
                overall_idx += 1
                print(
                    f"\n[{overall_idx}/{total_points}] "
                    f"({pass_cfg['label']}) Sweep @ {freq:.2f} Hz"
                )
                if fy_port:
                    try:
                        fy_apply(
                            port=fy_port,
                            proto="FY ASCII 9600",
                            freq_hz=freq,
                            amp_vpp=args.amp_vpp,
                            wave="Sine",
                            off_v=0.0,
                            duty=None,
                            ch=1,
                        )
                    except FYError as exc:
                        print(f"  FY retune warning: {exc}")
                explicit_span = span_hint if (span_hint is not None and span_hint > 0) else None
                span_to_use = compute_fft_span(
                    freq,
                    explicit_span=explicit_span,
                    min_span=min_span_hint,
                    factor=auto_factor_hint,
                )
                last_span_used = span_to_use
                last_center_used = freq
                last_zoom_used = zoom_value
                if timebase_value is not None:
                    try:
                        scope_configure_timebase(
                            resource=args.visa_resource,
                            seconds_per_div=timebase_value,
                        )
                        last_timebase_used = timebase_value
                    except Exception as exc:  # pragma: no cover - hardware path
                        print(f"  Timebase warning: {exc}")
                try:
                    scope_configure_fft(
                        resource=args.visa_resource,
                        center_hz=freq,
                        span_hz=span_to_use,
                        zoom=zoom_value,
                        scale=args.fft_scale,
                        window=args.fft_window,
                    )
                except TekError as exc:
                    print(f"  FFT configure warning: {exc}")
                if args.dwell > 0:
                    time.sleep(args.dwell)
                try:
                    fft_result = capture_fft_point(
                        resource=args.visa_resource,
                        freq_hz=freq,
                        source=scope_source,
                        window=args.fft_window,
                        scale=args.fft_scale,
                        vertical_scale=args.vertical_scale,
                        vertical_position=args.vertical_position,
                    )
                except TekError as exc:
                    print(f"  FFT capture error: {exc}")
                    continue

                freqs_fft = _coerce_float_list(fft_result.get("freqs"))
                values_fft = _coerce_float_list(fft_result.get("values"))
                x_unit_obj = fft_result.get("x_unit", "Hz")
                y_unit_obj = fft_result.get(
                    "y_unit", "dB" if args.fft_scale.upper() == "DB" else "V"
                )
                x_unit = str(x_unit_obj) if isinstance(x_unit_obj, str) else "Hz"
                y_unit = (
                    str(y_unit_obj)
                    if isinstance(y_unit_obj, str)
                    else ("dB" if args.fft_scale.upper() == "DB" else "V")
                )

                if not freqs_fft or not values_fft:
                    print("  FFT capture returned empty data.")
                    continue

                max_fft_freq = freqs_fft[-1]
                if freq > max_fft_freq * 0.98:
                    warning_msg = (
                        f"  Warning: drive {freq:.2f} Hz exceeds FFT axis max "
                        f"{max_fft_freq:.2f} {x_unit}; increase span or sampling."
                    )
                    print(warning_msg)

                label = format_freq(freq)
                filename = f"fft_{label}.csv"
                if len(passes) > 1:
                    filename = f"fft_{pass_cfg['label']}_{label}.csv"
                dest = output_dir / filename
                if args.timestamp:
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    dest = dest.with_name(dest.stem + f"_{stamp}{dest.suffix}")
                try:
                    with dest.open("w", newline="") as fh:
                        writer = csv.writer(fh)
                        writer.writerow([f"freq_{x_unit.lower()}", f"amplitude_{y_unit.lower()}"])
                        writer.writerows(zip(freqs_fft, values_fft, strict=False))
                    print(f"  Saved FFT trace → {dest}")
                except OSError as exc:
                    print(f"  Save error: {exc}")

            pairs = list(zip(freqs_fft, values_fft, strict=False))
            top_bins = rank_bins(freqs_fft, values_fft, scale=args.fft_scale, top=args.top)
            nearest_bin = None
            if top_bins:
                print(f"  Top {len(top_bins)} bins ({y_unit}):")
                for f_bin, val in top_bins:
                    print(f"    {f_bin:10.2f} {x_unit} → {val:9.3f} {y_unit}")
                nearest_bin = min(top_bins, key=lambda item: abs(item[0] - freq))
            else:
                print("  No significant bins detected.")
            bin_freq = float("nan")
            bin_val = float("nan")
            if nearest_bin is None and pairs:
                nearest_bin = min(pairs, key=lambda item: abs(item[0] - freq))
            if nearest_bin is not None:
                bin_freq, bin_val = nearest_bin
            drive_amp = interpolate_amplitude(freqs_fft, values_fft, freq, default=bin_val)
            bin_width = float("nan")
            if len(freqs_fft) > 1:
                bin_width = abs(freqs_fft[1] - freqs_fft[0])
            summary_rows.append([freq, drive_amp, bin_freq, bin_val, bin_width, dest.name])

        if len(summary_rows) > 1:
            summary_path = output_dir / "fft_sweep_summary.csv"
            try:
                with summary_path.open("w", newline="") as fh:
                    writer = csv.writer(fh)
                    if args.notes:
                        writer.writerow(["notes", args.notes])
                    writer.writerows(summary_rows)
                print(f"\nSummary saved to: {summary_path}")
            except OSError as exc:
                print(f"\nSummary save error: {exc}")
        else:
            print("\nSweep produced no captures; summary skipped.")
    finally:
        if not args.skip_restore:
            if fy_port:
                try:
                    fy_apply(
                        port=fy_port,
                        proto="FY ASCII 9600",
                        freq_hz=args.restore_freq,
                        amp_vpp=restore_amp,
                        wave="Sine",
                        off_v=0.0,
                        duty=None,
                        ch=1,
                    )
                    print(
                        f"\nRestored FY3200S → {args.restore_freq:.2f} Hz @ {restore_amp:.3f} Vpp"
                    )
                except FYError as exc:
                    print(f"\nFY restore warning: {exc}")
            try:
                restore_center = args.restore_freq if last_center_used is None else last_center_used
                if last_span_used is not None:
                    restore_span = last_span_used
                else:
                    restore_span = compute_fft_span(
                        restore_center,
                        explicit_span=args.fft_span,
                        min_span=args.min_span,
                        factor=args.auto_span_factor,
                    )
                scope_configure_fft(
                    resource=args.visa_resource,
                    center_hz=restore_center,
                    span_hz=restore_span,
                    zoom=last_zoom_used if last_zoom_used is not None else args.fft_zoom,
                    scale=args.fft_scale,
                    window=args.fft_window,
                )
            except TekError as exc:
                print(f"FFT restore warning: {exc}")
            if last_timebase_used is not None:
                try:
                    scope_configure_timebase(
                        resource=args.visa_resource,
                        seconds_per_div=last_timebase_used,
                    )
                except Exception as exc:  # pragma: no cover - hardware path
                    print(f"Timebase restore warning: {exc}")
            try:
                scope_resume_run(args.visa_resource)
            except Exception as exc:  # pragma: no cover - hardware path
                print(f"Scope resume warning: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


def _amp_type(value: str) -> float:
    try:
        return check_amp_vpp(float(value), allow_zero=False)
    except (ValueError, FYError) as exc:  # pragma: no cover - CLI parsing
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _amp_type_allow_zero(value: str) -> float:
    try:
        return check_amp_vpp(float(value), allow_zero=True)
    except (ValueError, FYError) as exc:  # pragma: no cover - CLI parsing
        raise argparse.ArgumentTypeError(str(exc)) from exc
