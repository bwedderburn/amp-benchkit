"""Command-line helper to run a THD sweep with sensible defaults.

This mirrors the `amp-benchkit thd-math-sweep` CLI but adds auto-detection for
the Tektronix VISA resource and FY port when possible, plus timestamped outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from amp_benchkit.calibration import load_calibration_curve
from amp_benchkit.deps import HAVE_PYVISA, _pyvisa, find_fy_port
from amp_benchkit.fy import FY_MAX_VPP, FYError, check_amp_vpp
from amp_benchkit.sweeps import thd_sweep

TEK_VENDOR_IDS = {0x0699}  # Tektronix default USB vendor ID


def _timestamp_path(path: Path, enable: bool) -> Path:
    if not enable:
        return path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def _guess_visa_resource() -> str | None:
    if not HAVE_PYVISA or _pyvisa is None:
        return None
    try:
        rm = _pyvisa.ResourceManager()
        resources = rm.list_resources()
    except Exception:
        return None
    for resource in resources:
        # Quick heuristic: Tektronix resources usually look like USB0::0x0699::...
        if "USB" in resource and any(f"{vid:#06x}" in resource for vid in TEK_VENDOR_IDS):
            return resource
    # Fallback: return first USB resource
    for resource in resources:
        if "USB" in resource:
            return resource
    return None


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


def _bounded_amp(text: str) -> float:
    try:
        return check_amp_vpp(float(text), allow_zero=False)
    except (ValueError, FYError) as exc:  # pragma: no cover - CLI parsing
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a THD sweep (Tek math channel).")
    p.add_argument(
        "--visa-resource",
        default=None,
        help="Tektronix VISA resource string (auto if omitted).",
    )
    p.add_argument("--fy-port", default=None, help="FY3200S serial port (auto if omitted).")
    p.add_argument(
        "--amp-vpp",
        type=_bounded_amp,
        default=0.5,
        help=f"Generator amplitude (Vpp, max {FY_MAX_VPP:.2f}).",
    )
    p.add_argument("--start", type=float, default=20.0, help="Sweep start frequency (Hz).")
    p.add_argument("--stop", type=float, default=20000.0, help="Sweep stop frequency (Hz).")
    p.add_argument(
        "--points",
        type=int,
        default=61,
        help="Number of logarithmic sweep points (>=2).",
    )
    p.add_argument(
        "--dwell",
        type=float,
        default=0.15,
        help="Dwell time per frequency (seconds).",
    )
    p.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Scope channel to capture (ignored if --math).",
    )
    p.add_argument(
        "--math",
        action="store_true",
        help="Capture math trace (CH1-CH2) instead of a single channel.",
    )
    p.add_argument(
        "--math-order", default="CH1-CH2", help="Math subtraction order when --math is enabled."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/thd_sweep.csv"),
        help="Output CSV path (frequency, Vrms, PkPk, THD%%).",
    )
    p.add_argument(
        "--timestamp", action="store_true", help="Append timestamp to the output filename."
    )
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable THD spike filtering (mirrors --keep-spikes in CLI).",
    )
    p.add_argument(
        "--scope-auto-scale",
        default=None,
        help="Auto vertical scale map (e.g. CH1=12,CH3=1). Values are relative gain.",
    )
    p.add_argument(
        "--scope-auto-scale-margin",
        type=float,
        default=1.25,
        help="Headroom multiplier for auto scope volts/div.",
    )
    p.add_argument(
        "--scope-auto-scale-min",
        type=float,
        default=1e-3,
        help="Minimum volts/div when auto scaling.",
    )
    p.add_argument(
        "--scope-auto-scale-divs",
        type=float,
        default=8.0,
        help="Vertical divisions assumed when auto-scaling.",
    )
    p.add_argument(
        "--apply-gold-calibration",
        action="store_true",
        help="Apply the packaged gold calibration curve to Vrms/PkPk/THD outputs.",
    )
    p.add_argument(
        "--cal-target-vpp",
        type=_bounded_amp,
        default=None,
        help=(
            "Target DUT amplitude when calibration is applied (adjust generator per frequency). "
            f"Must be <= {FY_MAX_VPP:.2f} Vpp."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    visa_resource = args.visa_resource or _guess_visa_resource()
    if not visa_resource:
        print("Unable to locate Tektronix VISA resource. Provide --visa-resource explicitly.")
        return 1
    fy_port = args.fy_port or find_fy_port()

    if fy_port:
        print("FY port:", fy_port)
    else:
        print("FY port not found; will rely on thd_sweep auto-detect.")
    print("Tek VISA resource:", visa_resource)

    output = _timestamp_path(args.output, args.timestamp)
    output.parent.mkdir(parents=True, exist_ok=True)

    calibration_curve = None
    if args.apply_gold_calibration or args.cal_target_vpp is not None:
        try:
            calibration_curve = load_calibration_curve()
        except Exception as exc:
            print(f"Calibration load error: {exc}")
            calibration_curve = None
    cal_target = args.cal_target_vpp if calibration_curve else None
    sweep_amp = cal_target if cal_target is not None else args.amp_vpp

    scope_scale_map = None
    if args.scope_auto_scale:
        try:
            scope_scale_map = _parse_auto_scale(args.scope_auto_scale)
        except ValueError as exc:
            print(f"Scope auto-scale error: {exc}")
            scope_scale_map = None

    rows, out_path, suppressed = thd_sweep(
        visa_resource=visa_resource,
        fy_port=fy_port,
        amp_vpp=sweep_amp,
        scope_channel=args.channel,
        start_hz=args.start,
        stop_hz=args.stop,
        points=args.points,
        dwell_s=args.dwell,
        use_math=args.math,
        math_order=args.math_order,
        output=output,
        filter_spikes=not args.no_filter,
        scope_scale_map=scope_scale_map,
        scope_scale_margin=args.scope_auto_scale_margin,
        scope_scale_min=args.scope_auto_scale_min,
        scope_scale_divs=args.scope_auto_scale_divs,
        calibration_curve=calibration_curve,
        calibrate_to_vpp=cal_target,
    )
    if out_path:
        print("Saved:", out_path)
    print(f"Captured {len(rows)} points.")
    if suppressed and not args.no_filter:
        print("Filtered spikes:")
        for freq, original, baseline in suppressed:
            print(f"  {freq:8.2f} Hz → {original:6.3f}% → {baseline:6.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
