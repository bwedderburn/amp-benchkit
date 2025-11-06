"""Headless THD sweep with Tektronix TDS2024B (single-ended or math).

This script mirrors the manual heredoc used earlier and saves a CSV
with Vrms, Vpp, and THD% across a 20 Hz to 20 kHz logarithmic sweep.
It assumes:
  - FY3200S generator connected over serial (FY_PORT env var or auto-detect).
  - Tektronix scope reachable via VISA resource (VISA_RESOURCE env var).
  - CH1 monitors the DUT output by default. Use ``--math`` for CH1-CH2 subtraction.

Example usage (macOS with pyvisa-py and libusb-package):
    PYUSB_LIBRARY=\"/path/to/libusb_package/libusb-1.0.dylib\" \\
    FY_PORT=/dev/cu.usbserial-XXXX \\
    VISA_RESOURCE=USB0::0x0699::0x036A::SERIAL::INSTR \\
    python scripts/thd_math_sweep.py

Add `sudo` if macOS blocks USBTMC access. The same sweep is also
available via the packaged CLI: `amp-benchkit thd-math-sweep`.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

# Ensure repository root on sys.path when running from a checkout.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amp_benchkit.fy import FY_MAX_VPP, FYError, check_amp_vpp  # noqa: E402
from amp_benchkit.sweeps import format_thd_rows, thd_sweep  # noqa: E402

ENV_DEFAULT_AMP = os.environ.get("AMP_VPP", "0.5")
try:
    DEFAULT_AMP_VPP = check_amp_vpp(float(ENV_DEFAULT_AMP), allow_zero=False)
except (ValueError, FYError):
    DEFAULT_AMP_VPP = min(0.5, FY_MAX_VPP)


def _bounded_amp(text: str) -> float:
    try:
        return check_amp_vpp(float(text), allow_zero=False)
    except (ValueError, FYError) as exc:  # pragma: no cover - CLI parsing
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep THD using Tektronix scope capture (single-ended or math)."
    )
    ap.add_argument(
        "--visa-resource",
        default=os.environ.get("VISA_RESOURCE", "USB0::0x0699::0x036A::C100563::INSTR"),
        help="Tektronix VISA resource string.",
    )
    ap.add_argument(
        "--fy-port",
        default=os.environ.get("FY_PORT"),
        help="FY3200S serial port (auto-detect if omitted).",
    )
    ap.add_argument(
        "--amp-vpp",
        type=_bounded_amp,
        default=DEFAULT_AMP_VPP,
        help=f"Generator amplitude in Vpp (max {FY_MAX_VPP:.2f}).",
    )
    ap.add_argument(
        "--start-hz",
        type=float,
        default=20.0,
        help="Sweep start frequency.",
    )
    ap.add_argument(
        "--stop-hz",
        type=float,
        default=20000.0,
        help="Sweep stop frequency.",
    )
    ap.add_argument(
        "--points",
        type=int,
        default=61,
        help="Number of logarithmic sweep points.",
    )
    ap.add_argument(
        "--dwell",
        type=float,
        default=0.15,
        help="Dwell time per frequency (seconds). Use 0.3+ for LF stability.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/thd_sweep.csv"),
        help="Destination CSV path.",
    )
    ap.add_argument(
        "--channel",
        type=int,
        default=1,
        help="Scope channel to capture (ignored when --math is set).",
    )
    ap.add_argument(
        "--math",
        action="store_true",
        help="Capture the scope MATH trace instead of a single channel.",
    )
    ap.add_argument(
        "--math-order",
        default="CH1-CH2",
        help="MATH subtraction order (used only when --math is set).",
    )
    ap.add_argument(
        "--keep-spikes",
        action="store_true",
        help="Disable automatic suppression of recurring THD spikes.",
    )
    ap.add_argument(
        "--filter-window",
        type=int,
        default=2,
        help="Neighbor window size for spike detection (default: 2).",
    )
    ap.add_argument(
        "--filter-factor",
        type=float,
        default=2.0,
        help="Spike threshold factor relative to local median (default: 2.0).",
    )
    ap.add_argument(
        "--filter-min",
        type=float,
        default=2.0,
        help="Minimum THD%% required before a spike may be filtered (default: 2.0).",
    )
    ap.add_argument(
        "--scope-auto-scale",
        default=None,
        help=(
            "Auto vertical scale map (e.g. CH1=12,CH3=1). "
            "Values represent expected Vpp gain relative to generator amplitude."
        ),
    )
    ap.add_argument(
        "--scope-auto-scale-margin",
        type=float,
        default=1.25,
        help="Headroom multiplier when computing auto scope V/div (default: 1.25).",
    )
    ap.add_argument(
        "--scope-auto-scale-min",
        type=float,
        default=1e-3,
        help="Minimum V/div when auto scaling (default: 1e-3).",
    )
    ap.add_argument(
        "--scope-auto-scale-divs",
        type=float,
        default=8.0,
        help="Vertical divisions assumed when auto scaling (default: 8).",
    )
    args = ap.parse_args()

    if args.points < 2:
        raise ValueError("points must be >= 2")
    if not math.isfinite(args.dwell) or args.dwell < 0:
        raise ValueError("dwell must be >= 0")

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

    scope_scale_map = None
    if args.scope_auto_scale:
        scope_scale_map = _parse_auto_scale(args.scope_auto_scale)

    rows, out_path, suppressed = thd_sweep(
        visa_resource=args.visa_resource,
        fy_port=args.fy_port,
        amp_vpp=args.amp_vpp,
        scope_channel=args.channel,
        start_hz=args.start_hz,
        stop_hz=args.stop_hz,
        points=args.points,
        dwell_s=args.dwell,
        use_math=args.math,
        math_order=args.math_order,
        output=args.output,
        filter_spikes=not args.keep_spikes,
        filter_window=args.filter_window,
        filter_factor=args.filter_factor,
        filter_min_percent=args.filter_min,
        scope_scale_map=scope_scale_map,
        scope_scale_margin=args.scope_auto_scale_margin,
        scope_scale_min=args.scope_auto_scale_min,
        scope_scale_divs=args.scope_auto_scale_divs,
    )
    if out_path:
        print("Saved:", out_path)
    for line in format_thd_rows(rows):
        print(line)
    if suppressed and not args.keep_spikes:
        print("Filtered spikes:")
        for freq, original, baseline in suppressed:
            print(f"  {freq:8.2f} Hz → {original:6.3f}% (replaced with {baseline:6.3f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
