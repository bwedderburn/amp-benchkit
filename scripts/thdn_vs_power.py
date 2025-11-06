#!/usr/bin/env python3
"""THD+N vs output power sweep at a fixed frequency (default 1 kHz)."""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

from amp_benchkit.calibration import load_calibration_curve
from amp_benchkit.fy import FY_MAX_VPP, FYError, check_amp_vpp, find_fy_port
from amp_benchkit.sweeps import thd_sweep


def vrms_from_power(power_w: float, load_ohms: float) -> float:
    return math.sqrt(power_w * load_ohms)


def vpp_from_vrms(vrms: float) -> float:
    return vrms * 2.0 * math.sqrt(2.0)


def ensure_output_dir(base: Path, timestamp: bool) -> Path:
    if timestamp:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = base / f"thdn_power_{stamp}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visa-resource",
        default="USB0::0x0699::0x036A::C100563::INSTR",
        help="Tektronix VISA resource string.",
    )
    parser.add_argument(
        "--fy-port",
        default=None,
        help="FY3200S port (auto-detect if omitted).",
    )
    parser.add_argument(
        "--load-ohms",
        type=float,
        default=8.0,
        help="Dummy load resistance in ohms.",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=1000.0,
        help="Test tone frequency in Hz.",
    )
    parser.add_argument(
        "--power-start",
        type=float,
        default=0.01,
        help="Starting power in watts.",
    )
    parser.add_argument(
        "--power-stop",
        type=float,
        default=150.0,
        help="Ending power in watts.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=20,
        help="Number of log-spaced power points (>= 2).",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=0.35,
        help="Settle time (seconds) before capturing each slice.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/thdn_power"),
        help="Directory where sweep CSVs are written.",
    )
    parser.add_argument(
        "--amp-gain",
        type=float,
        required=True,
        help=(
            "Amplifier voltage gain (Vout/Vin). Used to derive the safe generator amplitude "
            "for the requested load voltage."
        ),
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Place results in a timestamped subdirectory.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Optional calibration curve name.",
    )
    args = parser.parse_args()

    if args.points < 2:
        raise SystemExit("points must be >= 2")
    if args.power_start <= 0 or args.power_stop <= 0:
        raise SystemExit("power values must be > 0")
    if args.load_ohms <= 0:
        raise SystemExit("load-ohms must be > 0")

    powers = build_log_points(args.power_start, args.power_stop, args.points)
    output_dir = ensure_output_dir(args.output_dir, args.timestamp)
    fy_port = args.fy_port or find_fy_port()
    if fy_port:
        print(f"Using FY port: {fy_port}")
    else:
        print("Warning: FY generator port not found; attempting sweep anyway.")

    calibration_curve = None
    if args.calibration:
        calibration_curve = load_calibration_curve(args.calibration)
        print(f"Loaded calibration curve '{args.calibration}'.")

    summary_rows: list[list[float | str]] = [
        [
            "power_w",
            "target_vrms",
            "target_vpp",
            "generator_vpp",
            "freq_hz",
            "vrms",
            "pkpk",
            "thd_percent",
        ]
    ]

    for power_w in powers:
        if args.amp_gain <= 0:
            raise SystemExit("--amp-gain must be > 0")
        vrms_target = vrms_from_power(power_w, args.load_ohms)
        vpp_target = vpp_from_vrms(vrms_target)
        gen_vpp_raw = vpp_target / args.amp_gain
        try:
            gen_vpp = check_amp_vpp(gen_vpp_raw, allow_zero=False)
        except FYError as exc:
            raise SystemExit(
                f"Required generator amplitude {gen_vpp_raw:.3f} Vpp exceeds "
                f"the {FY_MAX_VPP:.2f} Vpp safety limit. Reduce target power or "
                f"increase --amp-gain. ({exc})"
            ) from exc
        print(
            f"\n=== THD sweep at {power_w:.4f} W -> {vrms_target:.4f} Vrms "
            f"(Vpp {vpp_target:.4f}, Vin {gen_vpp:.4f} Vpp) ==="
        )
        sweep_kwargs = dict(
            visa_resource=args.visa_resource,
            fy_port=fy_port,
            amp_vpp=gen_vpp,
            fy_proto="FY ASCII 9600",
            scope_channel=1,
            start_hz=args.frequency,
            stop_hz=args.frequency,
            points=1,
            dwell_s=args.dwell,
            use_math=False,
            math_order="CH1-CH2",
            output=output_dir / f"thdn_power_{power_w:.4f}W.csv",
        )
        if calibration_curve is not None:
            sweep_kwargs["calibrate_to_vpp"] = gen_vpp
            sweep_kwargs["calibration_curve"] = calibration_curve
        rows, _, _ = thd_sweep(**sweep_kwargs)
        if rows:
            freq_hz, vr, pk, thd_percent = rows[0]
            summary_rows.append(
                [power_w, vrms_target, vpp_target, gen_vpp, freq_hz, vr, pk, thd_percent]
            )

    summary_path = output_dir / "thdn_vs_power_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(summary_rows)
    print(f"\nSummary written to: {summary_path}")
    return 0


def build_log_points(start: float, stop: float, points: int) -> list[float]:
    """Return log-spaced points from start→stop inclusive."""

    lo = min(start, stop)
    hi = max(start, stop)
    if points == 1:
        return [hi]
    ratio = (hi / lo) ** (1.0 / (points - 1))
    out: list[float] = []
    value = lo
    for _ in range(points):
        out.append(value)
        value *= ratio
    if start > stop:
        out.reverse()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
