#!/usr/bin/env python3
"""Stepped single-tone THD+N vs frequency sweep at fixed output level(s).

Implements the workflow recommended for characterising class-AB amps:
run a log-spaced frequency sweep at one or more constant output powers and
capture THD+N for each tone after bias settles.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable
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
        base = base / f"thdn_freq_{stamp}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def parse_power_list(raw: Iterable[str]) -> list[float]:
    powers: list[float] = []
    for item in raw:
        try:
            value = float(item)
        except ValueError as exc:  # pragma: no cover - CLI guard
            raise argparse.ArgumentTypeError(f"Invalid power value '{item}'") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError("Power values must be > 0")
        powers.append(value)
    return powers


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
        help="FY3200S serial port (auto-detect if omitted).",
    )
    parser.add_argument(
        "--load-ohms",
        type=float,
        default=8.0,
        help="Dummy load resistance in ohms.",
    )
    parser.add_argument(
        "--power",
        action="append",
        default=["1.0", "140.0"],
        help="Target output power in watts (repeat for multiple levels). Default: 1 W and 140 W.",
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
        default=31,
        help="Number of log-spaced frequency points (>= 2).",
    )
    parser.add_argument(
        "--dwell",
        type=float,
        default=0.35,
        help="Settle time (seconds) before capturing each tone.",
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
        "--output-dir",
        type=Path,
        default=Path("results/thdn_frequency"),
        help="Directory where sweep CSVs are written.",
    )
    parser.add_argument(
        "--timestamp",
        action="store_true",
        help="Place results in a timestamped subdirectory.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Optional calibration curve name (see amp_benchkit.calibration_data).",
    )
    args = parser.parse_args()

    if args.points < 2:
        raise SystemExit("points must be >= 2")
    if args.load_ohms <= 0:
        raise SystemExit("load-ohms must be > 0")

    powers = parse_power_list(args.power)
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
                f"the {FY_MAX_VPP:.2f} Vpp safety limit. Adjust --power or --amp-gain. ({exc})"
            ) from exc
        print(
            f"\n=== Sweep at {power_w:.3f} W into {args.load_ohms:.2f} Ω "
            f"(Vrms {vrms_target:.3f} V, Vpp {vpp_target:.3f} V, Vin {gen_vpp:.3f} Vpp) ==="
        )
        sweep_kwargs = dict(
            visa_resource=args.visa_resource,
            fy_port=fy_port,
            amp_vpp=gen_vpp,
            fy_proto="FY ASCII 9600",
            scope_channel=1,
            start_hz=args.start,
            stop_hz=args.stop,
            points=args.points,
            dwell_s=args.dwell,
            use_math=False,
            math_order="CH1-CH2",
            output=output_dir / f"thdn_freq_{power_w:.3f}W.csv",
        )
        if calibration_curve is not None:
            sweep_kwargs["calibrate_to_vpp"] = gen_vpp
            sweep_kwargs["calibration_curve"] = calibration_curve
        rows, csv_path, suppressed = thd_sweep(**sweep_kwargs)
        if csv_path:
            print(f"Saved sweep CSV: {csv_path}")
        if suppressed:
            print(f"Spike filtering suppressed {len(suppressed)} THD outliers.")
        for freq_hz, vr, pk, thd_percent in rows:
            summary_rows.append(
                [power_w, vrms_target, vpp_target, gen_vpp, freq_hz, vr, pk, thd_percent]
            )

    summary_path = output_dir / "thdn_vs_frequency_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(summary_rows)
    print(f"\nSummary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
