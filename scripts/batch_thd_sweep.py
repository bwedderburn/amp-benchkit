"""Run a series of THD sweeps at multiple output amplitudes.

This helper wraps ``amp_benchkit.sweeps.thd_sweep`` and automates the
common “low/mid/high level” workflow.  Each amplitude gets its own CSV
named ``thd_<amp>Vpp_<timestamp>.csv`` inside a timestamped directory,
and a summary table (min/max/mean/median THD) is emitted to
``summary.csv``.  The console mirrors the per-level stats plus any spike
replacements performed by the median filter.

Example:

    python scripts/batch_thd_sweep.py \
        --amplitudes 0.2,0.5,0.9,1.2,1.35 \
        --dwell 0.5

"""

from __future__ import annotations

import argparse
import csv
import os
import time
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median

from amp_benchkit.fy import FY_MAX_VPP, FYError, check_amp_vpp
from amp_benchkit.sweeps import format_thd_rows, thd_sweep


def _parse_amplitudes(text: str) -> list[float]:
    try:
        amps = [float(tok) for tok in text.split(",") if tok.strip()]
    except ValueError as exc:  # pragma: no cover - CLI parsing
        raise argparse.ArgumentTypeError(f"Invalid amplitude list: {text!r}") from exc
    sanitized: list[float] = []
    for amp in amps:
        try:
            sanitized.append(check_amp_vpp(amp, allow_zero=False))
        except FYError as exc:  # pragma: no cover - CLI parsing
            raise argparse.ArgumentTypeError(str(exc)) from exc
    return sanitized


def _summaries(
    rows: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    values = [thd for *_ignored, thd in rows if not isinstance(thd, complex)]
    if not values:
        return float("nan"), float("nan"), float("nan"), float("nan")
    finite = [v for v in values if v == v]
    if not finite:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return min(finite), max(finite), mean(finite), median(finite)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run THD sweeps at multiple amplitudes.")
    p.add_argument(
        "--amplitudes",
        type=_parse_amplitudes,
        default=_parse_amplitudes("0.2,0.5,0.9,1.2,1.35"),
        help=(
            "Comma-separated list of Vpp levels "
            f"(default: 0.2,0.5,0.9,1.2,1.35; max {FY_MAX_VPP:.2f} Vpp)."
        ),
    )
    p.add_argument(
        "--visa-resource",
        default=os.environ.get("VISA_RESOURCE", ""),
        help="Tektronix VISA resource.",
    )
    p.add_argument(
        "--fy-port",
        default=os.environ.get("FY_PORT", ""),
        help="FY generator serial port.",
    )
    p.add_argument("--start", type=float, default=20.0, help="Sweep start frequency (Hz).")
    p.add_argument("--stop", type=float, default=20000.0, help="Sweep stop frequency (Hz).")
    p.add_argument(
        "--points",
        type=int,
        default=61,
        help="Number of logarithmic points (default 61).",
    )
    p.add_argument(
        "--dwell",
        type=float,
        default=0.5,
        help="Dwell/settling time at each point (seconds).",
    )
    p.add_argument("--scope-channel", type=int, default=1, help="Scope channel index (default 1).")
    p.add_argument(
        "--math",
        action="store_true",
        help="Capture scope MATH trace instead of a single channel.",
    )
    p.add_argument("--math-order", default="CH1-CH2", help="Scope MATH order (CH1-CH2 or CH2-CH1).")
    p.add_argument("--keep-spikes", action="store_true", help="Disable THD spike suppression.")
    p.add_argument(
        "--filter-window",
        type=int,
        default=2,
        help="Spike filter neighbour window (default 2).",
    )
    p.add_argument(
        "--filter-factor",
        type=float,
        default=2.0,
        help="Spike filter factor vs. median (default 2.0).",
    )
    p.add_argument(
        "--filter-min",
        type=float,
        default=2.0,
        help="Minimum THD%% before replacing.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory (default results/thd_batch_<timestamp>).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base_dir = Path(args.output_dir or Path("results") / f"thd_batch_{timestamp}")
    base_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[tuple[float, Path, float, float, float, float, int]] = []

    for amp_vpp in args.amplitudes:
        outfile = base_dir / f"thd_{amp_vpp:.2f}Vpp_{timestamp}.csv"
        rows, out_path, suppressed = thd_sweep(
            visa_resource=args.visa_resource,
            fy_port=args.fy_port,
            amp_vpp=amp_vpp,
            scope_channel=args.scope_channel,
            start_hz=args.start,
            stop_hz=args.stop,
            points=args.points,
            dwell_s=args.dwell,
            use_math=args.math,
            math_order=args.math_order,
            output=outfile,
            filter_spikes=not args.keep_spikes,
            filter_window=args.filter_window,
            filter_factor=args.filter_factor,
            filter_min_percent=args.filter_min,
        )
        out_path = Path(out_path or outfile)
        min_thd, max_thd, mean_thd, median_thd = _summaries(rows)
        summary_rows.append(
            (amp_vpp, out_path, min_thd, max_thd, mean_thd, median_thd, len(suppressed))
        )

        print(f"Saved: {out_path}")
        for line in format_thd_rows(rows):
            print(f"  {line}")
        if suppressed and not args.keep_spikes:
            print("  Filtered spikes:")
            for freq, original, baseline in suppressed:
                print(f"    {freq:8.2f} Hz → {original:6.3f}% → {baseline:6.3f}%")
        print(
            f"  Stats: min {min_thd:.3f}% | max {max_thd:.3f}% | "
            f"mean {mean_thd:.3f}% | median {median_thd:.3f}%\n"
        )

    summary_csv = base_dir / "summary.csv"
    with summary_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "amp_vpp",
                "csv_path",
                "thd_min_percent",
                "thd_max_percent",
                "thd_mean_percent",
                "thd_median_percent",
                "spikes_suppressed",
            ]
        )
        for amp_vpp, path, mn, mx, avg, med, suppressed_count in summary_rows:
            writer.writerow([amp_vpp, path, mn, mx, avg, med, suppressed_count])

    print(f"Summary written to {summary_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
