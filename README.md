# amp-benchkit

Unified GUI + LabJack instrumentation helper environment.

![Release](https://img.shields.io/github/v/release/bwedderburn/amp-benchkit)
[![CI](https://github.com/bwedderburn/amp-benchkit/actions/workflows/ci.yml/badge.svg)](https://github.com/bwedderburn/amp-benchkit/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/amp-benchkit.svg)
![PyPI](https://img.shields.io/pypi/v/amp-benchkit.svg)

> **Note on v0.3.2 (Maintenance Release):** This release restores the repository to the v0.3.1 baseline state to remove unintended changes introduced in subsequent commits. Key improvements include fixing LabJackPython import warnings in test output and updating Ruff configuration to modern format. All CI checks pass cleanly.


## Contents

- `unified_gui_layout.py` – Main multi‑tab GUI / CLI tool (PySide6/PyQt5, VISA, Serial, LabJack U3).
- `amp_benchkit/deps.py` – Dependency detection (Qt binding, pyvisa, pyserial, LabJack) + shared helpers.
- `amp_benchkit/fy.py` – FY32xx function generator helpers (command build, apply, sweep).
- `amp_benchkit/tek.py` – Tektronix scope SCPI helpers (setup, waveform capture, IEEE block parsing).
- `amp_benchkit/dsp.py` – DSP helpers (RMS, Vpp, THD FFT, bandwidth knees).
- `amp_benchkit/gui/` – Incremental extraction of GUI tabs (generator, scope, DAQ extracted).
- `amp_benchkit/u3util.py` – LabJack U3 safe‑open and feature detection utilities.

- `scripts/install_exodriver_alpine.sh` – Idempotent installer for Exodriver (liblabjackusb) on Alpine (musl) or glibc.
- `patches/exodriver-install-alpine.patch` – Patch capturing local enhancement to upstream `exodriver` install script (for reproducibility / PR prep).

## Documentation

The MkDocs site bundles setup recipes, hardware guides, and developer workflow notes:

- Published docs: <https://bwedderburn.github.io/amp-benchkit/>
- Build locally:
  ```bash
  pip install .[docs]
  mkdocs serve  # open http://127.0.0.1:8000
  ```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Build & install Exodriver (USB support)
./scripts/install_exodriver_alpine.sh

# Run CLI selftest (headless)
python unified_gui_layout.py selftest

# Launch GUI (if X / Wayland or VS Code desktop available)
python unified_gui_layout.py gui --gui
```

### Recreate a GUI-Ready Python 3.12 Environment

We ship `scripts/bootstrap-venv312.sh` to provision the Python 3.12 environment we use for GUI + hardware testing:

```bash
# Prereqs (once): brew install pyenv
./scripts/bootstrap-venv312.sh
source .venv312/bin/activate

# Launch GUI / run full suite
python unified_gui_layout.py gui
AMP_HIL=1 pytest -q -rs
```

The script installs Python 3.12.4 via `pyenv`, recreates `.venv312`, installs `.[dev,test,gui]`, and performs a quick Qt sanity check. Re-run it whenever you need a clean GUI-capable environment.

### Installed (Package) Usage

After `pip install .[gui]` you can use console scripts:

```bash
amp-benchkit selftest          # run selftests (headless)
amp-benchkit diag              # dependency diagnostics summary
amp-benchkit gui --gui         # launch GUI (or use: amp-benchkit-gui --gui)
amp-benchkit config-dump       # show persisted JSON config
amp-benchkit config-reset      # reset config to defaults

# Generate frequency sweep (headless list output)
amp-benchkit sweep --start 20 --stop 20000 --points 10 --mode log

# Headless THD sweep using Tek math (requires hardware)
amp-benchkit thd-math-sweep --amp-vpp 0.5 --output results/thd_sweep.csv --timestamp
# add --math to capture CH1-CH2 differential instead of a single channel
# add --scope-auto-scale CH1=12,CH3=1 --scope-auto-scale-margin 1.2 to auto-set volts/div

# Stepped THD+N vs frequency at fixed output power (1 W & just-under-rated default)
python scripts/thdn_vs_frequency.py \
  --visa-resource USB0::0x0699::0x036A::C100563::INSTR \
  --load-ohms 8 \
  --power 1 --power 140 \
  --points 31 --timestamp
# Override --power or --points to refine the grid; add --calibration gold_reference to compensate probe gain.

# THD+N vs output power at 1 kHz (log-spaced from 10 mW to 150 W by default)
python scripts/thdn_vs_power.py \
  --visa-resource USB0::0x0699::0x036A::C100563::INSTR \
  --frequency 1000 \
  --power-start 0.01 --power-stop 150 --points 25 \
  --timestamp
# Produces Vrms/THD tables you can plot to expose crossover knees, noise floor, and clipping onset.

# One-shot THD sweep with auto-detected instruments & scaling
python3 scripts/run_thd_sweep.py \
  --amp-vpp 0.5 \
  --start 20 --stop 20000 --points 61 \
  --scope-auto-scale CH1=12 \
  --apply-gold-calibration --cal-target-vpp 0.5 \
  --timestamp

# Headless bandwidth sweep to locate -3 dB knees (fake-hw friendly)
amp-benchkit knee-sweep --amp-vpp 1.0 --output results/knee_sweep.csv --timestamp
# add --knee-drop-db 1.5 to search for alternate bandwidth targets
# add --smoothing mean --smooth-window 7 --allow-non-monotonic to tweak the envelope fit

# Snapshot the scope FFT (math trace) with optional generator retune
amp-benchkit fft-capture --source CH1 --output results/fft_trace.csv --top 8 --timestamp
# add --fy-freq 1000 --fy-amp 0.5 to re-arm the FY generator before capture

# Sweep FFT captures across multiple fundamentals
amp-benchkit fft-sweep --start 100 --stop 5000 --points 10 \
  --amp-vpp 0.5 --output-dir results/fft_sweep --timestamp
# add --fft-span 500 or --fft-zoom 10 to match your scope configuration

# Compare FFT-derived THD against sweep output
python scripts/fft_thd_compare.py \
  --fft results/fft_trace.csv \
  --thd results/thd_sweep.csv \
  --auto-fundamental \
  --window 200 \
  --harmonics 8
# replace --auto-fundamental with --fundamental 1000 if the FFT bin lands exactly at 1 kHz

# Summarize an FFT sweep against the latest THD sweep
python scripts/fft_vs_thd_summary.py \
  --fft-dir results/fft_sweep \
  --thd results/thd_sweep.csv \
  --window 50 --harmonics 8

### Automatic Scope Scaling

`thd-math-sweep` can now drive Tektronix vertical scale automatically so the math trace
stays within the display window during amplitude sweeps. Provide a comma-separated map
of channel → expected Vpp gain relative to the FY3200S setting:

```bash
python3 unified_gui_layout.py thd-math-sweep \
  --math --math-order CH1-CH3 \
  --amp-vpp 0.4 \
  --scope-auto-scale CH1=13,CH3=1 \
  --scope-auto-scale-margin 0.8 \
  --apply-gold-calibration --cal-target-vpp 0.4 \
  --output results/thd_0p4_auto_gold.csv
```

- `CH1=13` assumes the bridged amplifier output is ~13× the generator amplitude (adjust for
  your probe factor and gain). Include additional channels or `MATH` if needed.
- `--scope-auto-scale-margin` controls headroom (values < 1 zoom out, > 1 zoom in). `--scope-auto-scale-min`
  and `--scope-auto-scale-divs` offer fine tuning for minimum volts/div or displays that show more than
  eight divisions.
- At the end of the run the helper restores the original volts/div settings and resets the generator to 1 kHz.
- For a reference dataset (Kenwood KAC-823, 0.2–0.5 Vpp) see `docs/examples/kenwood_baseline_auto_gold/`.

```

### Sweep CLI (Headless Frequency Lists)

The `sweep` subcommand prints a list of frequency points to stdout (one per line) for use in shell pipelines or other tooling.

Arguments:

* `--start` (Hz, float, required)
* `--stop` (Hz, float, required)
* `--points` (int >= 2, required)
* `--mode` (`log` | `linear`, default: `log`)

Spacing rules:
* `linear`: even arithmetic spacing including both endpoints.
* `log`: geometric progression including endpoints (requires start > 0, stop > 0).

Example (log spaced 6 points 10 Hz → 10 kHz):

```bash
amp-benchkit sweep --start 10 --stop 10000 --points 6 --mode log
```
Outputs (rounded to 6 decimals, trailing zeros trimmed):
```
10
46.415889
215.443469
1000
4641.588834
10000
```

Linear example:

```bash
amp-benchkit sweep --start 100 --stop 1000 --points 10 --mode linear
```

Exit codes:
* 0 on success
* 1 on invalid arguments (e.g. points < 2, stop < start)

Typical usage piping into another tool:

```bash
for f in $(amp-benchkit sweep --start 20 --stop 20000 --points 25 --mode log); do \
	echo "Would run measurement at $f Hz"; \
done
```

### THD Sweep CLI (Tek MATH)

Attach CH1/CH2 to the DUT input/output, enable the scope’s `MATH = CH1-CH2`, and provide VISA/FY connection details via flags or environment variables (`VISA_RESOURCE`, `FY_PORT`). When using the pure Python backend on macOS, also set `PYUSB_LIBRARY` to the `libusb-1.0.dylib` shipped with `libusb-package`, or run with `sudo` if USBTMC access is blocked.

```bash
PYUSB_LIBRARY="$PWD/.venv/lib/python3.13/site-packages/libusb_package/libusb-1.0.dylib" \
FY_PORT=/dev/cu.usbserial-XXXX \
VISA_RESOURCE=USB0::0x0699::0x036A::SERIAL::INSTR \
amp-benchkit thd-math-sweep --amp-vpp 0.5 --dwell 0.3 --channel 1
```

Results land in `results/thd_sweep.csv` by default and the console prints THD% per point (20 Hz–20 kHz, log-spaced).
Recurring 6 % spikes from the Tek math channel are suppressed automatically (window=2, factor=2.0, min=2.0 %).
Use `--keep-spikes` to disable filtering or tune the heuristics via `--filter-window`, `--filter-factor`, and `--filter-min`.
After each sweep the generator is returned to 1 kHz and the scope timebase to 100 µs/div for quick follow-up checks.
Pass `--math --math-order CH1-CH2` to re-enable differential capture with the scope's MATH trace.

#### Batch THD sweeps (multiple levels)

Automate the common “low/mid/high” check with the batch helper:

```bash
python scripts/batch_thd_sweep.py \
  --amplitudes 0.2,0.5,0.9,1.2,1.35 \
  --dwell 0.5
```

This produces a timestamped directory (e.g. `results/thd_batch_20251012-2018/`) containing:

* `thd_<amp>Vpp_<timestamp>.csv` for each amplitude.
* `summary.csv` with min/max/mean/median THD and the number of spikes replaced at each level.

All CLI options from `thd-math-sweep` are available (filter tuning, math capture, resource overrides).

#### Plot the results

Overlay one or more sweep CSVs with the plotting helper:

```bash
python scripts/plot_thd_sweep.py results/thd_batch_*/thd_*Vpp_*.csv \
  --output results/thd_plot.png --title "THD vs Frequency"
```

Labels default to the filename stem; provide `--labels` if you want custom legend text. The x‑axis is logarithmic so octave spacing is preserved.

#### One-off waveform capture

For deeper analysis, capture a single snapshot to CSV:

```bash
python scripts/capture_waveform.py 1000 \
  --amp-vpp 0.5 --dwell 0.3 --channel 1 \
  --visa-resource USB0::0x0699::0x036A::SERIAL::INSTR \
  --fy-port /dev/cu.usbserial-XXXX
```

Add `--math` to grab the MATH trace instead of a single channel. The script saves `results/waveform_<freq>.csv` and prints Vrms/Vpp/THD estimates for the captured waveform.

Future enhancements (planned): optional JSON/CSV output (`--format json|csv`) for the frequency list helper.

> Note: Direct invocation via `python unified_gui_layout.py ...` still works but is considered a legacy path. Prefer the installed console scripts (`amp-benchkit`, `amp-benchkit-gui`) for forward compatibility; future releases may relocate the legacy file.
```

Add `--verbose` to any of the above to elevate logging to DEBUG.

### Logging

A central logger (`amp_benchkit.logging`) is initialized by the CLI. It emits to stderr and a rotating file (`benchkit.log`) under `XDG_STATE_HOME` / `XDG_CACHE_HOME` (fallback `~/.cache/amp-benchkit`). Levels:

* INFO: High‑level actions (applied FY settings, captured scope block, etc.)
* DEBUG: Detailed SCPI / serial commands (enable with `--verbose`)
* WARNING/ERROR: Dependency or runtime issues

You can integrate programmatically:
```python
from amp_benchkit.logging import setup_logging, get_logger
setup_logging(verbose=True)
log = get_logger()
log.info("Starting scripted acquisition")
```

### Config Persistence

User preferences (FY port/protocol, last scope resource) persist to:
`~/.config/amp-benchkit/config.json` (XDG‑style). Functions:

```python
from amp_benchkit.config import load_config, save_config, update_config
cfg = load_config()
update_config(fy_port="/dev/ttyUSB0")
```
CLI helpers: `config-dump`, `config-reset`.

Corrupt config files are ignored gracefully and overwritten on next save.

### Programmatic Use (Headless Helpers)

```python
from amp_benchkit.fy import fy_apply
from amp_benchkit.tek import tek_capture_block, TEK_RSRC_DEFAULT
from amp_benchkit.u3util import open_u3_safely

# Apply a 1 kHz 1.0 Vpp sine on FY CH1 (auto port detection)
fy_apply(freq_hz=1000, amp_vpp=1.0, wave="Sine", ch=1)

# Capture a waveform from a Tek scope (returns t, volts, raw)
t, v, raw = tek_capture_block(TEK_RSRC_DEFAULT, ch=1)

# Open LabJack U3 safely
try:
	d = open_u3_safely()
	ain0 = d.getAIN(0)
	print("AIN0=", ain0)
finally:
	try: d.close()
	except Exception: pass
```

`fy_apply` clamps requests above `FY_MAX_VPP` (currently 1.35 Vpp) to protect downstream equipment; exceeding the limit raises `FYError`.

If LabJack USB still not detected on Alpine (musl) add:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

## Exodriver (LabJack USB Driver)

We vendor a minimal, source‑only snapshot of Exodriver under `exodriver/` (no `.git`, no compiled objects). See `EXODRIVER.md` for rationale, update procedure, and upstream attribution. For a full clone (e.g. to inspect history or contribute upstream), run:

```bash
git clone https://github.com/labjack/exodriver.git
```

Then either build directly or point our helper script at it:

```bash
EXO_DIR=exodriver ./scripts/install_exodriver_alpine.sh
```

### Health Check (after install)
```bash
python -c "import u3; print('u3 import OK')" || echo "LabJack Python import failed"
```
To probe a device (will fail gracefully if none):
```bash
python scripts/check_labjack_usb.py
```

## Make Targets (after Makefile added)
Planned:
- `make install-exodriver` – run wrapper.
- `make check-usb` – run health script.
- `make gui` – launch GUI.
- `make selftest` – run headless integrations.

## Development Hygiene

Install and enable pre-commit hooks to catch style and oversized file issues early:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files   # optional first full pass
```
Hooks block committing virtualenvs / `site-packages` and large (>5MB) binaries, and run ruff, formatting, and mypy.
- `make selftest` – headless tests.
 - See `ROADMAP.md` for planned milestones and how to propose new items (edit a single table row per PR).

## Development (Lint / Format / Type)

We use `ruff` (lint), `mypy` (types). Formatting is kept Black-compatible (you can optionally run `ruff format` in newer versions).

Manual invocations:

```bash
ruff check .          # lint
ruff check . --fix    # autofix simple issues
mypy .                 # type check
pytest -q             # tests
```

If you add a Makefile target locally, suggested grouping:

```bash
make lint   # ruff check
make type   # mypy
make test   # pytest
make ci-local  # run all of the above
```

Pre-commit is configured; enable with:

```bash
pre-commit install
```

Then commits will auto-run ruff / mypy (if configured) on changed files.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Could not load the Exodriver driver` | `liblabjackusb` not on search path | Run installer; set `LD_LIBRARY_PATH` |
| `No module named numpy` | Dependencies not installed | `pip install -r requirements.txt` (future) |
| GUI won’t launch | Missing Qt binding / no display | Install PySide6; ensure DISPLAY set or use VS Code desktop |
| `Permission denied` running script | Not executable | `chmod +x script.sh` |

## Future Improvements

Implemented so far:
* Phase 1 & 2 modularization: dependencies, instruments (FY / Tek / U3) extracted
* Core pytest suite (FFT THD, IEEE block decode, config roundtrip)
* Logging subsystem and JSON config persistence
* GitHub Actions CI (Python 3.11 / 3.12)
* DSP module extraction (`amp_benchkit.dsp`)
* GUI tab modularization (generator + scope + DAQ + automation + diagnostics extracted into `amp_benchkit.gui`)
* Version 0.2.0 released – full GUI tab extraction, shared Qt helper, removal of deprecated DSP wrappers.

Planned / open:
* Continue GUI tab extraction: diagnostics
* Additional error handling (timeouts around VISA / serial)
* Test extras & packaging polish
* Optional hardware-in-loop stage (conditional)
* Type hint expansion & static checks
* Potential REST / WebSocket automation bridge

---

**Support references**: LabJack Exodriver repo <https://github.com/labjack/exodriver>

## Changelog

See `CHANGELOG.md` for a full list of versions and changes. Highlight 0.2.0: all GUI tabs modular, deprecated DSP wrappers removed, shared lazy Qt import helper.

## Release & Publishing

Steps to cut a release (example for 0.2.0 already performed):

1. Update version in `pyproject.toml` & confirm tests pass.
2. Update `CHANGELOG.md` (add new heading, date, notes).
3. Commit: `git commit -am "release: vX.Y.Z"`.
4. Tag locally: `git tag vX.Y.Z` (lightweight or annotated `-a`).
5. Build artifacts locally (optional pre-check): `python -m build --sdist --wheel`.
6. Push branch and tag: `git push && git push origin vX.Y.Z`.
7. GitHub Actions CI will build & test; if a publish workflow exists with `PYPI_API_TOKEN` (an API token from PyPI stored as an Actions secret), it will upload.
8. Draft GitHub Release, attach release notes (can pull from changelog), verify assets.
9. Post-release: bump version to next dev (e.g., 0.2.1.dev0) if ongoing changes expected.

Dry-run publish (validate metadata):
```bash
python -m build
twine check dist/*
```

Install from a specific tag directly:
```bash
pip install git+https://github.com/bwedderburn/amp-benchkit@v0.2.0
```

## Next Steps / Roadmap

- Add `requirements.txt` or `pyproject.toml` for pinned dependencies.
- Continuous Integration: GitHub Actions workflow to run `make selftest` and build Docker image.
- Optional hardware-in-loop stage (tagged, skipped by default) for real LabJack tests.
- Improve `unified_gui_layout.py` modularity (further split GUI/automation logic – Phase 1 & 2 complete: deps + instruments extracted).
	(Logging + config persistence now implemented.)

	## Public API (Stability: Beta)

	The following functions / symbols are considered part of the provisional public API and will aim to avoid breaking changes without a minor version bump:

	Instrument helpers:
	* `amp_benchkit.fy.build_fy_cmds(freq_hz, amp_vpp, off_v, wave, duty=None, ch=1)`
	* `amp_benchkit.fy.fy_apply(...)`
	* `amp_benchkit.fy.fy_sweep(port, ch, proto, start=None, end=None, t_s=None, mode=None, run=None)`
	* `amp_benchkit.tek.tek_capture_block(resource, ch=1)`
	* `amp_benchkit.tek.scope_capture_calibrated(resource, timeout_ms=15000, ch=1)`
	* `amp_benchkit.tek.scope_screenshot(filename, resource=..., timeout_ms=15000, ch=1)`
	* `amp_benchkit.u3util.open_u3_safely()`

	Diagnostics / utilities:
	* `amp_benchkit.deps.find_fy_port()`
	* `amp_benchkit.logging.setup_logging(verbose=False, file_logging=True)`
	* `amp_benchkit.config.load_config()` / `save_config(cfg)` / `update_config(**kv)`

	DSP / analysis (`amp_benchkit.dsp`):
	* `vrms(v)`
	* `vpp(v)`
	* `thd_fft(t, v, f0=None, nharm=10, window='hann')`
	* `find_knees(freqs, amps, ref_mode='max', ref_hz=1000.0, drop_db=3.0)`

	Exceptions:
	* `amp_benchkit.fy.FYError`, `FYTimeoutError`
	* `amp_benchkit.tek.TekError`, `TekTimeoutError`

	Items not listed above should be treated as internal and are subject to change.
- Add type hints & mypy pass for critical modules.
- Provide wheel / PyPI packaging for headless CLI mode.
- Add a simple REST or WebSocket bridge for remote automation control.
- Extend Makefile with `format` / `lint` targets (black, ruff, mypy).
- Optional plugin architecture for new instrument tabs.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, style, tests, and PR guidelines.
