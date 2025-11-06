# Hardware Setup

This guide covers recommended wiring and configuration for each supported instrument. When hardware is unavailable, enable simulator mode via `AMPBENCHKIT_FAKE_HW=1`.

## FeelTech FY3200S / FY3224S

- Connect the FY3200S to your host via USB-to-serial adapter.
- Let the application auto-detect the serial port, or override with `FY_PORT=/dev/tty.usbserial-XXXX`.
- Recommended default settings:
  - Waveform: sine
  - Channel: CH1
  - Voltage: ≤1.35 Vpp (hardware-enforced safety limit; adjust DUT gain instead)
- The GUI exposes generator controls; CLI automation uses helper functions in `amp_benchkit.fy`.

## Tektronix TDS2024B Oscilloscope

- Connect over USB and ensure a VISA backend is installed (NI-VISA or `pyvisa-py`).
- Override discovery with `VISA_RESOURCE=USB0::...::INSTR` if multiple scopes are present.
- For external triggering:
  - Configure slope/level in automation via `use_ext`, `ext_slope`, and `ext_level`.
  - The GUI scope tab provides channel toggles and MATH subtraction helpers.

### FFT Math Function

The TDS 2024B's FFT (Math) function can be controlled programmatically with proper vertical scaling:

- **Vertical Scale**: Controls the amplitude scale of the FFT display
  - For dB mode: typically 1-20 dB/div (common values: 5, 10 dB/div)
  - For linear mode: depends on input signal amplitude
- **Vertical Position**: Controls the vertical offset of the FFT trace (-5.0 to +5.0 divisions)
- **Window Functions**: RECTANGULAR, HANNING, HAMMING, BLACKMAN, FLATTOP

Example using the enhanced FFT capture:

```python
from amp_benchkit.tek import scope_capture_fft_trace, scope_read_fft_vertical_params

# Read current FFT vertical parameters
params = scope_read_fft_vertical_params(resource)
print(f"Current scale: {params['scale']} units/div")

# Capture FFT with specific vertical scale (10 dB/div)
result = scope_capture_fft_trace(
    resource="USB0::0x0699::0x036A::C100563::INSTR",
    source=1,
    window="HANNING",
    scale="DB",
    vertical_scale=10.0,      # 10 dB/div
    vertical_position=0.0,     # centered
)
```

See `scripts/example_fft_capture.py` for a complete example with plotting and CSV export.

## LabJack U3-HV

- Install LabJack Exodriver (`liblabjackusb`) on macOS or Linux as described in `EXODRIVER.md`.
- Use USB for most setups; override connection with `U3_CONNECTION=ethernet` when required.
- The DAQ tab supports monitoring rails, temperatures, and digital pulses during sweeps.
- Simulator fallback returns deterministic values when `AMPBENCHKIT_FAKE_HW=1`.

## Session Directory

Set `AMPBENCHKIT_SESSION_DIR=/path/to/runs` to store captures (CSV/JSON/PNG). By default results are saved under `results/`, which is now `.gitignore`d for local experimentation.

## Safety Notes

- Never commit device serial numbers or lab credentials;-store them in environment variables.
- Document new lab setups in `EXODRIVER.md`, `SECURITY.md`, or the docs site so the team has a canonical reference.
