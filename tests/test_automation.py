import math

from amp_benchkit.automation import build_freq_list, sweep_audio_kpis, sweep_scope_fixed


def test_build_freq_list_basic():
    freqs = build_freq_list(100, 300, 100)
    assert freqs == [100, 200, 300]


def test_sweep_scope_fixed_minimal(monkeypatch):
    # Inject fakes
    calls = []

    def fake_fy_apply(**kw):
        calls.append(kw["freq_hz"])

    def fake_scope_measure(src, metric):
        # return simple function of src for determinism
        return 1.23 if metric == "RMS" else 4.56

    out = sweep_scope_fixed(
        freqs=[100, 200],
        channel=1,
        scope_channel=1,
        amp_vpp=1.0,
        dwell_s=0.0,
        metric="RMS",
        fy_apply=fake_fy_apply,
        scope_measure=fake_scope_measure,
    )
    assert len(out) == 2
    assert calls == [100, 200]
    assert out[0][1] == 1.23


def test_sweep_audio_kpis_basic(monkeypatch):
    freqs = [100, 200]

    def fake_fy_apply(**kw):
        pass

    # Provide a simple 1 kHz sine capture irrespective of freq to exercise DSP path
    import numpy as np

    fs = 10000.0
    f0 = 100.0
    N = 1024
    t = np.arange(N) / fs
    sig = 0.5 * np.sin(2 * math.pi * f0 * t)

    def fake_capture(res, ch):
        return t, sig

    def vrms(v):
        import numpy as np

        return float(np.sqrt((np.array(v) ** 2).mean()))

    def vpp(v):
        return float(max(v) - min(v))

    def thd_fft(t, v, f):
        # Return fake THD ratio 0.1
        return 0.1, f, None

    def find_knees(freqs, amps, ref_mode, ref_hz, drop_db):
        return freqs[0], freqs[-1], max(amps), 0.0

    res = sweep_audio_kpis(
        freqs,
        channel=1,
        scope_channel=1,
        amp_vpp=1.0,
        dwell_s=0.0,
        fy_apply=fake_fy_apply,
        scope_capture_calibrated=fake_capture,
        dsp_vrms=vrms,
        dsp_vpp=vpp,
        dsp_thd_fft=thd_fft,
        dsp_find_knees=find_knees,
        do_thd=True,
        do_knees=True,
    )
    rows = res["rows"]
    assert len(rows) == 2
    assert res["knees"][0] == 100 and res["knees"][1] == 200
    # Vrms of 0.5 * sin should be ~0.3535
    assert abs(rows[0][1] - 0.3535) < 0.02


def test_sweep_scope_fixed_calls_progress_on_fy_error():
    freqs = [100, 200, 300]
    attempted = []
    progress_calls = []

    def fake_fy_apply(freq_hz, **kw):
        attempted.append(freq_hz)
        if freq_hz == 200:
            raise RuntimeError("FY failure")

    def fake_scope_measure(src, metric):
        return 1.0

    out = sweep_scope_fixed(
        freqs=freqs,
        channel=1,
        scope_channel=1,
        amp_vpp=1.0,
        dwell_s=0.0,
        metric="RMS",
        fy_apply=fake_fy_apply,
        scope_measure=fake_scope_measure,
        progress=lambda i, n: progress_calls.append((i, n)),
    )
    assert attempted == freqs
    assert progress_calls == [(1, 3), (2, 3), (3, 3)]
    assert [row[0] for row in out] == [100, 300]


def test_sweep_audio_kpis_calls_progress_on_fy_error():
    freqs = [100, 200, 300]
    attempted = []
    progress_calls = []

    def fake_fy_apply(freq_hz, **kw):
        attempted.append(freq_hz)
        if freq_hz == 200:
            raise RuntimeError("FY failure")

    def fake_capture(res, ch):
        return [0.0, 0.001], [0.0, 0.5]

    def dsp_vrms(v):
        return float(math.sqrt(sum(x * x for x in v) / len(v)))

    def dsp_vpp(v):
        return float(max(v) - min(v)) if v else float("nan")

    res = sweep_audio_kpis(
        freqs=freqs,
        channel=1,
        scope_channel=1,
        amp_vpp=1.0,
        dwell_s=0.0,
        fy_apply=fake_fy_apply,
        scope_capture_calibrated=fake_capture,
        dsp_vrms=dsp_vrms,
        dsp_vpp=dsp_vpp,
        progress=lambda i, n: progress_calls.append((i, n)),
    )
    rows = res["rows"]
    assert attempted == freqs
    assert progress_calls == [(1, 3), (2, 3), (3, 3)]
    assert [row[0] for row in rows] == [100, 300]


def test_sweep_scope_fixed_applies_calibration():
    amps = []

    def fake_fy_apply(**kw):
        amps.append(kw["amp_vpp"])

    def fake_measure(src, metric):
        return 1.0

    out = sweep_scope_fixed(
        freqs=[100],
        channel=1,
        scope_channel=1,
        amp_vpp=1.0,
        dwell_s=0.0,
        metric="RMS",
        fy_apply=fake_fy_apply,
        scope_measure=fake_measure,
        amplitude_calibration=lambda freq, val: val / 2,
        amp_vpp_strategy=lambda freq: 0.5,
    )
    assert out[0][1] == 0.5
    assert amps == [0.5]


def test_sweep_audio_kpis_uses_amp_strategy_and_calibration():
    amps = []

    def fake_fy_apply(**kw):
        amps.append(kw["amp_vpp"])

    def fake_capture(resrc, ch):
        return [0.0, 0.001], [0.0, 1.0]

    res = sweep_audio_kpis(
        freqs=[100],
        channel=1,
        scope_channel=1,
        amp_vpp=1.0,
        dwell_s=0.0,
        fy_apply=fake_fy_apply,
        scope_capture_calibrated=fake_capture,
        dsp_vrms=lambda v: 2.0,
        dsp_vpp=lambda v: 4.0,
        progress=lambda i, n: None,
        amplitude_calibration=lambda freq, val: val / 2,
        amp_vpp_strategy=lambda freq: 0.5,
    )
    row = res["rows"][0]
    assert abs(row[1] - 1.0) < 1e-9
    assert abs(row[2] - 2.0) < 1e-9
    assert amps == [0.5]
