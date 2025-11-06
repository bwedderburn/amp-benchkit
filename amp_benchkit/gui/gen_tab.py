"""Generator tab UI construction.

Separated from monolithic unified_gui_layout for modularity.
"""

from __future__ import annotations

from typing import Any

from amp_benchkit.fy import FY_MAX_VPP, FY_PROTOCOLS  # central definition

from ._qt import require_qt  # headless-safe import helper

__all__ = ["build_generator_tab"]


def build_generator_tab(gui: Any) -> object | None:
    """Build generator tab and return QWidget.

    Expects `gui` to supply:
      - apply_gen_side(side)
      - start_sweep_side(side)
      - stop_sweep_side(side)
      - scan_serial_into(target_edit)
      - _log(target_widget, text)
    And attributes are attached to `gui` (wave1, freq1, etc.).
    """
    qt = require_qt()
    if qt is None:  # Qt not available
        return None
    QWidget = qt.QWidget
    QVBoxLayout = qt.QVBoxLayout
    QHBoxLayout = qt.QHBoxLayout
    QLabel = qt.QLabel
    QComboBox = qt.QComboBox
    QLineEdit = qt.QLineEdit
    QPushButton = qt.QPushButton
    QTextEdit = qt.QTextEdit
    Qt = qt.Qt
    w = QWidget()
    L = QVBoxLayout(w)
    row = QHBoxLayout()
    # CH1 panel
    c1 = QVBoxLayout()
    c1.addWidget(QLabel("CH1"))
    gui.wave1 = QComboBox()
    gui.wave1.addItems(["Sine", "Square", "Triangle", "Pulse"])
    c1.addWidget(QLabel("Waveform:"))
    c1.addWidget(gui.wave1)
    gui.freq1 = QLineEdit("1000")
    c1.addWidget(QLabel("Frequency (Hz):"))
    c1.addWidget(gui.freq1)
    gui.amp1 = QLineEdit("0.25")
    c1.addWidget(QLabel(f"Amplitude (Vpp, ≤{FY_MAX_VPP:.2f}):"))
    c1.addWidget(gui.amp1)
    gui.amp1.setPlaceholderText(f"≤ {FY_MAX_VPP:.2f}")
    gui.amp1.setToolTip(f"FY output is limited to {FY_MAX_VPP:.2f} Vpp for safety.")
    gui.off1 = QLineEdit("0.0")
    c1.addWidget(QLabel("Offset (V):"))
    c1.addWidget(gui.off1)
    gui.duty1 = QLineEdit("50.0")
    c1.addWidget(QLabel("Duty (%):"))
    c1.addWidget(gui.duty1)
    gui.proto1 = QComboBox()
    gui.proto1.addItems(FY_PROTOCOLS)
    c1.addWidget(QLabel("Protocol:"))
    c1.addWidget(gui.proto1)
    pr1 = QHBoxLayout()
    gui.port1 = QLineEdit("")
    pr1.addWidget(QLabel("Serial (auto/override):"))
    pr1.addWidget(gui.port1)
    b1 = QPushButton("Scan")
    b1.clicked.connect(lambda: gui.scan_serial_into(gui.port1))
    pr1.addWidget(b1)
    c1.addLayout(pr1)
    c1.addWidget(QLabel("Sweep controls:"))
    s1 = QHBoxLayout()

    def col(txt, wdg):
        lay = QVBoxLayout()
        lay.addWidget(QLabel(txt, alignment=Qt.AlignHCenter))
        wdg.setMaximumWidth(140)
        lay.addWidget(wdg)
        return lay

    gui.sw_start1 = QLineEdit("")
    s1.addLayout(col("Start Hz", gui.sw_start1))
    gui.sw_end1 = QLineEdit("")
    s1.addLayout(col("End Hz", gui.sw_end1))
    gui.sw_time1 = QLineEdit("10")
    s1.addLayout(col("Time s", gui.sw_time1))
    gui.sw_mode1 = QComboBox()
    gui.sw_mode1.addItems(["Linear", "Log"])
    s1.addLayout(col("Mode", gui.sw_mode1))
    gui.sw_amp1 = QLineEdit("")
    gui.sw_amp1.setPlaceholderText(f"≤ {FY_MAX_VPP:.2f}")
    gui.sw_amp1.setToolTip(f"Override sweep amplitude (max {FY_MAX_VPP:.2f} Vpp).")
    s1.addLayout(col("Amp Vpp", gui.sw_amp1))
    gui.sw_dwell1 = QLineEdit("")
    s1.addLayout(col("Dwell ms", gui.sw_dwell1))
    c1.addLayout(s1)
    ab1 = QHBoxLayout()
    a1 = QPushButton("Apply CH1")
    a1.clicked.connect(lambda: gui.apply_gen_side(1))
    ab1.addWidget(a1)
    rs1 = QPushButton("Start Sweep CH1")
    rs1.clicked.connect(lambda: gui.start_sweep_side(1))
    ab1.addWidget(rs1)
    st1 = QPushButton("Stop Sweep CH1")
    st1.clicked.connect(lambda: gui.stop_sweep_side(1))
    ab1.addWidget(st1)
    c1.addLayout(ab1)

    # CH2 panel
    c2 = QVBoxLayout()
    c2.addWidget(QLabel("CH2"))
    gui.wave2 = QComboBox()
    gui.wave2.addItems(["Sine", "Square", "Triangle", "Pulse"])
    c2.addWidget(QLabel("Waveform:"))
    c2.addWidget(gui.wave2)
    gui.freq2 = QLineEdit("1000")
    c2.addWidget(QLabel("Frequency (Hz):"))
    c2.addWidget(gui.freq2)
    gui.amp2 = QLineEdit("0.25")
    c2.addWidget(QLabel(f"Amplitude (Vpp, ≤{FY_MAX_VPP:.2f}):"))
    c2.addWidget(gui.amp2)
    gui.amp2.setPlaceholderText(f"≤ {FY_MAX_VPP:.2f}")
    gui.amp2.setToolTip(f"FY output is limited to {FY_MAX_VPP:.2f} Vpp for safety.")
    gui.off2 = QLineEdit("0.0")
    c2.addWidget(QLabel("Offset (V):"))
    c2.addWidget(gui.off2)
    gui.duty2 = QLineEdit("50.0")
    c2.addWidget(QLabel("Duty (%):"))
    c2.addWidget(gui.duty2)
    gui.proto2 = QComboBox()
    gui.proto2.addItems(FY_PROTOCOLS)
    c2.addWidget(QLabel("Protocol:"))
    c2.addWidget(gui.proto2)
    pr2 = QHBoxLayout()
    gui.port2 = QLineEdit("")
    pr2.addWidget(QLabel("Serial (auto/override):"))
    pr2.addWidget(gui.port2)
    b2 = QPushButton("Scan")
    b2.clicked.connect(lambda: gui.scan_serial_into(gui.port2))
    pr2.addWidget(b2)
    c2.addLayout(pr2)
    c2.addWidget(QLabel("Sweep controls:"))
    sweep_note = QLabel("FY3200S only supports sweeping on channel 1.")
    sweep_note.setWordWrap(True)
    c2.addWidget(sweep_note)
    ab2 = QHBoxLayout()
    a2 = QPushButton("Apply CH2")
    a2.clicked.connect(lambda: gui.apply_gen_side(2))
    ab2.addWidget(a2)
    c2.addLayout(ab2)

    row.addLayout(c1)
    row.addSpacing(20)
    row.addLayout(c2)
    L.addLayout(row)
    gui.gen_log = QTextEdit()
    gui.gen_log.setReadOnly(True)
    L.addWidget(gui.gen_log)
    return w
