#!/usr/bin/env python3
"""genzeltracker — a small Qt front-end for the HM Tracker runner.

Pick a data-root folder, tick the pipeline steps (or a preset), press Run.
It just drives ``runner.py`` (the single source of truth) via a subprocess,
passing the step selection through the ``HM_STEPS`` env var and streaming the
runner's output into a log pane. Launch it from a terminal with ``genzeltracker``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QGroupBox, QPlainTextEdit,
    QFileDialog, QSpinBox, QSizePolicy,
)

REPO = Path(__file__).resolve().parent
RUNNER = REPO / "runner.py"

# Single source of truth for the step list: import it from the runner.
try:
    sys.path.insert(0, str(REPO))
    from runner import MENU, SEQUENTIAL_STEPS
except Exception as exc:                      # pragma: no cover - fallback
    print(f"[genzeltracker] could not import MENU from runner.py: {exc}")
    MENU, SEQUENTIAL_STEPS = [], []

# Presets: a name -> ordered step-key string. Order follows MENU for parallel steps.
PRESETS = {
    "— Custom —": "",
    "Tracker implanted": "1e2345678d",
    "Tracker non-implanted": "23456d",
    "After manual curation": "wuvbm",
    "Retrack": "346d",
    "Full pipeline": "1e234567c89wuvnbmst",
    "Trodes export (DIO/raw/analog + LFP)": "1e",
    "Sync + stitch + track": "234",
    "Spike sorting (+ continue)": "7c",
    "LFP + motion + EMG (sleep)": "e8",
    "NWB packaging (nwb + units + visualise)": "wuv",
    "Analysis (decode / UMAP / summary)": "nbms",
    "Drive scan (QC)": "t",
}

DEFAULT_CONFIG = str(Path.home() / "Desktop" / "hm_tracker_paths.txt")


class TrackerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Genzel Tracker")
        self.resize(860, 760)
        self.proc: QProcess | None = None

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(10)

        title = QLabel("Genzel Tracker")
        title.setStyleSheet("font-size:18px; font-weight:700; color:#1f6f43;")
        v.addWidget(title)

        # --- data root + config ------------------------------------------------
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("data root folder (contains ip*/op* sub-folders)")
        v.addLayout(self._path_row("Data root:", self.root_edit, self._pick_root))

        self.cfg_edit = QLineEdit(DEFAULT_CONFIG)
        v.addLayout(self._path_row("Config:", self.cfg_edit, self._pick_cfg))

        # --- preset ------------------------------------------------------------
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Preset:"))
        self.preset = QComboBox()
        self.preset.addItems(PRESETS.keys())
        self.preset.currentTextChanged.connect(self._apply_preset)
        prow.addWidget(self.preset, 1)
        v.addLayout(prow)

        # --- steps -------------------------------------------------------------
        box = QGroupBox("Steps")
        grid = QGridLayout(box)
        grid.setVerticalSpacing(3)
        self.checks: dict[str, QCheckBox] = {}
        seq = set(SEQUENTIAL_STEPS)
        for i, (key, label) in enumerate(MENU):
            tag = " · seq" if key in seq else ""
            cb = QCheckBox(f"[{key}]  {label}{tag}")
            cb.stateChanged.connect(self._on_check_changed)
            self.checks[key] = cb
            grid.addWidget(cb, i % ((len(MENU) + 1) // 2), i // ((len(MENU) + 1) // 2))
        v.addWidget(box)

        # --- options -----------------------------------------------------------
        opt = QHBoxLayout()
        self.own_windows = QCheckBox("Each worker in its own terminal")
        self.own_windows.setChecked(True)
        opt.addWidget(self.own_windows)
        opt.addSpacing(16)
        opt.addWidget(QLabel("Max CPU%"))
        self.cpu = self._spin(90); opt.addWidget(self.cpu)
        opt.addWidget(QLabel("GPU%"))
        self.gpu = self._spin(90); opt.addWidget(self.gpu)
        opt.addWidget(QLabel("MEM%"))
        self.mem = self._spin(65); opt.addWidget(self.mem)
        opt.addStretch(1)
        v.addLayout(opt)

        # --- selection preview + buttons --------------------------------------
        self.sel_label = QLabel("steps: (none)")
        self.sel_label.setStyleSheet("color:#555; font-family:monospace;")
        v.addWidget(self.sel_label)

        brow = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setStyleSheet("background:#2e8b2e; color:white; font-weight:700; padding:8px 20px;")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        clear_btn = QPushButton("Clear log")
        clear_btn.clicked.connect(lambda: self.log.clear())
        brow.addWidget(self.run_btn)
        brow.addWidget(self.stop_btn)
        brow.addStretch(1)
        brow.addWidget(clear_btn)
        v.addLayout(brow)

        # --- log ---------------------------------------------------------------
        self.log = QPlainTextEdit(readOnly=True)
        self.log.setFont(QFont("Menlo", 10))
        self.log.setStyleSheet("background:#101418; color:#d6dbe0;")
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        v.addWidget(self.log, 1)

        self.statusBar().showMessage("Ready.")

    # --------------------------------------------------------------- ui helpers
    def _path_row(self, label, edit, cb):
        h = QHBoxLayout()
        lab = QLabel(label); lab.setMinimumWidth(64)
        h.addWidget(lab)
        h.addWidget(edit, 1)
        b = QPushButton("Browse…"); b.clicked.connect(cb)
        h.addWidget(b)
        return h

    def _spin(self, val):
        s = QSpinBox(); s.setRange(1, 100); s.setValue(val); s.setMaximumWidth(64)
        return s

    def _pick_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select data root")
        if d:
            self.root_edit.setText(d)

    def _pick_cfg(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select hm_tracker_paths.txt",
                                           str(Path(self.cfg_edit.text()).parent))
        if f:
            self.cfg_edit.setText(f)

    def _apply_preset(self, name):
        keys = PRESETS.get(name, "")
        if name == "— Custom —":
            return
        for key, cb in self.checks.items():
            cb.blockSignals(True)
            cb.setChecked(key in keys)
            cb.blockSignals(False)
        self._on_check_changed()

    def _selection(self) -> str:
        return "".join(k for k, _ in MENU if self.checks[k].isChecked())

    def _on_check_changed(self, *_):
        sel = self._selection()
        self.sel_label.setText(f"steps: {sel or '(none)'}")

    # ------------------------------------------------------------------- runner
    def _run(self):
        if self.proc is not None:
            return
        root = self.root_edit.text().strip()
        steps = self._selection()
        if not root or not Path(root).is_dir():
            self.statusBar().showMessage("Pick a valid data-root folder first.")
            return
        if not steps:
            self.statusBar().showMessage("Select at least one step.")
            return

        env = QProcessEnvironment.systemEnvironment()
        env.insert("HM_STEPS", steps)
        env.insert("HM_CONFIG_FILE", self.cfg_edit.text().strip())
        env.insert("MAX_CPU", str(self.cpu.value()))
        env.insert("MAX_GPU", str(self.gpu.value()))
        env.insert("MAX_MEM", str(self.mem.value()))
        env.insert("WORKER_WINDOWS", "1" if self.own_windows.isChecked() else "0")
        env.insert("PYTHONUNBUFFERED", "1")

        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(REPO))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.start(sys.executable, ["-u", str(RUNNER), root])

        self._append(f"$ HM_STEPS={steps}  {sys.executable} runner.py {root}\n")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage(f"Running steps: {steps}")

    def _stop(self):
        if self.proc is not None:
            self.proc.kill()

    def _read_output(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._append(data)

    def _append(self, text):
        self.log.moveCursor(QTextCursor.MoveOperation.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.MoveOperation.End)

    def _on_finished(self, code, _status):
        self._append(f"\n[genzeltracker] runner exited with code {code}.\n")
        self.statusBar().showMessage(f"Finished (exit {code}).")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.proc = None


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Genzel Tracker")
    win = TrackerGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
