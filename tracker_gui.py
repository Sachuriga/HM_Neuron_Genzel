#!/usr/bin/env python3
"""genzeltracker — a small Qt front-end for the HM Tracker runner.

Pick a data-root folder, tick the pipeline steps (or a preset), press Run.
It just drives ``runner.py`` (the single source of truth) via a subprocess,
passing the step selection through the ``HM_STEPS`` env var and streaming the
runner's output into a log pane. Launch it from a terminal with ``genzeltracker``.

A data root without ip* folders (e.g. the share's task folder, sessions at
<rat>/<YYYYMMDD>/) runs in the runner's tree mode: "Find sessions" lists the
session folders, and while it runs the "Session jobs" panel shows the machine's
load, the sessions done and each running job, read from the runner's output.
"Only failed of a run…" limits the next run to the sessions an earlier run did
not finish (read from its summary.csv; passed to the runner as HM_SESSIONS_FILE).
"""

from __future__ import annotations

import csv
import os
import re
import sys
import tempfile
from pathlib import Path

from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment
from PyQt6.QtGui import QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QGroupBox, QPlainTextEdit,
    QFileDialog, QSpinBox, QSizePolicy, QProgressBar,
)

try:
    import psutil                              # optional: Stop also ends the runner's steps
except Exception:                              # pragma: no cover
    psutil = None

REPO = Path(__file__).resolve().parent
RUNNER = REPO / "runner.py"

# Single source of truth for the step list: import it from the runner.
try:
    sys.path.insert(0, str(REPO))
    from runner import MENU, SEQUENTIAL_STEPS, POST_JOB_STEPS, find_session_folders
except Exception as exc:                      # pragma: no cover - fallback
    print(f"[genzeltracker] could not import MENU from runner.py: {exc}")
    MENU, SEQUENTIAL_STEPS, POST_JOB_STEPS, find_session_folders = [], [], [], None

# Presets: a name -> ordered step-key string. Order follows MENU for parallel steps.
PRESETS = {
    "— Maze —": "",
    "Tracker implanted (Video process + ephys)": "12345678d",
    "Tracker non-implanted (Video process, no ephys)": "23456d",
    "Retrack": "3456d",
    "LED Sync + stitch + track": "234", 
    "After manual curation (Maze)": "wuvbm",
    "— Sleep —": "",
    # step 1 exports the analog (IMU) data step 8's motion reads
    "Prepare sleep scoring and spike sorting": "178",
    "Prepare sleep scoring only": "18",
    "Prepare spike sorting only": "17",
    "After manual curation (Sleep)": "wu",
    "— Other —": "",
    "NWB packaging (nwb + units + visualise)": "wuv",
    "Analysis (decode / UMAP)": "nbm",
    "Spike sorting (+ continue)": "7c",
    "Full pipeline": "1234567c89wuvnbmt",
    # a root without ip* folders, e.g. the share's task folder: one job per session
    "— Session tree (e.g. the share) —": "",
    "Fix NWB LFP — check only (dry run)": "l",
    "Fix NWB LFP only (no figures)": "l",
    "Fix NWB LFP + remake theta pages": "lv",
    "Remake figures (visualise / decode / plots / UMAP)": "vb5m",
    "After manual curation (session tree)": "wuvb5m",
}
# presets that also set step l's dry-run box
PRESET_DRY_RUN = {"Fix NWB LFP — check only (dry run)": True,
                  "Fix NWB LFP only (no figures)": False,
                  "Fix NWB LFP + remake theta pages": False}

# The runner's tree-mode lines the "Session jobs" panel follows
_TREE_TOTAL = re.compile(r"^\[TREE\] (\d+) session\(s\) x steps \[(\w*)\]")
_TREE_EVENT = re.compile(r"^\[TREE\] (start|done|FAILED)\s+(\S+)")
_MON = re.compile(r"^\[MON [\d:]+\] (.*?) \| \d+ running, \d+ queued, \d+ done")
_JOB = re.compile(r"^    (\S+)\s+step (\S+)\s+([\d.]+) min(?:\s+([\d.]+) GB)?")

DEFAULT_CONFIG = str(Path.home() / "Desktop" / "hm_tracker_paths.txt")
# where the runner's tree mode keeps its per-run logs + summary.csv (POST_LOG_DIR)
POST_LOG_DIR = Path(tempfile.gettempdir()) / "hm_post"


def failed_sessions(summary_csv):
    """The session folders a tree-mode run did not finish: every row of its
    summary.csv whose status is not 'ok' (failed, interrupted, not started)."""
    folders = []
    with open(summary_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status = row.get("status", "")
            folder = row.get("folder", "")
            if status != "ok" and folder:
                folders.append(folder)
    return folders


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
        self.root_edit.setPlaceholderText(
            "data root: ip*/op* sub-folders, or a tree of session folders (e.g. the share's task folder)")
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

        # --- session tree (a root without ip* folders) -------------------------
        tbox = QGroupBox("Session tree — a data root without ip* folders, e.g. the share's task folder")
        tbox_rows = QVBoxLayout(tbox)
        trow = QHBoxLayout()
        tbox_rows.addLayout(trow)
        trow.addWidget(QLabel("Parallel sessions"))
        self.post_jobs = QSpinBox(); self.post_jobs.setRange(0, 32); self.post_jobs.setValue(0)
        self.post_jobs.setSpecialValueText("auto"); self.post_jobs.setMaximumWidth(72)
        self.post_jobs.setToolTip("Sessions processed at once (auto: half the CPU cores, "
                                  "one per 12 GB RAM, at most 6). New jobs also wait for the "
                                  "CPU/GPU/MEM limits above.")
        trow.addWidget(self.post_jobs)
        trow.addSpacing(16)
        self.lfp_dry_run = QCheckBox("Step l: check only (dry run — changes nothing)")
        self.lfp_dry_run.setToolTip("Rewrite and verify each NWB's LFP on a local copy, "
                                    "without replacing anything in the session folders.")
        trow.addWidget(self.lfp_dry_run)
        trow.addStretch(1)
        find_btn = QPushButton("Find sessions")
        find_btn.setToolTip("List the session folders below the data root.")
        find_btn.clicked.connect(self._find_sessions)
        trow.addWidget(find_btn)

        # which sessions: all found below the data root, or only the ones an
        # earlier run did not finish (read from that run's summary.csv)
        self.only_sessions: list[str] = []
        srow = QHBoxLayout()
        tbox_rows.addLayout(srow)
        self.only_label = QLabel("Sessions: all found below the data root")
        srow.addWidget(self.only_label)
        srow.addStretch(1)
        failed_btn = QPushButton("Only failed of a run…")
        failed_btn.setToolTip("Pick a run's summary.csv (in the hm_post log folder): the next "
                              "Run processes only its sessions that are not 'ok'.")
        failed_btn.clicked.connect(self._pick_failed_run)
        srow.addWidget(failed_btn)
        self.all_sessions_btn = QPushButton("All sessions")
        self.all_sessions_btn.setToolTip("Go back to every session found below the data root.")
        self.all_sessions_btn.clicked.connect(self._clear_only_sessions)
        self.all_sessions_btn.setEnabled(False)
        srow.addWidget(self.all_sessions_btn)
        v.addWidget(tbox)

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

        # --- session jobs (tree mode): load, progress, running jobs -------------
        self.jobs_box = QGroupBox("Session jobs")
        jv = QVBoxLayout(self.jobs_box)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m sessions done")
        jv.addWidget(self.progress)
        self.machine = QLabel("")
        self.jobs = QLabel("")
        for lab in (self.machine, self.jobs):
            lab.setStyleSheet("font-family:monospace; color:#333;")
            lab.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            jv.addWidget(lab)
        self.jobs_box.setVisible(False)
        v.addWidget(self.jobs_box)
        self._reset_jobs()

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
        if name in PRESET_DRY_RUN:
            self.lfp_dry_run.setChecked(PRESET_DRY_RUN[name])
        self._on_check_changed()

    def _find_sessions(self):
        """List the session folders the runner's tree mode would process."""
        root = Path(self.root_edit.text().strip())
        if not str(root) or not root.is_dir():
            self.statusBar().showMessage("Pick a valid data-root folder first.")
            return
        if any(d.is_dir() for d in root.glob("ip*")):
            self._append(f"[genzeltracker] {root} has ip* folders: it runs as ipN -> opN pairs "
                         f"(no session tree).\n")
            return
        if find_session_folders is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found = find_session_folders(root)
        finally:
            QApplication.restoreOverrideCursor()
        self._append(f"[genzeltracker] {len(found)} session folder(s) below {root}:\n"
                     + "".join(f"    {p.relative_to(root)}\n" for p in found))
        self.statusBar().showMessage(
            f"{len(found)} session folder(s); steps {' '.join(POST_JOB_STEPS)} run per session "
            f"in parallel.")

    def _pick_failed_run(self):
        """Ask for a run's summary.csv, then limit the next run to its unfinished sessions."""
        start_dir = POST_LOG_DIR if POST_LOG_DIR.is_dir() else Path.home()
        path, _ = QFileDialog.getOpenFileName(self, "Pick a run's summary.csv", str(start_dir),
                                              "Run summary (summary.csv);;CSV files (*.csv)")
        if path:
            self._use_failed_run(path)

    def _use_failed_run(self, summary_csv):
        """Limit the next run to the sessions in summary_csv that are not 'ok'."""
        folders = failed_sessions(summary_csv)
        run_name = Path(summary_csv).parent.name          # the run's start time
        if not folders:
            self._append(f"[genzeltracker] {summary_csv}: every session is ok — nothing to re-run.\n")
            return
        self.only_sessions = folders
        self.only_label.setText(f"Sessions: only the {len(folders)} not ok in run {run_name}")
        self.all_sessions_btn.setEnabled(True)
        self._append(f"[genzeltracker] the next run processes only these {len(folders)} session(s) "
                     f"from {summary_csv}:\n" + "".join(f"    {p}\n" for p in folders))

    def _clear_only_sessions(self):
        """Back to every session found below the data root."""
        self.only_sessions = []
        self.only_label.setText("Sessions: all found below the data root")
        self.all_sessions_btn.setEnabled(False)

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
        # tree mode; the HM_ names win over the same settings in the config file
        env.insert("HM_POST_JOBS", str(self.post_jobs.value() or "auto"))
        env.insert("HM_LFP_FIX_DRY_RUN", "1" if self.lfp_dry_run.isChecked() else "0")
        # only the picked sessions: the runner reads them from a list file
        # (always set, so an HM_SESSIONS_FILE left in the shell cannot leak in)
        sessions_file = ""
        if self.only_sessions:
            list_file = POST_LOG_DIR / "only_sessions.txt"
            list_file.parent.mkdir(parents=True, exist_ok=True)
            list_file.write_text("\n".join(self.only_sessions) + "\n", encoding="utf-8")
            sessions_file = str(list_file)
        env.insert("HM_SESSIONS_FILE", sessions_file)
        self._reset_jobs()

        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(REPO))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.start(sys.executable, ["-u", str(RUNNER), root])

        dry = " (step l: dry run)" if "l" in steps and self.lfp_dry_run.isChecked() else ""
        only = f" (only the {len(self.only_sessions)} listed sessions)" if self.only_sessions else ""
        self._append(f"$ HM_STEPS={steps}  {sys.executable} runner.py {root}{dry}{only}\n")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage(f"Running steps: {steps}")

    def _stop(self):
        if self.proc is not None:
            # the runner's own child processes (the step it is running, tree-mode
            # session jobs) would outlive it otherwise; worker terminal windows are
            # not its children and keep running, as before
            if psutil is not None:
                try:
                    for child in psutil.Process(int(self.proc.processId())).children(recursive=True):
                        child.kill()
                except Exception:
                    pass
            self.proc.kill()

    def _read_output(self):
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._append(data)
        self._follow(data)

    # ------------------------------------------------------------- session jobs
    def _reset_jobs(self):
        self._buf, self._total, self._done, self._failed = "", 0, 0, []
        self._running, self._job_rows, self._waits = set(), {}, ""
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.machine.setText("")
        self.jobs.setText("")

    def _follow(self, data):
        """Update the Session jobs panel from the runner's tree-mode lines."""
        lines = (self._buf + data).replace("\r", "").split("\n")
        self._buf = lines.pop()
        for line in lines:
            if m := _TREE_TOTAL.match(line):
                self._total = int(m.group(1))
                self.jobs_box.setTitle(f"Session jobs — steps [{m.group(2)}]")
                self.jobs_box.setVisible(True)
            elif m := _TREE_EVENT.match(line):
                kind, label = m.groups()
                if kind == "start":
                    self._running.add(label)
                else:
                    self._running.discard(label)
                    self._job_rows.pop(label, None)
                    self._done += 1
                    if kind == "FAILED":
                        self._failed.append(label)
            elif m := _MON.match(line):
                self.machine.setText(m.group(1))
                self._job_rows, self._waits = {}, ""
            elif m := _JOB.match(line):
                self._job_rows[m.group(1)] = m.groups()[1:]
            elif line.startswith("    next start waits:"):
                self._waits = line.strip()
            else:
                continue
            self._show_jobs()

    def _show_jobs(self):
        self.progress.setRange(0, max(self._total, 1))
        self.progress.setValue(self._done)
        self.progress.setFormat(f"%v / %m sessions done" + (f" — {len(self._failed)} failed: "
                                + ", ".join(self._failed) if self._failed else ""))
        rows = []
        for label in sorted(self._running):
            step, minutes, gb = self._job_rows.get(label, ("-", "", None))
            rows.append(f"{label:<28} step {step:<2} {minutes:>6} min" + (f"  {gb} GB" if gb else ""))
        if self._waits:
            rows.append(self._waits)
        self.jobs.setText("\n".join(rows))

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
