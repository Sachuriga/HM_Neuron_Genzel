#!/usr/bin/env python3
"""Drive-coverage checker — a Qt front-end that cross-checks a HexMaze
experiment spreadsheet against what is actually stored on the acquisition
drives.

Workflow:

  1. Browse to the experiment Excel (the ``Raw`` sheet, one row per trial).
     Every unique (subject, day, session, Date) becomes an *expected*
     rat-session, and the ``Implant`` column decides what data it should have:

         Implant == 0  ->  camera video only          (e.g. Rat3 / Rat4)
         Implant == 1  ->  video + ephys pre/task/post (e.g. Rat5 / Rat6)

  2. Browse to up to 4 drive-root folders (data is spread across several
     drives). Each root is laid out as ``Rat<N>_*/<YYYYMMDD>/...`` — the same
     layout ``scan_drive.py`` already understands, whose scanning helpers this
     GUI reuses.

  3. Press *Scan drives*. For every expected rat-session the tool looks for a
     matching ``Rat<N>/<YYYYMMDD>`` folder across all 4 roots and reports which
     data is present and which is MISSING.

Launch with:  python scan_drive_gui.py
"""

from __future__ import annotations

import sys
import csv
import datetime
import subprocess
from pathlib import Path

import pandas as pd

from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QBrush, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QComboBox,
)

# Reuse the drive-scanning helpers from the sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import scan_drive as sd  # noqa: E402

EXPECTED_PHASES = ("pre", "task", "post")
COLUMNS = ["Rat", "Date", "day", "session", "Expected",
           "Video", "pre", "task", "post", "Found in", "Status"]

# Status -> row background tint (light, works on default palette).
_STATUS_COLOR = {
    "OK": QColor(210, 244, 214),
    "PARTIAL": QColor(255, 244, 205),
    "MISSING": QColor(250, 214, 214),
    "?": QColor(238, 238, 238),
}


# ------------------------------------------------------------------
#                       spreadsheet -> roster
# ------------------------------------------------------------------
def parse_date8(val) -> str | None:
    """Normalise a spreadsheet Date cell to a ``YYYYMMDD`` drive-folder name.

    The column mixes ``DD.MM.YYYY`` strings and real Timestamps; handle both."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (pd.Timestamp, datetime.datetime, datetime.date)):
        return val.strftime("%Y%m%d")
    s = str(val).strip()
    if not s:
        return None
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S",
                "%Y%m%d", "%d.%m.%y"):
        try:
            return datetime.datetime.strptime(s, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    try:
        return pd.to_datetime(s, dayfirst=True).strftime("%Y%m%d")
    except Exception:
        return None


def build_roster(xlsx_path: str, sheet: str = "Raw") -> list[dict]:
    """Read the spreadsheet and return one dict per expected rat-session."""
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    required = ["subject", "day", "session", "Date", "Implant"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"sheet '{sheet}' is missing column(s): {', '.join(missing)}")

    sub = df.dropna(subset=["subject"]).copy()
    sub["date8"] = sub["Date"].apply(parse_date8)
    grouped = (sub.groupby(["subject", "day", "session", "date8", "Implant"], dropna=False)
                  .size().reset_index(name="trials"))

    roster = []
    for _, r in grouped.iterrows():
        implanted = int(r["Implant"]) == 1
        roster.append(dict(
            rat_no=int(r["subject"]),
            rat=f"Rat{int(r['subject'])}",
            date8=r["date8"],
            day=int(r["day"]),
            session=int(r["session"]),
            implanted=implanted,
            trials=int(r["trials"]),
            expected="video + ephys (pre/task/post)" if implanted else "video",
        ))
    roster.sort(key=lambda x: (x["rat_no"], x["day"]))
    return roster


# ------------------------------------------------------------------
#                       drives -> session index
# ------------------------------------------------------------------
def index_drives(roots: list[str]) -> dict:
    """Map (rat_no, YYYYMMDD) -> list of (drive_label, session_path) across all
    selected roots."""
    idx: dict = {}
    for i, root in enumerate(roots):
        if not root:
            continue
        p = Path(root)
        if not p.exists():
            continue
        label = f"Drive {i + 1}"
        for sess in sd.find_sessions(p):
            rat_no = sd._rat_of(sess.parent.name)
            if rat_no is None:
                continue
            idx.setdefault((rat_no, sess.name), []).append((label, sess))
    return idx


def inspect_session(sess: Path) -> dict:
    """Inventory a single located session folder. Returns n_video and the set of
    ephys phases present, by reusing scan_drive.scan_session."""
    issues, inv_rows, file_rows = [], [], []
    try:
        sd.scan_session(sess, issues, inv_rows, file_rows)
    except OSError:
        return dict(n_video=0, phases=set())
    if not inv_rows:
        return dict(n_video=0, phases=set())
    iv = inv_rows[0]
    phases = set(p for p in EXPECTED_PHASES if p in (iv.get("phases") or "").split("+"))
    return dict(n_video=int(iv.get("n_video", 0)), phases=phases,
                n_rec=int(iv.get("n_rec", 0)), n_merged=int(iv.get("n_merged", 0)),
                n_logger=int(iv.get("n_logger", 0)))


def evaluate(entry: dict, idx: dict) -> dict:
    """Compare one roster entry against the drive index; fill result fields."""
    res = dict(entry)
    res.update(n_video=0, phases=set(), found_in="", status="MISSING",
               n_rec=0, n_merged=0, n_logger=0, paths=[])

    if entry["date8"] is None:
        res["status"] = "MISSING"
        res["found_in"] = "(bad date in sheet)"
        return res

    matches = idx.get((entry["rat_no"], entry["date8"]), [])
    if not matches:
        return res

    labels, n_video, phases = [], 0, set()
    n_rec = n_merged = n_logger = 0
    paths = []
    for label, sess in matches:
        info = inspect_session(sess)
        n_video += info["n_video"]
        phases |= info["phases"]
        n_rec += info.get("n_rec", 0)
        n_merged += info.get("n_merged", 0)
        n_logger += info.get("n_logger", 0)
        labels.append(label)
        paths.append(str(sess))
    res.update(n_video=n_video, phases=phases, found_in=", ".join(sorted(set(labels))),
               n_rec=n_rec, n_merged=n_merged, n_logger=n_logger, paths=paths)

    # completeness
    have_video = n_video > 0
    if entry["implanted"]:
        have_all_phases = all(p in phases for p in EXPECTED_PHASES)
        if have_video and have_all_phases:
            res["status"] = "OK"
        else:
            res["status"] = "PARTIAL"
    else:
        res["status"] = "OK" if have_video else "PARTIAL"
    return res


# ------------------------------------------------------------------
#                       background scan worker
# ------------------------------------------------------------------
class ScanWorker(QObject):
    progress = pyqtSignal(int, int)          # done, total
    row_done = pyqtSignal(int, dict)         # row index, result
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, roster: list[dict], roots: list[str]):
        super().__init__()
        self.roster = roster
        self.roots = roots

    def run(self):
        try:
            idx = index_drives(self.roots)
            total = len(self.roster)
            for i, entry in enumerate(self.roster):
                res = evaluate(entry, idx)
                self.row_done.emit(i, res)
                self.progress.emit(i + 1, total)
            self.finished.emit()
        except Exception as exc:  # pragma: no cover
            self.failed.emit(str(exc))


# ------------------------------------------------------------------
#                              GUI
# ------------------------------------------------------------------
class ScanDriveGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HexMaze — Drive Coverage Checker")
        self.resize(1100, 720)
        self.roster: list[dict] = []
        self.results: list[dict] = []
        self.thread: QThread | None = None
        self.worker: ScanWorker | None = None

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(10)

        title = QLabel("Drive Coverage Checker")
        title.setStyleSheet("font-size:18px; font-weight:700; color:#1f6f43;")
        v.addWidget(title)

        # --- Excel picker -------------------------------------------------
        self.xlsx_edit = QLineEdit()
        self.xlsx_edit.setPlaceholderText("experiment spreadsheet (.xlsx) — the 'Raw' sheet")
        self.sheet_combo = QComboBox()
        self.sheet_combo.setEditable(True)
        self.sheet_combo.addItem("Raw")
        self.sheet_combo.setFixedWidth(140)
        row = QHBoxLayout()
        row.addWidget(QLabel("Excel:"))
        row.addWidget(self.xlsx_edit, 1)
        row.addWidget(QLabel("Sheet:"))
        row.addWidget(self.sheet_combo)
        b = QPushButton("Browse…")
        b.clicked.connect(self._pick_excel)
        row.addWidget(b)
        load = QPushButton("Load roster")
        load.clicked.connect(self._load_roster)
        row.addWidget(load)
        v.addLayout(row)

        # --- 4 drive pickers ----------------------------------------------
        box = QGroupBox("Drive folders (data is spread across up to 4 drives)")
        grid = QGridLayout(box)
        self.drive_edits: list[QLineEdit] = []
        for i in range(4):
            e = QLineEdit()
            e.setPlaceholderText(f"drive root {i + 1} — contains Rat<N>_*/<YYYYMMDD>/ …")
            btn = QPushButton("Browse…")
            btn.clicked.connect(lambda _, k=i: self._pick_drive(k))
            grid.addWidget(QLabel(f"Drive {i + 1}:"), i, 0)
            grid.addWidget(e, i, 1)
            grid.addWidget(btn, i, 2)
            self.drive_edits.append(e)
        v.addWidget(box)

        # --- actions ------------------------------------------------------
        act = QHBoxLayout()
        self.scan_btn = QPushButton("Scan drives")
        self.scan_btn.clicked.connect(self._scan)
        self.scan_btn.setEnabled(False)
        act.addWidget(self.scan_btn)
        self.export_btn = QPushButton("Export CSV…")
        self.export_btn.clicked.connect(self._export)
        self.export_btn.setEnabled(False)
        act.addWidget(self.export_btn)
        act.addStretch(1)
        self.status_lbl = QLabel("Load a spreadsheet to begin.")
        act.addWidget(self.status_lbl)
        v.addLayout(act)

        # --- results table ------------------------------------------------
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._open_folder)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(len(COLUMNS) - 2, QHeaderView.ResizeMode.Stretch)  # Found in
        v.addWidget(self.table, 1)

        hint = QLabel("Rows: green = complete · yellow = partial · red = missing. "
                      "Double-click a found row to open the folder.")
        hint.setStyleSheet("color:#666;")
        v.addWidget(hint)

    # -- pickers -----------------------------------------------------------
    def _pick_excel(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select experiment spreadsheet",
                                           str(Path.home()), "Excel (*.xlsx *.xlsm *.xls)")
        if f:
            self.xlsx_edit.setText(f)
            self._refresh_sheets(f)

    def _refresh_sheets(self, path: str):
        try:
            names = pd.ExcelFile(path).sheet_names
        except Exception:
            return
        cur = self.sheet_combo.currentText()
        self.sheet_combo.clear()
        self.sheet_combo.addItems(names)
        if "Raw" in names:
            self.sheet_combo.setCurrentText("Raw")
        elif cur in names:
            self.sheet_combo.setCurrentText(cur)

    def _pick_drive(self, k: int):
        d = QFileDialog.getExistingDirectory(self, f"Select drive root {k + 1}", str(Path.home()))
        if d:
            self.drive_edits[k].setText(d)

    # -- roster ------------------------------------------------------------
    def _load_roster(self):
        path = self.xlsx_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "No spreadsheet", "Pick a valid .xlsx file first.")
            return
        try:
            self.roster = build_roster(path, self.sheet_combo.currentText().strip() or "Raw")
        except Exception as exc:
            QMessageBox.critical(self, "Could not read spreadsheet", str(exc))
            return
        self.results = []
        self._populate_expected()
        n_imp = sum(1 for e in self.roster if e["implanted"])
        rats = sorted({e["rat"] for e in self.roster}, key=lambda s: int(s[3:]))
        self.status_lbl.setText(
            f"{len(self.roster)} rat-sessions · {', '.join(rats)} · "
            f"{n_imp} with ephys, {len(self.roster) - n_imp} video-only. "
            f"Now pick drives and press Scan.")
        self.scan_btn.setEnabled(True)
        self.export_btn.setEnabled(False)

    def _populate_expected(self):
        self.table.setRowCount(len(self.roster))
        for i, e in enumerate(self.roster):
            vals = [e["rat"], e["date8"] or "?", str(e["day"]), str(e["session"]),
                    e["expected"], "", "", "", "", "", "?"]
            for c, val in enumerate(vals):
                self._set(i, c, val, status="?")

    # -- scanning ----------------------------------------------------------
    def _scan(self):
        if not self.roster:
            return
        roots = [e.text().strip() for e in self.drive_edits]
        if not any(roots):
            QMessageBox.warning(self, "No drives", "Select at least one drive folder.")
            return
        self.scan_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.results = [None] * len(self.roster)
        self.status_lbl.setText("Scanning…")

        self.thread = QThread()
        self.worker = ScanWorker(self.roster, roots)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.row_done.connect(self._row_done)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._scan_finished)
        self.worker.failed.connect(self._scan_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.start()

    def _progress(self, done: int, total: int):
        self.status_lbl.setText(f"Scanning… {done}/{total}")

    def _row_done(self, i: int, res: dict):
        self.results[i] = res
        st = res["status"]
        video_cell = str(res["n_video"]) if res["n_video"] else "✗"
        if res["implanted"]:
            phase_cells = ["✓" if p in res["phases"] else "✗" for p in EXPECTED_PHASES]
        else:
            phase_cells = ["—", "—", "—"]
        vals = [res["rat"], res["date8"] or "?", str(res["day"]), str(res["session"]),
                res["expected"], video_cell, *phase_cells, res["found_in"], st]
        for c, val in enumerate(vals):
            self._set(i, c, val, status=st)

    def _scan_finished(self):
        ok = sum(1 for r in self.results if r and r["status"] == "OK")
        part = sum(1 for r in self.results if r and r["status"] == "PARTIAL")
        miss = sum(1 for r in self.results if r and r["status"] == "MISSING")
        self.status_lbl.setText(f"Done — {ok} complete · {part} partial · {miss} missing "
                                f"(of {len(self.results)}).")
        self.scan_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

    def _scan_failed(self, msg: str):
        QMessageBox.critical(self, "Scan failed", msg)
        self.status_lbl.setText("Scan failed.")
        self.scan_btn.setEnabled(True)

    # -- helpers -----------------------------------------------------------
    def _set(self, r: int, c: int, val: str, status: str = "?"):
        item = QTableWidgetItem(val)
        item.setBackground(QBrush(_STATUS_COLOR.get(status, Qt.GlobalColor.white)))
        if c in (5, 6, 7, 8, 10):
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if c == 10 and status in ("MISSING", "PARTIAL"):
            f = QFont()
            f.setBold(True)
            item.setFont(f)
        self.table.setItem(r, c, item)

    def _open_folder(self, r: int, _c: int):
        if r >= len(self.results) or not self.results[r]:
            return
        paths = self.results[r].get("paths") or []
        if not paths:
            return
        target = paths[0]
        try:
            if sys.platform == "darwin":
                subprocess.run(["open", target])
            elif sys.platform.startswith("win"):
                subprocess.run(["explorer", target])
            else:
                subprocess.run(["xdg-open", target])
        except Exception:
            pass

    def _export(self):
        if not any(self.results):
            return
        f, _ = QFileDialog.getSaveFileName(self, "Export coverage CSV",
                                           str(Path.home() / "drive_coverage.csv"), "CSV (*.csv)")
        if not f:
            return
        cols = ["rat", "date8", "day", "session", "expected", "implanted",
                "n_video", "pre", "task", "post", "n_rec", "n_merged", "n_logger",
                "found_in", "status"]
        try:
            with open(f, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                w.writeheader()
                for r in self.results:
                    if not r:
                        continue
                    w.writerow(dict(
                        rat=r["rat"], date8=r["date8"], day=r["day"], session=r["session"],
                        expected=r["expected"], implanted=int(r["implanted"]),
                        n_video=r["n_video"],
                        pre=int("pre" in r["phases"]) if r["implanted"] else "",
                        task=int("task" in r["phases"]) if r["implanted"] else "",
                        post=int("post" in r["phases"]) if r["implanted"] else "",
                        n_rec=r.get("n_rec", 0), n_merged=r.get("n_merged", 0),
                        n_logger=r.get("n_logger", 0),
                        found_in=r["found_in"], status=r["status"]))
            self.status_lbl.setText(f"Exported → {f}")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))


def main():
    app = QApplication(sys.argv)
    gui = ScanDriveGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
