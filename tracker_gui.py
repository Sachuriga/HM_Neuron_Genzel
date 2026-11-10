#!/usr/bin/env python3
"""genzeltracker — a small Qt front-end for the HM Tracker runner.

Pick a data-root folder, tick the pipeline steps (or a preset), press Run.
It just drives ``runner.py`` (the single source of truth) via a subprocess,
passing the step selection through the ``HM_STEPS`` env var and streaming the
runner's output into a log pane. Launch it from a terminal with ``genzeltracker``.

The window has two halves. Left: what to run (folders, preset, steps, options,
Run / Stop). Right: what is happening (the session jobs and the log).

A data root without ip* folders (e.g. the share's task folder, sessions at
<rat>/<YYYYMMDD>/) runs in the runner's tree mode. The window notices this by
itself and then shows the "Session tree" card: "Find sessions" lists the session
folders, and while it runs the "Session jobs" panel shows the machine's load,
the sessions done and each running job, read from the runner's output.
"Only failed of a run…" limits the next run to the sessions an earlier run did
not finish (read from its summary.csv; passed to the runner as HM_SESSIONS_FILE).

While a run is going, the Run button turns orange, shows a turning arrow and the
time since the run started ("Running…  12:34").

The window remembers its settings (folders, preset, limits, size) between
launches, and a folder can be dropped onto it to make it the data root.
"""

from __future__ import annotations

import csv
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

from PyQt6.QtCore import (
    Qt, QProcess, QProcessEnvironment, QTimer, QRectF, QPointF, QSize, QSettings, QUrl,
)
from PyQt6.QtGui import (
    QTextCursor, QIcon, QPixmap, QPainter, QPen, QColor, QPolygonF, QPalette,
    QFontDatabase, QKeySequence, QShortcut, QDesktopServices,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, QFrame, QPlainTextEdit,
    QFileDialog, QSpinBox, QSizePolicy, QProgressBar, QSplitter, QScrollArea,
    QToolButton, QMessageBox,
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

# ------------------------------------------------------------------ presets
# The preset list, in the order shown: (heading, {preset name: step keys}).
# The headings only organise the drop-down; they cannot be picked.
PRESET_GROUPS = [
    ("Maze", {
        "Tracker implanted (Video process + ephys)": "12345678d",
        "Tracker non-implanted (Video process, no ephys)": "23456d",
        "Retrack": "3456d",
        "LED Sync + stitch + track": "234",
        "After manual curation (Maze)": "wuvbm",
    }),
    ("Sleep", {
        # step 1 exports the analog (IMU) data step 8's motion reads
        "Prepare sleep scoring and spike sorting": "178",
        "Prepare sleep scoring only": "18",
        "Prepare spike sorting only": "17",
        "After manual curation (Sleep)": "wu",
    }),
    ("Other", {
        "NWB packaging (nwb + units + visualise)": "wuv",
        "Analysis (decode / UMAP)": "nbm",
        "Spike sorting (+ Phy auto-curation)": "7",
        "Auto-curate existing Phy exports": "a",
        "Full pipeline": "123456789wuvnbmt",
    }),
    # a root without ip* folders, e.g. the share's task folder: one job per session
    ("Session tree (e.g. the share)", {
        "Fix NWB LFP — check only (dry run)": "l",
        "Fix NWB LFP only (no figures)": "l",
        "Fix NWB LFP + remake theta pages": "lv",
        "Remake figures (visualise / decode / plots / UMAP)": "vb5m",
        "After manual curation (session tree)": "wuvb5m",
    }),
]
# every preset in one dict: name -> step keys
PRESETS = {name: steps for _, presets in PRESET_GROUPS for name, steps in presets.items()}
# what the drop-down shows when the ticked steps are the user's own choice
CUSTOM_PRESET = "Custom (tick the steps yourself)"
# presets that also set step l's dry-run box
PRESET_DRY_RUN = {"Fix NWB LFP — check only (dry run)": True,
                  "Fix NWB LFP only (no figures)": False,
                  "Fix NWB LFP + remake theta pages": False}

# -------------------------------------------------------------------- steps
# How the window arranges the runner's steps: (group title, column, {key: short name}).
# This is only for display. The runner's MENU still decides which steps exist:
# a key that is not in MENU is skipped, and a MENU step that is missing here
# shows up in an "Other" group — so a new runner step can never be forgotten.
# The runner's full description of a step is its tooltip.
STEP_GROUPS = [
    ("Export & video", 0, {
        "1": "Trodes export",
        "2": "LED sync",
        "3": "Stitching",
        "4": "Tracker (+ session NWB)",
        "5": "Plotting",
        "d": "DeepLabCut (eye)",
        "6": "Compression",
    }),
    ("Housekeeping", 0, {
        "9": "Cleaning",
        "t": "Drive scan",
    }),
    ("Ephys", 1, {
        "7": "Spike sorting (+ auto-curation)",
        "a": "Auto-curate Phy exports",
        "8": "LFP + motion + EMG (sleep)",
        "l": "Rewrite NWB LFP",
    }),
    ("NWB & analysis", 1, {
        "u": "Curated units → NWB",
        "v": "Visualise units",
        "b": "Bayesian decoder + videos",
        "m": "Population UMAP",
        "n": "Node analysis",
    }),
]


def grouped_steps():
    """The runner's steps, arranged for the window.

    Returns [(group title, column, [(key, short name, full description), ...]), ...]."""
    full_descriptions = dict(MENU)
    placed = set()
    groups = []
    for title, column, short_names in STEP_GROUPS:
        rows = [(key, short, full_descriptions[key])
                for key, short in short_names.items() if key in full_descriptions]
        placed.update(short_names)
        if rows:
            groups.append((title, column, rows))
    # steps the runner has but STEP_GROUPS does not know yet: the short name is
    # the description up to its first bracket
    leftover = [(key, description.split(" (")[0], description)
                for key, description in MENU if key not in placed]
    if leftover:
        groups.append(("Other", 1, leftover))
    return groups


# The runner's tree-mode lines the "Session jobs" panel follows
_TREE_TOTAL = re.compile(r"^\[TREE\] (\d+) session\(s\) x steps \[(\w*)\]")
_TREE_EVENT = re.compile(r"^\[TREE\] (start|done|FAILED)\s+(\S+)")
_MON = re.compile(r"^\[MON [\d:]+\] (.*?) \| \d+ running, \d+ queued, \d+ done")
_JOB = re.compile(r"^    (\S+)\s+step (\S+)\s+([\d.]+) min(?:\s+([\d.]+) GB)?")

DEFAULT_CONFIG = str(Path.home() / "Desktop" / "hm_tracker_paths.txt")
# where the runner's tree mode keeps its per-run logs + summary.csv (POST_LOG_DIR)
POST_LOG_DIR = Path(tempfile.gettempdir()) / "hm_post"

MAX_RECENT_ROOTS = 8               # data roots offered in the Data root drop-down
ROOT_CHECK_DELAY_MS = 400          # look at a typed data root this long after the last key
LOG_MAX_LINES = 50000              # older log lines are dropped, so a long run stays light

# ------------------------------------------------------------------- colours
GREEN = "#1f6f43"                  # titles
GREY = "#6b7785"                   # hints, group titles
RED = "#b3261e"                    # problems, the Stop button
HINT_COLORS = {"info": GREY, "ok": GREEN, "error": RED}

# The Run button's two looks. The colour is given for the :disabled state too:
# the button is disabled while a run is going, and without it Qt would keep
# drawing the same green, so a running GUI looked exactly like an idle one.
RUN_BUTTON_STYLE = ("QPushButton, QPushButton:disabled {{ background:{color}; color:white; "
                    "font-weight:700; padding:9px 20px; border:none; border-radius:6px; }}")
IDLE_GREEN = "#2e8b2e"
RUNNING_ORANGE = "#d9822b"
SPINNER_STEP_DEGREES = 30          # 12 frames per turn ...
SPINNER_INTERVAL_MS = 80           # ... x 80 ms = about one turn per second

# The window's own style rules. Colours of ordinary widgets (buttons, text
# fields) come from the palette in apply_look(); this only adds the "cards".
WINDOW_STYLE = f"""
QFrame#card {{ background:white; border:1px solid #dde2e7; border-radius:8px; }}
QLabel#appTitle {{ font-size:18px; font-weight:700; color:{GREEN}; }}
QLabel#cardTitle {{ font-weight:700; color:{GREEN}; }}
QLabel#groupTitle {{ font-size:11px; font-weight:600; color:{GREY}; }}
QLabel#hint {{ color:{GREY}; }}
QToolButton#optionsToggle {{ border:none; font-weight:700; color:{GREEN}; }}
QPushButton#stop:enabled {{ color:{RED}; font-weight:600; }}
QPlainTextEdit#log {{ background:#101418; color:#d6dbe0; border:none; border-radius:6px;
                      padding:6px; }}
QScrollArea {{ background:transparent; border:none; }}
QWidget#leftPanel {{ background:transparent; }}
"""


def apply_look(app):
    """Give the whole application one clean, light look.

    The "Fusion" style draws every widget itself, so the window looks the same
    on the Mac and on the lab's Windows PC. The colours are set here by hand
    (instead of taken from the system), so a computer in dark mode cannot end
    up with white text on our white cards."""
    app.setStyle("Fusion")
    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#f3f5f7",            # the window background
        QPalette.ColorRole.WindowText: "#1f2933",
        QPalette.ColorRole.Base: "#ffffff",              # text fields, lists
        QPalette.ColorRole.AlternateBase: "#f3f5f7",
        QPalette.ColorRole.Text: "#1f2933",
        QPalette.ColorRole.PlaceholderText: "#9aa5b1",
        QPalette.ColorRole.Button: "#fafbfc",
        QPalette.ColorRole.ButtonText: "#1f2933",
        QPalette.ColorRole.ToolTipBase: "#ffffff",
        QPalette.ColorRole.ToolTipText: "#1f2933",
        QPalette.ColorRole.Highlight: "#2e8b57",         # ticks, selections, progress bar
        QPalette.ColorRole.HighlightedText: "#ffffff",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    # greyed-out text for widgets that are switched off
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#a8b0b9"))
    app.setPalette(palette)


def format_elapsed(seconds):
    """Seconds as "m:ss", or "h:mm:ss" once a run passes an hour."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def spinner_icon(angle, size=18, color="white"):
    """One frame of the Run button's spinner: a circular "refresh" arrow, turned
    clockwise by `angle` degrees.

    It is drawn with QPainter instead of using a text symbol, so it looks the
    same on the Mac and on the lab's Windows PC (no font has to have the glyph)."""
    scale = 2                                    # draw at 2x so it stays sharp on retina screens
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.setDevicePixelRatio(scale)            # ... but keep using `size` as the coordinates
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.translate(size / 2, size / 2)        # (0, 0) is now the icon's centre ...
    painter.rotate(angle)                        # ... so the rotation turns it in place

    # an open circle: 300 of the 360 degrees, drawn clockwise from 3 o'clock.
    # Qt counts arc angles in 1/16 degree, counter-clockwise positive.
    radius = size / 2 - 3
    pen = QPen(QColor(color), 2.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawArc(QRectF(-radius, -radius, 2 * radius, 2 * radius), 0, -300 * 16)

    # the arrow head, where the arc ends (60 degrees = about 1 o'clock); it points
    # the way the arc was drawn (clockwise). Screen y points down, hence the minus signs.
    end = math.radians(60)
    end_x, end_y = radius * math.cos(end), -radius * math.sin(end)
    along_x, along_y = math.sin(end), math.cos(end)          # clockwise direction
    out_x, out_y = math.cos(end), -math.sin(end)             # away from the centre
    head = 3.5
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(color))
    painter.drawPolygon(QPolygonF([
        QPointF(end_x + along_x * head, end_y + along_y * head),   # tip
        QPointF(end_x + out_x * head, end_y + out_y * head),       # outer corner
        QPointF(end_x - out_x * head, end_y - out_y * head),       # inner corner
    ]))
    painter.end()

    icon = QIcon(pixmap)
    # the button is disabled while it spins: give the same white picture for the
    # disabled mode too, or Qt would draw a greyed-out copy of it
    icon.addPixmap(pixmap, QIcon.Mode.Disabled)
    return icon


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


def describe_data_root(text):
    """What kind of data root is `text`? Returns (kind, a sentence for the user).

    kind is "empty" (nothing typed), "missing" (no such folder), "classic"
    (it has ip* folders: each ipN is processed into its opN) or "tree" (no ip*
    folders: the runner's tree mode, one job per session folder)."""
    if not text:
        return "empty", "Pick the folder that holds the recordings (or drop it onto this window)."
    root = Path(text)
    if not root.is_dir():
        return "missing", "This folder does not exist."
    ip_folders = [d for d in root.glob("ip*") if d.is_dir()]
    if ip_folders:
        return "classic", f"{len(ip_folders)} ip folder(s) found — each ipN is processed into its opN."
    return "tree", "No ip* folders — runs as a session tree (one job per session folder)."


class TrackerGUI(QMainWindow):
    def __init__(self, settings=None):
        """`settings` is the QSettings the window remembers itself in (main()
        passes one). Without it nothing is read or saved — the tests use that,
        so they never touch the user's real settings."""
        super().__init__()
        self.settings = settings
        self.proc: QProcess | None = None
        self.root_kind = "empty"                 # see describe_data_root()
        self.only_sessions: list[str] = []       # tree mode: run only these (empty = all)
        self.run_started = 0.0                   # time.monotonic() when Run was pressed

        self.setWindowTitle("Genzel Tracker")
        self.resize(1240, 820)
        self.setStyleSheet(WINDOW_STYLE)
        self.setAcceptDrops(True)                # a dropped folder becomes the data root

        # left: what to run. right: what is happening. The bar between the two
        # can be dragged.
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.addWidget(self._build_left_side())
        self.splitter.addWidget(self._build_right_side())
        self.splitter.setStretchFactor(0, 0)     # extra window width goes to the log
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([520, 720])
        self.splitter.setChildrenCollapsible(False)
        self.setCentralWidget(self.splitter)

        # the Run button's spinner: a timer swaps in the next frame (and updates
        # the elapsed time) every SPINNER_INTERVAL_MS while the runner is running
        self.spinner_frames = [spinner_icon(angle) for angle in range(0, 360, SPINNER_STEP_DEGREES)]
        self.spinner_index = 0
        self.spinner_timer = QTimer(self)
        self.spinner_timer.setInterval(SPINNER_INTERVAL_MS)
        self.spinner_timer.timeout.connect(self._turn_spinner)

        # Looking at the data root means reading a folder, which can be slow on
        # the share. So it is not done on every key press, but once the typing
        # has paused for ROOT_CHECK_DELAY_MS.
        self.root_timer = QTimer(self)
        self.root_timer.setSingleShot(True)
        self.root_timer.setInterval(ROOT_CHECK_DELAY_MS)
        self.root_timer.timeout.connect(self._check_root)
        self.root_edit.textChanged.connect(self.root_timer.start)

        # Ctrl+Return (Cmd+Return on the Mac) = press Run
        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._run)

        self._reset_jobs()
        self._load_settings()
        self._check_root()
        self.statusBar().showMessage("Ready.")

    # ================================================================ building
    def _build_left_side(self):
        """Folders, steps and options (scrollable), with Run / Stop fixed below."""
        side = QWidget()
        side.setMinimumWidth(470)
        rows = QVBoxLayout(side)
        rows.setContentsMargins(14, 12, 6, 8)
        rows.setSpacing(10)

        title = QLabel("Genzel Tracker")
        title.setObjectName("appTitle")
        rows.addWidget(title)

        # the cards scroll if the window is too short for them; Run stays visible
        cards = QWidget()
        cards.setObjectName("leftPanel")
        card_rows = QVBoxLayout(cards)
        card_rows.setContentsMargins(0, 0, 4, 0)
        card_rows.setSpacing(10)
        card_rows.addWidget(self._build_data_card())
        card_rows.addWidget(self._build_steps_card())
        card_rows.addWidget(self._build_tree_card())
        card_rows.addWidget(self._build_options_card())
        card_rows.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(cards)
        rows.addWidget(scroll, 1)

        rows.addLayout(self._build_run_row())
        # one line under the buttons: what Run would do, or why it cannot yet
        self.run_hint = QLabel("")
        self.run_hint.setWordWrap(True)
        rows.addWidget(self.run_hint)
        return side

    def _card(self, title):
        """A white rounded box with a title. Returns (the box, the layout to fill)."""
        card = QFrame()
        card.setObjectName("card")
        rows = QVBoxLayout(card)
        rows.setContentsMargins(12, 10, 12, 12)
        rows.setSpacing(8)
        if title:
            label = QLabel(title)
            label.setObjectName("cardTitle")
            rows.addWidget(label)
        return card, rows

    def _hint_label(self, text=""):
        """A grey, word-wrapped line of explanation."""
        label = QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(True)
        return label

    def _build_data_card(self):
        card, rows = self._card("Data")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        rows.addLayout(grid)

        # The data root is a drop-down you can also type in: its list holds the
        # roots of earlier runs. self.root_edit is the text field inside it.
        self.root_box = QComboBox()
        self.root_box.setEditable(True)
        self.root_box.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)   # we fill the list ourselves
        # do not let a long path in the list make the whole window wider
        self.root_box.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.root_box.setMinimumContentsLength(12)
        self.root_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.root_edit = self.root_box.lineEdit()
        self.root_edit.setPlaceholderText("folder with ip*/op* sub-folders, or a tree of session folders")
        self.root_edit.setClearButtonEnabled(True)
        root_button = QPushButton("Browse…")
        root_button.clicked.connect(self._pick_root)
        grid.addWidget(QLabel("Data root"), 0, 0)
        grid.addWidget(self.root_box, 0, 1)
        grid.addWidget(root_button, 0, 2)

        # what the window found in that folder
        self.root_hint = self._hint_label()
        grid.addWidget(self.root_hint, 1, 1, 1, 2)

        self.cfg_edit = QLineEdit(DEFAULT_CONFIG)
        self.cfg_edit.setToolTip("The hm_tracker_paths.txt the runner reads its settings from.")
        cfg_button = QPushButton("Browse…")
        cfg_button.clicked.connect(self._pick_cfg)
        grid.addWidget(QLabel("Config"), 2, 0)
        grid.addWidget(self.cfg_edit, 2, 1)
        grid.addWidget(cfg_button, 2, 2)
        return card

    def _build_steps_card(self):
        card, rows = self._card("Steps")

        # --- preset: ticks a ready-made set of steps
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset"))
        self.preset = QComboBox()
        self._fill_preset_box()
        self.preset.currentTextChanged.connect(self._apply_preset)
        preset_row.addWidget(self.preset, 1)
        clear_button = QPushButton("Untick all")
        clear_button.clicked.connect(self._untick_all)
        preset_row.addWidget(clear_button)
        rows.addLayout(preset_row)

        # --- one checkbox per runner step, in two columns of groups
        columns = [QVBoxLayout(), QVBoxLayout()]
        for column in columns:
            column.setSpacing(3)
        self.checks: dict[str, QCheckBox] = {}
        sequential = set(SEQUENTIAL_STEPS)
        for group_title, column_index, steps in grouped_steps():
            column = columns[column_index]
            if column.count():
                column.addSpacing(8)             # a gap above every group but the first
            group_label = QLabel(group_title.upper())
            group_label.setObjectName("groupTitle")
            column.addWidget(group_label)
            for key, short_name, description in steps:
                box = QCheckBox(f"[{key}]  {short_name}")
                how = ("Runs once for the whole data root, after the per-folder steps."
                       if key in sequential else "Runs per folder, several folders at a time.")
                box.setToolTip(f"{description}\n\n{how}")
                box.toggled.connect(self._on_check_changed)
                self.checks[key] = box
                column.addWidget(box)
        step_columns = QHBoxLayout()
        step_columns.setSpacing(18)
        for column in columns:
            column.addStretch(1)
            step_columns.addLayout(column, 1)
        rows.addLayout(step_columns)

        # --- step l's own switch; only shown while step l is ticked
        self.lfp_dry_run = QCheckBox("Step l: check only (dry run — changes nothing)")
        self.lfp_dry_run.setToolTip("Rewrite and verify each NWB's LFP on a local copy, "
                                    "without replacing anything in the session folders.")
        self.lfp_dry_run.toggled.connect(self._refresh_run_hint)
        rows.addWidget(self.lfp_dry_run)
        return card

    def _fill_preset_box(self):
        """Custom first, then every preset under its heading. The headings are
        greyed out and cannot be picked; a preset's tooltip names its steps."""
        self.preset.addItem(CUSTOM_PRESET)
        model = self.preset.model()
        for heading, presets in PRESET_GROUPS:
            self.preset.addItem(f"— {heading} —")
            model.item(self.preset.count() - 1).setEnabled(False)
            for name, steps in presets.items():
                self.preset.addItem(name)
                self.preset.setItemData(self.preset.count() - 1, f"steps: {steps}",
                                        Qt.ItemDataRole.ToolTipRole)

    def _build_tree_card(self):
        """Settings that only matter for a root without ip* folders. The card is
        hidden for every other root (see _check_root)."""
        self.tree_card, rows = self._card("Session tree")

        # which sessions: all found below the data root, or only the ones an
        # earlier run did not finish (read from that run's summary.csv)
        top_row = QHBoxLayout()
        self.only_label = QLabel("Sessions: all found below the data root")
        self.only_label.setWordWrap(True)
        top_row.addWidget(self.only_label, 1)
        top_row.addWidget(QLabel("At once"))
        self.post_jobs = QSpinBox()
        self.post_jobs.setRange(0, 32)
        self.post_jobs.setValue(0)
        self.post_jobs.setSpecialValueText("auto")           # shown for the value 0
        self.post_jobs.setMaximumWidth(72)
        self.post_jobs.setToolTip("Sessions processed at once (auto: half the CPU cores, "
                                  "one per 12 GB RAM, at most 6). New jobs also wait for the "
                                  "CPU/GPU/memory limits under Options.")
        top_row.addWidget(self.post_jobs)
        rows.addLayout(top_row)

        buttons = QHBoxLayout()
        find_button = QPushButton("Find sessions")
        find_button.setToolTip("List the session folders below the data root.")
        find_button.clicked.connect(self._find_sessions)
        buttons.addWidget(find_button)
        failed_button = QPushButton("Only failed of a run…")
        failed_button.setToolTip("Pick a run's summary.csv (in the hm_post log folder): the next "
                                 "Run processes only its sessions that are not 'ok'.")
        failed_button.clicked.connect(self._pick_failed_run)
        buttons.addWidget(failed_button)
        self.all_sessions_btn = QPushButton("All sessions")
        self.all_sessions_btn.setToolTip("Go back to every session found below the data root.")
        self.all_sessions_btn.clicked.connect(self._clear_only_sessions)
        self.all_sessions_btn.setEnabled(False)
        buttons.addWidget(self.all_sessions_btn)
        buttons.addStretch(1)
        rows.addLayout(buttons)
        return self.tree_card

    def _build_options_card(self):
        """Rarely changed settings, folded away until "Options" is clicked."""
        card, rows = self._card("")
        self.options_toggle = QToolButton()
        self.options_toggle.setObjectName("optionsToggle")
        self.options_toggle.setText("Options")
        self.options_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.options_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.options_toggle.setCheckable(True)
        self.options_toggle.toggled.connect(self._show_options)
        rows.addWidget(self.options_toggle)

        self.options_panel = QWidget()
        panel_rows = QVBoxLayout(self.options_panel)
        panel_rows.setContentsMargins(0, 0, 0, 0)
        panel_rows.setSpacing(8)

        limits = QHBoxLayout()
        limits.addWidget(QLabel("Start new work only below"))
        self.cpu = self._percent_box(90)
        self.gpu = self._percent_box(90)
        self.mem = self._percent_box(65)
        for name, box in (("CPU", self.cpu), ("GPU", self.gpu), ("memory", self.mem)):
            limits.addSpacing(6)
            limits.addWidget(QLabel(name))
            limits.addWidget(box)
        limits.addStretch(1)
        panel_rows.addLayout(limits)

        self.own_windows = QCheckBox("Each worker in its own terminal window")
        self.own_windows.setChecked(True)
        panel_rows.addWidget(self.own_windows)

        self.options_panel.setVisible(False)
        rows.addWidget(self.options_panel)
        return card

    def _percent_box(self, value):
        box = QSpinBox()
        box.setRange(1, 100)
        box.setValue(value)
        box.setSuffix(" %")
        box.setMaximumWidth(76)
        return box

    def _build_run_row(self):
        row = QHBoxLayout()
        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setStyleSheet(RUN_BUTTON_STYLE.format(color=IDLE_GREEN))
        self.run_btn.setIconSize(QSize(18, 18))
        self.run_btn.setToolTip("Run the ticked steps (Ctrl+Return)")
        # wide enough for "Running…  1:23:45", so the Stop button next to it does
        # not jump around while the clock counts up
        self.run_btn.setMinimumWidth(200)
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("■  Stop")
        self.stop_btn.setObjectName("stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        row.addWidget(self.run_btn, 1)
        row.addWidget(self.stop_btn)
        return row

    def _build_right_side(self):
        """The session jobs (tree mode only) above the log."""
        side = QWidget()
        rows = QVBoxLayout(side)
        rows.setContentsMargins(6, 12, 14, 8)
        rows.setSpacing(10)
        fixed_font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)

        # --- session jobs (tree mode): load, progress, running jobs -------------
        self.jobs_box, job_rows = self._card("")
        header = QHBoxLayout()
        self.jobs_title = QLabel("Session jobs")
        self.jobs_title.setObjectName("cardTitle")
        header.addWidget(self.jobs_title)
        header.addStretch(1)
        logs_button = QPushButton("Open logs folder")
        logs_button.setToolTip("Every session's full log, one folder per run.")
        logs_button.clicked.connect(self._open_logs_folder)
        header.addWidget(logs_button)
        job_rows.addLayout(header)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m sessions done")
        job_rows.addWidget(self.progress)
        self.machine = QLabel("")
        self.jobs = QLabel("")
        for label in (self.machine, self.jobs):
            label.setFont(fixed_font)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            job_rows.addWidget(label)
        self.jobs_box.setVisible(False)          # shown by the first tree-mode line of a run
        rows.addWidget(self.jobs_box)

        # --- log ---------------------------------------------------------------
        header = QHBoxLayout()
        log_title = QLabel("Log")
        log_title.setObjectName("cardTitle")
        header.addWidget(log_title)
        header.addStretch(1)
        save_button = QPushButton("Save log…")
        save_button.clicked.connect(self._save_log)
        header.addWidget(save_button)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(lambda: self.log.clear())
        header.addWidget(clear_button)
        rows.addLayout(header)

        self.log = QPlainTextEdit(readOnly=True)
        self.log.setObjectName("log")
        self.log.setFont(fixed_font)
        self.log.setMaximumBlockCount(LOG_MAX_LINES)
        self.log.setPlaceholderText("The runner's output appears here.")
        rows.addWidget(self.log, 1)
        return side

    # ============================================================== data root
    def _pick_root(self):
        start = self.root_edit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Select data root", start)
        if folder:
            self.root_edit.setText(folder)
            self._check_root()                   # now, not after the typing delay

    def _pick_cfg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select hm_tracker_paths.txt",
                                              str(Path(self.cfg_edit.text()).parent))
        if path:
            self.cfg_edit.setText(path)

    def _check_root(self):
        """Look at the data root: say what it is, and show the Session tree card
        only for a root that runs in tree mode."""
        self.root_timer.stop()
        self.root_kind, message = describe_data_root(self.root_edit.text().strip())
        self.root_hint.setText(message)
        color = RED if self.root_kind == "missing" else GREY
        self.root_hint.setStyleSheet(f"color:{color};")
        self.tree_card.setVisible(self.root_kind == "tree")
        self._refresh_run_hint()

    def _remember_root(self, root):
        """Put `root` at the top of the Data root drop-down (newest first, no doubles)."""
        roots = [root] + [r for r in self._recent_roots() if r != root]
        self.root_box.blockSignals(True)         # refilling the list must not look like typing
        self.root_box.clear()
        self.root_box.addItems(roots[:MAX_RECENT_ROOTS])
        self.root_box.setCurrentIndex(0)
        self.root_box.blockSignals(False)

    def _recent_roots(self):
        return [self.root_box.itemText(i) for i in range(self.root_box.count())]

    # a folder dragged from the Finder / Explorer onto the window = the data root
    def dragEnterEvent(self, event):
        if self._dropped_folder(event):
            event.acceptProposedAction()

    def dropEvent(self, event):
        folder = self._dropped_folder(event)
        if folder:
            self.root_edit.setText(folder)
            self._check_root()

    def _dropped_folder(self, event):
        """The first folder among the dragged things, or "" if there is none."""
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).is_dir():
                return url.toLocalFile()
        return ""

    # ================================================================== steps
    def _apply_preset(self, name):
        """Tick exactly the steps of the preset `name`."""
        if name not in PRESETS:                  # "Custom": the ticks stay as they are
            return
        self._set_ticks(PRESETS[name])
        if name in PRESET_DRY_RUN:
            self.lfp_dry_run.setChecked(PRESET_DRY_RUN[name])
        self._refresh_run_hint()

    def _set_ticks(self, keys):
        """Tick the steps in `keys`, untick the rest — without each checkbox
        reporting it as a change made by the user."""
        for key, box in self.checks.items():
            box.blockSignals(True)
            box.setChecked(key in keys)
            box.blockSignals(False)

    def _untick_all(self):
        self._set_ticks("")
        self._show_custom_preset()
        self._refresh_run_hint()

    def _on_check_changed(self, *_):
        """The user ticked or unticked a step: the selection is now their own."""
        self._show_custom_preset()
        self._refresh_run_hint()

    def _show_custom_preset(self):
        self.preset.blockSignals(True)           # only the name changes, the ticks stay
        self.preset.setCurrentText(CUSTOM_PRESET)
        self.preset.blockSignals(False)

    def _selection(self) -> str:
        """The ticked step keys as one string, in the runner's menu order."""
        return "".join(k for k, _ in MENU if self.checks[k].isChecked())

    def _show_options(self, opened):
        self.options_panel.setVisible(opened)
        arrow = Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow
        self.options_toggle.setArrowType(arrow)

    # ================================================================ run hint
    def _problem(self):
        """Why Run cannot start yet ("" when it can)."""
        if self.root_kind in ("empty", "missing"):
            return "Pick a valid data-root folder first."
        if not self._selection():
            return "Tick at least one step, or pick a preset."
        return ""

    def _set_hint(self, text, kind="info"):
        """The line under the Run button; kind is "info", "ok" or "error"."""
        self.run_hint.setText(text)
        self.run_hint.setStyleSheet(f"color:{HINT_COLORS[kind]};")

    def _refresh_run_hint(self, *_):
        """Show what Run would do — and step l's dry-run box only with step l."""
        steps = self._selection()
        self.lfp_dry_run.setVisible("l" in steps)
        if self.proc is not None:                # a run is going: leave its message alone
            return
        problem = self._problem()
        if problem:
            self._set_hint(problem)
            return
        dry = "  (step l: dry run)" if "l" in steps and self.lfp_dry_run.isChecked() else ""
        self._set_hint(f"Ready to run steps:  {' '.join(steps)}{dry}")

    # ============================================================ session tree
    def _find_sessions(self):
        """List the session folders the runner's tree mode would process."""
        root = Path(self.root_edit.text().strip())
        self._check_root()
        if self.root_kind != "tree" or find_session_folders is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            found = find_session_folders(root)
        finally:
            QApplication.restoreOverrideCursor()
        self._append(f"[genzeltracker] {len(found)} session folder(s) below {root}:\n"
                     + "".join(f"    {p.relative_to(root)}\n" for p in found))
        self.root_hint.setText(f"Session tree — {len(found)} session folder(s) found "
                               f"(listed in the log).")
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
            self.statusBar().showMessage("Every session of that run is ok — nothing to re-run.")
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

    def _open_logs_folder(self):
        if POST_LOG_DIR.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(POST_LOG_DIR)))
        else:
            self.statusBar().showMessage(f"No logs yet: {POST_LOG_DIR} does not exist.")

    # ================================================================== runner
    def _run(self):
        if self.proc is not None:
            return
        self._check_root()                       # the folder may have changed since it was typed
        problem = self._problem()
        if problem:
            self._set_hint(problem, "error")
            self.statusBar().showMessage(problem)
            return
        root = self.root_edit.text().strip()
        steps = self._selection()
        self._remember_root(root)

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
        self._show_running(steps)
        self.stop_btn.setEnabled(True)
        self._set_hint(f"Running steps:  {' '.join(steps)}{dry}{only}")
        self.statusBar().showMessage(f"Running steps: {steps}")

    # ---------------------------------------------------------- run button looks
    def _show_running(self, steps):
        """Run button -> orange "Running…  m:ss" with a turning arrow."""
        self.run_started = time.monotonic()
        self.spinner_index = 0
        self.run_btn.setEnabled(False)           # one run at a time
        self.run_btn.setStyleSheet(RUN_BUTTON_STYLE.format(color=RUNNING_ORANGE))
        self._turn_spinner()                     # show the first frame now, not 80 ms later
        self.spinner_timer.start()
        # the title shows in the taskbar / Dock too, so a run is visible there as well
        self.setWindowTitle(f"Genzel Tracker — running steps {steps}")

    def _turn_spinner(self):
        """Next spinner frame + the time since Run was pressed."""
        self.run_btn.setIcon(self.spinner_frames[self.spinner_index])
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        elapsed = format_elapsed(time.monotonic() - self.run_started)
        self.run_btn.setText(f"Running…  {elapsed}")

    def _show_idle(self):
        """Run button back to the green "▶  Run"."""
        self.spinner_timer.stop()
        self.run_btn.setIcon(QIcon())            # an empty icon removes the spinner
        self.run_btn.setText("▶  Run")
        self.run_btn.setStyleSheet(RUN_BUTTON_STYLE.format(color=IDLE_GREEN))
        self.run_btn.setEnabled(True)
        self.setWindowTitle("Genzel Tracker")

    # -------------------------------------------------------------------- stop
    def _confirm(self, question):
        """Ask a yes/no question; True for yes. "No" is the default button, so a
        stray Enter cannot throw away hours of work."""
        answer = QMessageBox.question(
            self, "Genzel Tracker", question,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def _stop(self):
        """The Stop button: ask first, a run can be hours of work."""
        if self.proc is None:
            return
        elapsed = format_elapsed(time.monotonic() - self.run_started)
        if self._confirm(f"Stop the run? It has been going for {elapsed}."):
            self._kill_run()

    def _kill_run(self):
        if self.proc is None:                    # it ended while the question was open
            return
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
                self.jobs_title.setText(f"Session jobs — steps [{m.group(2)}]")
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

    # --------------------------------------------------------------------- log
    def _append(self, text):
        """Add `text` to the end of the log. The view follows the new text only
        if it was at the bottom already — scrolled up to read something, it
        stays where it is (and a selection is kept)."""
        bar = self.log.verticalScrollBar()
        was_at_bottom = bar.value() >= bar.maximum() - 4
        cursor = QTextCursor(self.log.document())    # our own cursor, not the user's
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        if was_at_bottom:
            bar.setValue(bar.maximum())

    def _save_log(self):
        default = str(Path.home() / f"genzeltracker_{time.strftime('%Y%m%d_%H%M%S')}.log")
        path, _ = QFileDialog.getSaveFileName(self, "Save the log", default, "Log files (*.log *.txt)")
        if path:
            Path(path).write_text(self.log.toPlainText(), encoding="utf-8")
            self.statusBar().showMessage(f"Log saved to {path}")

    def _on_finished(self, code, _status):
        self._append(f"\n[genzeltracker] runner exited with code {code}.\n")
        elapsed = format_elapsed(time.monotonic() - self.run_started)
        self.statusBar().showMessage(f"Finished (exit {code}) after {elapsed}.")
        self._show_idle()
        self.stop_btn.setEnabled(False)
        self.proc = None
        if code == 0:
            self._set_hint(f"✓ Finished after {elapsed}.", "ok")
        else:
            self._set_hint(f"✗ Stopped or failed after {elapsed} (exit code {code}) — see the log.",
                           "error")
        QApplication.alert(self)                 # flash the taskbar / bounce the Dock icon

    # ================================================================ settings
    def _load_settings(self):
        """Put the window back the way it was closed last time."""
        s = self.settings
        if s is None:
            return
        # type=... matters on Windows, where everything is stored as text
        recent = s.value("recent_roots", [], type=list)
        self.root_box.addItems([str(r) for r in recent][:MAX_RECENT_ROOTS])
        self.root_edit.setText(s.value("root", "", type=str))
        self.cfg_edit.setText(s.value("config", DEFAULT_CONFIG, type=str))
        self.cpu.setValue(s.value("max_cpu", 90, type=int))
        self.gpu.setValue(s.value("max_gpu", 90, type=int))
        self.mem.setValue(s.value("max_mem", 65, type=int))
        self.own_windows.setChecked(s.value("own_windows", True, type=bool))
        self.post_jobs.setValue(s.value("post_jobs", 0, type=int))
        self.options_toggle.setChecked(s.value("options_open", False, type=bool))

        # the steps: a preset by name, or the user's own ticks
        preset = s.value("preset", CUSTOM_PRESET, type=str)
        if preset in PRESETS:
            self.preset.setCurrentText(preset)
        else:
            self._set_ticks(s.value("steps", "", type=str))

        geometry = s.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter = s.value("splitter")
        if splitter is not None:
            self.splitter.restoreState(splitter)

    def _save_settings(self):
        s = self.settings
        if s is None:
            return
        s.setValue("recent_roots", self._recent_roots())
        s.setValue("root", self.root_edit.text().strip())
        s.setValue("config", self.cfg_edit.text().strip())
        s.setValue("max_cpu", self.cpu.value())
        s.setValue("max_gpu", self.gpu.value())
        s.setValue("max_mem", self.mem.value())
        s.setValue("own_windows", self.own_windows.isChecked())
        s.setValue("post_jobs", self.post_jobs.value())
        s.setValue("options_open", self.options_toggle.isChecked())
        s.setValue("preset", self.preset.currentText())
        s.setValue("steps", self._selection())
        s.setValue("geometry", self.saveGeometry())
        s.setValue("splitter", self.splitter.saveState())

    def closeEvent(self, event):
        """Closing the window ends a run that is still going — so ask first."""
        if self.proc is not None:
            if not self._confirm("A run is still going. Stop it and close the window?"):
                event.ignore()
                return
            self._kill_run()
        self._save_settings()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Genzel Tracker")
    apply_look(app)
    # where the window remembers itself: the registry on Windows, a .plist on the Mac
    settings = QSettings("GenzelLab", "GenzelTracker")
    win = TrackerGUI(settings)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
