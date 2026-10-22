"""tracker_gui: the session-tree presets, and the Session jobs panel following the
runner's tree-mode output."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PyQt6.QtWidgets import QApplication                 # noqa: E402

import tracker_gui as G                                   # noqa: E402

RUN = """[TREE] 3 session(s) x steps [lv], up to 2 at a time (POST_JOBS), 20 s apart
[TREE] start  Rat5_20260630  (1 running, 2 queued)

[MON 11:00:29] CPU 40% | MEM 56% (17.9/32 GB) | GPU 0% | net in 30.1 out 0.2 MB/s | 1 running, 2 queued, 0 done (0 failed)
    Rat5_20260630                step l     0.5 min    1.1 GB
    next start waits: MEM 70% > 65%
[TREE] start  Rat5_20260701  (2 running, 1 queued)
[TREE] FAILED Rat5_20260630 (3.1 min) — see /tmp/Rat5_20260630.log
"""


@pytest.fixture(scope="module")
def gui():
    app = QApplication.instance() or QApplication([])
    window = G.TrackerGUI()
    yield window                                           # the app must outlive its widgets
    window.close()
    del app


def test_session_tree_presets_set_steps_and_dry_run(gui):
    gui.preset.setCurrentText("Fix NWB LFP — check only (dry run)")
    assert gui._selection() == "l" and gui.lfp_dry_run.isChecked()
    gui.preset.setCurrentText("Fix NWB LFP + remake theta pages")
    assert gui._selection() == "lv" and not gui.lfp_dry_run.isChecked()


def test_jobs_panel_follows_the_runner(gui):
    gui._reset_jobs()
    for i in range(0, len(RUN), 50):                       # output arrives in pieces
        gui._follow(RUN[i:i + 50])
    assert (gui.progress.value(), gui.progress.maximum()) == (1, 3)
    assert "Rat5_20260630" in gui.progress.format()       # the failed session is named
    assert gui.machine.text().startswith("CPU 40% | MEM 56%")
    rows = gui.jobs.text().splitlines()
    assert rows[0].startswith("Rat5_20260701") and rows[-1].startswith("next start waits")
