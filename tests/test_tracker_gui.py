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


def test_fix_lfp_only_preset_runs_step_l_for_real(gui):
    gui.preset.setCurrentText("Fix NWB LFP only (no figures)")
    assert gui._selection() == "l" and not gui.lfp_dry_run.isChecked()


def test_only_the_unfinished_sessions_of_a_run(gui, tmp_path):
    summary = tmp_path / "20261022_120002" / "summary.csv"
    summary.parent.mkdir()
    summary.write_text(
        "session,status,minutes,folder,log\n"
        "Rat5_20260622,failed,7.8,/share/rat5/20260622,a.log\n"
        "Rat5_20260623,ok,23.6,/share/rat5/20260623,b.log\n"
        "Rat6_20260626,interrupted,3.0,/share/rat6/20260626,c.log\n"
        "Rat6_20260629,not started,0.0,/share/rat6/20260629,\n", encoding="utf-8")
    assert G.failed_sessions(summary) == ["/share/rat5/20260622", "/share/rat6/20260626",
                                          "/share/rat6/20260629"]

    gui._use_failed_run(summary)
    assert len(gui.only_sessions) == 3 and gui.all_sessions_btn.isEnabled()
    assert "only the 3 not ok in run 20261022_120002" in gui.only_label.text()

    gui._clear_only_sessions()
    assert gui.only_sessions == [] and not gui.all_sessions_btn.isEnabled()


def test_elapsed_time_on_the_run_button():
    assert G.format_elapsed(7) == "0:07"
    assert G.format_elapsed(754.9) == "12:34"
    assert G.format_elapsed(3 * 3600 + 5 * 60 + 9) == "3:05:09"


def test_run_button_shows_running_then_idle(gui):
    gui._show_running("lv")
    assert gui.run_btn.text().startswith("Running…") and not gui.run_btn.isEnabled()
    assert gui.spinner_timer.isActive() and not gui.run_btn.icon().isNull()
    assert G.RUNNING_ORANGE in gui.run_btn.styleSheet()
    assert "running steps lv" in gui.windowTitle()

    first_frame = gui.spinner_index
    gui._turn_spinner()                                   # the arrow moves on each tick
    assert gui.spinner_index == (first_frame + 1) % len(gui.spinner_frames)

    gui._show_idle()
    assert gui.run_btn.text() == "▶  Run" and gui.run_btn.isEnabled()
    assert not gui.spinner_timer.isActive() and gui.run_btn.icon().isNull()
    assert G.IDLE_GREEN in gui.run_btn.styleSheet()
    assert gui.windowTitle() == "Genzel Tracker"
