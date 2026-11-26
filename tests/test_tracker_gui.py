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


def test_every_runner_step_has_a_checkbox(gui):
    menu_keys = [key for key, _ in G.MENU]
    assert sorted(gui.checks) == sorted(menu_keys)
    # each step is shown once, with the runner's full description as its tooltip
    shown = [key for _, _, steps in G.grouped_steps() for key, _, _ in steps]
    assert sorted(shown) == sorted(menu_keys)
    assert dict(G.MENU)["7"] in gui.checks["7"].toolTip()


def test_ticking_a_step_by_hand_shows_the_custom_preset(gui):
    gui.preset.setCurrentText("Retrack")
    assert gui._selection() == "3456d"
    gui.checks["2"].setChecked(True)                      # the user's own choice now
    assert gui.preset.currentText() == G.CUSTOM_PRESET
    assert gui._selection() == "23456d"                   # ... and the ticks are kept

    gui._untick_all()
    assert gui._selection() == ""


def test_dry_run_box_only_shows_with_step_l(gui):
    gui.preset.setCurrentText("Retrack")
    assert gui.lfp_dry_run.isHidden()
    gui.preset.setCurrentText("Fix NWB LFP only (no figures)")
    assert not gui.lfp_dry_run.isHidden()


def test_super_resolution_box_only_shows_with_step_d(gui):
    gui._untick_all()
    assert gui.dlc_super_resolution.isHidden()
    gui.checks["d"].setChecked(True)
    assert not gui.dlc_super_resolution.isHidden()
    assert gui.dlc_super_resolution.isChecked()             # on unless switched off


def test_data_root_is_recognised(gui, tmp_path):
    classic = tmp_path / "classic"
    (classic / "ip1").mkdir(parents=True)
    tree = tmp_path / "tree"
    (tree / "rat5" / "20260630").mkdir(parents=True)
    assert G.describe_data_root("")[0] == "empty"
    assert G.describe_data_root(str(tmp_path / "nowhere"))[0] == "missing"
    assert G.describe_data_root(str(classic))[0] == "classic"
    assert G.describe_data_root(str(tree))[0] == "tree"

    # the Session tree card is only there for a tree root, and Run says what is missing
    gui.preset.setCurrentText("Retrack")
    gui.root_edit.setText(str(classic))
    gui._check_root()
    assert gui.tree_card.isHidden() and gui._problem() == ""
    gui.root_edit.setText(str(tree))
    gui._check_root()
    assert not gui.tree_card.isHidden()
    gui.root_edit.setText(str(tmp_path / "nowhere"))
    gui._check_root()
    assert "data-root" in gui._problem()


def test_log_stays_put_when_scrolled_up(gui):
    gui.show()
    gui.log.clear()
    gui._append("line\n" * 500)
    bar = gui.log.verticalScrollBar()
    assert bar.value() == bar.maximum()                   # at the bottom: it follows
    bar.setValue(0)                                       # scrolled up to read something
    gui._append("more\n" * 50)
    assert bar.value() == 0


def test_settings_are_remembered(tmp_path):
    from PyQt6.QtCore import QSettings
    app = QApplication.instance() or QApplication([])
    store = str(tmp_path / "settings.ini")

    first = G.TrackerGUI(QSettings(store, QSettings.Format.IniFormat))
    first.root_edit.setText(str(tmp_path))
    first._remember_root(str(tmp_path))
    first.preset.setCurrentText("Retrack")
    first.mem.setValue(50)
    first.close()                                         # closing saves

    second = G.TrackerGUI(QSettings(store, QSettings.Format.IniFormat))
    assert second.root_edit.text() == str(tmp_path)
    assert second.preset.currentText() == "Retrack" and second._selection() == "3456d"
    assert second.mem.value() == 50
    second.close()
    del app


def _fake_sorted_session(folder):
    """A session folder holding one finished sort whose units are in the NWB."""
    h5py = pytest.importorskip("h5py")
    import numpy as np
    out = folder / "Rat5_HM_20260623_105525_maze_merged.raw_group0_mountainsort4_sorting_output"
    (out / "processed_binary").mkdir(parents=True)
    (out / "processed_binary" / "traces_cached_seg0.raw").write_bytes(b"\0" * 4096)
    (out / "phy_export").mkdir()
    (out / "phy_export" / "params.py").write_text("dat_path = 'traces_cached_seg0.raw'\n")
    with h5py.File(folder / "Rat5_20260623.nwb", "w") as f:
        f.create_dataset("units/sorting_group", data=np.array([out.name.encode()], dtype="S"))
    return out / "processed_binary"


def _wait_for_the_process(gui):
    from PyQt6.QtCore import QCoreApplication
    import time
    deadline = time.monotonic() + 60
    while gui.proc is not None and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.05)
    assert gui.proc is None, "the clean-up tool did not finish"


def test_disk_card_checks_then_deletes_processed_binary(gui, tmp_path, monkeypatch):
    processed = _fake_sorted_session(tmp_path / "task" / "rat5_491390" / "20260623")
    gui.root_edit.setText(str(tmp_path))
    gui._check_root()
    gui.log.clear()

    gui._clean_processed_binary(delete=False)               # dry run
    assert not gui.pb_delete_btn.isEnabled()                # one process at a time
    _wait_for_the_process(gui)
    assert processed.is_dir()
    assert "DRY RUN" in gui.log.toPlainText()
    assert gui.pb_delete_btn.isEnabled()

    monkeypatch.setattr(gui, "_confirm", lambda question: False)   # "No": nothing runs
    gui._clean_processed_binary(delete=True)
    assert gui.proc is None and processed.is_dir()

    monkeypatch.setattr(gui, "_confirm", lambda question: True)
    gui._clean_processed_binary(delete=True)
    _wait_for_the_process(gui)
    assert not processed.exists()
    assert "Deleted 1 processed_binary folder" in gui.log.toPlainText()


# ------------------------------------------------------------------ disk guard
WORKER_RUN = """[DISK] Disk guard on: 20.0 GB kept free on each disk the jobs write to (MIN_FREE_GB)
[DISK] /data: 3 job(s) expect to write ~310.0 GB; 223.9 GB can be used (243.9 free - 20.0 kept free) — not enough for all: the last jobs wait until space is freed up

[QUEUE] Preparing: ip3
    [WAIT] DISK /data: job needs ~103.3 GB, 17.3 GB available (243.9 free - 206.6 for running jobs - 20.0 kept free). Pausing 30s...
[MASTER] pause  Rat5_20260630_101010_ip2 — DISK /data: only 15.0 GB free (MIN_FREE_GB 20.0); 1 job(s) keep running
"""


def test_disk_card_follows_the_guard_of_a_worker_run(gui):
    gui._reset_jobs()
    assert not gui.disk_guard_line.isVisibleTo(gui) and gui.disk_guard_line.text() == ""
    gui._follow(WORKER_RUN)
    text = gui.disk_guard_line.text()
    assert "3 job(s) expect to write ~310.0 GB" in text          # the runner's forecast
    assert "Waiting for room: the next job needs ~103.3 GB, 17.3 GB available." in text
    assert "Paused until there is room: Rat5_20260630_101010_ip2" in text
    assert G.ORANGE in gui.disk_guard_line.styleSheet()

    gui._follow("[MASTER] pause  Rat5_20260630_101010_ip1 — DISK /data: only 5.0 GB free "
                "(MIN_FREE_GB 20.0); 0 job(s) keep running\n"
                "[MASTER] every job is paused: free up space on /data — they go on by themselves.\n")
    assert "Every job is paused — free up space on /data" in gui.disk_guard_line.text()
    assert G.RED in gui.disk_guard_line.styleSheet()

    gui._follow("[MASTER] resume Rat5_20260630_101010_ip1 — DISK /data: 60.0 GB free\n"
                "[MASTER] resume Rat5_20260630_101010_ip2 — DISK /data: 60.0 GB free\n"
                "[MASTER] Job launched in its own window: Job-Rat5_20260630_101010_ip3\n")
    text = gui.disk_guard_line.text()
    assert "Paused" not in text and "Waiting" not in text and "Every job" not in text
    assert "3 job(s) expect to write" in text                     # the forecast stays to read


def test_disk_card_follows_the_guard_of_a_tree_run(gui):
    gui._reset_jobs()
    gui._follow("[TREE] 2 session(s) x steps [l], up to 2 at a time (POST_JOBS), 20 s apart\n"
                "[DISK] Disk guard on: 20.0 GB kept free on each disk the jobs write to (MIN_FREE_GB)\n"
                "[TREE] start  Rat5_20260630  (1 running, 1 queued)\n"
                "\n[MON 11:00:29] CPU 40% | MEM 56% (17.9/32 GB) | GPU 0% | DISK 31.0 GB free | "
                "1 running, 1 queued, 0 done (0 failed)\n"
                "    Rat5_20260630                step l     0.5 min    1.1 GB\n"
                "    next start waits: DISK /tmp: job needs ~60.0 GB, 11.0 GB available "
                "(31.0 free - 0.0 for running jobs - 20.0 kept free)\n")
    assert "DISK 31.0 GB free" in gui.machine.text()              # the machine line shows it too
    assert "Waiting for room: the next job needs ~60.0 GB, 11.0 GB available." in gui.disk_guard_line.text()
    gui._follow("[TREE] start  Rat5_20260701  (2 running, 0 queued)\n"
                "[TREE] pause  Rat5_20260701 — DISK /tmp: only 15.0 GB free (MIN_FREE_GB 20.0); "
                "1 job(s) keep running\n"
                "\n[MON 11:01:00] CPU 40% | MEM 56% | GPU 0% | DISK 15.0 GB free | "
                "2 running, 0 queued, 0 done (0 failed)\n"
                "    Rat5_20260630                step l     1.0 min    1.1 GB\n"
                "    Rat5_20260701                step l     0.2 min    0.4 GB  PAUSED (disk)\n")
    assert "Waiting" not in gui.disk_guard_line.text()
    assert "Paused until there is room: Rat5_20260701" in gui.disk_guard_line.text()
    assert gui.jobs.text().splitlines()[1].endswith("PAUSED (disk)")
    gui._follow("[TREE] FAILED Rat5_20260701 (3.1 min) — see /tmp/x.log\n")
    assert "Paused" not in gui.disk_guard_line.text()             # an ended job is not paused


def test_disk_card_shows_the_free_space_and_the_keep_free_setting(gui, monkeypatch):
    monkeypatch.setattr(G.disk_guard, "free_bytes", lambda folder: 30 * G.disk_guard.GB)
    gui.min_free.setValue(20)
    gui._show_free_space()                                        # as after a look at the data root
    assert gui.disk_free.text().startswith("Free now:  30.0 GB (")
    assert "temporary folder" in gui.disk_free.text() and G.GREY in gui.disk_free.styleSheet()
    gui.min_free.setValue(50)                                     # now below what must stay free
    assert G.RED in gui.disk_free.styleSheet()
    gui.min_free.setValue(20)
