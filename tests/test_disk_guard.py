"""The runner's disk guard: a parallel job starts only when its expected output
fits on the disk, and running jobs are paused when the disk still runs low.
No real disk is filled here: the free space the guard reads is a number the test
sets."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner  # noqa: E402  (chdirs to the repo root, as when it runs)
from tools import disk_guard  # noqa: E402  (runner put src/ on the path)

GB = disk_guard.GB


@pytest.fixture
def disk(monkeypatch):
    """One pretend disk: every folder is on it, and `disk["free"]` is its free space."""
    state = {"free": 100 * GB}
    monkeypatch.setattr(disk_guard, "free_bytes", lambda folder: state["free"])
    monkeypatch.setattr(disk_guard, "disk_of", lambda folder: "the-disk")
    monkeypatch.setenv("MIN_FREE_GB", "20")
    monkeypatch.delenv("JOB_DISK_GB", raising=False)
    monkeypatch.delenv("DISABLE_RESOURCE_CHECK", raising=False)
    return state


# ---------------------------------------------------------------- settings
def test_min_free_gb_setting(monkeypatch):
    monkeypatch.delenv("MIN_FREE_GB", raising=False)
    assert disk_guard.min_free_bytes() == 20 * GB            # the default
    monkeypatch.setenv("MIN_FREE_GB", "50")
    assert disk_guard.min_free_bytes() == 50 * GB
    monkeypatch.setenv("MIN_FREE_GB", "lots")                 # not a number: the default
    assert disk_guard.min_free_bytes() == 20 * GB


# ------------------------------------------------------- 1. before a job starts
def test_a_job_starts_only_when_its_output_fits(disk, tmp_path):
    budget = disk_guard.DiskBudget()
    need = {tmp_path: 30 * GB}
    # 100 free - 20 kept free = 80 GB to hand out: two jobs of 30 fit, a third does not
    for label in ("a", "b"):
        ok, _why = budget.room_for(need)
        assert ok
        budget.promise(label, need)
    ok, why = budget.room_for(need)
    assert not ok
    assert "DISK" in why and "needs ~30.0 GB" in why and "20.0 GB available" in why


def test_what_running_jobs_already_wrote_is_not_counted_twice(disk, tmp_path):
    budget = disk_guard.DiskBudget()
    need = {tmp_path: 30 * GB}
    budget.promise("a", need)
    budget.promise("b", need)
    disk["free"] = 40 * GB                                    # both wrote all of their 30 GB
    # nothing is left to write, so 40 - 20 kept free = 20 GB: a job of 30 does not fit...
    assert not budget.room_for(need)[0]
    # ...but one of 15 does (with the promises counted in full it would not)
    assert budget.room_for({tmp_path: 15 * GB})[0]


def test_a_finished_job_gives_its_space_back(disk, tmp_path):
    budget = disk_guard.DiskBudget()
    need = {tmp_path: 30 * GB}
    budget.promise("a", need)
    budget.promise("b", need)
    assert not budget.room_for(need)[0]                       # 100 - 60 - 20 = 20 < 30
    disk["free"] = 70 * GB                                    # job a wrote its 30 GB; they stay
    budget.release("a")
    # b still has its 30 to write: 70 - 30 - 20 = 20, so still no room for 30
    assert not budget.room_for(need)[0]
    disk["free"] = 40 * GB                                    # b wrote its 30 GB too
    budget.release("b")
    assert budget.room_for({tmp_path: 20 * GB})[0]            # 40 - 20 = 20
    assert not budget.room_for({tmp_path: 21 * GB})[0]


def test_temporary_files_come_back_when_the_job_ends(disk):
    # what a job writes into the temporary folder is deleted when it ends, also
    # when that folder is on the same disk as the outputs (as it is here)
    import tempfile
    budget = disk_guard.DiskBudget()
    temp = Path(tempfile.gettempdir())
    need = {temp: 30 * GB}
    budget.promise("a", need)
    budget.promise("b", need)
    disk["free"] = 40 * GB                                    # both at their peak
    disk["free"] = 70 * GB                                    # a deleted its temporary files
    budget.release("a")
    # b has written all of its 30 already: 70 - 0 - 20 = 50 GB for the next job
    assert budget.room_for({temp: 50 * GB})[0]
    assert not budget.room_for({temp: 51 * GB})[0]


def test_a_job_that_can_never_fit_says_so(disk, tmp_path):
    ok, why = disk_guard.DiskBudget().room_for({tmp_path: 500 * GB})
    assert not ok and "free up space" in why                  # nothing is running: only the user can help


def test_an_unreadable_disk_never_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr(disk_guard, "free_bytes", lambda folder: None)
    budget = disk_guard.DiskBudget()
    assert budget.room_for({tmp_path: 10 ** 6 * GB}) == (True, "")
    assert budget.fullest() == (None, None)


# ----------------------------------------------------------- 2. while jobs run
def test_pause_decision_zones():
    decide = lambda free_gb, active, paused: disk_guard.pause_decision(
        free_gb * GB, active, paused, reserve=20 * GB)
    assert decide(100, 3, 0) == ""                            # plenty, nothing paused
    assert decide(100, 1, 2) == "resume"                      # plenty: the paused go on
    assert decide(25, 2, 1) == ""                             # just enough: change nothing
    assert decide(25, 0, 2) == "resume"                       # ...but never leave all paused
    assert decide(15, 3, 0) == "pause"                        # low: pause the newest
    assert decide(15, 1, 2) == ""                             # low: one keeps running
    assert decide(15, 0, 3) == "resume"                       # low: one must run
    assert decide(5, 1, 2) == "pause"                         # critical: all pause
    assert decide(5, 0, 3) == ""                              # critical: all stay paused
    assert disk_guard.pause_decision(None, 1, 1) == "resume"  # unreadable disk: nothing to guard


def test_newest_job_is_paused_and_oldest_resumed(disk, monkeypatch, capsys, tmp_path):
    frozen = []
    monkeypatch.setattr(disk_guard, "can_pause", lambda: True)
    monkeypatch.setattr(disk_guard, "pause_job", lambda pid: frozen.append(pid) or True)
    monkeypatch.setattr(disk_guard, "resume_job", lambda pid: frozen.remove(pid) or True)
    budget = disk_guard.DiskBudget()
    budget.promise("x", {tmp_path: 1 * GB})                   # so that the disk is watched
    jobs = [{"label": name, "pid": pid, "paused": False} for name, pid in (("old", 1), ("mid", 2), ("new", 3))]

    disk["free"] = 15 * GB                                    # low
    runner.pause_for_disk(jobs, budget, "[TREE]")
    assert frozen == [3]                                      # the newest first
    runner.pause_for_disk(jobs, budget, "[TREE]")
    assert frozen == [3, 2]
    runner.pause_for_disk(jobs, budget, "[TREE]")
    assert frozen == [3, 2]                                   # the oldest keeps running
    disk["free"] = 5 * GB                                     # critical
    runner.pause_for_disk(jobs, budget, "[TREE]")
    assert frozen == [3, 2, 1]
    assert "every job is paused: free up space" in capsys.readouterr().out

    disk["free"] = 60 * GB                                    # room again
    runner.pause_for_disk(jobs, budget, "[TREE]")
    assert frozen == [3, 2]                                   # the oldest goes on first
    runner.pause_for_disk(jobs, budget, "[TREE]")
    runner.pause_for_disk(jobs, budget, "[TREE]")
    assert frozen == [] and not any(job["paused"] for job in jobs)
    out = capsys.readouterr().out
    assert out.count("[TREE] resume") == 3 and "resume old" in out


def test_the_gate_switch_also_switches_the_disk_guard_off(disk, monkeypatch, tmp_path):
    monkeypatch.setenv("DISABLE_RESOURCE_CHECK", "1")
    budget = disk_guard.DiskBudget()
    assert runner.disk_room(budget, {tmp_path: 10 ** 6 * GB}) == (True, "")


# -------------------------------------------------- how much will a job write?
def test_worker_job_need_follows_the_inputs(tmp_path, monkeypatch):
    monkeypatch.delenv("JOB_DISK_GB", raising=False)
    monkeypatch.delenv("KEEP_NPY", raising=False)
    ip, op = tmp_path / "ip1", tmp_path / "op1"
    ip.mkdir()
    (ip / "Rat5_20260630_101010.rec").write_bytes(b"r" * 100_000)
    (ip / "eye01_x.mp4").write_bytes(b"v" * 6_000)
    (ip / "eye02_x.MP4").write_bytes(b"v" * 4_000)             # upper-case suffix counts too
    (ip / "stitched.mp4").write_bytes(b"s" * 50_000)           # an output, not an input
    small = disk_guard.SMALL_STEPS_BYTES

    need = disk_guard.worker_job_need(ip, op, "1348")
    assert need[ip] == pytest.approx(0.10 * 100_000 + 10_000)            # export + stitched.mp4
    assert need[op] == pytest.approx(small + 10_000 + 0.15 * 100_000)    # tracker video + LFP
    assert disk_guard.worker_job_need(ip, op, "25n") == {ip: 0.0, op: float(small)}

    monkeypatch.setenv("JOB_DISK_GB", "40")                    # the user knows better
    assert disk_guard.worker_job_need(ip, op, "1348") == {op: 40 * GB}


def test_session_job_need_puts_step_l_copies_on_the_temp_disk(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.delenv("JOB_DISK_GB", raising=False)
    op = tmp_path / "20260630"
    (op / "LFP_Output").mkdir(parents=True)
    (op / "Rat5_20260630.nwb").write_bytes(b"n" * 20_000)
    (op / "._Rat5_20260630.nwb").write_bytes(b"x" * 999)       # a macOS sidecar is not the NWB
    (op / "LFP_Output" / "Rat5_20260630_lfp_data.npy").write_bytes(b"l" * 5_000)
    temp = Path(tempfile.gettempdir())

    need = disk_guard.session_job_need(op, "l")
    assert need[temp] == pytest.approx(2 * 20_000 + 2 * 5_000)
    assert need[op] == pytest.approx(disk_guard.SMALL_STEPS_BYTES + 0.5 * 20_000)
    assert disk_guard.session_job_need(op, "v")[temp] == 0     # only step l works on local copies


# ------------------------------------------------------------- in the runner
def _sleeping_job(stamps, seconds):
    """A stand-in session job: writes its start and end time, sleeping in between."""
    def job(op, steps, config):
        stamp = stamps / Path(op).name
        code = (f"import time; open(r'{stamp}', 'w').write(str(time.time()) + ' ');"
                f"time.sleep({seconds}); open(r'{stamp}', 'a').write(str(time.time()))")
        return [sys.executable, "-c", code]
    return job


def _tree(tmp_path, monkeypatch, n_sessions):
    monkeypatch.setenv("POST_JOBS", str(n_sessions))           # the disk is the only limit
    monkeypatch.setenv("POST_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(runner, "LAUNCH_GAP", 0)
    monkeypatch.setattr(runner, "MONITOR_SECONDS", 0)
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.1)
    monkeypatch.setattr(runner, "resources_free", lambda: (True, ""))    # CPU/GPU/MEM never wait
    targets = []
    for i in range(n_sessions):
        d = tmp_path / f"2026070{i}"
        d.mkdir()
        (d / f"Rat1_2026070{i}.nwb").write_text("x")
        targets.append((d, d))
    stamps = tmp_path / "stamps"
    stamps.mkdir()
    return targets, stamps


def test_tree_jobs_wait_for_disk_room(disk, tmp_path, monkeypatch, capsys):
    targets, stamps = _tree(tmp_path, monkeypatch, 3)
    monkeypatch.setenv("JOB_DISK_GB", "50")                    # 100 free - 20 kept: one job at a time
    results = runner.run_post_jobs(targets, "v", "cfg.txt", tmp_path, job_cmd=_sleeping_job(stamps, 0.5))
    assert [r["status"] for r in results] == ["ok"] * 3
    spans = sorted(tuple(map(float, f.read_text().split())) for f in stamps.iterdir())
    for (_start, end), (next_start, _end) in zip(spans, spans[1:]):
        assert next_start >= end                               # never two at once
    out = capsys.readouterr().out
    assert "next start waits: DISK" in out and "| DISK 100.0 GB free |" in out


def test_tree_jobs_are_paused_while_the_disk_is_low(disk, tmp_path, monkeypatch, capsys):
    pytest.importorskip("psutil")
    targets, stamps = _tree(tmp_path, monkeypatch, 2)
    monkeypatch.setattr(runner, "MONITOR_SECONDS", 1)          # a status block while one is paused
    monkeypatch.setenv("JOB_DISK_GB", "1")                     # both fit, so both start

    def job(op, steps, config):
        # 2 s of work in small steps: a frozen job makes no steps, so it ends later
        stamp = stamps / Path(op).name
        code = (f"import time; open(r'{stamp}', 'w').write(str(time.time()) + ' ');"
                f"[time.sleep(0.1) for _ in range(20)];"
                f"open(r'{stamp}', 'a').write(str(time.time()))")
        return [sys.executable, "-c", code]

    # The disk "runs low" as soon as both jobs are running and stays low for 1.2 s
    # after the first pause.
    real_pause, paused_at = disk_guard.pause_job, []

    def pause(pid):
        paused_at.append(time.time())
        return real_pause(pid)
    monkeypatch.setattr(disk_guard, "pause_job", pause)

    def free_now(folder):
        both_started = len(list(stamps.iterdir())) == 2
        still_low = not paused_at or time.time() - paused_at[0] < 1.2
        return 15 * GB if both_started and still_low else 100 * GB
    monkeypatch.setattr(disk_guard, "free_bytes", free_now)

    results = runner.run_post_jobs(targets, "v", "cfg.txt", tmp_path, job_cmd=job)
    assert [r["status"] for r in results] == ["ok", "ok"]      # a paused job still ends well
    out = capsys.readouterr().out
    assert "[TREE] pause  Rat1_20260701" in out and "[TREE] resume Rat1_20260701" in out
    assert "pause  Rat1_20260700" not in out                   # the oldest kept running
    assert len(paused_at) == 1 and "PAUSED (disk)" in out
    old, new = (tuple(map(float, (stamps / f"2026070{i}").read_text().split())) for i in (0, 1))
    assert old[1] - old[0] < 3.0                               # the old job was not held up
    assert (new[1] - new[0]) - (old[1] - old[0]) > 0.8         # the new one was, for about 1.2 s


# ------------------------------------------------- worker mode (ip*/op* folders)
def test_a_worker_waits_until_its_output_fits(disk, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WAIT_SECONDS", "7")
    budget = disk_guard.DiskBudget()
    disk["free"] = 30 * GB                                     # 30 - 20 kept free: no room for 25
    waits = []

    def waited(seconds):                                       # space is freed during the wait
        waits.append(seconds)
        disk["free"] = 60 * GB
    monkeypatch.setattr(runner.time, "sleep", waited)
    runner.wait_for_disk(budget, {tmp_path: 25 * GB})
    assert waits == [7]
    assert "[WAIT] DISK" in capsys.readouterr().out


def test_the_watch_gives_back_the_space_of_finished_workers(disk, tmp_path, monkeypatch):
    import threading
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.05)
    budget = disk_guard.DiskBudget()
    marker, pid_file = tmp_path / "hm_worker_a.done", tmp_path / "hm_worker_a.pid"
    workers = [{"label": "a", "pid": None, "paused": False,
                "marker": marker, "pid_file": pid_file, "thread": None}]
    budget.promise("a", {tmp_path: 70 * GB})
    assert not budget.room_for({tmp_path: 30 * GB})[0]         # a's promise is in the way
    stop = threading.Event()
    watch = threading.Thread(target=runner.watch_worker_disks, args=(budget, workers, stop), daemon=True)
    watch.start()
    try:
        pid_file.write_text("4242")                            # the worker says which process it is
        deadline = time.time() + 5
        while workers[0]["pid"] is None and time.time() < deadline:
            time.sleep(0.02)
        assert workers[0]["pid"] == 4242
        marker.write_text("done")                              # the worker finished
        while workers and time.time() < deadline:
            time.sleep(0.02)
        assert workers == [] and budget.room_for({tmp_path: 30 * GB})[0]
    finally:
        stop.set()
        watch.join(2)
