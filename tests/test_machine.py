"""The number of worker processes follows the computer (src/tools/machine.py).
No real hardware is measured here: the cores and the RAM are numbers the test
sets, so the same answers come out on every machine."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner  # noqa: E402  (chdirs to the repo root, as when it runs)
from tools import machine  # noqa: E402  (runner put src/ on the path)


@pytest.fixture
def computer(monkeypatch):
    """A pretend computer: `computer(cores, ram_gb)` sets what machine.py sees."""
    for name in ("N_JOBS", "HM_PARALLEL_JOBS", "AUTO_CURATE_JOBS", "POST_JOBS", "HM_POST_JOBS"):
        monkeypatch.delenv(name, raising=False)

    def build(cores, ram_gb):
        monkeypatch.setattr(machine, "cpu_cores", lambda: cores)
        monkeypatch.setattr(machine, "total_ram_gb", lambda: ram_gb)
    return build


# ------------------------------------------------------------ a step that runs alone
def test_the_lab_pc_uses_its_cores_but_leaves_two_free(computer):
    computer(cores=20, ram_gb=63.7)                # the lab PC ("HYDRA")
    assert machine.step_n_jobs() == 18


def test_small_machines_leave_one_core_or_none(computer):
    computer(cores=8, ram_gb=16)
    assert machine.step_n_jobs() == 7              # one core stays free
    computer(cores=2, ram_gb=8)
    assert machine.step_n_jobs() == 2              # none to spare
    computer(cores=1, ram_gb=1)
    assert machine.step_n_jobs() == 1              # never less than one worker


def test_little_ram_limits_the_workers(computer):
    computer(cores=20, ram_gb=8)
    assert machine.step_n_jobs() == 4              # half of 8 GB, 1 GB a worker
    # a step whose workers need more RAM gets fewer of them
    assert machine.step_n_jobs(gb_per_worker=2.0) == 2


# ------------------------------------------------------- jobs that run side by side
def test_jobs_side_by_side_share_the_cores(computer, monkeypatch):
    computer(cores=20, ram_gb=63.7)
    assert machine.step_n_jobs(side_by_side=2) == 9
    assert machine.step_n_jobs(side_by_side=5) == 4      # 3.6 each, rounded: 20 in all
    assert machine.step_n_jobs(side_by_side=6) == 3
    assert machine.step_n_jobs(side_by_side=40) == 1     # never less than one worker
    # the runner tells its jobs how many run side by side
    monkeypatch.setenv("HM_PARALLEL_JOBS", "5")
    assert machine.step_n_jobs() == 4
    assert machine.step_n_jobs(side_by_side=1) == 18     # an explicit number wins


def test_jobs_side_by_side_share_the_ram(computer):
    computer(cores=20, ram_gb=16)
    # 8 GB for the workers of all jobs, 2 jobs: 4 GB each, 2 GB a worker
    assert machine.step_n_jobs(gb_per_worker=2.0, side_by_side=2) == 2


# ------------------------------------------------------------------- the settings
def test_n_jobs_in_the_config_forces_a_number(computer, monkeypatch):
    computer(cores=20, ram_gb=63.7)
    monkeypatch.setenv("N_JOBS", "6")
    assert machine.step_n_jobs() == 6
    assert machine.step_n_jobs(side_by_side=5) == 6      # a forced number is not shared out
    for not_a_number in ("auto", "", "many", "0", "-3"):
        monkeypatch.setenv("N_JOBS", not_a_number)
        assert machine.step_n_jobs() == 18, not_a_number


def test_summary_names_a_setting_it_cannot_use(computer, monkeypatch):
    computer(cores=20, ram_gb=63.7)
    assert "20 CPU cores | 64 GB RAM | N_JOBS=auto -> 18 worker" in machine.summary()
    monkeypatch.setenv("N_JOBS", "many")
    assert "not a number" in machine.summary()


# ---------------------------------------------------------------------- tree mode
def test_session_jobs_limit(computer):
    computer(cores=20, ram_gb=63.7)
    assert machine.session_jobs_limit(n_sessions=30) == 5    # one job per 12 GB of RAM
    assert machine.session_jobs_limit(n_sessions=2) == 2     # never more than sessions
    computer(cores=64, ram_gb=512)
    assert machine.session_jobs_limit(n_sessions=30) == 6    # at most 6
    computer(cores=4, ram_gb=64)
    assert machine.session_jobs_limit(n_sessions=30) == 2    # half the cores
    computer(cores=1, ram_gb=4)
    assert machine.session_jobs_limit(n_sessions=30) == 1    # never less than one


def test_runner_post_jobs_auto_uses_the_machine(computer, monkeypatch):
    computer(cores=20, ram_gb=63.7)
    assert runner.post_jobs_limit(30) == 5
    monkeypatch.setenv("POST_JOBS", "3")                     # a number in the config wins
    assert runner.post_jobs_limit(30) == 3


def test_tree_jobs_are_told_how_many_run_side_by_side(computer, monkeypatch, tmp_path):
    """Each session job gets HM_PARALLEL_JOBS, so its steps share the cores."""
    computer(cores=20, ram_gb=63.7)
    monkeypatch.setenv("POST_JOBS", "2")
    monkeypatch.setenv("POST_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DISABLE_RESOURCE_CHECK", "1")
    monkeypatch.setattr(runner, "LAUNCH_GAP", 0)
    monkeypatch.setattr(runner, "MONITOR_SECONDS", 0)
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.05)
    sessions = [tmp_path / "s1", tmp_path / "s2", tmp_path / "s3"]
    for folder in sessions:
        folder.mkdir()

    # a stand-in job: it writes down the worker count a step would get in it
    script = tmp_path / "job.py"
    script.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {str(Path(machine.__file__).parent)!r})\n"
        "import machine\n"
        "machine.cpu_cores = lambda: 20\n"
        "machine.total_ram_gb = lambda: 63.7\n"
        "open(os.path.join(sys.argv[1], 'seen.txt'), 'w').write(\n"
        "    os.environ['HM_PARALLEL_JOBS'] + ' ' + str(machine.step_n_jobs()))\n")
    results = runner.run_post_jobs([(d, d) for d in sessions], "u", "cfg.txt", tmp_path,
                                   job_cmd=lambda op, steps, config: [sys.executable, str(script), str(op)])

    assert [r["status"] for r in results] == ["ok"] * 3
    for folder in sessions:
        assert (folder / "seen.txt").read_text() == "2 9"    # 2 side by side: 9 workers each


def test_a_worker_window_is_told_how_many_run_side_by_side(monkeypatch, tmp_path):
    """`runner.py --worker ... <marker> <n>`: the worker's steps see HM_PARALLEL_JOBS=<n>."""
    monkeypatch.delenv("HM_PARALLEL_JOBS", raising=False)
    seen = {}
    monkeypatch.setattr(runner, "load_config", lambda: "cfg.txt")
    monkeypatch.setattr(runner, "run_worker", lambda ip, op, steps, out: seen.update(
        jobs=runner.os.environ.get("HM_PARALLEL_JOBS")))
    marker = tmp_path / "w.done"
    runner._worker_main([str(tmp_path), str(tmp_path), "8", str(marker), "4"])
    assert seen["jobs"] == "4"
    assert marker.read_text() == "done"
    monkeypatch.delenv("HM_PARALLEL_JOBS", raising=False)     # _worker_main set it for real
