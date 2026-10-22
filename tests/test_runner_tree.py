"""runner.py tree mode: session folders found below a renamed/grouped root, and the
per-session steps run as parallel jobs. A root with ip* folders keeps its targets."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner  # noqa: E402  (chdirs to the repo root, as when it runs)


def touch(path, text="x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def share(tmp_path):
    """The share's layout: task/<rat>_<id>/<YYYYMMDD>/ with outputs inside."""
    root = tmp_path / "task"
    touch(root / "session_summary.pdf")
    touch(root / "figures" / "summary.pdf")
    touch(root / "rat5_491390" / "session_summary.xlsx")
    s1 = root / "rat5_491390" / "20260630"
    touch(s1 / "Rat5_20260630.nwb")
    touch(s1 / "LFP_Output" / "lfp_data.npy")
    touch(s1 / "decoding" / "Rat5_x.nwb")                 # outputs are not searched
    touch(root / "rat5_491390" / "20260701" / "Rat5_20260701.nwb")
    touch(root / "rat5_491390" / "20260701" / "._Rat5_20260701.nwb")
    touch(root / "rat6_491391" / "20260630" / "20260630_Rat6.mp4")   # date rule, no NWB yet
    touch(root / "rat6_491391" / "notes" / "readme.txt")
    touch(root / "op1" / "anything.csv")                  # a top-level op* folder
    touch(root / "a" / "b" / "c" / "d" / "20260101" / "Rat1_20260101.nwb")   # too deep
    return root


def test_session_folders_are_found_below_a_renamed_tree(share):
    found = [str(p.relative_to(share)) for p in runner.find_session_folders(share, max_depth=4)]
    assert found == ["op1", "rat5_491390/20260630", "rat5_491390/20260701", "rat6_491391/20260630"]


def test_session_labels(share):
    assert runner.session_label(share / "rat5_491390" / "20260701", share) == "Rat5_20260701"
    assert runner.session_label(share / "rat6_491391" / "20260630", share) == "rat6_491391_20260630"


def test_a_root_with_ip_folders_keeps_its_targets(tmp_path):
    root = tmp_path / "data"
    touch(root / "ip1" / "x.rec")
    touch(root / "op1" / "x.csv")
    touch(root / "20260629" / "Rat6_20260629.nwb")
    touch(root / "archive" / "20260101" / "Rat1_20260101.nwb")   # not a target in this layout
    ops = [(root / "ip1", root / "op1")]
    targets = sorted(p.name for _ip, p in runner.sequential_targets(root, ops))
    assert targets == ["20260629", "op1"]


def test_post_job_runs_the_steps_in_order(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run", lambda cmd, cwd=None, out=None: calls.append(
        [str(c) for c in cmd]) or 0)
    monkeypatch.setattr(runner, "decode_session", lambda op, config, tag, out=None: calls.append(["b"]) or [0])
    monkeypatch.setattr(runner, "umap_session", lambda op, config, tag, out=None: calls.append(["m"]) or [0])
    op = tmp_path / "20260630"
    op.mkdir()
    with open(tmp_path / "log.txt", "w") as out:
        failed = runner.run_post_job(op, "m5bvul", "cfg.txt", out)
    assert failed == []
    order = [c[0] if len(c) == 1 else Path(c[2]).name for c in calls]
    assert order == ["rewrite_nwb_lfp.py", "add_units.py", "visualize_nwb.py", "b", "plot_trials.py", "m"]
    plot = next(c for c in calls if c[-1:] and "plot_trials.py" in " ".join(c))
    assert plot[plot.index("--input_folder") + 1] == str(op) == plot[plot.index("--output_folder") + 1]


def test_jobs_run_in_parallel_up_to_the_limit_and_failures_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("POST_JOBS", "2")
    monkeypatch.setenv("DISABLE_RESOURCE_CHECK", "1")
    monkeypatch.setenv("POST_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(runner, "LAUNCH_GAP", 0)
    monkeypatch.setattr(runner, "MONITOR_SECONDS", 0)
    monkeypatch.setattr(runner, "POLL_SECONDS", 0.1)
    targets = []
    for i in range(4):
        d = tmp_path / f"2026070{i}"
        touch(d / f"Rat1_2026070{i}.nwb")
        targets.append((d, d))
    stamps = tmp_path / "stamps"
    stamps.mkdir()

    def job(op, steps, config):
        code = (f"import time, sys; open(r'{stamps}/{Path(op).name}', 'w').write(str(time.time()) + ' ');"
                f"time.sleep(0.8); open(r'{stamps}/{Path(op).name}', 'a').write(str(time.time()));"
                f"sys.exit({1 if Path(op).name.endswith('2') else 0})")
        return [sys.executable, "-c", code]

    t0 = time.time()
    results = runner.run_post_jobs(targets, "v", "cfg.txt", tmp_path, job_cmd=job)
    spans = [tuple(map(float, f.read_text().split())) for f in stamps.iterdir()]
    assert len(spans) == 4
    most = max(sum(a <= t < b for a, b in spans) for t, _ in spans)
    assert most == 2                                      # two at a time, never more
    assert time.time() - t0 < 4 * 0.8                     # so faster than one at a time
    status = {r["session"]: r["status"] for r in results}
    assert status == {"Rat1_20260700": "ok", "Rat1_20260701": "ok",
                      "Rat1_20260702": "failed", "Rat1_20260703": "ok"}
    assert (Path(results[0]["log"]).parent / "summary.csv").is_file()


def test_a_session_list_replaces_the_tree_walk(share, tmp_path, monkeypatch, capsys):
    # the GUI's "Only failed of a run…": only the listed sessions get the steps
    wanted = share / "rat5_491390" / "20260701"
    list_file = tmp_path / "only_sessions.txt"
    list_file.write_text(f"# failed in run 20261022_120002\n\n{wanted}\n{share / 'gone' / '20260101'}\n",
                         encoding="utf-8")
    assert runner.read_session_list(list_file) == [wanted]
    assert "not found, skipped" in capsys.readouterr().out

    jobs = {}

    def fake_run_post_jobs(targets, steps, config, root):
        jobs["targets"] = [op for _ip, op in targets]
        jobs["steps"] = steps
        return []

    monkeypatch.setattr(runner, "run_post_jobs", fake_run_post_jobs)
    monkeypatch.setattr(runner, "load_config", lambda: "cfg.txt")
    monkeypatch.setattr(sys, "argv", ["runner.py", str(share)])
    monkeypatch.setenv("HM_STEPS", "l")
    monkeypatch.setenv("HM_SESSIONS_FILE", str(list_file))
    runner.main()
    assert jobs == {"targets": [wanted], "steps": "l"}
