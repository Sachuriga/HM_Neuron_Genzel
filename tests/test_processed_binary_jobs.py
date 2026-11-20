"""Step 7 writes processed_binary/ with as many worker processes as fit the
computer (sorting.save_processed_binary). The file must not depend on how many
processes wrote it, and the shared-noise summary line must survive them."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("spikeinterface")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sorter"))
import sorting                                              # noqa: E402

FS = 30000
N_TETRODES = 16
GROUPS = np.repeat(np.arange(N_TETRODES), 4)                # tetrode of each of the 64 channels


def write_raw_file(path, seconds=3.4):
    """A raw int16 file: channel noise, plus a pulse train that reaches every
    tetrode with its own gain (what a common reference cannot remove)."""
    rng = np.random.default_rng(0)
    n = int(seconds * FS)
    data = rng.normal(0.0, 50.0, (n, GROUPS.size)).astype(np.float32)
    pulses = np.zeros(n, dtype=np.float32)
    pulses[::240] = 1500.0
    gain = np.repeat(rng.uniform(0.4, 2.4, N_TETRODES), 4)
    data += pulses[:, None] * gain[None, :]
    np.rint(data).astype(np.int16).tofile(path)


def preprocessed(raw_file, out_dir, in_memory=False):
    """The recording step 7 writes, built by sorting.py's own function."""
    import spikeinterface.full as si
    if in_memory:
        data = np.fromfile(raw_file, dtype=np.int16).reshape(-1, GROUPS.size)
        rec = si.NumpyRecording(traces_list=[data], sampling_frequency=FS)
    else:
        rec = si.read_binary(str(raw_file), sampling_frequency=FS, dtype="int16",
                             num_channels=GROUPS.size)
    rec.set_property("group", GROUPS)
    return sorting.preprocess_for_sorting(
        rec, freq_min=300.0, freq_max=6000.0, shared_noise_components=2,
        shared_noise_file=out_dir / sorting.SHARED_NOISE_STRENGTHS_FILE)


def test_worker_processes_write_the_same_file_and_summary(tmp_path, monkeypatch):
    raw_file = tmp_path / "raw.dat"
    write_raw_file(raw_file)
    written, summary = {}, {}
    for n_jobs in ("1", "2"):
        out_dir = tmp_path / f"jobs_{n_jobs}"
        out_dir.mkdir()
        monkeypatch.setenv("PREPROCESS_JOBS", n_jobs)
        rec = preprocessed(raw_file, out_dir)
        assert sorting.processed_binary_jobs(rec) == int(n_jobs)
        sorting.save_processed_binary(rec, out_dir / "processed_binary")
        written[n_jobs] = next((out_dir / "processed_binary").glob("*.raw")).read_bytes()
        summary[n_jobs] = rec.removed_summary()

    assert written["1"] == written["2"]                     # the same bytes
    # the 2 workers noted what they removed in the shared table: the same line
    assert "interference found and removed in 3 of 3 blocks" in summary["1"]
    assert summary["2"] == summary["1"]


def test_a_recording_held_in_memory_gets_one_worker(tmp_path, monkeypatch):
    """It would be copied whole into every worker process."""
    raw_file = tmp_path / "raw.dat"
    write_raw_file(raw_file, seconds=1.0)
    monkeypatch.setenv("PREPROCESS_JOBS", "8")
    assert sorting.processed_binary_jobs(preprocessed(raw_file, tmp_path)) == 8
    in_memory = preprocessed(raw_file, tmp_path, in_memory=True)
    assert not sorting.read_from_files(in_memory)
    assert sorting.processed_binary_jobs(in_memory) == 1


def test_the_number_of_workers_follows_the_computer(tmp_path, monkeypatch):
    raw_file = tmp_path / "raw.dat"
    write_raw_file(raw_file, seconds=1.0)
    rec = preprocessed(raw_file, tmp_path)
    for name in ("PREPROCESS_JOBS", "N_JOBS", "HM_PARALLEL_JOBS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sorting.machine, "cpu_cores", lambda: 20)       # the lab PC
    monkeypatch.setattr(sorting.machine, "total_ram_gb", lambda: 63.7)
    assert sorting.processed_binary_jobs(rec) == 18
    monkeypatch.setenv("N_JOBS", "6")                       # every step: 6
    assert sorting.processed_binary_jobs(rec) == 6
    monkeypatch.setenv("PREPROCESS_JOBS", "1")              # this part only: as it always was
    assert sorting.processed_binary_jobs(rec) == 1
    monkeypatch.setenv("PREPROCESS_JOBS", "auto")           # not a number: N_JOBS again
    assert sorting.processed_binary_jobs(rec) == 6


class RecordingWhoseWorkersFail:
    """Stands in for a recording: writing with worker processes fails."""

    def __init__(self):
        self.calls = []

    def check_serializability(self, kind):
        return True

    def save(self, folder, n_jobs, **kwargs):
        self.calls.append(n_jobs)
        if n_jobs > 1:
            Path(folder).mkdir()
            (Path(folder) / "half_written.raw").write_bytes(b"x")
            raise RuntimeError("a worker process died")
        assert not Path(folder).exists()                    # the half-written file is gone
        return "saved with 1 worker"


def test_failing_workers_fall_back_to_one_worker(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PREPROCESS_JOBS", "4")
    rec = RecordingWhoseWorkersFail()
    assert sorting.save_processed_binary(rec, tmp_path / "processed_binary") == "saved with 1 worker"
    assert rec.calls == [4, 1]
    assert "Starting again with 1 worker" in capsys.readouterr().out


def test_one_worker_failing_is_an_error(tmp_path, monkeypatch):
    """Nothing slower to fall back to: the error reaches the caller."""
    class Broken(RecordingWhoseWorkersFail):
        def save(self, folder, n_jobs, **kwargs):
            raise RuntimeError("disk full")

    monkeypatch.setenv("PREPROCESS_JOBS", "1")
    with pytest.raises(RuntimeError, match="disk full"):
        sorting.save_processed_binary(Broken(), tmp_path / "processed_binary")


# ------------------------------------------------- sorting the tetrodes side by side
class TetrodeRecording:
    """Stands in for the saved recording: 31 tetrodes, 5.6 h at 30 kHz."""

    def get_num_samples(self):
        return 606362611

    def get_property(self, name):
        return np.repeat(np.arange(31), 4)


def lab_pc(monkeypatch, free_disk_gb):
    for name in ("SORT_JOBS", "N_JOBS", "HM_PARALLEL_JOBS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sorting.machine, "cpu_cores", lambda: 20)
    monkeypatch.setattr(sorting.machine, "total_ram_gb", lambda: 63.7)
    monkeypatch.setattr(sorting.shutil, "disk_usage",
                        lambda folder: type("Usage", (), {"free": free_disk_gb * 2 ** 30})())


def test_tetrodes_sorted_side_by_side_fit_the_ram_and_the_disk(tmp_path, monkeypatch):
    rec = TetrodeRecording()
    assert round(sorting.one_tetrode_gb(rec), 1) == 9.0     # one tetrode as float32
    lab_pc(monkeypatch, free_disk_gb=500)
    # MountainSort4: half of 64 GB at 4 GB a sort
    assert sorting.sorter_jobs(rec, "mountainsort4", tmp_path) == 7
    # ...and its temporary files (13.6 GB a sort) must fit on top of 20 GB kept free
    lab_pc(monkeypatch, free_disk_gb=60)
    assert sorting.sorter_jobs(rec, "mountainsort4", tmp_path) == 2
    lab_pc(monkeypatch, free_disk_gb=10)
    assert sorting.sorter_jobs(rec, "mountainsort4", tmp_path) == 1
    # MountainSort5 holds the tetrode in RAM: 18 GB a sort, the disk does not matter
    assert sorting.sorter_jobs(rec, "mountainsort5", tmp_path) == 1
    monkeypatch.setenv("SORT_JOBS", "3")                    # a number in the config wins
    assert sorting.sorter_jobs(rec, "mountainsort4", tmp_path) == 3
    monkeypatch.setenv("SORT_JOBS", "99")                   # never more than tetrodes
    assert sorting.sorter_jobs(rec, "mountainsort4", tmp_path) == 31


def test_sorting_side_by_side_falls_back_to_one_after_the_other(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SORT_JOBS", "4")
    engines = []

    def fake_run_sorter_by_property(engine, engine_kwargs, **kwargs):
        engines.append((engine, engine_kwargs))
        if engine == "joblib":
            raise RuntimeError("a sorter process died")
        return "sorted in a loop"

    monkeypatch.setattr(sorting.si, "run_sorter_by_property", fake_run_sorter_by_property)
    result = sorting.sort_tetrodes(TetrodeRecording(), "mountainsort4", {}, tmp_path / "work")
    assert result == "sorted in a loop"
    assert engines == [("joblib", {"n_jobs": 4, "backend": "loky"}), ("loop", {})]
    assert "one after the other" in capsys.readouterr().out


# ------------------------------------- using processed_binary/ again after a stopped run
def recipe_for(raw_file, **changes):
    settings = dict(paths=[raw_file], bad_channel_ids=[3], ref_channels=[98],
                    eeg_channel_ids=[44, 45, 46, 47], freq_min=300.0, freq_max=6000.0,
                    fs=30000.0, gain=0.195, offset=0.0, shared_noise_components=2)
    settings.update(changes)
    return sorting.processed_binary_recipe(**settings)


def test_a_complete_processed_binary_is_used_again(tmp_path, monkeypatch):
    raw_file = tmp_path / "raw.dat"
    write_raw_file(raw_file, seconds=2.0)
    monkeypatch.setenv("PREPROCESS_JOBS", "1")
    folder = tmp_path / "out" / "processed_binary"
    (tmp_path / "out").mkdir()
    saved = sorting.save_processed_binary(preprocessed(raw_file, tmp_path / "out"), folder)
    recipe = recipe_for(raw_file)

    # written, but the run stopped before the note: not to be trusted
    rec, why = sorting.load_complete_processed_binary(folder, recipe)
    assert rec is None and "not written to the end" in why

    sorting.note_processed_binary_complete(folder, recipe, saved, "Shared noise: the line")
    rec, note = sorting.load_complete_processed_binary(folder, recipe)
    assert rec.get_num_samples() == saved.get_num_samples()
    assert np.array_equal(rec.get_traces(end_frame=5000), saved.get_traces(end_frame=5000))
    assert list(rec.get_property("group")) == list(GROUPS)          # the sorter needs the tetrodes
    assert note["shared_noise_summary"] == "Shared noise: the line"

    # other settings today: the file is not what a new sort would write
    rec, why = sorting.load_complete_processed_binary(folder, recipe_for(raw_file, freq_min=600.0))
    assert rec is None and "freq_min" in why
    rec, why = sorting.load_complete_processed_binary(folder, recipe_for(raw_file, bad_channel_ids=[]))
    assert rec is None and "bad_channel_ids" in why

    # the raw file is another one now (its size is in the recipe)
    with open(raw_file, "ab") as f:
        f.write(b"\0" * 128)
    rec, why = sorting.load_complete_processed_binary(folder, recipe_for(raw_file))
    assert rec is None and "raw_files" in why

    # the samples are no longer all on disk
    del rec, saved
    traces = next(folder.glob("*.raw"))
    with open(traces, "r+b") as f:
        f.truncate(traces.stat().st_size - 64)
    rec, why = sorting.load_complete_processed_binary(folder, recipe)
    assert rec is None


def test_leftovers_are_deleted_but_processed_binary_is_kept(tmp_path):
    out = tmp_path / "out"
    for folder in ("processed_binary", "sorting_work_folder", "ms4_temp"):
        (out / folder).mkdir(parents=True)
    for name in ("spliced_gaps.json", "shared_noise_strengths.npy"):
        (out / name).write_text("x")
    sorting.keep_only_processed_binary(out)
    assert sorted(p.name for p in out.iterdir()) == ["processed_binary", "spliced_gaps.json"]
