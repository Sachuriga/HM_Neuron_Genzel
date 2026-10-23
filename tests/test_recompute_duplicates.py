"""Step r / step u: duplicated spikes are dropped with spikeinterface's
'keep_first' rule (same spikes in the metrics and in the NWB), and redundant
units are reported but never removed."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

si = pytest.importorskip("spikeinterface.full")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sorter"))
import sorter_common  # noqa: E402

FS = 30000.0


def test_duplicate_rule_by_hand():
    # 0.3 ms at 30 kHz = 9 samples. Unit 1: 105 and 108 follow 100 within 9
    # samples (non-iterative keep_first drops both), 200 is far. Unit 2's spike
    # at 104 is close in time but belongs to another unit -> kept.
    samples = np.array([100, 104, 105, 108, 200, 209, 219])
    clusters = np.array([1, 2, 1, 1, 1, 1, 1])
    is_dup = sorter_common.find_duplicated_phy_spikes(samples, clusters, FS)
    # 209 is exactly 9 samples after 200 -> duplicate (<=); 219 is 10 after 209.
    assert is_dup.tolist() == [False, False, True, True, False, True, False]


def test_duplicate_rule_matches_spikeinterface():
    from spikeinterface.curation.curation_tools import find_duplicated_spikes

    rng = np.random.default_rng(0)
    samples, clusters = [], []
    for unit in (3, 7, 12):
        train = np.sort(rng.integers(0, 3_000_000, 2000))
        close = rng.choice(train, 60, replace=False) + rng.integers(0, 15, 60)
        samples.append(np.r_[train, close])
        clusters.append(np.full(len(train) + len(close), unit))
    samples = np.concatenate(samples)
    clusters = np.concatenate(clusters)
    shuffle = rng.permutation(len(samples))          # phy order is not per unit
    samples, clusters = samples[shuffle], clusters[shuffle]

    is_dup = sorter_common.find_duplicated_phy_spikes(samples, clusters, FS)

    censored = int(round(sorter_common.DUPLICATE_CENSORED_PERIOD_MS * 1e-3 * FS))
    for unit in (3, 7, 12):
        own = clusters == unit
        expected = np.sort(samples[own])
        expected = np.delete(expected, find_duplicated_spikes(
            expected, censored, method="keep_first"))
        assert np.array_equal(np.sort(samples[own & ~is_dup]), expected)


def _write_phy(folder, samples, clusters):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "params.py").write_text(f"sample_rate = {FS}\n")
    np.save(folder / "spike_times.npy", np.asarray(samples, dtype=np.uint64))
    np.save(folder / "spike_clusters.npy", np.asarray(clusters, dtype=np.int32))


def test_step_r_and_step_u_drop_the_same_spikes(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "nwb"))
    import add_units

    samples = [100, 104, 105, 300, 600, 900]
    clusters = [1, 2, 1, 1, 2, 2]
    _write_phy(tmp_path, samples, clusters)

    sorting = sorter_common._load_curated_sorting_from_phy(tmp_path)
    times = add_units._spike_times_by_unit(tmp_path, [1, 2])

    for unit in (1, 2):
        train = np.asarray(sorting.get_unit_spike_train(unit))
        assert np.allclose(times[unit], train / FS)
    assert sorting.get_unit_spike_train(1).tolist() == [100, 300]   # 105 dropped
    assert sorting.get_unit_spike_train(2).tolist() == [104, 600, 900]


def test_redundant_units_are_reported_not_removed(tmp_path):
    pytest.importorskip("numba")   # spikeinterface's unit comparison needs it
    recording, truth = si.generate_ground_truth_recording(
        durations=[20.0], num_units=3, sampling_frequency=FS, seed=1)
    # Phy-like folder with units 0, 1, 2 plus unit 99 = unit 0's spikes again
    # (the same neuron sorted twice), loaded the way step r loads it.
    trains = [truth.get_unit_spike_train(u) for u in truth.unit_ids]
    samples = np.concatenate(trains + [trains[0]])
    clusters = np.concatenate([np.full(len(t), i) for i, t in enumerate(trains)]
                              + [np.full(len(trains[0]), 99)])
    order = np.argsort(samples, kind="stable")
    _write_phy(tmp_path, samples[order], clusters[order])
    sorting = sorter_common._load_curated_sorting_from_phy(tmp_path)

    analyzer = si.create_sorting_analyzer(sorting, recording, format="memory", sparse=False)
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=200, seed=0)
    analyzer.compute("templates")

    wrote = sorter_common._flag_redundant_units(analyzer, tmp_path)

    assert wrote == ["redundant_units.csv", "cluster_redundant_with.tsv"]
    pairs = pd.read_csv(tmp_path / "redundant_units.csv")
    assert len(pairs) == 1
    assert {int(pairs.unit_a[0]), int(pairs.unit_b[0])} == {0, 99}
    column = pd.read_csv(tmp_path / "cluster_redundant_with.tsv", sep="\t")
    assert set(column.cluster_id) == {0, 99}
    assert len(analyzer.unit_ids) == 4                  # nothing removed


def test_no_redundant_units_rewrites_empty_files(tmp_path):
    pytest.importorskip("numba")   # spikeinterface's unit comparison needs it
    # A pair fixed in Phy must disappear from the report on the next run.
    (tmp_path / "cluster_redundant_with.tsv").write_text("cluster_id\tredundant_with\n5\t9\n")
    recording, sorting = si.generate_ground_truth_recording(
        durations=[10.0], num_units=2, sampling_frequency=FS, seed=2)
    analyzer = si.create_sorting_analyzer(sorting, recording, format="memory", sparse=False)
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=200, seed=0)
    analyzer.compute("templates")

    sorter_common._flag_redundant_units(analyzer, tmp_path)

    assert pd.read_csv(tmp_path / "redundant_units.csv").empty
    assert pd.read_csv(tmp_path / "cluster_redundant_with.tsv", sep="\t").empty
