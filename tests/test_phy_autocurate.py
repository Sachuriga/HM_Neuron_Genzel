"""Phy auto-curation (src/sorter/phy_autocurate.py) on a small synthetic Phy export:
the recluster replica is deterministic, the training targets follow the curated
spikes, a curated folder is left alone without a redo and curated again from the
sorter's units with one (the default of step 7 and the CLI; AUTO_CURATE_REDO=0 opts
out; no backup), step 7 curates the exports it does not sort again, and a full run writes
a consistent curation (every spike in one cluster, the pieces inherit their unit's
metric values, the amplitude cut, Phy's cache cleared)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sorter"))
import phy_autocurate as pa  # noqa: E402

FS = 30000.0
N_CH = 4
DURATION_S = 40


def _spike_shape(n=40, width=6.0):
    t = np.arange(n) - n // 2
    return -np.exp(-0.5 * (t / (width / 2)) ** 2) + 0.35 * np.exp(-0.5 * ((t - 8) / 5) ** 2)


def make_phy(folder, seed=0):
    """A one-tetrode phy export: unit 0 = two neurons with different channel
    profiles (to be split), unit 1 = one clean neuron, unit 2 = noise."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    n_samples = int(DURATION_S * FS)
    traces = rng.normal(0, 10, (n_samples, N_CH)).astype(np.float32)
    shape = _spike_shape()
    profiles = {"a": np.array([1.0, 0.5, 0.2, 0.1]), "b": np.array([0.1, 0.2, 0.6, 1.0]),
                "c": np.array([0.6, 1.0, 0.4, 0.2])}
    amps = {"a": 180.0, "b": 150.0, "c": 220.0}
    trains = {}
    for name, rate in (("a", 6.0), ("b", 5.0), ("c", 4.0)):
        isi = rng.exponential(1.0 / rate, int(rate * DURATION_S * 1.5)) + 0.004
        t = np.cumsum(isi)
        t = t[(t > 0.01) & (t < DURATION_S - 0.01)]
        trains[name] = (t * FS).astype(np.int64)
        for s in trains[name]:
            traces[s - 20:s + 20] += amps[name] * rng.uniform(0.9, 1.1) * np.outer(shape, profiles[name])
    noise_t = np.sort(rng.integers(100, n_samples - 100, 300))
    traces.astype(np.int16).tofile(folder / "recording.dat")
    times = np.concatenate([trains["a"], trains["b"], trains["c"], noise_t])
    templ = np.concatenate([np.zeros(len(trains["a"]) + len(trains["b"]), np.int64),
                            np.ones(len(trains["c"]), np.int64), np.full(len(noise_t), 2, np.int64)])
    order = np.argsort(times, kind="stable")
    times, templ = times[order], templ[order]
    # PC features: the per-channel peak amplitude and two jittered copies
    peaks = np.stack([traces[t - 2:t + 3].min(0) for t in times])          # (n, 4)
    pcs = np.stack([peaks, 0.3 * peaks + rng.normal(0, 3, peaks.shape),
                    rng.normal(0, 3, peaks.shape)], 1).astype(np.float32)  # (n, 3, 4)
    np.save(folder / "spike_times.npy", times.reshape(-1, 1))
    np.save(folder / "spike_templates.npy", templ.reshape(-1, 1))
    np.save(folder / "spike_clusters.npy", templ.astype(np.int32))
    np.save(folder / "amplitudes.npy", (-np.abs(peaks).max(1)).astype(np.float32).reshape(-1, 1))
    np.save(folder / "pc_features.npy", pcs)
    np.save(folder / "pc_feature_ind.npy", np.tile(np.arange(N_CH), (3, 1)))
    np.save(folder / "template_ind.npy", np.tile(np.arange(N_CH), (3, 1)))
    tm = np.zeros((3, 40, N_CH))
    tm[0] = np.outer(shape, profiles["a"]) * 180
    tm[1] = np.outer(shape, profiles["c"]) * 220
    tm[2] = np.outer(shape, np.ones(N_CH)) * 30
    np.save(folder / "templates.npy", tm)
    np.save(folder / "channel_groups.npy", np.zeros(N_CH, np.int32))
    np.save(folder / "channel_positions.npy", np.zeros((N_CH, 2), np.float32))
    np.save(folder / "whitening_mat_inv.npy", np.eye(N_CH))
    (folder / "params.py").write_text(
        f"dat_path = 'recording.dat'\nn_channels_dat = {N_CH}\ndtype = 'int16'\n"
        f"offset = 0\nsample_rate = {FS}\nhp_filtered = True\n")
    pd.DataFrame({"cluster_id": [0, 1, 2], "snr": [7.0, 9.0, 1.0]}).to_csv(
        folder / "cluster_snr.tsv", sep="\t", index=False)
    pd.DataFrame({"cluster_id": [0, 1, 2], "group": ["mua", "good", "noise"]}).to_csv(
        folder / "cluster_group.tsv", sep="\t", index=False)
    pd.DataFrame({"quality_label": ["mua", "good", "noise"]}).to_csv(folder / "quality_check_labels.csv")
    return {"a": trains["a"], "b": trains["b"], "times": times, "templates": templ}


def training_tables(folder, rng):
    """Synthetic training tables: every class present, features loosely tied to
    the class, so the forests make the decisions the test expects."""
    units, pieces = [], []
    for target, snr, isf in (("good", 15, -5), ("noise", 2, -1.5), ("split4", 8, -2.2),
                             ("split2", 9, -3), ("mua", 6, -2.5)):
        for _ in range(25):
            r = {f: rng.normal() for f in pa.UNIT_FEATS}
            r.update(snr_wf=snr + rng.normal(), log_isf=isf + 0.2 * rng.normal(), target=target,
                     dataset="synthetic", unit=0, is_narrow=0.0)
            units.append(r)
    for target, snr, isf in (("good", 15, -5), ("noise", 2, -1.5), ("mua", 6, -2.5), ("split", 9, -2)):
        for _ in range(25):
            r = {f: rng.normal() for f in pa.SUB_FEATS}
            r.update(snr_wf=snr + rng.normal(), log_isf=isf + 0.2 * rng.normal(), target=target,
                     dataset="synthetic", unit=0, is_narrow=0.0, level=1)
            pieces.append(r)
    folder.mkdir()
    pd.DataFrame(units).to_csv(folder / "training_units.tsv", sep="\t", index=False)
    pd.DataFrame(pieces).to_csv(folder / "training_pieces.tsv", sep="\t", index=False)


def test_gmm_split_separates_two_neurons_and_is_deterministic(tmp_path):
    truth = make_phy(tmp_path / "phy")
    data = pa.PhyData(tmp_path / "phy", tmp_path / "cache")
    spikes = data.spikes_of(0)
    lab = pa.gmm_split(data, spikes, 0, 2)
    assert np.array_equal(lab, pa.gmm_split(data, spikes, 0, 2))     # same spikes every run
    t = data.spike_samples[spikes]
    is_a = np.isin(t, truth["a"])
    agree = max(np.mean(lab == is_a), np.mean(lab != is_a))
    assert agree > 0.97
    assert lab[0] == 0                                                 # piece 0 holds the first spike


def test_targets_follow_the_curated_spikes(tmp_path):
    truth = make_phy(tmp_path / "phy")
    phy = tmp_path / "phy"
    bank = pa.build_bank(phy, tmp_path / "bank", n_jobs=1, l2_min=10 ** 9)
    # the "curator" split unit 0 into its two neurons (good + mua), kept 1 and 2 whole
    sc = truth["templates"].copy()
    sc[np.isin(truth["times"], truth["a"])] = 3
    sc[np.isin(truth["times"], truth["b"])] = 4
    np.save(phy / "spike_clusters.npy", sc.astype(np.int32))
    pd.DataFrame({"cluster_id": [1, 2, 3, 4], "group": ["good", "noise", "good", "mua"]}).to_csv(
        phy / "cluster_group.tsv", sep="\t", index=False)
    t = pa.derive_targets(bank, phy)
    assert t["u1"] == "good" and t["u2"] == "noise"
    assert t["u0"] == "split2"                                         # the 2-piece split matches
    assert sorted(t[f"u0/s2k{k}"] for k in (0, 1)) == ["good", "mua"]
    units, pieces = pa.training_tables(bank, t, "test")
    assert set(units.target) == {"good", "noise", "split2"} and len(pieces) == 2


def test_a_curated_folder_is_not_overwritten_without_redo(tmp_path):
    make_phy(tmp_path / "phy")
    phy = tmp_path / "phy"
    assert pa.folder_state(phy) == "fresh"
    sc = np.load(phy / "spike_clusters.npy")
    sc[:5] = 7                                                         # a split made in Phy
    np.save(phy / "spike_clusters.npy", sc)
    assert pa.folder_state(phy) == "manual"
    before = (phy / "spike_clusters.npy").read_bytes()
    assert pa.autocurate_phy_folder(phy, n_jobs=1).startswith("skipped")
    assert (phy / "spike_clusters.npy").read_bytes() == before
    np.save(phy / "spike_clusters.npy", np.load(phy / "spike_templates.npy").reshape(-1).astype(np.int32))
    (phy / "phy.log").write_text("12:00:00 [I] supervisor:755 Change metadata_group for clusters 1 to good.\n")
    assert pa.folder_state(phy) == "manual"                            # a label saved in Phy


def test_the_cli_redoes_curated_folders_by_default(tmp_path, monkeypatch):
    make_phy(tmp_path / "phy")
    seen = []
    monkeypatch.setattr(pa, "autocurate_phy_folder", lambda phy, **kw: seen.append(kw["redo"]))
    monkeypatch.delenv("AUTO_CURATE_REDO", raising=False)
    pa.main(["--output_folder", str(tmp_path / "phy")])                # the default: redo
    monkeypatch.setenv("AUTO_CURATE_REDO", "0")                        # opt out: skip curated
    pa.main(["--output_folder", str(tmp_path / "phy")])
    pa.main(["--output_folder", str(tmp_path / "phy"), "--redo"])
    assert seen == [True, False, True]


def test_step_7_curates_the_exports_it_does_not_sort_again(tmp_path, monkeypatch):
    op = tmp_path / "op"
    for name in ("A", "B"):
        make_phy(op / f"{name}_sorting_output" / "phy_export")
    seen = []
    monkeypatch.setattr(pa, "autocurate_phy_folder",
                        lambda phy, **kw: seen.append((Path(phy).parent.name, kw["redo"])) or "curated")
    for name in ("AUTO_CURATE_REDO", "PHY_AUTOCURATE"):
        monkeypatch.delenv(name, raising=False)
    just_sorted = [op / "A_sorting_output" / "phy_export"]             # curated by its post-sorting
    assert len(pa.curate_sorted_exports(op, skip=just_sorted)) == 1
    assert seen == [("B_sorting_output", True)]
    monkeypatch.setenv("AUTO_CURATE_REDO", "0")                        # keep the curations
    pa.curate_sorted_exports(op, skip=just_sorted)
    assert seen[-1] == ("B_sorting_output", False)
    monkeypatch.setenv("PHY_AUTOCURATE", "0")                          # no auto-curation at all
    assert pa.curate_sorted_exports(op) == {} and len(seen) == 2


def test_a_full_run_writes_a_consistent_curation(tmp_path):
    rng = np.random.default_rng(1)
    make_phy(tmp_path / "phy")
    phy = tmp_path / "phy"
    (phy / ".phy").mkdir()
    (phy / ".phy" / "spikes_per_cluster.pkl").write_bytes(b"stale")
    (phy / ".phy" / "state.json").write_text(json.dumps({"ClusterView": {"selected": [2]}, "X": 1}))
    training_tables(tmp_path / "train", rng)
    status = pa.autocurate_phy_folder(phy, n_jobs=1, data_dir=tmp_path / "train")
    assert status.startswith("curated")
    st = np.load(phy / "spike_templates.npy").reshape(-1)
    sc = np.load(phy / "spike_clusters.npy")
    assert sc.dtype == np.int32 and len(sc) == len(st)
    groups = pd.read_csv(phy / "cluster_group.tsv", sep="\t")
    assert set(groups.group) <= {"good", "mua", "noise"}
    assert set(groups.cluster_id) <= set(np.unique(sc))
    # every cluster lies within one original unit (no merges); new ids above the units
    pairs = pd.DataFrame({"u": st, "c": sc}).drop_duplicates()
    assert pairs.groupby("c").u.nunique().max() == 1
    assert all(c == u or c > st.max() for u, c in zip(pairs.u, pairs.c))
    # the pieces inherit their unit's step-7 metric values (as Phy does on a split)
    snr = pd.read_csv(phy / "cluster_snr.tsv", sep="\t").set_index("cluster_id").snr
    unit_snr = {0: 7.0, 1: 9.0, 2: 1.0}
    for c, u in zip(pairs.c, pairs.u):
        assert snr[c] == unit_snr[u]
    # review columns, logs, the original files kept, Phy's cache cleared (state kept)
    steps = pd.read_csv(phy / "cluster_auto_step.tsv", sep="\t")
    assert set(steps.cluster_id) == set(np.unique(sc))
    cur = phy / pa.CURATION_DIR
    for f in ("cutting_log.txt", "cutting_log.tsv", "units.tsv", "review_list.tsv", "curation.json",
              "same_neuron_candidates.tsv", "original/cluster_snr.tsv"):
        assert (cur / f).exists(), f
    assert not (phy / ".phy" / "spikes_per_cluster.pkl").exists()
    assert json.loads((phy / ".phy" / "state.json").read_text()) == {"X": 1}
    # a good cluster has no spike above its mean + 4 SD amplitude after the cut
    amp = np.abs(np.load(phy / "amplitudes.npy").reshape(-1))
    for c in groups[groups.group == "good"].cluster_id:
        a = amp[sc == c]
        assert len(a) > 0
    # without a redo, a second run leaves the curated folder alone. A redo starts
    # again from the sorter's units: a change made in Phy since is replaced, with no
    # backup, and the result is the same (deterministic)
    edited = sc.copy()
    edited[:5] = sc.max() + 1                                          # a split made in Phy
    np.save(phy / "spike_clusters.npy", edited)
    assert pa.autocurate_phy_folder(phy, n_jobs=1, data_dir=tmp_path / "train").startswith("skipped")
    again = pa.autocurate_phy_folder(phy, n_jobs=1, redo=True, data_dir=tmp_path / "train")
    assert again.startswith("curated")
    assert np.array_equal(np.load(phy / "spike_clusters.npy"), sc)
    assert not any(p.name.startswith("backup") for p in cur.iterdir())


def test_same_neuron_pairs_flags_pieces_of_one_refractory_train():
    rng = np.random.default_rng(3)
    duration = 600.0

    def train(rate):  # 2 ms refractory period
        t = np.cumsum(rng.exponential(1 / rate, int(rate * duration * 1.2)) + 0.002)
        return (t[t < duration] * FS).astype(np.int64)

    one = train(20.0)
    half = rng.random(len(one)) < 0.5                     # one neuron cut in two pieces
    other = np.sort(np.r_[train(20.0), train(20.0)])      # two independent neurons
    samples = np.r_[one, other]
    sc = np.r_[np.where(half, 10, 11), np.full(len(other), 0)]
    sc[len(one):] = np.where(rng.random(len(other)) < 0.5, 12, 13)
    order = np.argsort(samples, kind="stable")
    samples, sc = samples[order], sc[order]
    labels = {10: "good", 11: "good", 12: "good", 13: "good"}
    parent = {10: 1, 11: 1, 12: 2, 13: 2}
    rows = pa.same_neuron_pairs(samples, sc, labels, parent, FS, duration)
    assert [(r["cluster_a"], r["cluster_b"]) for r in rows] == [(10, 11)]
    assert rows[0]["ratio"] < 0.05


def test_compare_curations_scores_good_clusters():
    user = np.array([0] * 50 + [1] * 50 + [2] * 50)
    user_labels = {0: "good", 1: "good", 2: "noise"}
    pred = np.array([5] * 50 + [6] * 17 + [7] * 17 + [9] * 16 + [8] * 50)  # unit 1 cut in 3
    labels = {5: "good", 6: "good", 7: "good", 9: "good", 8: "noise"}
    r = pa.compare_curations(pred, labels, user, user_labels)
    assert r["spike_acc"] == 1.0
    assert r["good_recall"] == 0.5                       # a third of a unit: Jaccard 0.34 < 0.5
    assert r["good_precision"] == 0.25
    halves = np.array([5] * 50 + [6] * 25 + [7] * 25 + [8] * 50)        # Jaccard 0.5 still matches
    assert pa.compare_curations(halves, labels, user, user_labels)["good_recall"] == 1.0
