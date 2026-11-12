"""Phy auto-curation (src/sorter/phy_autocurate.py) on a small synthetic Phy export:
the recluster replica is deterministic and its pieces are the mixture components, a
phy.log replays into the curator's actions (undo, a session closed without saving, a
merge back), the curator bank built from that log marks the kept recluster and gives
the training tables, a curated folder is left alone without a redo and curated again
from the sorter's units with one (the default of step 7 and the CLI;
AUTO_CURATE_REDO=0 opts out; no backup), step 7 curates the exports it does not sort
again, and a full run writes a consistent curation (every spike in one cluster, ids
level by level, the pieces inherit their unit's metric values, the amplitude cut,
Phy's cache cleared)."""
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
    """Synthetic training tables: the label loosely tied to snr / ISI violations, and
    a recluster kept when the cluster is big (unit 0 of make_phy, not the 300-spike
    noise unit), so the forests make the decisions the test expects."""
    clusters, splits = [], []
    for label, snr, isf in (("good", 15, -5), ("noise", 2, -1.5), ("mua", 6, -2.5)):
        for _ in range(30):
            r = {f: rng.normal() for f in pa.LABEL_FEATS}
            r.update(snr_wf=snr + rng.normal(), log_isf=isf + 0.2 * rng.normal(), dataset="synthetic",
                     key=f"c{len(clusters)}", label=label, explicit=True, depth=0, n_spikes=1000)
            clusters.append(r)
    for kept in (True, False):
        for i in range(40):
            r = dict.fromkeys(pa.SPLIT_FEATS, 0.0)
            r.update(c_log_n=rng.uniform(2.56, 3.5) if kept else rng.uniform(1.5, 2.52), dataset="synthetic",
                     split=f"a{len(splits)}", parent=f"c{i}", source="user", kept=kept, n=2)
            splits.append(r)
    folder.mkdir()
    pd.DataFrame(clusters)[pa.CLUSTER_TABLE_COLS].to_csv(folder / "training_clusters.tsv.gz", sep="\t", index=False)
    pd.DataFrame(splits)[pa.SPLIT_TABLE_COLS].to_csv(folder / "training_splits.tsv.gz", sep="\t", index=False)


class PhyLog:
    """Writes a phy.log as the phy_genzel Template GUI does (every line twice,
    ANSI colours, CRLF)."""

    def __init__(self):
        self.t = 10 * 3600.0
        self.lines = []

    def add(self, level, source, msg):
        self.t += 1.25
        hh, rest = divmod(self.t, 3600)
        mm, ss = divmod(rest, 60)
        colour = "\x1b[0m" if level == "I" else "\x1b[90m"
        line = f"{colour}{int(hh):02d}:{int(mm):02d}:{ss:06.3f} [{level}] {source:<20} {msg}\x1b[0m"
        self.lines += [line, line]

    def launch(self):
        self.add("D", "context:99", r"Initialize joblib cache dir at `D:\data\phy_export\.phy`.")
        self.add("D", "config:30", r"Load config file `C:\Users\someone\.phy\phy_config.py`.")
        self.add("D", "context:181", r"Save data to `D:\data\phy_export\.phy\spikes_per_cluster.pkl`.")  # start-up
        self.add("D", "gui:521", "Creating GUI.")

    def select(self, ids, similar=None):
        self.add("D", "supervisor:865", f"Clusters selected: {list(ids)} ({similar})")

    def recluster_n(self, n_spikes, children):
        self.add("D", "actions:67", "Prompting arguments for recluster_n")
        self.add("I", "recluster:148", f"Reclustering {n_spikes} spikes on 4 channels.")
        self.add("I", "supervisor:740", f"Assigned {n_spikes} spikes.")
        self.add("D", "context:181", r"Save data to `D:\data\phy_export\.phy\new_cluster_id.pkl`.")
        self.select(children)

    def merge(self, ids, to):
        self.add("I", "supervisor:744", f"Merge clusters {', '.join(map(str, ids))} to {to}.")
        self.select([to])

    def label(self, ids, group):
        self.add("D", "supervisor:1151", f"Move {list(ids)} to {group}.")
        self.add("I", "supervisor:755", f"Change metadata_group for clusters {', '.join(map(str, ids))} to {group}.")

    def undo(self):
        self.add("I", "supervisor:736", "Undo cluster assign.")

    def redo(self):
        self.add("I", "supervisor:742", "Redo cluster assign.")

    def undo_move(self):
        self.add("I", "supervisor:753", "Undo move.")

    def save(self):
        self.add("D", "context:181", r"Save data to `D:\data\phy_export\.phy\spikes_per_cluster.pkl`.")
        self.add("D", "context:133", "Save memcache for `phy.apps.base._get_mean_waveforms`.")

    def write(self, path):
        Path(path).write_bytes(("\r\n".join(self.lines) + "\r\n").encode())


def curation_log(n0, n1, n2):
    """A hand curation of make_phy's export (n0, n1, n2 = the spikes of its units).
    Session 1: unit 0 reclustered n=3, undone, reclustered n=2 (-> 6, 7) and labelled;
    saved; unit 2 reclustered n=2 (-> 8, 9) and merged back (-> 10), noise; saved;
    unit 1 reclustered (-> 11, 12) and the GUI closed without saving.
    Session 2: unit 1 good; reclustered (-> 13, 14), undone, redone, undone; saved."""
    log = PhyLog()
    log.launch()
    log.select([0], 2)
    log.recluster_n(n0, [3, 4, 5])
    log.undo()
    log.select([0])
    log.recluster_n(n0, [6, 7])
    log.select([6])
    log.label([6], "good")
    log.select([7])
    log.label([7], "mua")
    log.label([7], "noise")
    log.undo_move()
    log.save()
    log.select([2])
    log.recluster_n(n2, [8, 9])
    log.select([8, 9])
    log.merge([8, 9], 10)
    log.label([10], "noise")
    log.save()
    log.select([1])
    log.recluster_n(n1, [11, 12])
    log.launch()                                                       # closed without saving
    log.select([1])
    log.label([1], "good")
    log.select([1])
    log.recluster_n(n1, [13, 14])
    log.undo()
    log.redo()
    log.undo()
    log.save()
    return log


def make_curated_phy(folder, cache):
    """make_phy + the curation of curation_log: its phy.log, the spike_clusters.npy
    and cluster_group.tsv Phy would have saved (unit 0 -> 6 + 7, unit 2 -> 10)."""
    make_phy(folder)
    data = pa.PhyData(folder, cache)
    sp0 = data.spikes_of(0)
    lab, _x, _info = pa.recluster(data, sp0, np.unique(data.unit_channels[0]), 2)
    sc = np.asarray(data.spike_templates).astype(np.int32)
    sc[sp0[lab == 0]], sc[sp0[lab == 1]] = 6, 7
    sc[data.spikes_of(2)] = 10
    np.save(folder / "spike_clusters.npy", sc)
    pd.DataFrame({"cluster_id": [1, 6, 7, 10], "group": ["good", "good", "mua", "noise"]}).to_csv(
        folder / "cluster_group.tsv", sep="\t", index=False)
    curation_log(len(sp0), len(data.spikes_of(1)), len(data.spikes_of(2))).write(folder / "phy.log")
    pa._release(cache)
    return sp0, lab


def _files(folder):
    return sorted((str(p.relative_to(folder)), p.stat().st_size) for p in Path(folder).rglob("*"))


def test_recluster_is_deterministic_and_piece_j_is_component_j(tmp_path):
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    truth = make_phy(tmp_path / "phy")
    data = pa.PhyData(tmp_path / "phy", tmp_path / "cache")
    spikes = data.spikes_of(0)
    channels = np.unique(data.unit_channels[0])
    lab, x, info = pa.recluster(data, spikes, channels, 2)
    assert np.array_equal(lab, pa.recluster(data, spikes, channels, 2)[0])   # same spikes every run
    t = data.spike_samples[spikes]
    is_a = np.isin(t, truth["a"])
    assert max(np.mean(lab == is_a), np.mean(lab != is_a)) > 0.97            # the two neurons
    # the label of a spike is its mixture component (no renumbering): the order of Phy's new ids
    xr = PCA(n_components=min(10, x.shape[1]), whiten=False, random_state=0).fit_transform(x)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(xr)
    assert np.array_equal(lab, gmm.predict(xr))
    assert info["w_min"] == pytest.approx(gmm.weights_.min()) and info["bic_gain"] > 0
    # the worker's piece j of a candidate recluster is component j, with its feature row
    res = pa._unit_task(str(tmp_path / "phy"), str(tmp_path / "cache"), 0, True,
                        [dict(key="c0", depth=0, chain=[], ns=[2, 3])])
    assert np.array_equal(res["labels"][("c0", 2)], lab)
    rows = {r["key"]: r for r in res["rows"]}
    assert [rows[f"c0/n2k{j}"]["n_spikes"] for j in (0, 1)] == [int(np.sum(lab == 0)), int(np.sum(lab == 1))]
    assert {s["split"] for s in res["splits"]} == {"c0/n2", "c0/n3"} and len(res["pairs"]) == 1 + 3
    assert rows["c0"]["depth"] == 0 and rows["c0/n3k2"]["parent"] == "c0" and rows["c0/n3k2"]["split"] == "c0/n3"


def test_a_phy_log_replays_into_the_curators_actions(tmp_path):
    curation_log(430, 160, 300).write(tmp_path / "phy.log")
    ev = pa.log_events(tmp_path / "phy.log")
    assert [k for _t, k, _v in ev].count("gui") == 2 and [k for _t, k, _v in ev].count("save") == 3
    assert sum(1 for _t, k, _v in ev if k == "assign") == 5              # doubled lines count once
    r = pa.replay_log(tmp_path / "phy.log", initial_ids=[0, 1, 2],
                      initial_groups={0: "mua", 1: "good", 2: "noise"})
    acts = r["actions"]
    assert [(a["kind"], a["parents"], a["children"]) for a in acts] == [
        ("recluster_n", [0], [3, 4, 5]), ("recluster_n", [0], [6, 7]), ("recluster_n", [2], [8, 9]),
        ("merge", [8, 9], [10]), ("recluster_n", [1], [11, 12]), ("recluster_n", [1], [13, 14])]
    assert [a["undone"] for a in acts] == [True, False, False, False, False, True]   # undo; undo-redo-undo
    assert [a["saved"] for a in acts] == [True, True, True, True, False, True]       # the unsaved session
    assert [a["n_spikes"] for a in acts] == [430, 430, 300, None, 160, 160]
    assert [a["session"] for a in acts] == [1, 1, 1, 1, 1, 2]
    assert r["live"] == {1: "good", 6: "good", 7: "mua", 10: "noise"}    # as saved; 7 -> noise was undone
    assert [(lb["ids"], lb["group"], lb["undone"]) for lb in r["labels"]] == [
        ([6], "good", False), ([7], "mua", False), ([7], "noise", True), ([10], "noise", False),
        ([1], "good", False)]
    assert r["problems"] == [] and r["sessions"] == 2


def test_the_curator_bank_marks_the_kept_split_and_gives_the_training_tables(tmp_path):
    phy = tmp_path / "phy"
    sp0, lab = make_curated_phy(phy, tmp_path / "cache")
    before = _files(phy)
    bank_dir = pa.build_bank(phy, tmp_path / "banks" / "synth", n_jobs=1)
    assert _files(phy) == before                                         # the phy folder is only read
    for f in ("clusters.tsv", "splits.tsv", "pairs.tsv", "actions.tsv", "sets.json", "replay.json"):
        assert (bank_dir / f).exists(), f
    assert not (bank_dir / "cache").exists()
    sets = json.loads((bank_dir / "sets.json").read_text())
    assert sets["verified"] == {"1": True, "6": True, "7": True, "10": True}   # the replica is exact
    assert sets["sizes"]["6"] == int(np.sum(lab == 0)) and sets["sizes"]["7"] == int(np.sum(lab == 1))
    acts = pd.read_csv(bank_dir / "actions.tsv", sep="\t")
    assert list(acts.sim) == ["replica", "exact", "replica", "merge", "replica", "replica"]
    bank = pa.load_bank(bank_dir)
    C, S = bank["C"], bank["S"]
    user = S[S.source == "user"].set_index("split")
    assert list(user.index) == ["a0", "a1", "a2", "a5"]                  # a4 was never saved: dropped
    assert list(user.kept.astype(bool)) == [False, True, False, False]   # undone / kept / merged back / undone
    assert list(user.merged_back) == [False, False, True, False] and list(user.n) == [3, 2, 2, 2]
    # merged back = undo: the merged cluster 10 IS unit 2, which gets its label; candidates hang on it
    final = C[C.final].set_index("key")
    assert dict(final.label) == {"c1": "good", "c6": "good", "c7": "mua", "c2": "noise"}
    assert final.loc["c2", "id"] == 10 and "c10" not in set(C.key)
    assert dict(final.depth) == {"c1": 0, "c6": 1, "c7": 1, "c2": 0}
    cand = S[S.source == "cand"]
    assert len(cand) and set(cand.parent) == {"c2"} and not cand.kept.astype(bool).any()
    assert set(pa.LABEL_FEATS) <= set(C.columns) and set(pa.SPLIT_BASE_FEATS) <= set(bank["T"].columns)
    # the training tables: label rows = the final clusters, recluster rows = attempts + candidates
    data_dir = tmp_path / "train"
    pa.cmd_train([phy], ["synth"], tmp_path / "banks", n_jobs=1, data_dir=data_dir, cand_frac=1.0)
    assert _files(phy) == before
    clusters, splits = pa.load_training_tables(data_dir)
    assert list(clusters.columns) == pa.CLUSTER_TABLE_COLS and list(splits.columns) == pa.SPLIT_TABLE_COLS
    assert sorted(clusters.label) == ["good", "good", "mua", "noise"] and set(clusters.dataset) == {"synth"}
    assert len(splits) == 4 + len(cand) and int(splits.kept.sum()) == 1
    assert splits[pa.STACK_FEATS].notna().all().all()
    assert pd.read_csv(data_dir / "training_datasets.tsv", sep="\t").dataset.tolist() == ["synth"]
    # adding the same dataset again replaces its rows; the tables curate another export
    pa.cmd_train([phy], ["synth"], tmp_path / "banks", n_jobs=1, data_dir=data_dir, cand_frac=1.0)
    again = pa.load_training_tables(data_dir)
    assert again[0].equals(clusters) and again[1].equals(splits)
    make_phy(tmp_path / "fresh", seed=1)
    assert pa.autocurate_phy_folder(tmp_path / "fresh", n_jobs=1, data_dir=data_dir).startswith("curated")


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
    # a redo without training tables fails before the folder is touched
    files = _files(phy)
    with pytest.raises(FileNotFoundError):
        pa.autocurate_phy_folder(phy, n_jobs=1, redo=True, data_dir=tmp_path / "no_tables")
    assert _files(phy) == files


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


def test_the_settings_come_from_the_environment(monkeypatch):
    for name in ("AUTO_GOOD_MIN_CONF", "AUTO_GOOD_BOOST_NARROW", "AUTO_SPLIT_BIAS", "AUTO_AMP_CUT_SD"):
        monkeypatch.delenv(name, raising=False)
    assert pa.env_params() == {}
    monkeypatch.setenv("AUTO_GOOD_MIN_CONF", "0.5")
    monkeypatch.setenv("AUTO_SPLIT_BIAS", "1.5")
    monkeypatch.setenv("AUTO_AMP_CUT_SD", "0")
    assert pa.env_params() == {"good_min_conf": 0.5, "split_bias": 1.5, "amp_cut_sd": None}
    assert set(pa.env_params()) <= set(pa.DEFAULT_PARAMS)


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
    steps = pd.read_csv(phy / "cluster_auto_step.tsv", sep="\t").set_index("cluster_id").auto_step
    assert set(steps.index) == set(np.unique(sc))
    assert set(pd.read_csv(phy / "cluster_auto_conf.tsv", sep="\t").cluster_id) == set(np.unique(sc))
    cur = phy / pa.CURATION_DIR
    for f in ("cutting_log.txt", "cutting_log.tsv", "units.tsv", "review_list.tsv", "curation.json",
              "same_neuron_candidates.tsv", "original/cluster_snr.tsv"):
        assert (cur / f).exists(), f
    assert not (phy / ".phy" / "spikes_per_cluster.pkl").exists()
    assert json.loads((phy / ".phy" / "state.json").read_text()) == {"X": 1}
    assert json.loads((cur / "curation.json").read_text())["version"] == pa.VERSION
    # the two-neuron unit was reclustered (the keep model keeps reclusters of big clusters),
    # the 300-spike noise unit and the small unit were not
    cut = pd.read_csv(cur / "cutting_log.tsv", sep="\t", keep_default_na=False)
    assert set(cut[cut.unit == 0].depth) == {1} and set(cut[cut.unit != 0].path) == {""}
    assert all(s.startswith("L1-n") for s in cut[cut.unit == 0].step)
    assert all(steps[c].startswith("asis") for c in cut[cut.unit != 0].cluster)
    units = pd.read_csv(cur / "units.tsv", sep="\t").set_index("unit")
    assert units.loc[0, ["pkeep_n2", "pkeep_n3", "pkeep_n4"]].max() >= pa.KEEP_THR
    assert units.loc[1, ["pkeep_n2", "pkeep_n3", "pkeep_n4"]].isna().all()       # too few spikes to recluster
    assert "P(keep)" in (cur / "cutting_log.txt").read_text()
    # the path of every row rebuilds the cluster's spikes (with the spikes its amplitude cut took)
    amp_cuts = pd.read_csv(cur / "amplitude_cuts.tsv", sep="\t") if (cur / "amplitude_cuts.tsv").exists() \
        else pd.DataFrame(columns=["cluster", "noise_id", "good_id"])
    data = pa.PhyData(phy, tmp_path / "cache")
    for r in cut.itertuples():
        ids = {r.final_cluster} | set(amp_cuts[amp_cuts.cluster == r.cluster].noise_id)
        assert np.array_equal(np.sort(pa.spikes_of_path(data, r.unit, r.path)), np.nonzero(np.isin(sc, list(ids)))[0])
    # a good cluster has no spike above its mean + 4 SD amplitude left in it by the cut
    amp = np.abs(np.load(phy / "amplitudes.npy").reshape(-1)).astype(np.float64)
    for r in amp_cuts.itertuples():
        a = amp[np.isin(sc, [r.noise_id, r.good_id])]
        assert amp[sc == r.noise_id].min() > a.mean() + 4 * a.std() >= amp[sc == r.good_id].max()
    for c in groups[groups.group == "good"].cluster_id:
        assert np.sum(sc == c) > 0
    pa._release(tmp_path / "cache")
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


class FakeModels:
    """Stands in for pa.Models: P(keep) by (cluster key, n), noise for the keys in `noise`."""
    n_label_rows = n_split_rows = 0

    def __init__(self, keep, noise=()):
        self.keep, self.noise = keep, set(noise)

    def keep_proba(self, T, C):
        return np.array([self.keep.get((p, int(n)), 0.0) for p, n in zip(T.parent, T.n)])

    def label_proba(self, rows):
        return np.array([[0.1, 0.1, 0.8] if k in self.noise else [0.1, 0.8, 0.1] for k in rows.key])


def test_pieces_of_any_depth_get_their_ids_level_by_level(tmp_path, monkeypatch):
    make_phy(tmp_path / "phy")
    phy = tmp_path / "phy"
    monkeypatch.setattr(pa, "MIN_SPIKES_TO_SPLIT", 120)
    # unit 0: n=2 beats n=3, its piece 0 is reclustered again; unit 1: both pieces noise -> undone;
    # unit 2: n=3
    fake = FakeModels({("c0", 2): 0.9, ("c0", 3): 0.5, ("c0/n2k0", 2): 0.8, ("c0/n2k1", 2): 0.2,
                       ("c1", 2): 0.6, ("c2", 3): 0.7, ("c2", 4): 0.4},
                      noise=("c1/n2k0", "c1/n2k1"))
    sess = pa.Session(phy, tmp_path / "scratch", n_jobs=1)
    leaves, tree = pa.curate(sess, fake)
    assert tree["accepted"] == {"c0": (2, 0.9), "c1": (2, 0.6), "c2": (3, 0.7), "c0/n2k0": (2, 0.8)}
    assert tree["reverted"] == {"c1": 2}
    assert sorted(lf["key"] for lf in leaves) == ["c0/n2k0/n2k0", "c0/n2k0/n2k1", "c0/n2k1", "c1",
                                                  "c2/n3k0", "c2/n3k1", "c2/n3k2"]
    lab = sess.labels_of(0, "c0", 2)
    leaf = {lf["key"]: lf for lf in leaves}
    assert np.array_equal(pa.leaf_spikes(sess.data, leaf["c0/n2k1"]), sess.data.spikes_of(0)[lab == 1])
    # a higher bar (split_bias < 1): only the two best reclusters are kept
    fewer, tree2 = pa.curate(sess, fake, {"split_bias": 0.4})
    assert set(tree2["accepted"]) == {"c0", "c0/n2k0"} and len(fewer) == 5
    pred, plab = pa.leaves_to_pred(sess.data, leaves)
    assert (pred >= 0).all() and len(plab) == 7 and sorted(set(plab.values())) == ["mua", "noise"]
    pa._release(tmp_path / "scratch")
    # the same through the step: ids as Phy would give them
    monkeypatch.setattr(pa, "load_training_tables", lambda *a, **k: (pd.DataFrame({"dataset": ["fake"]}),) * 2)
    monkeypatch.setattr(pa, "Models", lambda *a, **k: fake)
    assert pa.autocurate_phy_folder(phy, n_jobs=1) == "curated: {'good': 0, 'mua': 6, 'noise': 1, 'unsorted': 0}"
    cut = pd.read_csv(phy / pa.CURATION_DIR / "cutting_log.tsv", sep="\t", keep_default_na=False)
    got = {(r.unit, r.path): (r.cluster, r.step, r.label, r.depth) for r in cut.itertuples()}
    assert got == {
        (0, "n2k1"): (3, "L1-n2", "mua", 1),                 # level 1, in unit order: 3; 4, 5, 6
        (2, "n3k0"): (4, "L1-n3", "mua", 1), (2, "n3k1"): (5, "L1-n3", "mua", 1), (2, "n3k2"): (6, "L1-n3", "mua", 1),
        (0, "n2k0/n2k0"): (7, "L2-n2", "mua", 2), (0, "n2k0/n2k1"): (8, "L2-n2", "mua", 2),   # then level 2
        (1, ""): (1, "asis(split2-reverted)", "noise", 0)}   # whole again: the sorter's id
    assert list(cut.final_cluster) == list(cut.cluster)      # no good cluster, no amplitude cut
    st = np.load(phy / "spike_templates.npy").reshape(-1)
    sc = np.load(phy / "spike_clusters.npy")
    assert {u: sorted(set(sc[st == u])) for u in (0, 1, 2)} == {0: [3, 7, 8], 1: [1], 2: [4, 5, 6]}
    steps = pd.read_csv(phy / "cluster_auto_step.tsv", sep="\t").set_index("cluster_id").auto_step
    assert steps[8] == "L2-n2" and steps[1] == "asis(split2-reverted)"
    text = (phy / pa.CURATION_DIR / "cutting_log.txt").read_text()
    assert "recluster_n 2 of n2k0: P(keep) 0.80" in text and "undone, its pieces are all noise" in text


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
    # why they disagree: one good cluster found, the other cut in pieces (which are fragments of it)
    d = pa.diagnose(pred, labels, user, user_labels)
    assert (d["u_good"], d["u_found"], d["u_fragmented"]) == (2, 1, 1)
    assert (d["p_good"], d["p_right"], d["p_fragment_of_good"]) == (4, 1, 3)
    assert pa.spike_scores(pred, labels, user, user_labels)["sp_recall_good"] == 1.0


def test_an_automatic_curation_checked_in_phy_is_learned_from(tmp_path):
    rng = np.random.default_rng(0)
    training_tables(tmp_path / "train", rng)
    phy = tmp_path / "phy"
    make_phy(phy)
    assert pa.autocurate_phy_folder(phy, n_jobs=1, data_dir=tmp_path / "train").startswith("curated")
    cur = phy / pa.CURATION_DIR
    cut = pd.read_csv(cur / "cutting_log.tsv", sep="\t", keep_default_na=False)
    pieces = cut[cut.unit == 0]
    n = int(pieces.path.iloc[0][1:].split("k")[0])
    # its cuts, as actions: the recluster of unit 0 (pieces in component order), the amplitude cuts
    auto = pa.auto_lineage(phy)
    recl = [a for a in auto["actions"] if a["kind"] == "recluster_n"]
    assert [(a["parents"], a["children"], a["n"]) for a in recl] == [([0], list(pieces.cluster), n)]
    amp = pd.read_csv(cur / "amplitude_cuts.tsv", sep="\t") if (cur / "amplitude_cuts.tsv").exists() else []
    assert [a["children"] for a in auto["actions"] if a["kind"] == "amp_cut"] == \
        [[r.noise_id, r.good_id] for r in (amp.itertuples() if len(amp) else [])]
    groups = pd.read_csv(phy / "cluster_group.tsv", sep="\t")
    assert {c: g for c, g in auto["groups"].items() if g in pa.LABELS} == dict(zip(groups.cluster_id, groups.group))
    R = pa.replay_curation(phy)                              # not opened in Phy yet: it ends where it is
    assert R["auto"] == pa.VERSION and not (R["missing"] or R["wrong"] or R["problems"][1:])
    # the check in Phy: one label changed by hand, saved
    target = int(pieces.final_cluster.iloc[0])
    new = "noise" if auto["groups"][target] != "noise" else "mua"
    log = PhyLog()
    log.launch()
    log.select([target])
    log.label([target], new)
    log.save()
    log.write(phy / "phy.log")
    groups.loc[groups.cluster_id == target, "group"] = new
    groups.to_csv(phy / "cluster_group.tsv", sep="\t", index=False)
    before = _files(phy)
    bank_dir = pa.build_bank(phy, tmp_path / "banks" / "checked", n_jobs=1)
    assert _files(phy) == before
    bank = pa.load_bank(bank_dir)
    C, S = bank["C"], bank["S"]
    assert bank["R"]["auto"] == pa.VERSION
    assert all(json.loads((bank_dir / "sets.json").read_text())["verified"].values())   # rebuilt exactly
    final = C[C.final]
    assert set(final.id) == set(groups.cluster_id) and (final.depth >= 0).all()       # with their history
    assert final.set_index("id").label[target] == new and list(final[final.explicit].id) == [target]
    made = S[S.source == "user"]                             # the recluster it made and the curator left: kept
    assert list(made.auto) == [True] and list(made.kept.astype(bool)) == [True] and list(made.n) == [n]
    assert list(made.parent) == ["c0"] and "amp_cut" not in set(S.kind)
    assert not ((S.source == "cand") & (S.parent == "c0")).any()         # unit 0 is not a final cluster
    pa.cmd_train([phy], ["checked"], tmp_path / "banks", n_jobs=1, data_dir=tmp_path / "train2")
    clusters, splits = pa.load_training_tables(tmp_path / "train2")
    assert len(clusters) == len(final) and int(splits.kept.sum()) == 1
    # the curator merges the pieces of its recluster together again: that recluster was not kept
    phy2 = tmp_path / "phy2"
    make_phy(phy2)
    pa.autocurate_phy_folder(phy2, n_jobs=1, data_dir=tmp_path / "train", params={"amp_cut_sd": None})
    cut = pd.read_csv(phy2 / pa.CURATION_DIR / "cutting_log.tsv", sep="\t", keep_default_na=False)
    ids = list(cut[cut.unit == 0].final_cluster)
    sc = np.load(phy2 / "spike_clusters.npy")
    merged = int(sc.max()) + 1
    sc[np.isin(sc, ids)] = merged
    np.save(phy2 / "spike_clusters.npy", sc)
    groups = pd.read_csv(phy2 / "cluster_group.tsv", sep="\t")
    groups = pd.concat([groups[~groups.cluster_id.isin(ids)],
                        pd.DataFrame({"cluster_id": [merged], "group": ["mua"]})], ignore_index=True)
    groups.to_csv(phy2 / "cluster_group.tsv", sep="\t", index=False)
    log = PhyLog()
    log.launch()
    log.select(ids)
    log.merge(ids, merged)
    log.label([merged], "mua")
    log.save()
    log.write(phy2 / "phy.log")
    bank = pa.load_bank(pa.build_bank(phy2, tmp_path / "banks" / "merged", n_jobs=1))
    made = bank["S"][bank["S"].source == "user"]
    assert list(made.auto) == [True] and list(made.kept.astype(bool)) == [False] and list(made.merged_back) == [True]
    whole = bank["C"][bank["C"].key == "c0"].iloc[0]                     # the merged cluster is unit 0 again
    assert whole.final and whole.label == "mua" and whole.id == merged and whole.depth == 0
    # an automatic curation of v1 (no paths in its log) cannot be rebuilt: replayed as before
    cut.drop(columns="path").to_csv(phy2 / pa.CURATION_DIR / "cutting_log.tsv", sep="\t", index=False)
    assert pa.auto_lineage(phy2) is None and pa.replay_curation(phy2)["auto"] is None


def _eap(cap=0.05, na_sd=4.0, k=0.3, k_delay=16, k_sd=9.0, n=90, n_before=30):
    """A model extracellular action potential at 30 kHz, trough = -1 at sample n_before: a small
    positive (capacitive) peak, the Na+ trough, a longer positive K+ phase (Gold et al. 2006)."""
    t = np.arange(n) - n_before
    return (cap * np.exp(-0.5 * ((t + 3.2 * na_sd) / (0.8 * na_sd)) ** 2) - np.exp(-0.5 * (t / na_sd) ** 2)
            + k * np.exp(-0.5 * ((t - k_delay) / k_sd) ** 2))


def test_a_waveform_is_measured_against_the_rules_of_an_action_potential():
    profile = np.array([1.0, 0.6, 0.3, 0.1])
    f = pa.ap_features(100 * np.outer(_eap(), profile), FS, 30)
    assert set(f) == set(pa.AP_FEATS)
    assert f["ap_viol"] == pytest.approx(0, abs=0.02)                    # follows every rule
    assert 0.03 < f["ap_cap_ratio"] < 0.06 and 0.35 <= f["ap_na_w25"] <= 0.6
    assert f["ap_k_ratio"] == pytest.approx(0.3, abs=0.03) and f["ap_k_t"] == pytest.approx(16 / 30, abs=0.07)
    assert f["ap_k_na_dur"] > 1 and f["ap_n_neg"] == 1                   # the K+ phase is the longer one
    assert f["ap_ch_rank1"] == pytest.approx(1) and f["ap_ch_lag"] == 0  # one shape on every channel
    # the width at 25 % of the Na+ peak is the paper's measure: a Gaussian of SD s is 3.33 s wide there
    for sd in (3.0, 6.0):
        assert pa.ap_features(np.outer(_eap(cap=0, k=0, na_sd=sd), profile), FS, 30)["ap_na_w25"] == \
            pytest.approx(2 * sd * np.sqrt(2 * np.log(4)) / 30, abs=0.02)
    # what is not an action potential: each breaks a rule
    t = np.arange(90) - 30
    w_shape = _eap() - 0.8 * np.exp(-0.5 * ((t - 14) / 3.0) ** 2)        # a "W": two negative peaks
    fw = pa.ap_features(np.outer(w_shape, profile), FS, 30)
    assert fw["ap_n_neg"] == 2 and fw["ap_viol"] >= 1
    assert pa.ap_features(np.outer(_eap(cap=0.9), profile), FS, 30)["ap_viol"] > 1       # capacitive peak
    assert pa.ap_features(np.outer(_eap(na_sd=0.5, k=0), profile), FS, 30)["ap_viol"] >= 1    # too narrow
    assert pa.ap_features(np.outer(_eap(na_sd=20, k=0), profile), FS, 30)["ap_viol"] >= 1     # too wide
    assert pa.ap_features(np.outer(-_eap(), profile), FS, 30)["ap_viol"] >= 1                 # positive
    two = np.stack([_eap(), np.roll(_eap(), 9), 0 * _eap(), 0 * _eap()], 1)  # another shape per channel
    f2 = pa.ap_features(two, FS, 30)
    assert f2["ap_ch_lag"] == pytest.approx(0.3) and f2["ap_ch_rank1"] < 0.8 and f2["ap_viol"] >= 1
    assert all(np.isnan(v) for v in pa.ap_features(None, FS, 30).values())
    # two pieces: one waveform at two amplitudes (one neuron) or two waveforms (two neurons)
    a = 100 * np.outer(_eap(), profile)
    same = pa.ap_pair_features(a, 0.6 * a, f, pa.ap_features(0.6 * a, FS, 30), FS, 30)
    assert same["wf_sin"] == pytest.approx(0, abs=1e-6) and same["wf_amp_ratio"] == pytest.approx(0.6)
    assert same["wf_d_w25"] == pytest.approx(0, abs=1e-9) and same["wf_prof_cos"] == pytest.approx(1)
    b = 100 * np.outer(_eap(na_sd=6.5, k=0.1), profile)                  # other ion channels: wider, less K+
    other = pa.ap_pair_features(a, b, f, pa.ap_features(b, FS, 30), FS, 30)
    assert other["wf_shape_sin"] > 0.3 and other["wf_d_w25"] > 0.2 and other["wf_d_k"] > 0.1
    assert other["wf_prof_cos"] == pytest.approx(1, abs=0.02)            # at the same place
    c = 100 * np.outer(_eap(), profile[::-1])                            # the same cell type elsewhere
    moved = pa.ap_pair_features(a, c, f, pa.ap_features(c, FS, 30), FS, 30)
    assert moved["wf_shape_sin"] == pytest.approx(0, abs=1e-6) and moved["wf_sin"] > 0.5
    assert all(np.isnan(v) for v in pa.ap_pair_features(a, None, f, f, FS, 30).values())


def test_the_keep_model_sees_the_action_potential_of_every_piece(tmp_path, monkeypatch):
    make_phy(tmp_path / "phy")
    phy, cache = tmp_path / "phy", tmp_path / "cache"
    before = _files(phy)
    data = pa.PhyData(phy, cache)
    c = data.waveforms()                                                 # one pass over the recording
    assert _files(phy) == before and data.wf_window() == (30, 60)
    assert c["W"].shape == (data.n_spikes, 90, N_CH) and c["W"].dtype == np.int16
    raw = np.fromfile(phy / "recording.dat", np.int16).reshape(-1, N_CH)
    for unit in data.units:                                              # a unit's snippets: its spikes, in order
        sp, w = data.unit_waveforms(unit)
        assert np.array_equal(sp, data.spikes_of(unit))
        for i in (0, len(sp) // 2, len(sp) - 1):
            t = int(data.spike_samples[sp[i]])
            assert np.array_equal(w[i], raw[t - 30:t + 60])
    wf = pa.Waveforms(data, [0])
    half = data.spikes_of(0)[::2]
    t = np.asarray(data.spike_samples[half])
    assert np.allclose(wf.mean(half), np.mean([raw[s - 30:s + 60] for s in t], 0))
    assert wf.mean(half[:5]) is None and wf.mean(data.spikes_of(1)) is None   # too few; another unit
    pa._release(cache / "nothing")
    with monkeypatch.context() as m:                                     # a cache that is there is used
        m.setattr(pa, "build_waveforms", lambda *a, **k: pytest.fail("the recording is read again"))
        assert pa.PhyData(phy, cache).waveforms() is not None
    # the pieces of the two-neuron unit: two action potentials on other channels; of the one-neuron
    # unit: one action potential at two amplitudes
    monkeypatch.setattr(pa, "MIN_SPIKES_TO_SPLIT", 120)
    sess = pa.Session(phy, cache, n_jobs=1)
    assert sess.has_waveforms
    sess.ensure([dict(unit=u, key=f"c{u}", depth=0, chain=[], ns=[2]) for u in (0, 1)], want_units=[0, 1, 2])
    C, S, P = sess.tables()
    rows = C.set_index("key")
    assert rows.loc[["c0", "c0/n2k0", "c0/n2k1", "c1", "c1/n2k0", "c1/n2k1"], pa.AP_FEATS].notna().all().all()
    assert rows.loc[["c0/n2k0", "c0/n2k1", "c1"], "ap_viol"].max() < 0.5 < rows.loc["c2", "ap_viol"]  # c2: noise
    pair = P.set_index("split")
    assert pair.loc["c0/n2", "wf_sin"] > 0.5 > pair.loc["c0/n2", "wf_prof_cos"]
    assert pair.loc["c0/n2", "wf_shape_sin"] < 0.2                       # the same cell type
    assert pair.loc["c1/n2", "wf_sin"] < 0.2 and pair.loc["c1/n2", "wf_prof_cos"] > 0.98
    T = pa.split_table(pa.add_ctx(C, P), S, P).set_index("split")
    assert set(pa.AP_SPLIT_FEATS) <= set(pa.SPLIT_FEATS) and set(pa.AP_SPLIT_FEATS) <= set(T.columns)
    assert T[pa.AP_SPLIT_FEATS].notna().all().all()
    assert np.allclose(T.ap_viol_gain, T.c_ap_viol - T.pmin_ap_viol)     # is a piece more of a spike
    pa._release(cache)


def test_without_the_recording_the_keep_model_decides_without_action_potentials(tmp_path):
    rng = np.random.default_rng(0)
    make_phy(tmp_path / "phy")
    phy = tmp_path / "phy"
    training_tables(tmp_path / "train", rng)
    clusters, splits = pa.load_training_tables(tmp_path / "train")
    sess = pa.Session(phy, tmp_path / "with", n_jobs=1)
    sess.ensure([dict(unit=0, key="c0", depth=0, chain=[], ns=[2])], want_units=[0])
    C, S, P = sess.tables()
    Cx = pa.add_ctx(C, P)
    models = pa.Models(clusters, splits)
    models.keep_proba(pa.split_table(Cx, S, P), Cx)
    assert list(models._keep) == [True] and models.keep_forest()[1] == pa.SPLIT_FEATS
    off = pa.Models(clusters, splits, {"use_ap": False})                 # AUTO_USE_AP=0
    off.keep_proba(pa.split_table(Cx, S, P), Cx)
    assert list(off._keep) == [False] and off.keep_forest()[1] == pa.SPLIT_FEATS_NO_AP
    old = pa.Models(clusters, splits.drop(columns=pa.AP_SPLIT_FEATS))    # training tables of before 2.1
    assert not old.trained_ap and old.keep_forest()[1] == pa.SPLIT_FEATS_NO_AP
    pa._release(tmp_path / "with")
    quiet = pa.Session(phy, tmp_path / "off", n_jobs=1, waveforms=False)     # then the recording is not read
    assert not quiet.has_waveforms and not (tmp_path / "off" / "waveforms.npy").exists()
    pa._release(tmp_path / "off")
    (phy / "recording.dat").unlink()                                     # the export alone
    sess = pa.Session(phy, tmp_path / "without", n_jobs=1)
    assert not sess.has_waveforms
    sess.ensure([dict(unit=0, key="c0", depth=0, chain=[], ns=[2])], want_units=[0])
    C, S, P = sess.tables()
    assert C[pa.AP_FEATS].isna().all().all() and P[pa.AP_PAIR_FEATS].isna().all().all()
    Cx = pa.add_ctx(C, P)
    models = pa.Models(clusters, splits)
    p = models.keep_proba(pa.split_table(Cx, S, P), Cx)
    assert list(models._keep) == [False] and len(p) == 1                 # the forest fit without them
    pa._release(tmp_path / "without")
    assert pa.autocurate_phy_folder(phy, n_jobs=1, data_dir=tmp_path / "train").startswith("curated")
    cur = phy / pa.CURATION_DIR
    assert json.loads((cur / "curation.json").read_text())["action_potentials"] is False
    assert "WITHOUT the action potentials" in (cur / "cutting_log.txt").read_text()
