"""
Phy auto-curation: recluster + cut + label the units of a tetrode Phy export,
the way they are curated by hand in Phy, with models trained on what the
curator did in Phy (the action history every curation leaves in phy.log).

What it does to a phy_export (starting from the sorter's original units):
  1. every cluster of >= 300 spikes - a sorter unit first, then every piece
     that is kept - is reclustered with the phy_genzel fork's recluster_n
     (Gaussian mixture on the cluster's PC features, the exact computation of
     the Template GUI action) with n = 2, 3 and 4. The "keep" model gives each
     of the three splits the probability that the curator would have kept it;
     the best one is kept if that is >= 0.30 (divided by AUTO_SPLIT_BIAS) and
     its pieces are tried in the same way, up to 4 reclusters deep. Otherwise
     the cluster stays as it is;
  2. every cluster that is left is labelled good / mua / noise by the "label"
     model. Interneurons (narrow waveforms, trough-to-peak <= 0.425 ms) get
     their P(good) weighted x1.5: fast-spiking cells are kept with more short
     ISIs than a pyramidal cell may have. A 'good' call below 0.4 confidence
     (AUTO_GOOD_MIN_CONF) is left unsorted;
  3. a split whose pieces are all noise is undone (the cluster is noise
     whole). No merges;
  4. in every good cluster, spikes with |amplitude| > mean + 4 SD are cut off as
     a separate noise cluster (amplitude artefacts);
  5. the result is written as Phy would save it: spike_clusters.npy,
     cluster_group.tsv, cluster_<metric>.tsv (new clusters inherit their unit's
     values, as in Phy), cluster_auto_conf.tsv / cluster_auto_step.tsv (two
     review columns), and a log of every cut in phy_export/_auto_curation/.

The two models are random forests fit at run time on the training tables in
phy_autocurate_data/, which are made from hand curations and their phy.log:
  training_clusters.tsv.gz  the final clusters of the hand curations with the
                            curator's label                    -> the label model
  training_splits.tsv.gz    every recluster_n the curator made, kept or undone
                            (undo, or all pieces merged back), and the
                            reclusters (n = 2, 3, 4) the curator did NOT make
                            on a final cluster (= not kept)    -> the keep model
The log is replayed into the curator's actions (replay_log), the spikes of
every cluster the curator saw are rebuilt from them (the recluster replica is
exact) and checked against the saved curation (build_bank). Add a curation and
re-train with `python phy_autocurate.py train --curated <phy_export> ...`.

Runs as part of step 7 (sorting.py): at the end of a new sort's post-sorting
(sorter_common.analyze_and_export), and on the export of every recording step 7
finds already sorted (curate_sorted_exports). PHY_AUTOCURATE=0 turns it off.
On its own:

    python phy_autocurate.py --output_folder <op|sorting_output|phy_export> [--n_jobs 4]

A phy_export that was already curated (by hand or by this script) is curated
again from the sorter's units and its curation is replaced; there is no backup.
AUTO_CURATE_REDO=0 leaves curated exports alone instead (--redo overrides it).

Needs only the export: the waveform features come from the sorter's templates
(templates.npy) and the per-spike PC features, the noise level from the exported
snr column; the recording (params.py dat_path) is read for its length only.

numpy / scipy / pandas / scikit-learn / joblib only (no phy, no spikeinterface).
"""

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings
import zlib
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_type_metrics as ctm  # noqa: E402

VERSION = "2.0"
DATA_DIR = Path(__file__).resolve().parent / "phy_autocurate_data"
CURATION_DIR = "_auto_curation"

# ---- reclustering: the phy_genzel ReclusterPlugin (recluster_n) -------------
GMM_MAX_SPIKES_FIT = 20000   # the mixture is fit on at most this many spikes
GMM_MAX_DIMS = 10            # PCA dimensions before the mixture
RECLUSTER_NS = (2, 3, 4)     # the n tried on every cluster
MIN_SPIKES_TO_SPLIT = 300    # smaller clusters are never reclustered
MAX_DEPTH = 4                # reclusters on top of each other, at most
# ---- features ---------------------------------------------------------------
N_ISOLATION_SPIKES = 500     # spikes per cluster in the isolation distance (as SI)
N_PAIR_SPIKES = 20000        # spikes per piece in the statistics of a pair of pieces
N_NOISE_CHUNKS = 40          # 1-s chunks for the per-channel noise (MAD; only without an snr column)
SHAPE_T_MS = np.arange(-0.6, 1.41, 0.1)   # where the template is sampled for the models (ms from the trough)
ISI_THRESHOLD_MS = 1.5
DUPLICATE_MS = 0.3
N_TIME_BINS = 60
# ---- decisions ----------------------------------------------------------------
KEEP_THR = 0.30              # a recluster is kept if P(keep) >= this / split_bias
AMP_CUT_SD = 4.0             # amplitude cut: |amp| > mean + 4 SD -> noise
GOOD_MIN_CONF = 0.4          # a 'good' call below this confidence -> unsorted
GOOD_BOOST_NARROW = 1.5      # P(good) weight of narrow-waveform (interneuron) clusters
N_TREES = 300
CAND_FRAC = 0.3              # share of the never-made candidate reclusters kept in the training table
TABLE_DIGITS = 5             # significant digits of the floats in the training tables
LABELS = ("good", "mua", "noise")

# The waveform: the unit's template (templates.npy, uV) on its best channel, around the
# trough - shape metrics and the shape itself (template_features). A piece of a unit
# shares its unit's template; how its own waveform differs is in the PC features.
TEMPLATE_FEATS = [
    "trough_to_peak", "wf_ptp_ratio2", "wf_pos_neg_ratio", "wf_flatness", "tpl_ptp",
    "tpl_ptp_ratio4", "tpl_spread", "tpl_trough_hw", "tpl_after_hw", "tpl_before_ratio",
    "tpl_after_ratio", "tpl_repol_slope", "tpl_recov_slope", "tpl_n_neg", "tpl_n_pos",
    "tpl_energy_pre", "tpl_trough_pos",
] + [f"tpl_s{i:02d}" for i, t in enumerate(SHAPE_T_MS) if abs(t) > 1e-9]   # the trough itself is -1
# A piece in PC space (pc_features.npy), relative to the spike set it was cut from and
# to its sibling pieces (pc_relative).
PC_FEATS = ["pc_d_parent", "pc_d_sib", "pc_spread_rel", "pc_d_ch_max", "pc_d_ch_min"]
UNIT_FEATS = [
    "snr_wf", "amp_iqr_rel", "amp_p90_rel", "log_n", "log_fr", "log_isr", "log_isf",
    "log_iso", "log_lr", "acg_tau_rise", "presence", "rate_cv",
    "top10_bins_frac", "isi_lt2", "isi_2_4", "isi_4_10", "isi_10_50", "refrac_contrast",
    "tet_max_snr", "tet_n_snr15", "amp_rel_tetmax",
] + TEMPLATE_FEATS
# Burstiness, amplitude drift / clipping, the spread in PC space (extra_cluster_features).
NEW_FEATS = ["pc1_ch_max_frac", "pc_cv", "burst_idx", "isi_cv2", "isi_p05_ms", "amp_drift", "amp_cutoff",
             "amp_skew"]
CLUSTER_FEATS = list(UNIT_FEATS) + NEW_FEATS
# A cluster in its context (add_ctx): the sorter unit (u_), the cluster it was cut from
# (p_, d_ = the difference), its sibling pieces (rel_, sib_ = the nearest sibling).
PIECE_FEATS = ["depth", "n_frac", "amp_rel_unit", "u_snr", "u_log_isf", "u_log_fr", "u_log_n",
               "pn_frac", "amp_rel_parent", "p_log_isf", "p_log_iso", "p_log_fr", "p_presence",
               "d_log_isf", "d_log_iso", "d_presence", "d_lt2",
               "n_siblings", "amp_rank", "n_rank", "amp_rel_maxsib", "rel_isf", "rel_iso", "rel_snr",
               "rel_presence", "rel_lt2"] + list(PC_FEATS) + \
              ["sib_ccg_min", "sib_ccg_local_min", "sib_cos_max", "sib_pc1cos_max", "sib_dprime_min",
               "sib_ratecorr_max", "sib_ampratio_max"]
LABEL_FEATS = CLUSTER_FEATS + PIECE_FEATS            # the label model's features
# Two pieces of one recluster: one neuron or two (pair_features).
PAIR_FEATS = ["ccg_ratio", "ccg_ratio_local", "ccg_1p5_10", "pc_cos", "pc_norm_ratio", "pc1_cos", "pc_dprime",
              "pc_dprime_1d", "amp_ratio", "rate_corr", "time_overlap"]
# A recluster (split_table): the mixture, the cluster that is cut (c_), its pieces (min /
# max / spike-weighted mean) and its pairs of pieces (min / max / median) ...
SPLIT_STATS = ["n", "n_out", "ll_gain", "bic_gain", "post_conf", "post_lt90", "w_min", "w_max"]
AGG_PIECE = ["log_isf", "log_isr", "isi_lt2", "refrac_contrast", "snr_wf", "log_iso", "log_lr", "log_fr",
             "log_n", "presence", "rate_cv", "amp_iqr_rel", "amp_p90_rel", "pn_frac", "amp_rel_parent",
             "pc_d_parent", "pc_d_sib", "pc_spread_rel", "pc_d_ch_max", "pc_d_ch_min", "burst_idx",
             "amp_cutoff", "amp_drift", "acg_tau_rise", "d_log_isf", "d_log_iso", "pc_cv", "pc1_ch_max_frac"]
SPLIT_BASE_FEATS = SPLIT_STATS + [f"c_{f}" for f in LABEL_FEATS] + \
    [f"{a}_{f}" for f in AGG_PIECE for a in ("pmin", "pmax", "pavg")] + \
    [f"{a}_{f}" for f in PAIR_FEATS for a in ("qmin", "qmax", "qmed")]
# ... and what the label model says of the cluster and of its pieces (stack_features).
STACK_FEATS = ["c_pgood", "c_pmua", "c_pnoise", "pmax_pgood", "pmin_pgood", "pavg_pgood", "pmax_pnoise",
               "pmin_pnoise", "pavg_pnoise", "pmax_pmua", "n_arg_good", "n_arg_mua", "n_arg_noise",
               "n_arg_labels", "gain_pgood", "gain_pnoise", "frac_arg_good", "frac_arg_noise"]
SPLIT_FEATS = SPLIT_BASE_FEATS + STACK_FEATS         # the keep model's features
# The columns of the training tables (phy_autocurate_data/).
CLUSTER_TABLE_COLS = list(dict.fromkeys(["dataset", "key", "label", "explicit", "depth", "n_spikes"] + LABEL_FEATS))
SPLIT_TABLE_COLS = list(dict.fromkeys(["dataset", "split", "parent", "source", "kept", "n"] + SPLIT_FEATS))
# Settings that can be changed per run (env / config file / CLI); defaults here.
DEFAULT_PARAMS = {
    "good_min_conf": GOOD_MIN_CONF,
    "amp_cut_sd": AMP_CUT_SD,
    "good_boost_narrow": GOOD_BOOST_NARROW,  # weight on P(good) of narrow-waveform clusters
    "split_bias": 1.0,          # > 1 reclusters more readily, < 1 less (keep_thr / split_bias)
    "keep_thr": KEEP_THR,
    "max_depth": MAX_DEPTH,
}


# =============================================================================
# Small helpers
# =============================================================================
def log(msg):
    print(f"[autocurate] {msg}", flush=True)


def seed_of(*parts):
    """A stable 32-bit seed for a cluster key (same spikes every run)."""
    return zlib.crc32("/".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


def read_params(phy_dir):
    ns = {}
    exec((Path(phy_dir) / "params.py").read_text(), {}, ns)
    return {k: v for k, v in ns.items() if not k.startswith("__")}


def resolve_dat(phy_dir, dat_path):
    """The recording params.py names, also after the op folder moved (same rule
    as sorter_common._resolve_phy_dat)."""
    phy = Path(phy_dir)
    if isinstance(dat_path, (list, tuple)):
        dat_path = dat_path[0]
    raw = str(dat_path)
    name = PureWindowsPath(raw).name if "\\" in raw else Path(raw).name
    direct = Path(raw)
    cands = [direct] if direct.is_absolute() else []
    cands += [phy / name, phy.parent / "processed_binary" / name]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"recording '{name}' not found (dat_path {raw!r}; looked in "
                            f"{phy} and {phy.parent / 'processed_binary'})")


def spike_hash(spikes):
    return hashlib.md5(np.sort(np.asarray(spikes, dtype=np.int64)).tobytes()).hexdigest()


# =============================================================================
# Data access (memory-mapped, read-only)
# =============================================================================
class PhyData:
    """The arrays of a phy export, memory-mapped, plus per-unit spike lists.

    'unit' = an original sorter unit = a template id (spike_templates.npy)."""

    def __init__(self, phy_dir, cache_dir=None, need_traces=False):
        self.dir = Path(phy_dir)
        p = read_params(self.dir)
        self.fs = float(p["sample_rate"])
        n_ch = int(p["n_channels_dat"])
        self.n_channels = n_ch
        ld = lambda name, **kw: np.load(self.dir / name, mmap_mode="r", **kw)
        self.spike_samples = ld("spike_times.npy").reshape(-1)
        self.spike_templates = ld("spike_templates.npy").reshape(-1)
        self.amplitudes = ld("amplitudes.npy").reshape(-1)
        self.pc_features = ld("pc_features.npy")            # (n_spikes, n_pcs, n_ch_loc)
        self.pc_feature_ind = np.load(self.dir / "pc_feature_ind.npy")
        self.templates = np.load(self.dir / "templates.npy")
        ti = self.dir / "template_ind.npy"
        self.template_ind = np.load(ti) if ti.exists() else self.pc_feature_ind
        wmi = self.dir / "whitening_mat_inv.npy"
        self.wmi = np.load(wmi) if wmi.exists() else None
        cg = self.dir / "channel_groups.npy"
        self.channel_groups = np.load(cg).reshape(-1) if cg.exists() else np.arange(n_ch) // 4
        self.n_spikes = len(self.spike_samples)
        self.traces = None
        self._dat = None
        self.duration = float(self.spike_samples[-1]) / self.fs     # if the recording is gone
        try:   # the recording: its length, and the noise if the export has no snr column
            dat = resolve_dat(self.dir, p["dat_path"])
            dtype = np.dtype(p.get("dtype", "int16"))
            offset = int(p.get("offset", 0))
            n_samples = (dat.stat().st_size - offset) // (n_ch * dtype.itemsize)
            self._dat = (dat, dtype, offset, n_samples)
            self.duration = n_samples / self.fs
        except OSError:
            pass
        if need_traces:
            self.open_traces()
        self._template_info()
        self._unit_spikes(cache_dir)
        self.noise = noise_from_snr(self, cache_dir)

    def open_traces(self):
        """Memory-map the recording (needed only for the noise-level fallback)."""
        if self.traces is None:
            if self._dat is None:
                raise FileNotFoundError(f"the recording of {self.dir} is not there")
            dat, dtype, offset, n_samples = self._dat
            self.traces = np.memmap(dat, dtype=dtype, mode="r", offset=offset,
                                    shape=(n_samples, self.n_channels))
        return self.traces

    def _template_info(self):
        """Per template: channels by decreasing amplitude (phylib's order), the
        peak channel and its tetrode (channel group), the template on those
        channels and its waveform features."""
        self.unit_channels, self.unit_peak = {}, {}
        self.unit_template, self.unit_template_feats = {}, {}
        for t in range(self.templates.shape[0]):
            w = np.asarray(self.templates[t], dtype=np.float64)
            ch = np.asarray(self.template_ind[t]).astype(np.int64)
            ok = (ch >= 0) & (np.abs(w).max(0) > np.abs(w).max() * 1e-6)
            if not ok.any():
                ok = ch >= 0
            w, ch = w[:, ok], ch[ok]
            if self.wmi is not None and len(ch):
                w = w @ self.wmi[np.ix_(ch, ch)]
            amp = w.max(0) - w.min(0)
            order = np.argsort(amp)[::-1]
            self.unit_channels[t] = ch[order]
            self.unit_peak[t] = int(ch[order][0]) if len(ch) else 0
            self.unit_template[t] = w[:, order]
            self.unit_template_feats[t] = template_features(w[:, order], self.fs)
        self.unit_tetrode = {t: int(self.channel_groups[c]) for t, c in self.unit_peak.items()}

    def _unit_spikes(self, cache_dir):
        """order / bounds so that the spikes of unit u are order[b[u]:b[u+1]] (sorted)."""
        n_t = int(max(self.templates.shape[0], int(np.max(self.spike_templates)) + 1))
        cache = Path(cache_dir) if cache_dir else None
        if cache is not None and (cache / "unit_order.npy").exists():
            self.order = np.load(cache / "unit_order.npy", mmap_mode="r")
            self.bounds = np.load(cache / "unit_bounds.npy")
        else:
            st = np.asarray(self.spike_templates)
            self.order = np.argsort(st, kind="stable")
            self.bounds = np.searchsorted(st[self.order], np.arange(n_t + 1))
            if cache is not None:
                cache.mkdir(parents=True, exist_ok=True)
                np.save(cache / "unit_order.npy", self.order)
                np.save(cache / "unit_bounds.npy", self.bounds)
        self.units = [u for u in range(n_t) if self.bounds[u + 1] > self.bounds[u]]

    def spikes_of(self, unit):
        return np.asarray(self.order[self.bounds[unit]:self.bounds[unit + 1]], dtype=np.int64)

    def units_on_tetrode(self, tet):
        return [u for u in self.units if self.unit_tetrode.get(u) == tet]


_DATA_CACHE = {}


def get_data(phy_dir, cache_dir=None):
    key = (str(phy_dir), str(cache_dir))
    if key not in _DATA_CACHE:
        _DATA_CACHE[key] = PhyData(phy_dir, cache_dir)
    return _DATA_CACHE[key]


def _descendants(spike_templates, spike_clusters):
    """{sorter unit: [(current cluster, n_spikes), ...]}."""
    pairs, counts = np.unique(np.stack([spike_templates, spike_clusters], 1), axis=0,
                              return_counts=True)
    kids = {}
    for (t, c), n in zip(pairs, counts):
        kids.setdefault(int(t), []).append((int(c), int(n)))
    return kids


def unit_metric_values(phy_dir, metric, spike_templates, original_dir=None):
    """{sorter unit: value} of a step-7 cluster_<metric>.tsv column: from the
    original file (_auto_curation/original/, or `original_dir`: the same files
    rebuilt in a scratch folder, see _scratch_originals) when there is one, else
    from the current one, where a unit split in Phy is read from its largest
    descendant (a split inherits the value). Reads only."""
    import pandas as pd
    phy = Path(phy_dir)
    f = phy / CURATION_DIR / "original" / f"cluster_{metric}.tsv"
    if not f.exists() and original_dir is not None:
        f = Path(original_dir) / f"cluster_{metric}.tsv"
    current_file = not f.exists()
    if current_file:
        f = phy / f"cluster_{metric}.tsv"
    if not f.exists():
        return {}
    try:
        t = pd.read_csv(f, sep="\t")
        vals = {int(c): float(v) for c, v in zip(t.iloc[:, 0], pd.to_numeric(t.iloc[:, 1], errors="coerce"))}
    except Exception:
        return {}
    st = np.asarray(spike_templates, np.int64)
    units = np.unique(st).tolist()
    direct = {u: vals[u] for u in units if u in vals}
    scp = phy / "spike_clusters.npy"
    if not current_file or not scp.exists():
        return direct
    sc = np.load(scp, mmap_mode="r").reshape(-1)
    if len(sc) != len(st) or np.array_equal(np.asarray(sc), st):
        return direct
    kids = _descendants(st, np.asarray(sc, np.int64))
    current = set(np.unique(np.asarray(sc)).tolist())
    out = {}
    for u in units:
        if u in vals and (u in current or not kids.get(u)):
            out[u] = vals[u]
            continue
        for c, _n in sorted(kids.get(u, []), key=lambda x: -x[1]):
            if c in vals:
                out[u] = vals[c]
                break
    return out


def noise_from_snr(data, cache_dir=None):
    """Per-channel noise level (uV, as SpikeInterface's noise_levels) recovered
    from the exported snr column: snr = |template extremum| / noise on the
    extremum channel. A channel without a unit gets its tetrode's median, then
    the recording's. Without an snr column the noise is measured on the
    recording (MAD), if that is still there."""
    n_ch = data.n_channels
    snr = unit_metric_values(data.dir, "snr", data.spike_templates,
                             Path(cache_dir) / "original" if cache_dir else None)
    per = {}
    for u, s in snr.items():
        w = data.unit_template.get(u)
        if w is None or w.size == 0 or not (np.isfinite(s) and s > 0):
            continue
        j = int(np.argmax(np.abs(w).max(0)))
        amp = float(np.abs(w[:, j]).max())
        if amp > 0:
            per.setdefault(int(data.unit_channels[u][j]), []).append(amp / s)
    if not per:
        try:
            return noise_levels(data, cache_dir)
        except Exception as e:
            log(f"{data.dir}: no snr column and no recording for the noise level ({e})")
            return np.full(n_ch, np.nan)
    noise = np.full(n_ch, np.nan)
    for c, v in per.items():
        if 0 <= c < n_ch:
            noise[c] = float(np.median(v))
    groups = np.asarray(data.channel_groups)
    if len(groups) == n_ch:
        for tet in np.unique(groups):
            m = groups == tet
            if np.isnan(noise[m]).any() and np.isfinite(noise[m]).any():
                noise[m & np.isnan(noise)] = np.nanmedian(noise[m])
    if np.isfinite(noise).any():
        noise[np.isnan(noise)] = np.nanmedian(noise)
    return noise


def noise_levels(data, cache_dir=None):
    """Per-channel noise, MAD / 0.6745 of the (filtered) recording, 40 x 1 s.
    The fallback of noise_from_snr for an export without an snr column."""
    if cache_dir and (Path(cache_dir) / "noise.npy").exists():
        return np.load(Path(cache_dir) / "noise.npy")
    data.open_traces()
    rng = np.random.default_rng(0)
    n = data.traces.shape[0]
    chunk = int(data.fs)
    starts = np.sort(rng.integers(0, max(1, n - chunk), N_NOISE_CHUNKS))
    per_chunk = []
    for s in starts:  # one chunk in memory at a time
        x = np.asarray(data.traces[s:s + chunk], dtype=np.float32)
        per_chunk.append(np.median(np.abs(x - np.median(x, 0)), 0))
    mad = np.median(np.array(per_chunk), 0) / 0.6744897501960817
    mad[mad <= 0] = np.nan
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(Path(cache_dir) / "noise.npy", mad)
    return mad


# =============================================================================
# Metrics (the same formulas as the fork's live columns / step u)
# =============================================================================
def isi_violations(spike_samples, fs, duration):
    """(ratio, count, n_spikes) after dropping duplicated spikes (<= 0.3 ms)."""
    s = np.sort(np.asarray(spike_samples, dtype=np.int64))
    if len(s) == 0:
        return np.nan, 0, 0
    dup = np.zeros(len(s), bool)
    dup[1:] = np.diff(s) <= int(round(DUPLICATE_MS * 1e-3 * fs))
    s = s[~dup]
    n = len(s)
    count = int(np.sum(np.diff(s / fs) < ISI_THRESHOLD_MS / 1000))
    ratio = (count / (2 * n * ISI_THRESHOLD_MS / 1000)) / (n / duration) if duration > 0 else np.nan
    return ratio, count, n


def sample_spikes(spikes, max_spikes, seed):
    spikes = np.asarray(spikes)
    if len(spikes) <= max_spikes:
        return spikes
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(spikes, size=max_spikes, replace=False))


def mahalanobis_metrics(x, y):
    """(isolation distance, L-ratio), as SpikeInterface's mahalanobis metric."""
    from scipy.stats import chi2
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if min(len(x), len(y)) < 2 or len(x) <= x.shape[1]:
        return np.nan, np.nan
    try:
        vi = np.linalg.inv(np.atleast_2d(np.cov(x, rowvar=False)))
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    z = y - x.mean(0)
    d2 = np.einsum("ij,ij->i", z @ vi, z)
    d2[d2 < 0] = np.nan
    d2 = np.sort(d2)
    n = min(len(x), len(y))
    return float(d2[n - 1]), float(np.nansum(chi2.sf(d2, x.shape[1])) / len(x))


def pc_features(data, spikes, channels):
    """(n_spikes, n_channels * n_pcs) PC features on `channels` (0 where a
    spike's template has no feature), as phylib get_features + NaN -> 0."""
    spikes = np.asarray(spikes, dtype=np.int64)
    raw = np.asarray(data.pc_features[spikes], dtype=np.float64)        # (n, pcs, loc)
    cols = data.pc_feature_ind[np.asarray(data.spike_templates[spikes], dtype=np.int64)]
    n, n_pcs = len(spikes), raw.shape[1]
    out = np.zeros((n, len(channels), n_pcs), np.float64)   # float64, as phylib's
    for j, ch in enumerate(channels):
        m = cols == ch
        has = m.any(1)
        if has.any():
            idx = m.argmax(1)
            out[has, j, :] = raw[np.nonzero(has)[0], :, idx[has]]
    return out.reshape(n, -1)


def recluster(data, spikes, channels, n):
    """phy_genzel recluster_n: PCA (<= 10 dims, not whitened) of the PC features
    on `channels` (np.unique of the unit's channels, as the GUI passes them),
    then a full-covariance Gaussian mixture with n components fit on <= 20000
    spikes (seed 0) and applied to all. Piece j = mixture component j, which is
    the order of the cluster ids Phy creates.

    Returns (component of every spike, or None if the mixture cannot be fit -
    the Phy action would fail too; the PC features; statistics of the split)."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    x = pc_features(data, spikes, channels)
    if len(spikes) < 2:
        return None, x, {}
    xr = PCA(n_components=min(GMM_MAX_DIMS, x.shape[1], x.shape[0]), whiten=False,
             random_state=0).fit_transform(x)
    xf = xr
    if len(xr) > GMM_MAX_SPIKES_FIT:
        xf = xr[np.random.default_rng(0).choice(len(xr), GMM_MAX_SPIKES_FIT, replace=False)]
    try:
        g = GaussianMixture(n_components=n, covariance_type="full", random_state=0).fit(xf)
    except ValueError:
        return None, x, {}
    lab = g.predict(xr)
    info = {}
    try:
        g1 = GaussianMixture(n_components=1, covariance_type="full", random_state=0).fit(xf)
        post = g.predict_proba(xf)
        info = dict(ll_gain=float(g.score(xf) - g1.score(xf)),
                    bic_gain=float((g1.bic(xf) - g.bic(xf)) / len(xf)),
                    post_conf=float(post.max(1).mean()),
                    post_lt90=float(np.mean(post.max(1) < 0.9)),
                    w_min=float(g.weights_.min()), w_max=float(g.weights_.max()),
                    gmm_converged=float(g.converged_))
    except Exception:
        pass
    return lab, x, info


def template_features(w, fs):
    """Waveform features of a mean waveform `w` (n_samples, n_channels; the
    channels by decreasing amplitude; uV), on its best channel, around the
    trough (a positive spike is flipped): CellExplorer's trough-to-peak, the
    widths, the peaks before / after the trough, the slopes after it, the number
    of peaks, baseline flatness, the energy before the spike, the trough's
    position, the amplitude profile over the channels, and the shape itself
    (SHAPE_T_MS samples, trough = -1)."""
    from scipy.signal import find_peaks
    f = dict.fromkeys(TEMPLATE_FEATS, np.nan)
    w = np.asarray(w, np.float64)
    if w.ndim != 2 or w.size == 0 or not np.all(np.isfinite(w)):
        return f
    ptp = w.max(0) - w.min(0)
    b = int(np.argmax(ptp))
    if ptp[b] <= 0:
        return f
    x = w[:, b]
    n = len(x)
    ms = 1000.0 / fs
    t0 = int(np.argmax(np.abs(x)))
    y = x if x[t0] < 0 else -x                   # the trough is negative
    depth = float(-y[t0])
    k05, k15 = int(round(0.5 / ms)), int(round(1.5 / ms))
    f["trough_to_peak"] = ctm.trough_to_peak(y, fs, t0)
    f["tpl_ptp"] = float(ptp[b])
    f["wf_ptp_ratio2"] = float(np.sort(ptp)[-2] / ptp[b]) if len(ptp) > 1 else 0.0
    f["tpl_ptp_ratio4"] = float(ptp.min() / ptp[b])
    f["tpl_spread"] = float(np.sum(ptp >= 0.5 * ptp[b]))
    f["wf_pos_neg_ratio"] = float(x.max() / max(-x.min(), 1e-9))
    edge = np.r_[y[:max(t0 - k05, 0)], y[t0 + k15:]]
    if len(edge) < 3:
        edge = np.r_[y[:3], y[-3:]]
    f["wf_flatness"] = float(np.std(edge) / ptp[b])
    f["tpl_trough_pos"] = float((t0 - n / 2) * ms)
    half = y[t0] / 2                             # the trough's width at half depth
    lo = t0
    while lo > 0 and y[lo] < half:
        lo -= 1
    hi = t0
    while hi < n - 1 and y[hi] < half:
        hi += 1
    f["tpl_trough_hw"] = float((hi - lo) * ms)
    before = y[max(t0 - k05, 0):t0]
    f["tpl_before_ratio"] = float(before.max() / depth) if len(before) else 0.0
    after = y[t0 + 1:t0 + k15 + 1]
    if len(after):
        ia = int(np.argmax(after))
        pk = t0 + 1 + ia                         # the peak after the trough
        f["tpl_after_ratio"] = float(after.max() / depth)
        f["tpl_repol_slope"] = float((y[pk] - y[t0]) / (ia + 1) / ms / depth)
        k = min(k05, n - 1 - pk)
        if k >= 2:
            f["tpl_recov_slope"] = float((y[pk + k] - y[pk]) / k / ms / depth)
        if y[pk] > 0:
            hl = y[pk] / 2
            lo = pk
            while lo > t0 and y[lo] > hl:
                lo -= 1
            hi = pk
            while hi < n - 1 and y[hi] > hl:
                hi += 1
            f["tpl_after_hw"] = float((hi - lo) * ms)
    prom = 0.1 * ptp[b]
    f["tpl_n_neg"] = float(len(find_peaks(-y, prominence=prom)[0]))
    f["tpl_n_pos"] = float(len(find_peaks(y, prominence=prom)[0]))
    e = y ** 2
    pre = e[max(t0 - int(round(0.75 / ms)), 0):max(t0 - int(round(0.25 / ms)), 0)]
    f["tpl_energy_pre"] = float(pre.sum() / max(e.sum(), 1e-9))
    idx = np.clip(np.round(t0 + SHAPE_T_MS / ms).astype(int), 0, n - 1)
    for i, v in enumerate(y[idx] / depth):
        f[f"tpl_s{i:02d}"] = float(v)
    return f


def pc_relative(x, parent, siblings, n_pcs):
    """A piece's waveform in PC space, relative to the spike set it was cut from
    (`parent`) and to its sibling pieces: pc_d_parent = Mahalanobis distance of
    its mean from the parent's mean (parent covariance), pc_d_sib = the same to
    the nearest sibling with the pooled covariance (a d'), pc_spread_rel = its
    spread / the parent's, pc_d_ch_max / _min = the largest and smallest of the
    per-channel distances from the parent (is the difference on one channel or
    on all). Rows of `x`: (n_channels * n_pcs) PC features, channel-major."""
    out = dict.fromkeys(PC_FEATS, np.nan)
    x, parent = np.asarray(x, np.float64), np.asarray(parent, np.float64)
    if len(x) < 5 or len(parent) < 10:
        return out

    def inv(c):
        c = np.atleast_2d(c)
        c = c + np.eye(len(c)) * (1e-6 * np.trace(c) / len(c) + 1e-12)
        try:
            return np.linalg.inv(c)
        except np.linalg.LinAlgError:
            return None

    mu, pm = x.mean(0), parent.mean(0)
    d = mu - pm
    cov = np.cov(parent, rowvar=False)
    vi = inv(cov)
    if vi is None:
        return out
    out["pc_d_parent"] = float(np.sqrt(max(d @ vi @ d, 0.0)))
    out["pc_spread_rel"] = float(np.sqrt(x.var(0).mean() / max(parent.var(0).mean(), 1e-12)))
    per = []
    for c in range(x.shape[1] // n_pcs):
        sl = slice(c * n_pcs, (c + 1) * n_pcs)
        vic = inv(cov[sl, sl])
        if vic is not None:
            per.append(float(np.sqrt(max(d[sl] @ vic @ d[sl], 0.0))))
    if per:
        out["pc_d_ch_max"], out["pc_d_ch_min"] = max(per), min(per)
    cx = np.cov(x, rowvar=False) * (len(x) - 1)
    best = np.nan
    for s in siblings:
        s = np.asarray(s, np.float64)
        if len(s) < 5:
            continue
        vis = inv((cx + np.cov(s, rowvar=False) * (len(s) - 1)) / (len(x) + len(s) - 2))
        if vis is None:
            continue
        dd = mu - s.mean(0)
        v = float(np.sqrt(max(dd @ vis @ dd, 0.0)))
        best = v if not np.isfinite(best) else min(best, v)
    out["pc_d_sib"] = best
    return out


INTRINSIC_KEYS = (
    "n_spikes", "num_spikes", "fr", "isi_violations_ratio", "isi_violations_count",
    "isi_violations_fraction", "presence", "rate_cv", "top10_bins_frac", "isi_lt2", "isi_2_4",
    "isi_4_10", "isi_10_50", "isi_median_ms", "refrac_contrast", "amp_med", "amp_iqr_rel",
    "amp_p90_rel", "peak_channel", "wf_ptp", "snr_wf", "acg_tau_rise",
    "isolation_distance", "l_ratio") + tuple(TEMPLATE_FEATS) + tuple(PC_FEATS)


def intrinsic_features(data, spikes, unit, neighbors, key, noise, unit_amp_med=None):
    """Features of one spike set that do not depend on its siblings.
    `neighbors`: spike arrays of the clusters it is compared with (isolation).
    `unit_amp_med`: the median spike amplitude of the unit the set belongs to
    (its template's amplitude is scaled by the set's own amplitude)."""
    spikes = np.asarray(spikes, dtype=np.int64)
    f = dict.fromkeys(INTRINSIC_KEYS, np.nan)
    f["n_spikes"] = len(spikes)
    t = np.asarray(data.spike_samples[spikes], dtype=np.int64)
    ratio, count, n_nodup = isi_violations(t, data.fs, data.duration)
    f.update(num_spikes=n_nodup, fr=len(spikes) / data.duration, isi_violations_ratio=ratio,
             isi_violations_count=count,
             isi_violations_fraction=count / (n_nodup - 1) if n_nodup > 1 else np.nan)
    tt = t / data.fs
    h = np.histogram(tt, np.linspace(0, data.duration, N_TIME_BINS + 1))[0]
    f["presence"] = float(np.mean(h > 0))
    f["rate_cv"] = float(np.std(h) / max(np.mean(h), 1e-9))
    f["top10_bins_frac"] = float(np.sort(h)[::-1][:N_TIME_BINS // 10].sum() / max(h.sum(), 1))
    isi = np.diff(tt) * 1000.0
    if len(isi):
        f.update(isi_lt2=float(np.mean(isi < 2)), isi_2_4=float(np.mean((isi >= 2) & (isi < 4))),
                 isi_4_10=float(np.mean((isi >= 4) & (isi < 10))),
                 isi_10_50=float(np.mean((isi >= 10) & (isi < 50))),
                 isi_median_ms=float(np.median(isi)),
                 refrac_contrast=float(np.sum(isi < 1.5) / max(np.sum((isi >= 3) & (isi < 6)), 1)))
    a = np.abs(np.asarray(data.amplitudes[spikes], dtype=np.float64))
    if len(a):
        med = float(np.median(a))
        q5, q25, q75, q95 = np.percentile(a, [5, 25, 75, 95])
        f.update(amp_med=med, amp_iqr_rel=(q75 - q25) / max(med, 1e-9),
                 amp_p90_rel=(q95 - q5) / max(med, 1e-9))
    channels = data.unit_channels[unit]
    # the waveform: the unit's template, its amplitude scaled by this set's own
    f.update(data.unit_template_feats.get(unit, {}))
    scale = f["amp_med"] / unit_amp_med if unit_amp_med and np.isfinite(f["amp_med"]) else 1.0
    f["peak_channel"] = int(channels[0]) if len(channels) else -1
    f["wf_ptp"] = f["tpl_ptp"] * scale
    nz = noise[int(channels[0])] if noise is not None and len(channels) else np.nan
    f["snr_wf"] = float(f["wf_ptp"] / nz) if np.isfinite(nz) and nz > 0 and np.isfinite(f["wf_ptp"]) \
        else np.nan
    f["acg_tau_rise"] = ctm.acg_tau_rise(t, data.fs) if len(t) > 1 else np.nan
    own = sample_spikes(spikes, N_ISOLATION_SPIKES, seed_of(key, "iso"))
    other = [sample_spikes(nb, N_ISOLATION_SPIKES, seed_of(key, "nb", i))
             for i, nb in enumerate(neighbors) if len(nb)]
    if other and len(own) > 1:
        other = np.concatenate(other)
        f["isolation_distance"], f["l_ratio"] = mahalanobis_metrics(
            pc_features(data, own, channels), pc_features(data, other, channels))
    else:
        f["isolation_distance"], f["l_ratio"] = np.nan, np.nan
    return f


def pair_features(data, a, b, xa, xb):
    """Two sibling pieces (spike arrays a, b; their PC features xa, xb): are they
    one neuron or two? Refractory cross-correlogram, distance in PC space,
    channel profile, amplitude, time course."""
    f = {}
    fs, dur = data.fs, data.duration
    ta = np.sort(np.asarray(data.spike_samples[a], np.int64))
    tb = np.sort(np.asarray(data.spike_samples[b], np.int64))
    tau, dup = ISI_THRESHOLD_MS / 1000 * fs, DUPLICATE_MS / 1000 * fs
    lo = np.searchsorted(tb, ta - tau, side="left")
    hi = np.searchsorted(tb, ta + tau, side="right")
    lo_d = np.searchsorted(tb, ta - dup, side="left")
    hi_d = np.searchsorted(tb, ta + dup, side="right")
    close = int(np.sum(hi - lo) - np.sum(hi_d - lo_d))
    expected = 2 * (tau - dup) / fs * len(ta) * len(tb) / max(dur, 1e-9)
    f["ccg_close"] = close
    f["ccg_expected"] = expected
    f["ccg_ratio"] = close / expected if expected > 0 else np.nan
    # the same within 1.5-10 ms (bursts / common drive) and locally expected (20-50 ms)
    w1, w2 = 10e-3 * fs, 50e-3 * fs
    n10 = int(np.sum(np.searchsorted(tb, ta + w1, side="right") - np.searchsorted(tb, ta - w1, side="left")))
    n50 = int(np.sum(np.searchsorted(tb, ta + w2, side="right") - np.searchsorted(tb, ta - w2, side="left")))
    side = max(n50 - n10, 0) / (2 * (w2 - w1))          # pairs per sample of lag, 10-50 ms
    f["ccg_ratio_local"] = close / (side * 2 * (tau - dup)) if side > 0 else np.nan
    f["ccg_1p5_10"] = (n10 - close) / (side * 2 * (w1 - tau)) if side > 0 else np.nan
    ma, mb = xa.mean(0), xb.mean(0)
    na, nb = np.linalg.norm(ma), np.linalg.norm(mb)
    f["pc_cos"] = float(ma @ mb / (na * nb)) if na > 0 and nb > 0 else np.nan
    f["pc_norm_ratio"] = float(min(na, nb) / max(na, nb)) if max(na, nb) > 0 else np.nan
    n_pcs = int(data.pc_features.shape[1])
    pa1, pb1 = ma[::n_pcs], mb[::n_pcs]                  # PC1 on each channel
    n1a, n1b = np.linalg.norm(pa1), np.linalg.norm(pb1)
    f["pc1_cos"] = float(pa1 @ pb1 / (n1a * n1b)) if n1a > 0 and n1b > 0 else np.nan
    if len(xa) > 5 and len(xb) > 5:
        cov = (np.cov(xa, rowvar=False) * (len(xa) - 1) + np.cov(xb, rowvar=False) * (len(xb) - 1)) / (
            len(xa) + len(xb) - 2)
        cov = cov + np.eye(len(cov)) * (1e-6 * np.trace(cov) / len(cov) + 1e-12)
        try:
            d = ma - mb
            f["pc_dprime"] = float(np.sqrt(max(d @ np.linalg.solve(cov, d), 0.0)))
        except np.linalg.LinAlgError:
            f["pc_dprime"] = np.nan
        # along the line joining the two means: overlap of the two distributions
        w = d / max(np.linalg.norm(d), 1e-12)
        pa_, pb_ = xa @ w, xb @ w
        sd = np.sqrt((pa_.var() * len(pa_) + pb_.var() * len(pb_)) / (len(pa_) + len(pb_)))
        f["pc_dprime_1d"] = float(abs(pa_.mean() - pb_.mean()) / max(sd, 1e-12))
    aa = np.abs(np.asarray(data.amplitudes[a], np.float64))
    ab = np.abs(np.asarray(data.amplitudes[b], np.float64))
    if len(aa) and len(ab):
        f["amp_ratio"] = float(min(np.median(aa), np.median(ab)) / max(np.median(aa), np.median(ab), 1e-9))
    bins = np.linspace(0, dur, N_TIME_BINS + 1)
    ha = np.histogram(ta / fs, bins)[0].astype(float)
    hb = np.histogram(tb / fs, bins)[0].astype(float)
    if ha.std() > 0 and hb.std() > 0:
        f["rate_corr"] = float(np.corrcoef(ha, hb)[0, 1])
    f["time_overlap"] = float(np.minimum(ha / max(ha.sum(), 1), hb / max(hb.sum(), 1)).sum())
    f["n_a"], f["n_b"] = len(a), len(b)
    return f


def extra_cluster_features(data, spikes, x):
    """Per-cluster additions to intrinsic_features: the mean PC vector (channel
    profile), burstiness, amplitude drift and a cut-off amplitude distribution.
    `x` = the cluster's PC features."""
    f = {}
    n_pcs = int(data.pc_features.shape[1])
    m = x.mean(0) if len(x) else np.zeros(x.shape[1])
    for i, v in enumerate(m[:12]):
        f[f"pcm{i:02d}"] = float(v)
    p1 = np.abs(m[::n_pcs])
    f["pc1_ch_max_frac"] = float(p1.max() / max(p1.sum(), 1e-12)) if len(p1) else np.nan
    if len(x) > 5:
        sd = x.std(0)
        f["pc_cv"] = float(np.linalg.norm(sd) / max(np.linalg.norm(m), 1e-9))
    t = np.asarray(data.spike_samples[spikes], np.int64) / data.fs
    isi = np.diff(np.sort(t)) * 1000.0
    if len(isi) > 5:
        f["burst_idx"] = float(np.mean((isi >= 2) & (isi < 8)))
        f["isi_cv2"] = float(np.mean(2 * np.abs(np.diff(isi)) / np.maximum(isi[1:] + isi[:-1], 1e-9)))
        f["isi_p05_ms"] = float(np.percentile(isi, 5))
    a = np.abs(np.asarray(data.amplitudes[spikes], np.float64))
    if len(a) > 20:
        # amplitude drift over the recording and a cut-off (clipped) amplitude distribution
        k = max(len(a) // 10, 1)
        o = np.argsort(t)
        meds = [np.median(a[o[i * k:(i + 1) * k]]) for i in range(10) if len(o[i * k:(i + 1) * k])]
        f["amp_drift"] = float((max(meds) - min(meds)) / max(np.median(a), 1e-9))
        h = np.histogram(a, 30)[0]
        f["amp_cutoff"] = float(h[0] / max(h.max(), 1))
        f["amp_skew"] = float(((a - a.mean()) ** 3).mean() / max(a.std() ** 3, 1e-12))
    return f


# =============================================================================
# Feature tables: clusters in their context, reclusters, the label model's opinion
# =============================================================================
# Tables: C = one row per cluster (key, unit, tetrode, depth, parent = the key of the
# cluster it was cut from, split = the key of the recluster it is a piece of), S = one
# row per recluster (split, parent, n, ...), P = one row per pair of pieces of a
# recluster (split, a, b). Keys: "c<unit>" for a sorter unit, "<key>/n<n>k<j>" for
# piece j of its recluster_n n ("<key>/n<n>" is the recluster).
def add_ctx(C, P):
    """Derived and context features: logs; the tetrode; the sorter unit; the cluster a
    piece was cut from; its sibling pieces (same split)."""
    import pandas as pd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return _add_ctx(C, P)


def _add_ctx(C, P):
    import pandas as pd
    C = C.copy()
    C["log_n"] = np.log10(C.num_spikes.clip(lower=1))
    C["log_fr"] = np.log10(C.fr.clip(lower=1e-4))
    C["log_isr"] = np.log10(C.isi_violations_ratio.fillna(0) + 1e-3)
    C["log_isf"] = np.log10(C.isi_violations_fraction.fillna(0) + 1e-5)
    C["log_iso"] = np.log10(C.isolation_distance.clip(lower=0.1))
    C["log_lr"] = np.log10(C.l_ratio.clip(lower=1e-4))
    C["is_narrow"] = (C.trough_to_peak <= ctm.TROUGH_TO_PEAK_BOUNDARY_MS).astype(float)
    U = C[(C.depth == 0) & (C.parent == "")].drop_duplicates("unit").set_index("unit")
    tet = U.groupby("tetrode").snr_wf
    C["tet_max_snr"] = C.tetrode.map(tet.max())
    C["tet_n_snr15"] = C.tetrode.map(tet.apply(lambda s: float((s > 15).sum())))
    C["amp_rel_tetmax"] = C.snr_wf / C.tet_max_snr
    C["n_frac"] = C.n_spikes / C.unit.map(U.n_spikes)
    C["amp_rel_unit"] = C.amp_med / C.unit.map(U.amp_med)
    C["u_snr"] = C.unit.map(U.snr_wf)
    C["u_log_isf"] = C.unit.map(U.log_isf)
    C["u_log_fr"] = C.unit.map(U.log_fr)
    C["u_log_n"] = C.unit.map(U.log_n)
    K = C.drop_duplicates("key").set_index("key")
    par = C.parent.where(C.parent != "", np.nan)
    for src, dst in (("log_isf", "p_log_isf"), ("log_iso", "p_log_iso"), ("log_fr", "p_log_fr"),
                     ("presence", "p_presence")):
        C[dst] = par.map(K[src])
    C["pn_frac"] = C.n_spikes / par.map(K.n_spikes)
    C["amp_rel_parent"] = C.amp_med / par.map(K.amp_med)
    C["d_log_isf"] = C.log_isf - C.p_log_isf
    C["d_log_iso"] = C.log_iso - C.p_log_iso
    C["d_presence"] = C.presence - C.p_presence
    C["d_lt2"] = C.isi_lt2 - par.map(K.isi_lt2)
    # siblings: the pieces of the same split (a merged cluster is compared with the pieces)
    piece = (C.split != "")
    base = C[piece & (C.merged_from == "")]
    g = base.groupby("split")
    stats = dict(n_siblings=g.key.size(), amp_max=g.amp_med.max(), isf_min=g.log_isf.min(),
                 iso_max=g.log_iso.max(), snr_max=g.snr_wf.max(), pres_max=g.presence.max(),
                 lt2_min=g.isi_lt2.min())
    sp = C.split.where(piece, np.nan)
    C["n_siblings"] = sp.map(stats["n_siblings"])
    C["amp_rel_maxsib"] = C.amp_med / sp.map(stats["amp_max"])
    C["rel_isf"] = C.log_isf - sp.map(stats["isf_min"])
    C["rel_iso"] = C.log_iso - sp.map(stats["iso_max"])
    C["rel_snr"] = C.snr_wf / sp.map(stats["snr_max"])
    C["rel_presence"] = C.presence - sp.map(stats["pres_max"])
    C["rel_lt2"] = C.isi_lt2 - sp.map(stats["lt2_min"])
    C["amp_rank"] = np.nan
    C["n_rank"] = np.nan
    C.loc[base.index, "amp_rank"] = g.amp_med.rank(ascending=False, method="first") - 1
    C.loc[base.index, "n_rank"] = g.n_spikes.rank(ascending=False, method="first") - 1
    # the nearest sibling in each pair statistic
    if len(P):
        both = pd.concat([P.rename(columns={"a": "key"}), P.rename(columns={"b": "key"})])
        q = both.groupby("key")
        for src, dst, how in (("ccg_ratio", "sib_ccg_min", "min"), ("ccg_ratio_local", "sib_ccg_local_min", "min"),
                              ("pc_cos", "sib_cos_max", "max"), ("pc1_cos", "sib_pc1cos_max", "max"),
                              ("pc_dprime", "sib_dprime_min", "min"), ("rate_corr", "sib_ratecorr_max", "max"),
                              ("amp_ratio", "sib_ampratio_max", "max")):
            C[dst] = C.key.map(getattr(q[src], how)())
    else:
        for dst in ("sib_ccg_min", "sib_ccg_local_min", "sib_cos_max", "sib_pc1cos_max", "sib_dprime_min",
                    "sib_ratecorr_max", "sib_ampratio_max"):
            C[dst] = np.nan
    return C


def split_table(C, S, P):
    """One row per (candidate) recluster: the cluster that is cut, the mixture, its
    pieces (min / max / weighted mean of their features) and its pairs of pieces."""
    import pandas as pd
    K = C.drop_duplicates("key").set_index("key")
    pieces = C[(C.split != "") & (C.merged_from == "")]
    out = S.reset_index(drop=True)
    par = K.reindex(out.parent.values)
    new = {}
    for f in LABEL_FEATS:
        new[f"c_{f}"] = par[f].values if f in par else np.full(len(out), np.nan)
    g = pieces.groupby("split")
    w = pieces.n_spikes / g.n_spikes.transform("sum")
    for f in AGG_PIECE:
        if f not in pieces:
            continue
        new[f"pmin_{f}"] = out.split.map(g[f].min()).values
        new[f"pmax_{f}"] = out.split.map(g[f].max()).values
        new[f"pavg_{f}"] = out.split.map((pieces[f] * w).groupby(pieces.split).sum(min_count=1)).values
    q = P.groupby("split")
    for f in PAIR_FEATS:
        new[f"qmin_{f}"] = out.split.map(q[f].min()).values
        new[f"qmax_{f}"] = out.split.map(q[f].max()).values
        new[f"qmed_{f}"] = out.split.map(q[f].median()).values
    return pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)


def stack_features(C, S, p):
    """Per recluster: the label model's opinion of the cluster and of its pieces.
    C: cluster rows (key, split, n_spikes, merged_from); p: (len(C), 3) label
    probabilities; S: the recluster rows (split, parent)."""
    import pandas as pd
    D = pd.DataFrame(dict(key=C.key.values, split=C.split.values, n=C.n_spikes.values.astype(float),
                          pg=p[:, 0], pm=p[:, 1], pn=p[:, 2], arg=np.argmax(p, 1),
                          base=(C.merged_from.values == "") if "merged_from" in C else True))
    K = D.drop_duplicates("key").set_index("key")
    pieces = D[(D.split != "") & D.base]
    g = pieces.groupby("split")
    w = pieces.n / g.n.transform("sum")
    par = K.reindex(S.parent.values)
    f = dict(c_pgood=par.pg.values, c_pmua=par.pm.values, c_pnoise=par.pn.values)
    m = lambda s: S.split.map(s).values
    f["pmax_pgood"], f["pmin_pgood"] = m(g.pg.max()), m(g.pg.min())
    f["pavg_pgood"] = m((pieces.pg * w).groupby(pieces.split).sum())
    f["pmax_pnoise"], f["pmin_pnoise"] = m(g.pn.max()), m(g.pn.min())
    f["pavg_pnoise"] = m((pieces.pn * w).groupby(pieces.split).sum())
    f["pmax_pmua"] = m(g.pm.max())
    for j, c in enumerate(LABELS):
        f[f"n_arg_{c}"] = m((pieces.arg == j).groupby(pieces.split).sum())
    f["n_arg_labels"] = m(g.arg.nunique())
    f["gain_pgood"] = f["pmax_pgood"] - f["c_pgood"]
    f["gain_pnoise"] = f["pmax_pnoise"] - f["c_pnoise"]
    f["frac_arg_good"] = m((w * (pieces.arg == 0)).groupby(pieces.split).sum())
    f["frac_arg_noise"] = m((w * (pieces.arg == 2)).groupby(pieces.split).sum())
    return pd.DataFrame(f, index=S.index)


def _X(rows, feats):
    x = rows.reindex(columns=feats).astype(float).values
    x[~np.isfinite(x)] = -1
    return x


# =============================================================================
# Candidate reclusters of a recording (worker processes, kept in memory)
# =============================================================================
def _unit_task(phy_dir, cache_dir, unit, want_unit, clusters):
    """Worker: the candidate reclusters of some clusters of ONE sorter unit.

    clusters: [dict(key, depth, chain=[(labels, k), ...], ns=[...])]; the spikes of
    the cluster are the unit's spikes masked down the chain (labels over the
    previous set, piece k). want_unit: also the feature row of the unit itself.
    Returns dict(rows, splits, pairs, labels={(key, n): int8 piece of every spike
    of the cluster})."""
    data = get_data(phy_dir, cache_dir)
    noise = data.noise
    n_pcs = int(data.pc_features.shape[1])
    tet = data.unit_tetrode[unit]
    channels = np.unique(data.unit_channels[unit])
    U = data.spikes_of(unit)
    others = [(f"c{v}", data.spikes_of(v)) for v in data.units_on_tetrode(tet) if v != unit]
    uamp = float(np.median(np.abs(np.asarray(data.amplitudes[U], np.float64)))) if len(U) else 0.0
    out = dict(rows=[], splits=[], pairs=[], labels={})
    xU = None

    def cluster_row(key, spikes, d, parent_key, split_key, nbs, x, parent_x=None, sib_x=()):
        f = intrinsic_features(data, spikes, unit, [s for _, s in nbs], key, noise, uamp)
        f.update(extra_cluster_features(data, spikes, x))
        if parent_x is not None:
            f.update(pc_relative(x, parent_x, list(sib_x), n_pcs))
        f.update(key=key, id=-1, unit=unit, tetrode=tet, depth=d, parent=parent_key, split=split_key)
        out["rows"].append(f)

    if want_unit or clusters:
        xU = pc_features(data, U, channels)
    if want_unit:
        cluster_row(f"c{unit}", U, 0, "", "", others, xU)
    for cl in clusters:
        key, d = cl["key"], cl["depth"]
        sp, x, nbs = U, xU, list(others)
        kk = f"c{unit}"
        for lab, k in cl["chain"]:
            m = lab == k
            sib = [(f"{kk}/k{j}", sp[lab == j]) for j in np.unique(lab) if j != k]
            nbs = sib + nbs
            sp, x = sp[m], x[m]
            kk = f"{kk}/k{k}"
        for n in cl["ns"]:
            if len(sp) < max(MIN_SPIKES_TO_SPLIT, 10 * n):
                continue
            lab, _x, info = recluster(data, sp, channels, n)
            if lab is None:
                continue
            uniq = np.unique(lab)
            if len(uniq) < 2:
                continue
            lab = np.searchsorted(uniq, lab).astype(np.int8)
            m = len(uniq)
            pieces = [sp[lab == j] for j in range(m)]
            xs = [x[lab == j] for j in range(m)]
            pkeys = [f"{key}/n{n}k{j}" for j in range(m)]
            skey = f"{key}/n{n}"
            for j in range(m):
                sib = [(pkeys[i], pieces[i]) for i in range(m) if i != j]
                cluster_row(pkeys[j], pieces[j], d + 1, key, skey, sib + nbs, xs[j], x,
                            [xs[i] for i in range(m) if i != j])
            for i in range(m):
                for j in range(i + 1, m):
                    si = sample_spikes(pieces[i], N_PAIR_SPIKES, seed_of(pkeys[i], "pair"))
                    sj = sample_spikes(pieces[j], N_PAIR_SPIKES, seed_of(pkeys[j], "pair"))
                    f = pair_features(data, pieces[i], pieces[j], x[np.searchsorted(sp, si)],
                                      x[np.searchsorted(sp, sj)])
                    f.update(split=skey, a=pkeys[i], b=pkeys[j])
                    out["pairs"].append(f)
            s = dict(split=skey, parent=key, unit=unit, depth=d, n_out=m, n_parent=len(sp), source="rt",
                     kind="candidate", n=n)
            s.update(info or {})
            out["splits"].append(s)
            out["labels"][(key, n)] = lab
    return out


class Session:
    """One recording while it is curated: its data and the candidate reclusters
    computed so far (in memory; with `cache_dir` also kept there as one pickle per
    unit, so that an evaluation can curate the recording again with other models
    or settings without computing them again)."""

    def __init__(self, phy_dir, tmp, n_jobs=1, cache_dir=None):
        self.phy, self.tmp, self.n_jobs = Path(phy_dir), Path(tmp), n_jobs
        self.dir = Path(cache_dir) if cache_dir else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.data = get_data(self.phy, self.tmp)
        self.mem = {}                    # unit -> dict(rows, splits, pairs, labels, done={(key, n)}, unit_row)

    def _load(self, unit):
        if unit not in self.mem:
            f = self.dir / f"u{unit}.pkl" if self.dir is not None else None
            if f is not None and f.exists():
                with open(f, "rb") as fh:
                    self.mem[unit] = pickle.load(fh)
            else:
                self.mem[unit] = dict(rows=[], splits=[], pairs=[], labels={}, done=set(), unit_row=False)
        return self.mem[unit]

    def ensure(self, requests, want_units=()):
        """requests: [dict(unit, key, depth, chain, ns)]. Computes what is not there yet."""
        todo = {}
        for u in want_units:
            if not self._load(u)["unit_row"]:
                todo.setdefault(u, dict(want=True, clusters=[]))["want"] = True
        for r in requests:
            c = self._load(r["unit"])
            ns = [n for n in r["ns"] if (r["key"], n) not in c["done"]]
            if ns:
                todo.setdefault(r["unit"], dict(want=False, clusters=[]))["clusters"].append(
                    dict(key=r["key"], depth=r["depth"], chain=r["chain"], ns=ns))
        if not todo:
            return
        units = sorted(todo, key=lambda u: -len(self.data.spikes_of(u)))      # largest first
        args = [(str(self.phy), str(self.tmp), u, todo[u]["want"], todo[u]["clusters"]) for u in units]
        if self.n_jobs and self.n_jobs > 1 and len(units) > 1:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(
                delayed(_unit_task)(*a) for a in args)
        else:
            res = [_unit_task(*a) for a in args]
        for u, r in zip(units, res):
            c = self._load(u)
            c["rows"] += r["rows"]
            c["splits"] += r["splits"]
            c["pairs"] += r["pairs"]
            c["labels"].update(r["labels"])
            if todo[u]["want"]:
                c["unit_row"] = True
            for cl in todo[u]["clusters"]:
                for n in cl["ns"]:
                    c["done"].add((cl["key"], n))
            if self.dir is not None:
                with open(self.dir / f"u{u}.pkl", "wb") as fh:
                    pickle.dump(c, fh, protocol=4)

    def tables(self, units=None):
        """(C, S, P) of everything computed so far."""
        import pandas as pd
        units = list(self.mem) if units is None else units
        C = pd.DataFrame([r for u in units for r in self.mem[u]["rows"]])
        S = pd.DataFrame([r for u in units for r in self.mem[u]["splits"]])
        P = pd.DataFrame([r for u in units for r in self.mem[u]["pairs"]])
        for col in ("parent", "split"):
            C[col] = C[col].fillna("")
        C["merged_from"] = ""
        return C, S, P

    def labels_of(self, unit, key, n):
        return self.mem[unit]["labels"].get((key, n))


# =============================================================================
# Models
# =============================================================================
def _forest(n_jobs=1):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=N_TREES, min_samples_leaf=2, class_weight="balanced_subsample",
                                  random_state=0, n_jobs=n_jobs)


def _label_rows_for_fit(clusters):
    """The rows the label model learns from: final clusters labelled good / mua /
    noise whose history is known (depth >= 0; see build_bank)."""
    L = clusters[clusters.label.isin(LABELS) & (clusters.depth >= 0)]
    if "final" in L:
        L = L[L.final.astype(bool)]
    return L


def _fit_label(clusters, n_jobs=1):
    L = _label_rows_for_fit(clusters)
    return _forest(n_jobs).fit(_X(L, LABEL_FEATS), L.label.values)


def _proba3(clf, x):
    """predict_proba in the order of LABELS (0 for a label the forest never saw)."""
    p = clf.predict_proba(x)
    out = np.zeros((len(x), 3))
    for j, c in enumerate(clf.classes_):
        out[:, LABELS.index(c)] = p[:, j]
    return out


class Models:
    """Two random forests fit on the training tables.
    label: good / mua / noise of a cluster (training_clusters: the final clusters
    of the hand curations).
    keep: is this recluster kept (training_splits: the curator's reclusters, kept
    or undone, and the candidate reclusters the curator did not make). The keep
    model also sees the label model's opinion of the cluster and of its pieces
    (STACK_FEATS; in the training table they are cross-fitted: computed with a
    label forest that did not see that recording)."""

    def __init__(self, clusters, splits, params=None, n_jobs=1):
        self.params = dict(DEFAULT_PARAMS, **(params or {}))
        self.n_jobs = n_jobs
        self.label = _fit_label(clusters, n_jobs)
        self.n_label_rows = len(_label_rows_for_fit(clusters))
        self.keep = _forest(n_jobs).fit(_X(splits, SPLIT_FEATS), splits.kept.astype(int).values,
                                        sample_weight=np.ones(len(splits)))
        self.n_split_rows = len(splits)

    def label_proba(self, rows):
        """(len(rows), 3) P(good / mua / noise) of cluster rows (with context)."""
        return _proba3(self.label, _X(rows, LABEL_FEATS))

    def keep_proba(self, T, C):
        """P(keep) of the recluster rows T (split_table); C: the cluster rows (with
        context) they refer to."""
        import pandas as pd
        T = T.reset_index(drop=True)
        T = pd.concat([T, stack_features(C, T, self.label_proba(C))], axis=1)
        classes = list(self.keep.classes_)
        if 1 not in classes:
            return np.zeros(len(T))
        return self.keep.predict_proba(_X(T, SPLIT_FEATS))[:, classes.index(1)]


def load_training_tables(data_dir=DATA_DIR, exclude=()):
    """(clusters, splits): the label rows and the recluster rows of the training
    tables, without the datasets in `exclude`."""
    import pandas as pd
    data_dir = Path(data_dir)
    fc, fs = data_dir / "training_clusters.tsv.gz", data_dir / "training_splits.tsv.gz"
    if not (fc.exists() and fs.exists()):
        raise FileNotFoundError(
            f"no training tables in {data_dir} (training_clusters.tsv.gz / training_splits.tsv.gz): "
            f"build them with `python phy_autocurate.py train --curated <phy_export> ... --names ... "
            f"--cache <folder>`")
    # round_trip: exact floats, so `train` rewrites the rows it keeps unchanged
    c = pd.read_csv(fc, sep="\t", float_precision="round_trip", low_memory=False)
    s = pd.read_csv(fs, sep="\t", float_precision="round_trip", low_memory=False)
    if exclude:
        c, s = c[~c.dataset.isin(exclude)], s[~s.dataset.isin(exclude)]
    return c.reset_index(drop=True), s.reset_index(drop=True)


# =============================================================================
# Decisions: recluster or keep, down the tree; then the labels
# =============================================================================
def _path_of(key):
    """'c12/n4k2/n2k0' -> 'n4k2/n2k0' ('' for a whole unit)."""
    return key.partition("/")[2]


def _path_order(path):
    """Sort key of a path: [(n, k), ...]."""
    return [tuple(int(v) for v in s[1:].split("k")) for s in str(path).split("/") if s]


def _step_of(key, depth, reverted_n=None):
    """cluster_auto_step: 'asis' (a unit kept whole), 'L<depth>-n<n>' (a piece, n =
    the last recluster), + '(split<n>-reverted)' when its own recluster was undone."""
    step = "asis" if depth == 0 else f"L{depth}-n{_path_order(_path_of(key))[-1][0]}"
    return step + (f"(split{reverted_n}-reverted)" if reverted_n else "")


def curate(sess, models, params=None, verbose=False):
    """The curation of one recording (nothing is written).

    Level by level: every cluster of the level (first the sorter units) with
    >= MIN_SPIKES_TO_SPLIT spikes gets its candidate reclusters n = 2, 3, 4; the
    one with the highest P(keep) is accepted if that is >= keep_thr / split_bias
    and its pieces form the next level (at most max_depth levels); the other
    clusters are leaves. The leaves are labelled; a split whose leaves are all
    noise is undone, bottom-up.

    Returns (leaves, info). leaves: dict(unit, key, path, chain=[(labels, k), ...],
    label, conf, p=(good, mua, noise), depth, step); info: accepted = {key: (n,
    P(keep))}, reverted = {key: n}, units = the unit rows with the label
    probabilities and P(keep) of n = 2, 3, 4, n_units."""
    import pandas as pd
    P_ = dict(DEFAULT_PARAMS, **(params or {}))
    data = sess.data
    units = list(data.units)
    sess.ensure([], want_units=units)
    node = {}                                   # key -> dict(unit, depth, chain, n)
    for u in units:
        node[f"c{u}"] = dict(unit=u, depth=0, chain=[], n=len(data.spikes_of(u)))
    frontier = [k for k, v in node.items() if v["n"] >= MIN_SPIKES_TO_SPLIT]
    leaves = [k for k, v in node.items() if v["n"] < MIN_SPIKES_TO_SPLIT]
    accepted = {}                               # key -> (n, pkeep)
    children = {}
    ns = list(RECLUSTER_NS)
    bias = float(P_.get("split_bias") or 0.0)
    thr = P_["keep_thr"] / bias if bias > 0 else np.inf
    unit_pkeep = {}
    for depth in range(int(P_["max_depth"])):
        if not frontier:
            break
        sess.ensure([dict(unit=node[k]["unit"], key=k, depth=node[k]["depth"], chain=node[k]["chain"],
                          ns=ns) for k in frontier])
        C, S, Pq = sess.tables()
        T = pd.DataFrame()
        if len(S):
            Cx = add_ctx(C, Pq)
            S = S[S.parent.isin(frontier) & S.n.isin(ns)]
            T = split_table(Cx, S, Pq)
            T["pkeep"] = models.keep_proba(T, Cx) if len(T) else []
        if depth == 0 and len(T):
            unit_pkeep = {(str(p), int(n)): float(v) for p, n, v in zip(T.parent, T.n, T.pkeep)}
        new = []
        best = T.sort_values("pkeep", ascending=False).drop_duplicates("parent").set_index("parent") if len(T) \
            else pd.DataFrame()
        for k in frontier:
            if k in best.index and best.loc[k, "pkeep"] >= thr:
                n = int(best.loc[k, "n"])
                lab = sess.labels_of(node[k]["unit"], k, n)
                accepted[k] = (n, float(best.loc[k, "pkeep"]))
                kids = []
                for j in range(int(lab.max()) + 1):
                    ck = f"{k}/n{n}k{j}"
                    node[ck] = dict(unit=node[k]["unit"], depth=node[k]["depth"] + 1,
                                    chain=node[k]["chain"] + [(lab, j)], n=int(np.sum(lab == j)))
                    kids.append(ck)
                    (new if node[ck]["n"] >= MIN_SPIKES_TO_SPLIT else leaves).append(ck)
                children[k] = kids
            else:
                leaves.append(k)
        frontier = new
        if verbose:
            log(f"  level {depth}: {len(accepted)} reclusters kept so far, {len(frontier)} pieces to try next")
    leaves += frontier
    C, S, Pq = sess.tables()
    Cx = add_ctx(C, Pq).drop_duplicates("key").set_index("key", drop=False)
    L = Cx.loc[leaves]
    p = models.label_proba(L)
    nar = (L.is_narrow.values == 1)
    b = P_["good_boost_narrow"]
    if b != 1.0:
        p = p.copy()
        p[nar, 0] *= b
        p = p / p.sum(1, keepdims=True)
    out = {}
    for i, k in enumerate(leaves):
        j = int(np.argmax(p[i]))
        lab, conf = LABELS[j], float(p[i, j])
        if lab == "good" and conf < P_["good_min_conf"]:
            lab = "unsorted"
        out[k] = dict(unit=node[k]["unit"], key=k, path=_path_of(k), chain=node[k]["chain"], label=lab,
                      conf=conf, p=tuple(float(v) for v in p[i]), depth=node[k]["depth"],
                      step=_step_of(k, node[k]["depth"]))
    # a split whose leaves are all noise is undone (the cluster is noise whole), bottom-up
    reverted = {}
    for k in sorted(accepted, key=lambda s: -s.count("/")):
        kids = children[k]
        if all(c in out and out[c]["label"] == "noise" for c in kids):
            conf = float(np.mean([out[c]["conf"] for c in kids]))
            for c in kids:
                out.pop(c)
            reverted[k] = accepted[k][0]
            out[k] = dict(unit=node[k]["unit"], key=k, path=_path_of(k), chain=node[k]["chain"], label="noise",
                          conf=conf, p=(0.0, 0.0, conf), depth=node[k]["depth"],
                          step=_step_of(k, node[k]["depth"], accepted[k][0]), reverted=True)
    # the unit rows for the log: what the label model says of the whole unit, P(keep) of n = 2, 3, 4
    U = Cx.loc[[f"c{u}" for u in units]].reset_index(drop=True)
    pu = models.label_proba(U) if len(U) else np.zeros((0, 3))
    for j, c in enumerate(LABELS):
        U[f"p_{c}"] = pu[:, j]
    for n in ns:
        U[f"pkeep_n{n}"] = [unit_pkeep.get((k, n), np.nan) for k in U.key]
    return list(out.values()), dict(accepted=accepted, reverted=reverted, n_units=len(units), units=U)


def leaf_spikes(data, leaf):
    """The spikes (indices) of a leaf of curate()."""
    sp = data.spikes_of(leaf["unit"])
    for lab, k in leaf["chain"]:
        sp = sp[lab == k]
    return sp


def spikes_of_path(data, unit, path):
    """The spikes of the cluster cutting_log.tsv names by its unit and path, e.g.
    'n4k2/n2k0' = piece 2 of the unit's recluster_n 4, then piece 0 of that
    piece's recluster_n 2 ('' = the whole unit), computed again."""
    sp = data.spikes_of(unit)
    channels = np.unique(data.unit_channels[unit])
    for n, k in _path_order(path if isinstance(path, str) else ""):
        lab, _x, _info = recluster(data, sp, channels, n)
        sp = sp[np.searchsorted(np.unique(lab), lab) == k]
    return sp


def amplitude_cut(amplitudes, spikes, k_sd=AMP_CUT_SD):
    """Spikes of a cluster whose |amplitude| exceeds mean + k SD."""
    a = np.abs(np.asarray(amplitudes[spikes], dtype=np.float64))
    if len(a) < 3:
        return np.zeros(len(spikes), bool)
    return a > a.mean() + k_sd * a.std()


# =============================================================================
# Evaluation against a curation
# =============================================================================
def curated_labels(phy_dir):
    """(spike_clusters, {cluster: label}) of a curated phy folder."""
    import pandas as pd
    phy = Path(phy_dir)
    sc = np.load(phy / "spike_clusters.npy").reshape(-1).astype(np.int64)
    grp = {}
    if (phy / "cluster_group.tsv").exists():
        g = pd.read_csv(phy / "cluster_group.tsv", sep="\t")
        col = "group" if "group" in g.columns else g.columns[1]
        grp = {int(c): str(v) for c, v in zip(g.cluster_id, g[col])}
    return sc, grp


def leaves_to_pred(data, leaves, amp_cut_sd=None):
    """Per-spike predicted cluster (int) and {cluster: label} of the leaves of
    curate(), with the amplitude cut."""
    pred = np.full(data.n_spikes, -1, np.int64)
    labels = {}
    cid = 0
    for lf in leaves:
        sp = leaf_spikes(data, lf)
        lab = lf["label"]
        if lab == "good" and amp_cut_sd and len(sp):
            cut = amplitude_cut(data.amplitudes, sp, amp_cut_sd)
            if cut.any() and not cut.all():
                pred[sp[cut]] = cid
                labels[cid] = "noise"
                cid += 1
                sp = sp[~cut]
        pred[sp] = cid
        labels[cid] = lab
        cid += 1
    return pred, labels


def compare_curations(pred, pred_labels, user, user_labels):
    """Agreement of a predicted curation with a manual one: spike-label accuracy,
    and recall / precision / F1 of the good clusters (Jaccard >= 0.5)."""
    import pandas as pd
    pl = pd.Series(pred).map(pred_labels).fillna("unsorted").values
    ul = pd.Series(user).map(user_labels).fillna("unsorted").values
    m = np.isin(ul, LABELS)
    acc = float(np.mean(pl[m] == ul[m]))
    key = pred.astype(np.int64) * (int(user.max()) + 1) + user
    k, cnt = np.unique(key, return_counts=True)
    ct = pd.DataFrame({"p": k // (int(user.max()) + 1), "u": k % (int(user.max()) + 1), "n": cnt})
    np_ = ct.groupby("p").n.sum()
    nu = ct.groupby("u").n.sum()
    ct["jac"] = ct.n / (ct.p.map(np_) + ct.u.map(nu) - ct.n)
    ct["pl"] = ct.p.map(pred_labels).fillna("unsorted")
    ct["ul"] = ct.u.map(user_labels).fillna("unsorted")
    best_u = ct.sort_values("n", ascending=False).drop_duplicates("u")
    best_p = ct.sort_values("n", ascending=False).drop_duplicates("p")
    ug = best_u[best_u.ul == "good"]
    pg = best_p[best_p.pl == "good"]
    rec = float(np.mean((ug.pl == "good") & (ug.jac >= 0.5))) if len(ug) else np.nan
    prec = float(np.mean((pg.ul == "good") & (pg.jac >= 0.5))) if len(pg) else np.nan
    f1 = 2 * rec * prec / (rec + prec) if rec + prec > 0 else 0.0
    n_unsorted = int(sum(1 for v in pred_labels.values() if v == "unsorted"))
    return dict(spike_acc=round(acc, 4), good_recall=round(rec, 4), good_precision=round(prec, 4),
                good_f1=round(f1, 4), n_good_pred=int(len(pg)), n_good_user=int(len(ug)),
                n_unsorted=n_unsorted, n_clusters=len(pred_labels))


def spike_scores(pred, pred_labels, user, user_labels):
    """Spike-level agreement: the share of the curator's good / mua / noise spikes
    given the same label, and the share of the predicted good spikes that the
    curator called good."""
    import pandas as pd
    pl = pd.Series(pred).map(pred_labels).fillna("unsorted").values
    ul = pd.Series(user).map(user_labels).fillna("unsorted").values
    out = {}
    for c in LABELS:
        m = ul == c
        out[f"sp_recall_{c}"] = round(float(np.mean(pl[m] == c)), 4) if m.any() else np.nan
    m = pl == "good"
    out["sp_prec_good"] = round(float(np.mean(ul[m] == "good")), 4) if m.any() else np.nan
    return out


def diagnose(pred, pred_labels, user, user_labels):
    """Why a predicted curation and the hand curation disagree on the good
    clusters. For every good cluster of the curator (u_, the recall side) and
    every predicted good cluster (p_, the precision side): found (Jaccard >= 0.5,
    both good), the same cluster with another label, or a different cut."""
    import pandas as pd
    key = pred.astype(np.int64) * (int(user.max()) + 1) + user
    k, cnt = np.unique(key, return_counts=True)
    ct = pd.DataFrame({"p": k // (int(user.max()) + 1), "u": k % (int(user.max()) + 1), "n": cnt})
    ct["np"] = ct.p.map(ct.groupby("p").n.sum())
    ct["nu"] = ct.u.map(ct.groupby("u").n.sum())
    ct["jac"] = ct.n / (ct.np + ct.nu - ct.n)
    ct["pl"] = ct.p.map(pred_labels).fillna("unsorted")
    ct["ul"] = ct.u.map(user_labels).fillna("unsorted")
    out = {}
    # recall side: the curator's good clusters
    best_u = ct.sort_values("n", ascending=False).drop_duplicates("u")
    ug = best_u[best_u.ul == "good"]
    out["u_good"] = len(ug)
    out["u_found"] = int(((ug.pl == "good") & (ug.jac >= 0.5)).sum())
    out["u_label_miss"] = int(((ug.pl != "good") & (ug.jac >= 0.5)).sum())       # same cluster, other label
    lo = ug[ug.jac < 0.5]
    # cut mismatch: the cluster is spread over several predicted clusters (fragmented) or
    # sits inside a larger predicted cluster (not cut out)
    frag = lo.n / lo.nu < 0.5 + 1e-9
    out["u_fragmented"] = int(frag.sum())
    out["u_inside_bigger"] = int((~frag).sum())
    out["u_inside_bigger_good"] = int(((~frag) & (lo.pl == "good")).sum())
    for lab in ("mua", "noise", "unsorted"):
        out[f"u_label_miss_{lab}"] = int(((ug.pl == lab) & (ug.jac >= 0.5)).sum())
    # of the good spikes of the curator, the share in predicted good clusters
    g = ct[ct.ul == "good"]
    out["u_good_spikes_in_pred_good"] = round(float(g[g.pl == "good"].n.sum() / max(g.n.sum(), 1)), 4)
    # precision side: the predicted good clusters
    best_p = ct.sort_values("n", ascending=False).drop_duplicates("p")
    pg = best_p[best_p.pl == "good"]
    out["p_good"] = len(pg)
    out["p_right"] = int(((pg.ul == "good") & (pg.jac >= 0.5)).sum())
    for lab in ("mua", "noise", "unsorted"):
        out[f"p_user_{lab}"] = int(((pg.ul == lab) & (pg.jac >= 0.5)).sum())     # same cluster, the curator says ...
    lo = pg[pg.jac < 0.5]
    frag = lo.n / lo.np < 0.5 + 1e-9
    out["p_mixes_several"] = int(frag.sum())                                     # holds several of the curator's clusters
    out["p_fragment_of_good"] = int(((~frag) & (lo.ul == "good")).sum())         # a piece of one good cluster
    out["p_fragment_of_other"] = int(((~frag) & (lo.ul != "good")).sum())
    g = ct[ct.pl == "good"]
    out["p_good_spikes_user_good"] = round(float(g[g.ul == "good"].n.sum() / max(g.n.sum(), 1)), 4)
    return out


# =============================================================================
# Writing the curation into the phy folder
# =============================================================================
def folder_state(phy_dir):
    """'fresh' (the sorter's units, labels as exported), 'auto' (curated by
    this script), or 'manual' (splits/merges or label changes made in Phy)."""
    import pandas as pd
    phy = Path(phy_dir)
    if (phy / CURATION_DIR / "curation.json").exists():
        return "auto"
    st = np.load(phy / "spike_templates.npy", mmap_mode="r").reshape(-1)
    scp = phy / "spike_clusters.npy"
    if scp.exists():
        sc = np.load(scp, mmap_mode="r").reshape(-1)
        if len(sc) != len(st) or not np.array_equal(np.asarray(sc), np.asarray(st)):
            return "manual"
    log_file = phy / "phy.log"
    if log_file.exists():  # a label change, split or merge saved in Phy is logged
        try:
            text = log_file.read_text(errors="replace")
            if any(s in text for s in ("Change metadata_group", "Assigned ", "Merge clusters")):
                return "manual"
        except OSError:
            pass
    qc, grp = phy / "quality_check_labels.csv", phy / "cluster_group.tsv"
    if grp.exists() and qc.exists():
        try:
            g = pd.read_csv(grp, sep="\t")
            q = pd.read_csv(qc, index_col=0)
            ids = np.unique(np.asarray(st))
            qlab = dict(zip(range(len(q)), q["quality_label"].astype(str)))
            glab = dict(zip(g.cluster_id.astype(int), g.iloc[:, 1].astype(str)))
            if any(glab.get(int(c), "unsorted") != qlab.get(int(c), "unsorted") for c in ids):
                return "manual"
        except Exception:
            pass
    return "fresh"


_NOT_METRIC_TSVS = {"cluster_group.tsv", "cluster_info.tsv", "cluster_auto_conf.tsv",
                    "cluster_auto_step.tsv"}


def _original_tsvs(phy, cur_dir):
    """The step-7 cluster_<metric>.tsv files of the sorter's units, kept in
    _auto_curation/original/ (restored from there on --redo). Made on the first
    run: a copy of the files of a fresh export, or, for a folder curated in Phy
    before, rebuilt from the curated files (reconstruct_original_tsvs)."""
    orig = cur_dir / "original"
    if not orig.exists():
        orig.mkdir(parents=True)
        st = np.load(phy / "spike_templates.npy", mmap_mode="r").reshape(-1)
        scp = phy / "spike_clusters.npy"
        sc = np.load(scp, mmap_mode="r").reshape(-1) if scp.exists() else st
        reconstruct_original_tsvs(phy, orig, np.asarray(st), np.asarray(sc))
        if (phy / "quality_check_labels.csv").exists():
            shutil.copy2(phy / "quality_check_labels.csv", orig / "quality_check_labels.csv")
    return orig


def reconstruct_original_tsvs(phy, dest, spike_templates, spike_clusters):
    """cluster_<metric>.tsv rows for the sorter's units from the current files.

    Phy saves only the current clusters, and a split cluster inherits its
    parent's value, so the value of a unit that was split is read from its
    largest descendant. Unchanged units keep their own rows."""
    import pandas as pd
    st = np.asarray(spike_templates, np.int64)
    sc = np.asarray(spike_clusters, np.int64)
    units = np.unique(st)
    current = set(np.unique(sc).tolist())
    pairs, counts = np.unique(np.stack([st, sc], 1), axis=0, return_counts=True)
    kids = {}
    for (t, c), n in zip(pairs, counts):
        kids.setdefault(int(t), []).append((int(c), int(n)))
    for f in sorted(Path(phy).glob("cluster_*.tsv")):
        if f.name in _NOT_METRIC_TSVS:
            continue
        try:
            t = pd.read_csv(f, sep="\t")
        except Exception:
            continue
        if t.shape[1] < 2:
            continue
        vals = dict(zip(t.iloc[:, 0].astype(int), t.iloc[:, 1]))
        rows = []
        for u in units.tolist():
            if u in vals and (u in current or not kids.get(u)):
                rows.append((u, vals[u]))
                continue
            for c, _n in sorted(kids.get(u, []), key=lambda x: -x[1]):
                if c in vals:
                    rows.append((u, vals[c]))
                    break
        pd.DataFrame(rows, columns=t.columns[:2]).to_csv(Path(dest) / f.name, sep="\t", index=False)


def write_curation(phy_dir, data, leaves, unit_rows, leaf_spikes, params, info, tree=None):
    """Write spike_clusters.npy, cluster_group.tsv, inherited cluster_<metric>.tsv,
    the review columns, the logs; clear Phy's cache. Cluster ids follow Phy's
    rule: the sorter's id for a unit kept whole, new ids (max + 1, ...) for pieces
    in the order Phy would create them, level by level (the pieces left by the
    first reclusters in unit order, then those of the second ones, ...), then the
    amplitude cuts. leaves / leaf_spikes: the leaves of curate() and their spikes;
    tree: its `accepted` and `reverted` (for the log)."""
    import pandas as pd
    phy = Path(phy_dir)
    cur = phy / CURATION_DIR
    cur.mkdir(exist_ok=True)
    orig = _original_tsvs(phy, cur)
    st = np.asarray(data.spike_templates, dtype=np.int64)
    sc = st.copy()
    next_id = int(st.max()) + 1
    ids = {}
    for i in sorted(range(len(leaves)), key=lambda i: (leaves[i]["depth"], leaves[i]["unit"],
                                                       _path_order(leaves[i]["path"]))):
        if leaves[i]["depth"] == 0:
            ids[i] = leaves[i]["unit"]
        else:
            ids[i] = next_id
            next_id += 1
    for i in range(len(leaves)):
        sc[leaf_spikes[i]] = ids[i]
    labels = {ids[i]: lf["label"] for i, lf in enumerate(leaves)}
    conf = {ids[i]: lf["conf"] for i, lf in enumerate(leaves)}
    step = {ids[i]: lf["step"] for i, lf in enumerate(leaves)}
    parent = {ids[i]: lf["unit"] for i, lf in enumerate(leaves)}
    cuts = []
    k_sd = params.get("amp_cut_sd")
    if k_sd:
        for cid in sorted(c for c, l in labels.items() if l == "good"):
            sp = np.nonzero(sc == cid)[0]
            cut = amplitude_cut(data.amplitudes, sp, k_sd)
            if not cut.any() or cut.all():
                continue
            noise_id, good_id = next_id, next_id + 1
            next_id += 2
            sc[sp[cut]] = noise_id
            sc[sp[~cut]] = good_id
            labels[noise_id], labels[good_id] = "noise", "good"
            conf[noise_id], conf[good_id] = np.nan, conf[cid]
            step[noise_id], step[good_id] = f"amp>{k_sd:g}SD cut of {cid}", f"{step[cid]}+amp{k_sd:g}SD"
            parent[noise_id] = parent[good_id] = parent[cid]
            a = np.abs(np.asarray(data.amplitudes[sp], np.float64))
            cuts.append(dict(cluster=cid, n_spikes=len(sp), threshold=round(a.mean() + k_sd * a.std(), 2),
                             n_cut=int(cut.sum()), noise_id=noise_id, good_id=good_id))
            for d in (labels, conf, step, parent):
                d.pop(cid)
    np.save(phy / "spike_clusters.npy", sc.astype(np.int32))
    present = sorted(labels)
    pd.DataFrame({"cluster_id": [c for c in present if labels[c] in LABELS],
                  "group": [labels[c] for c in present if labels[c] in LABELS]}
                 ).to_csv(phy / "cluster_group.tsv", sep="\t", index=False)
    pd.DataFrame({"cluster_id": present, "auto_conf": [round(conf[c], 3) if np.isfinite(conf[c]) else ""
                                                        for c in present]}
                 ).to_csv(phy / "cluster_auto_conf.tsv", sep="\t", index=False)
    pd.DataFrame({"cluster_id": present, "auto_step": [step[c] for c in present]}
                 ).to_csv(phy / "cluster_auto_step.tsv", sep="\t", index=False)
    # metric columns: the sorter's values, inherited by the pieces (as Phy does)
    for f in phy.glob("cluster_*.tsv"):
        if f.name not in _NOT_METRIC_TSVS and not (orig / f.name).exists():
            f.unlink()  # a column of the old curation that has no original
    for f in orig.glob("cluster_*.tsv"):
        if f.name in _NOT_METRIC_TSVS:
            continue
        try:
            t = pd.read_csv(f, sep="\t")
            vals = dict(zip(t.iloc[:, 0].astype(int), t.iloc[:, 1]))
            out = [(c, vals[parent[c]]) for c in present if parent[c] in vals]
            pd.DataFrame(out, columns=t.columns[:2]).to_csv(phy / f.name, sep="\t", index=False)
        except Exception as e:
            log(f"could not carry {f.name} over ({e})")
    if (phy / "cluster_info.tsv").exists():
        (phy / "cluster_info.tsv").unlink()       # stale; Phy rewrites it on save
    # Phy's cache describes the old clusters: drop it (state.json minus the selection)
    dphy = phy / ".phy"
    if dphy.exists():
        state = None
        if (dphy / "state.json").exists():
            try:
                state = json.loads((dphy / "state.json").read_text())
                state.pop("ClusterView", None)
            except Exception:
                state = None
        shutil.rmtree(dphy, ignore_errors=True)
        if state is not None:
            dphy.mkdir()
            (dphy / "state.json").write_text(json.dumps(state, indent=2))
    same = same_neuron_pairs(data.spike_samples, sc, labels, parent, data.fs, data.duration)
    _write_logs(cur, leaves, ids, labels, conf, step, parent, unit_rows, cuts, sc, info, same, tree or {})
    counts = {l: sum(1 for c in present if labels[c] == l) for l in ("good", "mua", "noise", "unsorted")}
    return counts, len(present)


def same_neuron_pairs(spike_samples, sc, labels, parent, fs, duration, max_ratio=0.2):
    """Good clusters cut from the same unit that look like ONE neuron: their
    spikes avoid each other within 1.5 ms (a shared refractory period).

    ratio = cross-pairs closer than 1.5 ms (0.3 ms duplicates aside) / the
    number expected if the two fired independently. One neuron ~ 0, two ~ 1.
    Returns rows for the pairs with ratio < max_ratio (merge candidates)."""
    by_unit = {}
    for c, lab in labels.items():
        if lab == "good":
            by_unit.setdefault(parent[c], []).append(c)
    tau, dup = ISI_THRESHOLD_MS / 1000 * fs, DUPLICATE_MS / 1000 * fs
    rows = []
    for unit, cl in by_unit.items():
        if len(cl) < 2:
            continue
        times = {c: np.sort(np.asarray(spike_samples[np.nonzero(sc == c)[0]], np.int64)) for c in cl}
        for i, a in enumerate(cl):
            for b in cl[i + 1:]:
                ta, tb = times[a], times[b]
                if len(ta) < 50 or len(tb) < 50:
                    continue
                lo = np.searchsorted(tb, ta - tau, side="left")
                hi = np.searchsorted(tb, ta + tau, side="right")
                lo_d = np.searchsorted(tb, ta - dup, side="left")
                hi_d = np.searchsorted(tb, ta + dup, side="right")
                close = int(np.sum(hi - lo) - np.sum(hi_d - lo_d))
                expected = 2 * (tau - dup) / fs * len(ta) * len(tb) / max(duration, 1e-9)
                ratio = close / expected if expected > 0 else np.nan
                if ratio < max_ratio:
                    rows.append(dict(unit=unit, cluster_a=a, cluster_b=b, n_a=len(ta), n_b=len(tb),
                                     close_pairs=close, expected=round(expected, 1),
                                     ratio=round(ratio, 3)))
    return rows


def _write_logs(cur, leaves, ids, labels, conf, step, parent, unit_rows, cuts, sc, info, same=(), tree=None):
    import pandas as pd
    tree = tree or {}
    accepted, reverted = tree.get("accepted", {}), tree.get("reverted", {})
    lines = [f"Phy auto-curation v{VERSION} - {info['time']}",
             f"Models trained on: {', '.join(info['datasets'])}",
             f"Settings: {info['params']}", "",
             "Each unit, in id order: what the label model says of the whole unit and P(keep) of its",
             f"recluster_n {' / '.join(str(n) for n in RECLUSTER_NS)} (- = too few spikes to recluster), the reclusters "
             "that were kept, then the clusters.",
             "A cluster is named by its path: n4k2/n2k0 = piece 2 of the unit's recluster_n 4, then piece 0 of",
             "that piece's recluster_n 2. Numbers after a label are the model confidence.", ""]
    rows = []
    by_unit, splits_of = {}, {}
    cut_to = {c["cluster"]: c["good_id"] for c in cuts}
    for i, lf in enumerate(leaves):
        by_unit.setdefault(lf["unit"], []).append(i)
    for k, (n, pk) in accepted.items():
        splits_of.setdefault(int(k.partition("/")[0][1:]), []).append((_path_of(k), n, pk, k in reverted))
    num = lambda v: f"{v:.2f}" if v is not None and np.isfinite(v) else "-"
    for _, u in unit_rows.sort_values("unit").iterrows():
        unit = int(u.unit)
        p = "/".join(num(u.get("p_" + c, np.nan)) for c in LABELS)
        pk = "/".join(num(u.get(f"pkeep_n{n}", np.nan)) for n in RECLUSTER_NS)
        lines.append(f"unit {unit:4d}  {int(u.n_spikes):>9,} sp  snr~{u.snr_wf:6.2f}  "
                     f"t2p {u.trough_to_peak:.2f} ms  P(good/mua/noise)={p}  "
                     f"P(keep) n={'/'.join(str(n) for n in RECLUSTER_NS)}={pk}")
        for path, n, pkeep, rev in sorted(splits_of.get(unit, []), key=lambda s: (len(_path_order(s[0])),
                                                                               _path_order(s[0]))):
            lines.append(f"    recluster_n {n} of {path or 'the unit'}: P(keep) {pkeep:.2f}" +
                         (" - undone, its pieces are all noise" if rev else ""))
        for i in sorted(by_unit.get(unit, []), key=lambda i: _path_order(leaves[i]["path"])):
            lf = leaves[i]
            cid = ids[i]
            where = lf["path"] or "kept whole"
            final = cut_to.get(cid, cid)
            moved = f" -> {final} after the amplitude cut" if final != cid else ""
            lines.append(f"    {where}: cluster {cid} {lf['label']} {lf['conf']:.2f} ({lf['step']}){moved}")
            n_last = _path_order(lf["path"])[-1][0] if lf["path"] else 0
            rows.append(dict(unit=unit, path=lf["path"], depth=lf["depth"], n=n_last, cluster=cid,
                             final_cluster=final, label=lf["label"], conf=round(lf["conf"], 3),
                             step=lf["step"]))
    lines += ["", f"Amplitude cut (|amplitude| > mean + {info['params'].get('amp_cut_sd')} SD) on good clusters:"]
    for c in cuts:
        lines.append(f"    cluster {c['cluster']} ({c['n_spikes']:,} sp, threshold {c['threshold']}): "
                     f"{c['n_cut']} sp -> {c['noise_id']} noise; rest -> {c['good_id']} good")
    lines += ["", "Good clusters cut from one unit that share a refractory period (probably one "
              "neuron: merge them in Phy if the waveforms agree); ratio = cross-ISIs < 1.5 ms / chance:"]
    for r in same:
        lines.append(f"    unit {r['unit']}: clusters {r['cluster_a']} + {r['cluster_b']}  ratio {r['ratio']}")
    if not same:
        lines.append("    none")
    (cur / "cutting_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.DataFrame(list(same), columns=["unit", "cluster_a", "cluster_b", "n_a", "n_b", "close_pairs",
                                      "expected", "ratio"]
                 ).to_csv(cur / "same_neuron_candidates.tsv", sep="\t", index=False)
    pd.DataFrame(rows, columns=["unit", "path", "depth", "n", "cluster", "final_cluster", "label", "conf", "step"]
                 ).to_csv(cur / "cutting_log.tsv", sep="\t", index=False)
    if cuts:
        pd.DataFrame(cuts).to_csv(cur / "amplitude_cuts.tsv", sep="\t", index=False)
    else:
        (cur / "amplitude_cuts.tsv").unlink(missing_ok=True)    # of an earlier curation
    unit_rows.to_csv(cur / "units.tsv", sep="\t", index=False)
    review = [dict(cluster_id=c, label=labels[c], auto_conf=round(conf[c], 3), auto_step=step[c],
                   unit=parent[c], n_spikes=int(np.sum(sc == c)))
              for c in sorted(labels) if labels[c] == "unsorted" or
              (labels[c] in LABELS and np.isfinite(conf[c]) and conf[c] < 0.6)]
    pd.DataFrame(review, columns=["cluster_id", "label", "auto_conf", "auto_step", "unit", "n_spikes"]
                 ).to_csv(cur / "review_list.tsv", sep="\t", index=False)
    (cur / "curation.json").write_text(json.dumps(info, indent=2, default=str))


# =============================================================================
# The step: one phy folder
# =============================================================================
def autocurate_phy_folder(phy_dir, n_jobs=4, redo=False, params=None, data_dir=DATA_DIR,
                          reason="phy_autocurate.py"):
    """Auto-curate one phy_export. Returns a short status string; never raises
    for a folder it skips."""
    phy = Path(phy_dir)
    params = dict(DEFAULT_PARAMS, **(params or {}))
    if not (phy / "params.py").exists():
        return "not a phy folder"
    state = folder_state(phy)
    cur = phy / CURATION_DIR
    if state != "fresh" and not redo:
        log(f"{phy}: already curated ({state}) - left alone (redo is off: AUTO_CURATE_REDO=0)")
        return f"skipped ({state})"
    clusters_tab, splits_tab = load_training_tables(data_dir)    # before the folder is touched
    if state != "fresh":  # start again from the sorter's units; the old curation is replaced
        log(f"{phy}: already curated ({state}) - curating it again from the sorter's units "
            f"(its curation is replaced; no backup)")
        orig = cur / "original"
        if orig.exists():
            for f in phy.glob("cluster_*.tsv"):
                f.unlink()
            for f in orig.glob("cluster_*.tsv"):
                shutil.copy2(f, phy / f.name)
        (cur / "curation.json").unlink(missing_ok=True)
    t0 = time.time()
    models = Models(clusters_tab, splits_tab, params, n_jobs=n_jobs)
    datasets = sorted(set(clusters_tab.dataset) | set(splits_tab.dataset))
    log(f"{phy}: models fit on {models.n_label_rows} labelled clusters / {models.n_split_rows} reclusters "
        f"of {len(datasets)} curation(s)")
    # the sorter units' metric files (the snr column gives the noise level), kept from now on
    cur.mkdir(exist_ok=True)
    _original_tsvs(phy, cur)
    tmp = Path(tempfile.mkdtemp(prefix="autocurate_"))  # per-unit spike lists for the workers
    try:
        sess = Session(phy, tmp, n_jobs)
        data = sess.data
        leaves, tree = curate(sess, models, params)
        n_rev = len(tree["reverted"])
        log(f"{tree['n_units']} units: {len(tree['accepted'])} recluster(s) kept" +
            (f", {n_rev} of them undone again (all pieces noise)" if n_rev else ""))
        spikes = [leaf_spikes(data, lf) for lf in leaves]
        info = dict(version=VERSION, time=_dt.datetime.now().isoformat(timespec="seconds"),
                    reason=reason, params=params, datasets=datasets,
                    n_units=tree["n_units"], n_reclusters=len(tree["accepted"]) - len(tree["reverted"]),
                    runtime_s=round(time.time() - t0))
        counts, n_clusters = write_curation(phy, data, leaves, tree["units"], spikes, params, info, tree)
    finally:
        _release(tmp)
    log(f"{phy}: {n_clusters} clusters {counts} in {(time.time() - t0) / 60:.1f} min. "
        f"Log: {cur / 'cutting_log.txt'}")
    return f"curated: {counts}"


def _release(tmp):
    """Close the workers and the memory maps, then delete the scratch folder
    (Windows cannot delete a file that is still mapped)."""
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:
        pass
    _DATA_CACHE.clear()
    _FINAL_CLUSTERS.clear()
    import gc
    gc.collect()
    shutil.rmtree(tmp, ignore_errors=True)


def find_phy_folders(output_folder):
    op = Path(output_folder)
    if (op / "params.py").exists():
        return [op]
    phys = sorted(op.glob("*_sorting_output/phy_export"))
    if not phys and (op / "phy_export" / "params.py").exists():
        phys = [op / "phy_export"]
    return phys


def curate_sorted_exports(output_folder, skip=(), n_jobs=None):
    """Step 7 on the recordings it does not sort again: curate the phy_export of
    each one under output_folder again from the sorter's units (its curation is
    replaced; no backup). `skip`: the exports step 7 has just made, which its
    post-sorting has curated already. AUTO_CURATE_REDO=0 leaves curated exports
    alone, PHY_AUTOCURATE=0 the whole step-7 auto-curation. Returns
    {phy_export: status}; never raises for one export."""
    if os.environ.get("PHY_AUTOCURATE", "1").strip().lower() in ("0", "false", "no", "off"):
        log("PHY_AUTOCURATE is off; the sorted Phy exports are left as they are.")
        return {}
    done = {Path(p).resolve() for p in skip}
    phys = [p for p in find_phy_folders(output_folder) if p.resolve() not in done]
    if not phys:
        return {}
    redo = flag("AUTO_CURATE_REDO", default=True)
    n_jobs = n_jobs or int(os.environ.get("AUTO_CURATE_JOBS", "4") or 4)
    out = {}
    for i, phy in enumerate(phys, 1):
        log(f"[{i}/{len(phys)}] already sorted: {phy}")
        try:
            out[str(phy)] = autocurate_phy_folder(phy, n_jobs=n_jobs, redo=redo,
                                                  params=env_params(), reason="step 7")
        except Exception as e:
            log(f"FAILED on {phy}: {e}")
            traceback.print_exc()
            out[str(phy)] = f"failed ({e})"
    return out


def env_params():
    """Settings from the environment (the runner exports hm_tracker_paths.txt)."""
    p = {}
    for key, name in (("AUTO_GOOD_MIN_CONF", "good_min_conf"),
                      ("AUTO_GOOD_BOOST_NARROW", "good_boost_narrow"),
                      ("AUTO_SPLIT_BIAS", "split_bias")):
        v = os.environ.get(key, "").strip()
        if v:
            try:
                p[name] = float(v)
            except ValueError:
                log(f"ignoring {key}={v!r} (not a number)")
    v = os.environ.get("AUTO_AMP_CUT_SD", "").strip()
    if v:
        try:
            p["amp_cut_sd"] = float(v) if float(v) > 0 else None
        except ValueError:
            log(f"ignoring AUTO_AMP_CUT_SD={v!r} (not a number)")
    return p


def flag(name, default=False):
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


# =============================================================================
# phy.log: the curator's actions
# =============================================================================
# Every clustering action of the phy_genzel Template GUI is logged: selections,
# recluster prompts, the recluster itself (spike count), the assignment, merges,
# undo / redo, label moves and saves. Cluster ids are never reused (new_cluster_id
# only grows), so the log gives the lineage of every cluster that ever existed.
_LOG_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LOG_LINE = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d+) \[(\w)\] (\S+)\s+(.*)$")
_LOG_SEL = re.compile(r"^Clusters selected: \[([^\]]*)\] \((.*)\)$")
_LOG_SIM = re.compile(r"^Similar clusters selected: \[([^\]]*)\] \((.*)\)$")
_LOG_PROMPT = re.compile(r"^Prompting arguments for (\w+)")
_LOG_RECL = re.compile(r"^Reclustering (\d+) spikes on (\d+) channels\.")
_LOG_ASSIGN = re.compile(r"^Assigned (\d+) spikes\.")
_LOG_MERGE = re.compile(r"^Merge clusters (.+) to (\S+)\.$")
_LOG_GROUP = re.compile(r"^Change metadata_group for clusters (.+) to (\w+)\.$")
_LOG_BIC = re.compile(r"^Selected (\d+) subclusters by BIC")
_LOG_NPINT = re.compile(r"np\.u?int\d+\((\d+)\)")


def _log_ids(s):
    s = _LOG_NPINT.sub(r"\1", s)
    return [int(x) for x in re.findall(r"\d+", s)]


def log_events(path):
    """A phy.log as a list of (t_seconds, kind, payload), doubled lines collapsed."""
    out = []
    prev = None
    in_gui = False
    for raw in Path(path).read_text(errors="replace").splitlines():
        s = _LOG_ANSI.sub("", raw)
        if s == prev:
            continue
        prev = s
        m = _LOG_LINE.match(s)
        if not m:
            continue
        hh, mm, ss, ms, _lvl, _src, msg = m.groups()
        t = int(hh) * 3600 + int(mm) * 60 + int(ss) + float("0." + ms)
        if msg.startswith("Initialize joblib cache dir"):
            out.append((t, "gui", None))          # a phy launch: unsaved actions are gone
            in_gui = False
        elif msg.startswith("Load config file"):
            out.append((t, "config", "\\claude\\" in msg))
        elif msg.startswith("Creating GUI"):
            in_gui = True
        elif msg.startswith("Save data to") and "spikes_per_cluster" in msg:
            if in_gui:              # (the supervisor also writes this cache when it starts)
                out.append((t, "save", None))
        elif (m2 := _LOG_SEL.match(msg)):
            out.append((t, "select", _log_ids(m2.group(1))))
        elif (m2 := _LOG_SIM.match(msg)):
            out.append((t, "similar", _log_ids(m2.group(1))))
        elif (m2 := _LOG_PROMPT.match(msg)):
            out.append((t, "prompt", m2.group(1)))
        elif msg.startswith("User cancelled input prompt"):
            out.append((t, "cancel", None))
        elif (m2 := _LOG_RECL.match(msg)):
            out.append((t, "recluster", (int(m2.group(1)), int(m2.group(2)))))
        elif msg.startswith("Reclustering found a single cluster"):
            out.append((t, "single", None))
        elif (m2 := _LOG_BIC.match(msg)):
            out.append((t, "bic", int(m2.group(1))))
        elif (m2 := _LOG_ASSIGN.match(msg)):
            out.append((t, "assign", int(m2.group(1))))
        elif (m2 := _LOG_MERGE.match(msg)):
            out.append((t, "merge", (_log_ids(m2.group(1)), _log_ids(m2.group(2))[0])))
        elif (m2 := _LOG_GROUP.match(msg)):
            out.append((t, "group", (_log_ids(m2.group(1)), m2.group(2))))
        elif msg.startswith("Undo cluster assign"):
            out.append((t, "undo_assign", None))
        elif msg.startswith("Redo cluster assign"):
            out.append((t, "redo_assign", None))
        elif msg.startswith("Undo move"):
            out.append((t, "undo_move", None))
        elif msg.startswith("Redo move"):
            out.append((t, "redo_move", None))
        elif msg.startswith("Lasso has"):
            out.append((t, "lasso", None))
    return out


def replay_log(path, initial_ids=None, initial_groups=None):
    """Replay a phy.log into the curator's actions.

    State = the live clusters {id: group} and, for every cluster that was split
    or merged away, the action that consumed it. A phy launch returns to the last
    save. Where the log contradicts the state (a cluster is selected, labelled or
    cut that the state thinks is gone: a save that did not happen, files restored
    from a copy), the actions that consumed it are rolled back.

    Returns dict(actions, labels, live = {id: group} as saved, sessions, problems, ...).
    action: idx, session, t, kind (recluster_n | isosplit | dip | lasso | merge),
    parents [ids], children [ids], n_spikes (reclusters), undone (undo, not redone),
    saved (False: the GUI was closed without saving it)."""
    ev = log_events(path)
    live = dict.fromkeys(initial_ids or [], "unsorted")     # id -> group
    if initial_groups:
        live.update({c: g for c, g in initial_groups.items() if c in live})
    cur = dict(live=live, consumed={})
    saved = dict(live=copy.copy(live), consumed={})
    n_saved_actions = n_saved_labels = 0
    actions, labels, problems, unknown = [], [], [], set()
    session = 0
    scripted = False
    sel, sim = [], []
    pending_prompt = None
    pending_recl = None           # (kind, n_spikes, n_ch, parents)
    undo_c, redo_c, undo_m, redo_m = [], [], [], []
    n = len(ev)

    def next_select(j):
        """ids of the first cluster-view selection after event j (before another action)."""
        for k in range(j + 1, min(j + 12, n)):
            kind = ev[k][1]
            if kind == "select":
                return ev[k][2]
            if kind in ("assign", "merge", "recluster", "gui", "undo_assign", "redo_assign"):
                return None
        return None

    def rollback(a, st):
        """Take action `a` out of state `st` (and whatever was built on its clusters)."""
        for c in a["children"]:
            if c not in st["live"]:
                b = st["consumed"].get(c)
                if b is not None:
                    b["saved"] = False
                    b["rolled_back"] = True
                    rollback(b, st)
            st["live"].pop(c, None)
        for p, g in zip(a["parents"], a["parent_groups"]):
            st["live"][p] = g
            st["consumed"].pop(p, None)

    def ensure_live(ids, t, what):
        for x in ids:
            if x in cur["live"]:
                continue
            a = cur["consumed"].get(x)
            if a is None:
                unknown.add(x)
                problems.append(f"session {session} t={t:.0f}: {what} cluster {x}, of unknown origin")
                for st in (cur, saved):
                    st["live"].setdefault(x, "unsorted")
                continue
            problems.append(f"session {session} t={t:.0f}: {what} cluster {x}, cut earlier by action "
                            f"{a['idx']} ({a['kind']}, session {a['session']}): that action was not kept")
            a["saved"] = False
            a["rolled_back"] = True
            for st in (cur, saved):
                if st["consumed"].get(x) is a:
                    rollback(a, st)
                else:
                    st["live"].setdefault(x, "unsorted")

    def apply(a):
        grp = a["parent_groups"]
        for p in a["parents"]:
            cur["live"].pop(p, None)
            cur["consumed"][p] = a
        inherit = grp[0] if grp and all(g == grp[0] for g in grp) else "unsorted"
        for c in a["children"]:
            cur["live"][c] = inherit

    def unapply(a):
        for c in a["children"]:
            cur["live"].pop(c, None)
        for p, g in zip(a["parents"], a["parent_groups"]):
            cur["live"][p] = g
            cur["consumed"].pop(p, None)

    for i, (t, kind, val) in enumerate(ev):
        if kind == "gui":
            session += 1
            scripted = False
            for a in actions[n_saved_actions:]:      # after the last save: lost
                a["saved"] = False
            for lb in labels[n_saved_labels:]:
                lb["saved"] = False
            cur = dict(live=copy.copy(saved["live"]), consumed=copy.copy(saved["consumed"]))
            n_saved_actions, n_saved_labels = len(actions), len(labels)
            sel, sim, pending_prompt, pending_recl = [], [], None, None
            undo_c, redo_c, undo_m, redo_m = [], [], [], []
        elif kind == "config":
            scripted = bool(val)
        elif kind == "save":
            saved = dict(live=copy.copy(cur["live"]), consumed=copy.copy(cur["consumed"]))
            n_saved_actions, n_saved_labels = len(actions), len(labels)
        elif kind == "select":
            sel, sim = list(val), []
            ensure_live(sel, t, "selected")
        elif kind == "similar":
            sim = list(val)
            ensure_live(sim, t, "selected")
        elif kind == "prompt":
            pending_prompt = val
        elif kind == "cancel":
            pending_prompt = None
        elif kind == "recluster":
            how = {"recluster_n": "recluster_n", "recluster_dip": "dip"}.get(pending_prompt, "isosplit")
            pending_recl = (how, val[0], val[1], list(dict.fromkeys(sel + sim)))
            pending_prompt = None
        elif kind == "single":
            if pending_recl:
                actions.append(dict(idx=len(actions), session=session, t=t, kind=pending_recl[0],
                                    parents=pending_recl[3], children=[], n_spikes=pending_recl[1],
                                    parent_groups=[cur["live"].get(p, "unsorted") for p in pending_recl[3]],
                                    undone=False, saved=True, single=True, scripted=scripted))
            pending_recl = None
        elif kind == "assign":
            kids = next_select(i)
            if pending_recl and pending_recl[1] == val:
                how, n_sp, _n_ch, parents = pending_recl
            else:
                how, n_sp, parents = "lasso", val, list(dict.fromkeys(sel + sim))
            pending_recl = None
            if kids is None:
                problems.append(f"session {session} t={t:.0f}: assign of {val} spikes without a selection after it")
                kids = []
            ensure_live(parents, t, f"{how} of")
            a = dict(idx=len(actions), session=session, t=t, kind=how, parents=parents, children=list(kids),
                     n_spikes=n_sp, undone=False, saved=True, scripted=scripted,
                     parent_groups=[cur["live"].get(p, "unsorted") for p in parents])
            apply(a)
            actions.append(a)
            undo_c.append(a)
            redo_c = []
        elif kind == "merge":
            parents, to = val
            ensure_live(parents, t, "merge of")
            a = dict(idx=len(actions), session=session, t=t, kind="merge", parents=list(parents),
                     children=[to], n_spikes=None, undone=False, saved=True, scripted=scripted,
                     parent_groups=[cur["live"].get(p, "unsorted") for p in parents])
            apply(a)
            actions.append(a)
            undo_c.append(a)
            redo_c = []
        elif kind == "undo_assign":
            if undo_c:
                a = undo_c.pop()
                a["undone"] = True
                unapply(a)
                redo_c.append(a)
            else:
                problems.append(f"session {session} t={t:.0f}: undo of a cluster action with an empty stack")
        elif kind == "redo_assign":
            if redo_c:
                a = redo_c.pop()
                a["undone"] = False
                apply(a)
                undo_c.append(a)
            else:
                problems.append(f"session {session} t={t:.0f}: redo of a cluster action with an empty stack")
        elif kind == "group":
            ids, g = val
            ensure_live(ids, t, "label of")
            lb = dict(idx=len(labels), session=session, t=t, ids=list(ids), group=g,
                      old=[cur["live"].get(c) for c in ids], undone=False, saved=True,
                      after_action=len(actions), scripted=scripted)
            for c in ids:
                cur["live"][c] = g
            labels.append(lb)
            undo_m.append(lb)
            redo_m = []
        elif kind == "undo_move":
            if undo_m:
                lb = undo_m.pop()
                lb["undone"] = True
                for c, g in zip(lb["ids"], lb["old"]):
                    if c in cur["live"] and g is not None:
                        cur["live"][c] = g
                redo_m.append(lb)
        elif kind == "redo_move":
            if redo_m:
                lb = redo_m.pop()
                lb["undone"] = False
                for c in lb["ids"]:
                    if c in cur["live"]:
                        cur["live"][c] = lb["group"]
                undo_m.append(lb)
    for a in actions[n_saved_actions:]:              # never saved at the end of the log
        a["saved"] = False
    for lb in labels[n_saved_labels:]:
        lb["saved"] = False
    return dict(actions=actions, labels=labels, live=saved["live"], live_unsaved=cur["live"],
                sessions=session, problems=problems, n_events=n, unknown=sorted(unknown))


def replay_curation(phy_dir):
    """The replay of a curated phy folder's phy.log, checked against the saved
    curation: dict(n_units, actions, labels, live = the labels the replay ends
    with, final = cluster_group.tsv, missing / extra / wrong = the labelled
    clusters the replay does not give / gives but the curation has not / gives
    another label, problems). As it is stored in a bank folder (replay.json).
    Reads only."""
    import pandas as pd
    phy = Path(phy_dir)
    st = np.load(phy / "spike_templates.npy", mmap_mode="r").reshape(-1)
    units = [int(u) for u in np.unique(np.asarray(st))]
    final = {}
    if (phy / "cluster_group.tsv").exists():
        g = pd.read_csv(phy / "cluster_group.tsv", sep="\t")
        final = dict(zip(g.cluster_id.astype(int), g.iloc[:, 1].astype(str)))
    init_groups = {}
    qc = phy / "quality_check_labels.csv"
    if qc.exists():
        q = pd.read_csv(qc, index_col=0)
        init_groups = dict(zip(range(len(q)), q["quality_label"].astype(str)))
    if (phy / "phy.log").exists():
        r = replay_log(phy / "phy.log", initial_ids=units, initial_groups=init_groups)
    else:
        r = dict(actions=[], labels=[], live=dict.fromkeys(units, "unsorted"), problems=["no phy.log"])
        r["live"].update({c: g for c, g in init_groups.items() if c in r["live"]})
    live = r["live"]
    labelled = {c: v for c, v in final.items() if v in LABELS}
    missing = sorted(set(labelled) - set(live))             # in the curation, not in the replay
    extra = sorted(c for c in live if c not in final and live[c] in LABELS)
    wrong = sorted(c for c in labelled if c in live and live[c] != labelled[c])
    R = dict(phy=str(phy), n_units=len(units), actions=r["actions"], labels=r["labels"],
             live={int(k): v for k, v in live.items()}, final={int(k): v for k, v in final.items()},
             missing=missing, extra=extra, wrong=wrong, problems=r["problems"])
    return json.loads(json.dumps(R))                        # as it is read back from replay.json


# =============================================================================
# The curator bank: what the curator saw and decided
# =============================================================================
# For one hand-curated phy_export, from the replayed phy.log:
#   * the spike set of every cluster that ever existed (the sorter's units; the pieces of
#     every recluster_n, kept or undone - the recluster replica is exact; lasso pieces from
#     the final curation), checked against the final spike_clusters.npy;
#   * features of every such cluster, of every candidate recluster (n = 2, 3, 4) the
#     curator did NOT make on a final cluster, and of every pair of sibling pieces;
#   * the decisions: recluster attempts (n, kept / undone), merges, labels.
# Tables (tab-separated, in the bank folder):
#   clusters.tsv  one row per spike set: key, id, unit, depth, parent, split + features
#   splits.tsv    one row per recluster / candidate: split, parent, n, source, kept, ...
#   pairs.tsv     one row per pair of sibling pieces: split, a, b + pair features
#   actions.tsv   the curator's actions with the simulation status
#   sets.json     the size of every spike set; which final clusters were reproduced exactly
#   replay.json   the replay of phy.log (replay_curation)
_FINAL_CLUSTERS = {}


def _final_clusters(phy_dir):
    if str(phy_dir) not in _FINAL_CLUSTERS:
        _FINAL_CLUSTERS[str(phy_dir)] = np.load(Path(phy_dir) / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    return _FINAL_CLUSTERS[str(phy_dir)]


def _scratch_originals(phy, cache):
    """The sorter units' metric files (the snr column gives the noise level) of a
    folder curated by hand, which has none in _auto_curation/original/: rebuilt
    from the curated files into the scratch folder `cache` (noise_from_snr looks
    there), so that the phy folder is only read."""
    if not (Path(phy) / CURATION_DIR / "original").exists():
        _original_tsvs(Path(phy), Path(cache))


def _group_task(phy_dir, cache_dir, grp):
    """Worker: one connected group of sorter units (usually one): simulate the
    curator's actions on it and compute every feature row. grp: units, actions."""
    data = get_data(phy_dir, cache_dir)
    sc = _final_clusters(phy_dir)
    noise = data.noise
    n_pcs = int(data.pc_features.shape[1])
    units = grp["units"]
    acts = sorted(grp["actions"], key=lambda a: a["idx"])
    sets = {u: data.spikes_of(u) for u in units}
    root = {u: u for u in units}
    depth = {u: 0 for u in units}
    all_sp = np.sort(np.concatenate([sets[u] for u in units])) if units else np.zeros(0, np.int64)
    fin = np.asarray(sc[all_sp])
    truth = {}
    order = np.argsort(fin, kind="stable")
    ids, starts = np.unique(fin[order], return_index=True)
    for c, s0, s1 in zip(ids, starts, list(starts[1:]) + [len(order)]):
        truth[int(c)] = all_sp[np.sort(order[s0:s1])]
    final_ids = set(truth)
    tet = data.unit_tetrode[units[0]]
    channels = np.unique(np.concatenate([data.unit_channels[u] for u in units]))
    others = [(f"u{v}", data.spikes_of(v)) for v in data.units_on_tetrode(tet) if v not in units]
    ctx = {u: others + [(f"c{v}", sets[v]) for v in units if v != u] for u in units}
    uamp = {u: float(np.median(np.abs(np.asarray(data.amplitudes[sets[u]], np.float64)))) if len(sets[u]) else 0.0
            for u in units}

    kept = [a for a in acts if a["saved"] and not a["undone"] and not a.get("single") and a["children"]]
    consumed = {}
    for a in kept:
        for p in a["parents"]:
            consumed[p] = a
    seen = set()

    def get_truth(c):
        if c in truth:
            return truth[c]
        if c in seen:
            return None
        seen.add(c)
        a = consumed.get(c)
        if a is None or a["kind"] == "merge" or len(a["parents"]) != 1:
            return None
        parts = [get_truth(k) for k in a["children"]]
        if any(p is None for p in parts):
            return None
        truth[c] = np.sort(np.concatenate(parts))
        return truth[c]

    xcache = {}

    def X(key, spikes):
        if key not in xcache:
            xcache[key] = pc_features(data, spikes, channels)
        return xcache[key]

    rows, splits, pairs, status = [], [], [], {}

    def cluster_row(key, cid, spikes, unit, d, parent_key, split_key, nbs, x, parent_x=None, sib_x=()):
        f = intrinsic_features(data, spikes, unit, [s for _, s in nbs], key, noise, uamp.get(unit))
        f.update(extra_cluster_features(data, spikes, x))
        if parent_x is not None:
            f.update(pc_relative(x, parent_x, list(sib_x), n_pcs))
        f.update(key=key, id=cid, unit=unit, tetrode=tet, depth=d, parent=parent_key, split=split_key)
        rows.append(f)

    def split_rows(split_key, parent_key, parent_spikes, unit, d, pieces, piece_keys, piece_ids, nbs,
                   meta, info, parent_x=None):
        """Feature rows of the pieces of one (candidate) recluster + its pairs."""
        px = parent_x if parent_x is not None else X(parent_key, parent_spikes)
        idx = [np.searchsorted(parent_spikes, p) for p in pieces]
        xs = [px[i] for i in idx]
        for k, (p, pk, pid) in enumerate(zip(pieces, piece_keys, piece_ids)):
            sib = [(piece_keys[j], pieces[j]) for j in range(len(pieces)) if j != k]
            cluster_row(pk, pid, p, unit, d + 1, parent_key, split_key, sib + nbs, xs[k], px,
                        [xs[j] for j in range(len(pieces)) if j != k])
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                si = sample_spikes(pieces[i], N_PAIR_SPIKES, seed_of(piece_keys[i], "pair"))
                sj = sample_spikes(pieces[j], N_PAIR_SPIKES, seed_of(piece_keys[j], "pair"))
                f = pair_features(data, pieces[i], pieces[j],
                                  px[np.searchsorted(parent_spikes, si)], px[np.searchsorted(parent_spikes, sj)])
                f.update(split=split_key, a=piece_keys[i], b=piece_keys[j])
                pairs.append(f)
        s = dict(split=split_key, parent=parent_key, unit=unit, depth=d, n_out=len(pieces),
                 n_parent=len(parent_spikes))
        s.update(meta)
        s.update(info or {})
        splits.append(s)

    # the sorter's units
    for u in units:
        cluster_row(f"c{u}", u, sets[u], u, 0, "", "", ctx[u], X(f"c{u}", sets[u]))
    done_rows = {f"c{u}" for u in units}
    nbs_of = {u: ctx[u] for u in units}
    key_of = {u: f"c{u}" for u in units}
    made = {}                                # piece id -> (split key, parent id, sibling ids)

    # the curator's actions, in order
    tried = defaultdict(set)                 # id -> {n tried on it}
    for a in acts:
        kind, kids, parents = a["kind"], a["children"], a["parents"]
        st = dict(idx=a["idx"], sim="")
        status[a["idx"]] = st
        if a.get("scripted"):
            st["sim"] = "scripted"
            continue
        if kind == "merge":
            if all(p in sets for p in parents) and kids:
                c = kids[0]
                sets[c] = np.sort(np.concatenate([sets[p] for p in parents]))
                root[c] = root[parents[0]]
                depth[c] = max(depth[p] for p in parents)
                gone = {f"c{p}" for p in parents}
                nbs_of[c] = [(k, s) for k, s in nbs_of[parents[0]] if k not in gone]
                key_of[c] = f"c{c}"
                st["sim"] = "merge"
                tk = get_truth(c)
                st["verified"] = bool(tk is not None and np.array_equal(tk, sets[c]))
                # the merged cluster as a piece of the split its parts came from (if siblings)
                mk = {made.get(p, (None,))[0] for p in parents}
                skey, par_id, sib = "", None, []
                if len(mk) == 1 and None not in mk:
                    skey, par_id, sibs0 = made[parents[0]]
                    sib = [j for j in sibs0 if j not in parents]
                    made[c] = (skey, par_id, sib + [c])
                    for j in sib:
                        made[j] = (skey, par_id, sib + [c])
                if f"c{c}" not in done_rows:
                    xm = pc_features(data, sets[c], channels)
                    pxm = X(key_of[par_id], sets[par_id]) if par_id is not None and par_id in sets else None
                    sxs = []
                    if pxm is not None:
                        sxs = [pxm[np.searchsorted(sets[par_id], sets[j])] for j in sib if j in sets]
                    cluster_row(f"c{c}", c, sets[c], root[c], depth[c],
                                key_of[par_id] if par_id is not None else "", skey, nbs_of[c], xm, pxm, sxs)
                    rows[-1]["merged_from"] = ",".join(f"c{p}" for p in parents)
                    done_rows.add(f"c{c}")
            else:
                st["sim"] = "parent unknown"
            continue
        if a.get("single") or not kids:
            st["sim"] = "single"
            continue
        if not all(p in sets for p in parents):
            st["sim"] = "parent unknown"
            continue
        P = np.sort(np.concatenate([sets[p] for p in parents])) if len(parents) > 1 else sets[parents[0]]
        if a["n_spikes"] is not None and len(P) != a["n_spikes"]:
            st["sim"] = f"size mismatch ({len(P)} vs {a['n_spikes']})"
            continue
        p0 = parents[0]
        unit, d = root[p0], max(depth[p] for p in parents)
        pkey = key_of[p0] if len(parents) == 1 else "+".join(key_of[p] for p in parents)
        tk = {k: get_truth(k) for k in kids}
        cand, info, px = None, {}, None
        if kind == "recluster_n":
            for p in parents:
                tried[p].add(len(kids))
            lab, px, info = recluster(data, P, channels, len(kids))
            if lab is not None:
                uniq = np.unique(lab)
                if len(uniq) == len(kids):
                    cand = {k: P[lab == uu] for k, uu in zip(kids, uniq)}
        known = [k for k in kids if tk[k] is not None]
        pieces = None
        if cand is not None:
            if all(np.array_equal(cand[k], tk[k]) for k in known):
                pieces, st["sim"] = cand, ("exact" if known else "replica")
                st["n_verified"] = len(known)
            else:
                st["sim"] = "replica differs"
        if pieces is None:
            # from the final curation: every piece known, or all but one (the remainder)
            if len(known) == len(kids):
                pieces = {k: tk[k] for k in kids}
            elif len(known) == len(kids) - 1:
                rest = np.setdiff1d(P, np.concatenate([tk[k] for k in known]) if known else np.zeros(0, np.int64),
                                    assume_unique=True)
                miss = [k for k in kids if tk[k] is None][0]
                pieces = {k: (tk[k] if tk[k] is not None else rest) for k in kids}
                pieces[miss] = rest
            if pieces is not None:
                if sum(len(v) for v in pieces.values()) != len(P):
                    pieces = None
                    st["sim"] += " / truth inconsistent"
                else:
                    st["sim"] = (st["sim"] + " / " if st["sim"] else "") + "from the final curation"
        if pieces is None:
            st["sim"] = st["sim"] or "unknown pieces"
            continue
        if len(parents) > 1 and px is None:
            px = pc_features(data, P, channels)
        for k in kids:
            sets[k] = pieces[k]
            root[k] = unit
            depth[k] = d + 1
            key_of[k] = f"c{k}"
        nb_parent = nbs_of[p0] if len(parents) == 1 else \
            [(k, s) for k, s in nbs_of[p0] if k not in {f"c{p}" for p in parents}]
        for k in kids:
            nbs_of[k] = [(f"c{j}", pieces[j]) for j in kids if j != k] + nb_parent
        skey = f"a{a['idx']}"
        if len(parents) == 1:
            for k in kids:
                made[k] = (skey, p0, list(kids))
        is_kept = bool(a["saved"] and not a["undone"])
        new = [k for k in kids if f"c{k}" not in done_rows]
        if len(new) == len(kids):
            split_rows(skey, pkey, P, unit, d, [pieces[k] for k in kids], [f"c{k}" for k in kids], kids,
                       nb_parent, dict(source="user", kind=kind, action=a["idx"], kept=is_kept,
                                       undone=bool(a["undone"]), saved=bool(a["saved"]), n=len(kids),
                                       n_parents=len(parents), session=a["session"], t=a["t"]),
                       info, px if px is not None else None)
            done_rows.update(f"c{k}" for k in kids)
        else:                                 # a redo: the same pieces again
            st["sim"] += " (redo)"

    # final clusters whose history the log does not give (pieces of an unlogged cut)
    for c in sorted(final_ids):
        if c in sets:
            continue
        sets[c] = truth[c]
        tpl, cnt = np.unique(np.asarray(data.spike_templates[truth[c]]), return_counts=True)
        root[c] = int(tpl[np.argmax(cnt)])
        depth[c] = -1
        key_of[c] = f"c{c}"
        nbs_of[c] = ctx.get(root[c], others)
        cluster_row(f"c{c}", c, sets[c], root[c], -1, "", "", nbs_of[c], X(f"c{c}", sets[c]))
        done_rows.add(f"c{c}")
    # candidate reclusters the curator did not make, on the final clusters
    for c in sorted(final_ids):
        sp = sets[c]
        if len(sp) < MIN_SPIKES_TO_SPLIT:
            continue
        unit, d = root[c], depth[c]
        ckey = key_of[c]
        px = X(ckey, sp)
        for n in RECLUSTER_NS:
            if n in tried.get(c, ()):
                continue
            lab, _x, info = recluster(data, sp, channels, n)
            if lab is None:
                continue
            uniq = np.unique(lab)
            if len(uniq) < 2:
                continue
            pieces = [sp[lab == uu] for uu in uniq]
            pkeys = [f"{ckey}/n{n}k{k}" for k in range(len(pieces))]
            split_rows(f"{ckey}/n{n}", ckey, sp, unit, d, pieces, pkeys, [-1] * len(pieces), nbs_of[c],
                       dict(source="cand", kind="candidate", action=-1, kept=False, undone=False, saved=False,
                            n=n, n_parents=1, session=-1, t=np.nan), info, px)
    sizes = {int(c): int(len(s)) for c, s in sets.items()}
    verified = {int(c): bool(c in sets and np.array_equal(sets[c], truth[c])) for c in final_ids}
    return dict(rows=rows, splits=splits, pairs=pairs, status=list(status.values()), sizes=sizes,
                verified=verified, roots={int(c): int(r) for c, r in root.items()},
                depths={int(c): int(v) for c, v in depth.items()})


def build_bank(phy_dir, cache_dir, n_jobs=1, name=None):
    """The curator bank of one hand-curated phy_export, in `cache_dir` (see
    above; `name` = the dataset name written into the tables, default: the
    folder's name). A folder that already has clusters.tsv is reused, not
    computed again. The phy folder is only read: scratch files (per-unit spike
    lists, the sorter units' metric files) go into cache_dir/cache, which is
    deleted at the end. Returns cache_dir."""
    import pandas as pd
    out = Path(cache_dir)
    phy = Path(phy_dir)
    name = name or out.name
    if (out / "clusters.tsv").exists():
        if not (out / "replay.json").exists():       # a bank of before the replay was kept with it
            (out / "replay.json").write_text(json.dumps(replay_curation(phy)))
        return out
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R = replay_curation(phy)
    cache = out / "cache"
    _scratch_originals(phy, cache)
    data = get_data(phy, cache)
    # group the sorter units that the curator's actions connect (merges across units)
    par = {u: u for u in data.units}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    root = {u: u for u in data.units}
    for a in R["actions"]:
        rs = {root[p] for p in a["parents"] if p in root}
        if not rs:
            continue
        r0 = min(rs)
        for r in rs:
            par[find(r)] = find(r0)
        for c in a["children"]:
            root[c] = r0
    groups = defaultdict(lambda: dict(units=[], actions=[]))
    for u in data.units:
        groups[find(u)]["units"].append(u)
    for a in R["actions"]:
        rs = {root[p] for p in a["parents"] if p in root}
        if rs:
            groups[find(min(rs))]["actions"].append(a)
    glist = sorted(groups.values(), key=lambda g: -sum(len(data.spikes_of(u)) for u in g["units"]))
    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_group_task)(str(phy), str(cache), g) for g in glist)
    else:
        res = [_group_task(str(phy), str(cache), g) for g in glist]
    rows = [r for x in res for r in x["rows"]]
    splits = [r for x in res for r in x["splits"]]
    pairs = [r for x in res for r in x["pairs"]]
    status = {s["idx"]: s for x in res for s in x["status"]}
    sizes = {c: n for x in res for c, n in x["sizes"].items()}
    verified = {c: v for x in res for c, v in x["verified"].items()}
    C = pd.DataFrame(rows)
    C["session"] = name
    S = pd.DataFrame(splits)
    S["session"] = name
    Pq = pd.DataFrame(pairs)
    Pq["session"] = name
    A = pd.DataFrame([dict(idx=a["idx"], session_no=a["session"], kind=a["kind"],
                           parents=",".join(map(str, a["parents"])), children=",".join(map(str, a["children"])),
                           n_spikes=a["n_spikes"], undone=a["undone"], saved=a["saved"],
                           scripted=bool(a.get("scripted")), **{k: v for k, v in status.get(a["idx"], {}).items()
                                                                if k != "idx"})
                      for a in R["actions"]])
    S.to_csv(out / "splits.tsv", sep="\t", index=False)
    Pq.to_csv(out / "pairs.tsv", sep="\t", index=False)
    A.to_csv(out / "actions.tsv", sep="\t", index=False)
    with open(out / "sets.json", "w") as fh:
        json.dump(dict(sizes=sizes, verified=verified), fh)
    (out / "replay.json").write_text(json.dumps(R))
    C.to_csv(out / "clusters.tsv", sep="\t", index=False)      # last: its presence marks a finished bank
    n_ok = sum(verified.values())
    sim = A.sim.value_counts().to_dict() if len(A) else {}
    n_user = int((S.source == "user").sum()) if len(S) else 0
    log(f"bank of {name}: {len(C)} cluster rows, {len(S)} reclusters ({n_user} by the curator), "
        f"{len(Pq)} pairs; final clusters reproduced exactly: {n_ok}/{len(verified)}; "
        f"actions: {sim}; {time.time() - t0:.0f} s")
    _release(cache)
    return out


def load_bank(cache_dir):
    """The tables of a bank folder, ready for the models: dict(C = cluster rows
    with context features and final / label / explicit (the curator's label of a
    final cluster; explicit = set by hand, not inherited), S = recluster rows
    (source user | cand, kept), P = pair rows, R = the replay, T = split_table).

    A recluster whose pieces were all merged together again counts as undone
    (undo_merge_backs); one that was neither undone nor saved (the GUI was closed
    on it) says nothing and is dropped."""
    import pandas as pd
    d = Path(cache_dir)
    C = pd.read_csv(d / "clusters.tsv", sep="\t", low_memory=False)
    S = pd.read_csv(d / "splits.tsv", sep="\t")
    P = pd.read_csv(d / "pairs.tsv", sep="\t")
    with open(d / "replay.json") as fh:
        R = json.load(fh)
    for col in ("parent", "split", "merged_from"):
        if col not in C:
            C[col] = ""
        C[col] = C[col].fillna("")
    final = {int(k): v for k, v in R["final"].items()}
    explicit = set()
    for lb in R["labels"]:
        if lb["saved"] and not lb["undone"] and not lb.get("scripted"):
            explicit.update(lb["ids"])
    C["final"] = C.id.isin(set(final))
    C["label"] = C.id.map(final).where(C.final, "")
    C["explicit"] = C.id.isin(explicit) & C.final
    C, S = undo_merge_backs(C, S)
    # a recluster that was neither undone nor saved (the GUI closed on it) says nothing
    lost = (S.source == "user") & (~S.saved.astype(bool)) & (~S.undone.astype(bool))
    S = S[~lost].reset_index(drop=True)
    C = add_ctx(C, P)
    return dict(C=C, S=S, P=P, R=R, T=split_table(C, S, P))


def undo_merge_backs(C, S):
    """A recluster whose pieces were ALL merged together again is an undo done with the
    merge key: the split counts as not kept, the merged cluster is the cluster that was
    cut (same spikes), and what was done to it afterwards was done to that cluster."""
    C = C.copy()
    S = S.copy()
    S["merged_back"] = False
    M = C[C.merged_from != ""]
    if not len(M):
        return C, S
    parts = {}                                   # merged key -> the base pieces it holds

    def expand(key):
        if key in parts:
            return parts[key]
        row = M[M.key == key]
        if not len(row):
            return {key}
        out = set()
        for k in row.merged_from.iloc[0].split(","):
            out |= expand(k)
        parts[key] = out
        return out

    base = C[(C.split != "") & (C.merged_from == "")].groupby("split").key.apply(set).to_dict()
    remap = {}                                   # merged-back key -> the key of the cluster that was cut
    for _, m in M.iterrows():
        if m.split and expand(m.key) == base.get(m.split, set()) and m.parent:
            remap[m.key] = m.parent
            S.loc[S.split == m.split, ["kept", "merged_back"]] = [False, True]
    if not remap:
        return C, S

    def target(k):
        seen = set()
        while k in remap and k not in seen:
            seen.add(k)
            k = remap[k]
        return k

    ix = C.drop_duplicates("key").set_index("key").index
    for mk in remap:
        tk = target(mk)
        if tk not in ix:
            continue
        mrow = C[C.key == mk].iloc[0]
        if mrow.final:                           # the merged cluster is a final one: its label belongs to tk
            C.loc[C.key == tk, ["id", "final", "label", "explicit"]] = [mrow.id, True, mrow.label, mrow.explicit]
    C = C[~C.key.isin(set(remap))].copy()
    C["parent"] = C.parent.map(lambda k: target(k) if k in remap else k)
    S["parent"] = S.parent.map(lambda k: target(k) if k in remap else k)
    # depths again, from the parent links
    par = dict(zip(C.key, C.parent))
    d0 = dict(zip(C.key, C.depth))
    memo = {}

    def depth(k, guard=0):
        if k in memo:
            return memo[k]
        p = par.get(k, "")
        if not p or p not in par or guard > 50:
            memo[k] = d0.get(k, 0)
        else:
            memo[k] = depth(p, guard + 1) + 1
        return memo[k]

    C["depth"] = [depth(k) if par.get(k, "") else d for k, d in zip(C.key, C.depth)]
    kd = dict(zip(C.key, C.depth))
    S["depth"] = [kd.get(p, d) for p, d in zip(S.parent, S.depth)]
    return C, S


# =============================================================================
# Training tables and evaluation on curated folders
# =============================================================================
def bank_label_rows(bank, dataset):
    """Training rows of the label model: the final clusters of a bank labelled
    good / mua / noise. Those whose history the log does not give (depth -1: no
    parent / sibling context) are in the table but not used for the fit."""
    C = bank["C"]
    L = C[C.final & C.label.isin(LABELS)].copy()
    L["dataset"] = dataset
    return L.reindex(columns=CLUSTER_TABLE_COLS).reset_index(drop=True)


def bank_split_rows(bank, dataset, label_forest):
    """Training rows of the keep model: the reclusters of a bank (the curator's
    attempts and the candidates never made) with the opinion of `label_forest` (a
    label forest that did not learn from this bank) of the cluster and its pieces."""
    import pandas as pd
    C, T = bank["C"], bank["T"].reset_index(drop=True)
    p = _proba3(label_forest, _X(C, LABEL_FEATS))
    T = pd.concat([T, stack_features(C, T, p)], axis=1)
    T["dataset"] = dataset
    return T.reindex(columns=SPLIT_TABLE_COLS)


def tables_from_banks(banks, names, n_jobs=1, cand_frac=1.0):
    """(clusters, splits) training rows of the banks `names`, straight from the
    banks (the evaluation refits everything inside each fold). The label model's
    opinion in the split rows is cross-fitted over at most 5 folds of the
    recordings; cand_frac < 1 keeps that share of the never-made candidates."""
    import pandas as pd
    L = {n: bank_label_rows(banks[n], n) for n in names}
    k = min(5, len(names))
    Ts = []
    for fold in [names[i::k] for i in range(k)]:
        rest = [n for n in names if n not in fold]
        clf = _fit_label(pd.concat([L[n] for n in (rest or names)], ignore_index=True), n_jobs)
        for n in fold:
            Ts.append(bank_split_rows(banks[n], n, clf))
    T = pd.concat(Ts, ignore_index=True)
    if cand_frac < 1.0:
        rng = np.random.default_rng(0)
        T = T[(T.source == "user") | (rng.random(len(T)) < cand_frac)].reset_index(drop=True)
    return pd.concat([L[n] for n in names], ignore_index=True), T


def _table_round_trip(df, digits=TABLE_DIGITS):
    """A table as it is read back from its file (floats with `digits` significant digits)."""
    import io
    import pandas as pd
    if not len(df):
        return df
    text = df.to_csv(sep="\t", index=False, float_format=f"%.{digits}g" if digits else None)
    return pd.read_csv(io.StringIO(text), sep="\t", float_precision="round_trip", low_memory=False)


def _concat(frames, columns):
    """pd.concat of the frames that have rows (an empty frame with `columns` if none has)."""
    import pandas as pd
    frames = [f for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def _write_table(df, path, digits=TABLE_DIGITS):
    df.to_csv(path, sep="\t", index=False, float_format=f"%.{digits}g" if digits else None,
              compression={"method": "gzip", "mtime": 0} if str(path).endswith(".gz") else None)


def cmd_train(curated, names, cache_root, n_jobs, data_dir=DATA_DIR, replace=False, cand_frac=CAND_FRAC,
              digits=TABLE_DIGITS):
    """Add hand curations (phy folders with their phy.log) to the training tables
    (a dataset already in the tables is replaced; replace=True starts the tables
    afresh). Their banks are built in cache_root/<name> (or reused). The label
    model's opinion in the recluster rows of a new dataset comes from a label
    forest fit on all OTHER datasets and is stored (the rows of the datasets
    already in the tables are not touched). cand_frac: the share of the candidate
    reclusters the curator never made that is written (seeded per dataset)."""
    import pandas as pd
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    old_c = pd.DataFrame(columns=CLUSTER_TABLE_COLS)
    old_s = pd.DataFrame(columns=SPLIT_TABLE_COLS)
    meta = []
    if not replace and (data_dir / "training_clusters.tsv.gz").exists():
        old_c, old_s = load_training_tables(data_dir, exclude=names)
        if (data_dir / "training_datasets.tsv").exists():
            old_m = pd.read_csv(data_dir / "training_datasets.tsv", sep="\t")
            meta += old_m[old_m.dataset.isin(set(old_c.dataset)) & ~old_m.dataset.isin(names)].to_dict("records")
    banks, L = {}, {}
    for phy, name in zip(curated, names):
        banks[name] = load_bank(build_bank(phy, Path(cache_root) / name, n_jobs, name=name))
        L[name] = _table_round_trip(bank_label_rows(banks[name], name), digits)
    new_s = []
    for phy, name in zip(curated, names):
        R = banks[name]["R"]
        others = _concat([old_c] + [L[m] for m in names if m != name], CLUSTER_TABLE_COLS)
        if not len(_label_rows_for_fit(others)):
            others = L[name]
            log(f"{name}: no other dataset to fit the label model on - its opinion of this dataset's "
                f"reclusters is not cross-fitted (add a second curation)")
        T = bank_split_rows(banks[name], name, _fit_label(others, n_jobs))
        n_cand = int((T.source != "user").sum())
        if cand_frac < 1.0:
            rng = np.random.default_rng(seed_of(name, "cand"))
            T = T[(T.source == "user") | (rng.random(len(T)) < cand_frac)]
        new_s.append(T)
        user = T[T.source == "user"]
        n_lab, n_fit = len(L[name]), len(_label_rows_for_fit(L[name]))
        log(f"{name}: {n_lab} labelled final clusters {L[name].label.value_counts().to_dict()}, "
            f"{len(user)} recluster(s) by the curator ({int(user.kept.astype(bool).sum())} kept), "
            f"{len(T) - len(user)} of {n_cand} candidate reclusters never made")
        if n_fit < n_lab:
            log(f"{name}: phy.log does not give the history of {n_lab - n_fit} of these {n_lab} final clusters "
                f"(the log cannot be replayed into them: cut or merged outside the logged actions, or a log "
                f"that starts after the first curation). They are in the table as label rows with depth -1 "
                f"and are NOT used for training; the other {n_fit} are")
        if R.get("missing") or R.get("wrong"):
            log(f"{name}: the replay of phy.log does not end in the saved labels ({len(R['missing'])} labelled "
                f"clusters it does not give, {len(R['wrong'])} with another label); the saved labels are used")
        meta.append(dict(dataset=name, phy_export=str(phy), n_clusters=n_lab, n_clusters_used=n_fit,
                         n_reclusters=len(user), n_kept=int(user.kept.astype(bool).sum()),
                         n_candidates=len(T) - len(user), built=_dt.date.today().isoformat()))
    clusters = _concat([old_c] + [L[n] for n in names], CLUSTER_TABLE_COLS)
    splits = _concat([old_s] + new_s, SPLIT_TABLE_COLS)
    for col in ("depth", "n_spikes"):
        clusters[col] = clusters[col].astype(int)
    splits["n"] = splits["n"].astype(int)
    _write_table(clusters, data_dir / "training_clusters.tsv.gz", digits)
    _write_table(splits, data_dir / "training_splits.tsv.gz", digits)
    pd.DataFrame(meta).to_csv(data_dir / "training_datasets.tsv", sep="\t", index=False)
    for old in ("training_units.tsv", "training_pieces.tsv"):       # the tables of v1
        (data_dir / old).unlink(missing_ok=True)
    log(f"training tables written to {data_dir}: {len(clusters)} clusters, {len(splits)} reclusters of "
        f"{clusters.dataset.nunique()} curation(s)")


def evaluate_lodo(curated, names, cache_root, n_jobs=1, params=None, cand_frac=CAND_FRAC, out=None):
    """Leave-one-recording-out: fit the models on the banks of the other
    recordings (the label model's opinion in the recluster rows cross-fitted
    inside the fold), curate the held-out recording from its sorter units,
    compare with its hand curation. The curated folders are only read; the
    candidate reclusters of each recording are kept in cache_root/<name>/candidates."""
    import pandas as pd
    params = dict(DEFAULT_PARAMS, **(params or {}))
    banks = {}
    for phy, name in zip(curated, names):
        banks[name] = load_bank(build_bank(phy, Path(cache_root) / name, n_jobs, name=name))
    rows = []
    for phy, test in zip(curated, names):
        t0 = time.time()
        train = [n for n in names if n != test]
        clusters, splits = tables_from_banks(banks, train, n_jobs, cand_frac)
        models = Models(clusters, splits, params, n_jobs=n_jobs)
        tmp = Path(tempfile.mkdtemp(prefix="autocurate_"))
        try:
            _scratch_originals(phy, tmp)
            sess = Session(phy, tmp, n_jobs, cache_dir=Path(cache_root) / test / "candidates")
            sc, grp = curated_labels(phy)
            leaves, tree = curate(sess, models, params)
            pred, plab = leaves_to_pred(sess.data, leaves, params.get("amp_cut_sd"))
        finally:
            _release(tmp)
        r = compare_curations(pred, plab, sc, grp)
        r.update(spike_scores(pred, plab, sc, grp))
        r.update(diagnose(pred, plab, sc, grp))
        r.update(test=test, n_splits=len(tree["accepted"]), n_train=len(train), secs=round(time.time() - t0))
        rows.append(r)
        log(f"test={test} {r}")
        if out:
            pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    return pd.DataFrame(rows)


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--output_folder", help="op folder (or *_sorting_output / phy_export)")
    ap.add_argument("--n_jobs", type=int, default=int(os.environ.get("AUTO_CURATE_JOBS", "4") or 4))
    ap.add_argument("--redo", action="store_true",
                    help="curate curated exports again even if AUTO_CURATE_REDO=0 (the default "
                         "already does: the old curation is replaced)")
    ap.add_argument("--config", default=None, help=argparse.SUPPRESS)
    tr = sub.add_parser("train", help="add hand curations (phy folders with their phy.log) to the training tables")
    tr.add_argument("--curated", nargs="+", required=True)
    tr.add_argument("--names", nargs="+", required=True)
    tr.add_argument("--cache", required=True, help="folder for the curator banks (one sub-folder per name)")
    tr.add_argument("--n_jobs", type=int, default=4)
    tr.add_argument("--data_dir", default=str(DATA_DIR))
    tr.add_argument("--replace", action="store_true",
                    help="start the training tables afresh instead of adding to them")
    tr.add_argument("--cand_frac", type=float, default=CAND_FRAC,
                    help="share of the candidate reclusters the curator never made that is written")
    ev = sub.add_parser("evaluate", help="leave-one-recording-out agreement with curated folders")
    ev.add_argument("--curated", nargs="+", required=True)
    ev.add_argument("--names", nargs="+", required=True)
    ev.add_argument("--cache", required=True)
    ev.add_argument("--n_jobs", type=int, default=4)
    ev.add_argument("--out", default=None)
    ev.add_argument("--cand_frac", type=float, default=CAND_FRAC)
    args = ap.parse_args(argv)
    if args.cmd in ("train", "evaluate") and len(args.curated) != len(args.names):
        ap.error("give one --names entry per --curated folder")
    if args.cmd == "train":
        cmd_train(args.curated, args.names, args.cache, args.n_jobs, args.data_dir, args.replace,
                  args.cand_frac)
        return
    if args.cmd == "evaluate":
        res = evaluate_lodo(args.curated, args.names, args.cache, args.n_jobs, env_params(),
                            args.cand_frac, args.out)
        cols = ["test", "good_f1", "good_recall", "good_precision", "spike_acc", "n_good_pred", "n_good_user",
                "n_unsorted", "n_clusters", "n_splits"]
        print(res[cols].to_string(index=False))
        print("\nmean over the held-out recordings:")
        print(res[cols[1:]].mean().round(3).to_string())
        if args.out:
            res.to_csv(args.out, sep="\t", index=False)
        return
    if not args.output_folder:
        ap.error("--output_folder is required")
    redo = args.redo or flag("AUTO_CURATE_REDO", default=True)  # always redo (no backup)
    phys = find_phy_folders(args.output_folder)
    if not phys:
        log(f"no phy_export under {args.output_folder}")
        return
    for i, phy in enumerate(phys, 1):
        log(f"[{i}/{len(phys)}] {phy}")
        try:
            autocurate_phy_folder(phy, n_jobs=args.n_jobs, redo=redo, params=env_params())
        except Exception as e:
            log(f"FAILED on {phy}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # exit 0 so the batch runner continues (repo convention)
        print(f"\n[FATAL] phy_autocurate crashed: {e}")
        traceback.print_exc()
        sys.exit(0)
