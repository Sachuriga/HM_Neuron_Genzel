"""
Phy auto-curation: recluster + cut + label the units of a tetrode Phy export,
the way they are curated by hand in Phy, with models trained on Phy curations.

What it does to a phy_export (starting from the sorter's original units):
  1. every unit is kept whole (good / mua / noise) or reclustered with the
     phy_genzel fork's recluster_n (Gaussian mixture on the unit's PC features,
     n = 2 or 4, the exact computation of the Template GUI action);
  2. every piece is labelled good / mua / noise, or reclustered once more (n = 4);
     a split whose pieces are all noise is undone (the unit is noise whole);
  3. no merges. Interneurons (narrow waveforms, trough-to-peak <= 0.425 ms) get
     their P(good) weighted x1.5: fast-spiking cells are kept with more short
     ISIs than a pyramidal cell may have. A 'good' call below 0.4 confidence
     (AUTO_GOOD_MIN_CONF) is left unsorted;
  4. in every good cluster, spikes with |amplitude| > mean + 4 SD are cut off as
     a separate noise cluster (amplitude artefacts);
  5. the result is written as Phy would save it: spike_clusters.npy,
     cluster_group.tsv, cluster_<metric>.tsv (new clusters inherit their unit's
     values, as in Phy), cluster_auto_conf.tsv / cluster_auto_step.tsv (two
     review columns), and a log of every cut in phy_export/_auto_curation/.

The decisions come from random forests fit at run time (a few seconds) on the
training tables in phy_autocurate_data/: one row per original unit and per
candidate piece of the Phy curations listed in training_datasets.tsv, with the
curator's decision derived from the final curation by spike overlap. Add a
curation and re-train with `python phy_autocurate.py train --curated <phy_export> ...`.

Runs as the last part of step 7's post-sorting (sorter_common.analyze_and_export,
PHY_AUTOCURATE=0 turns it off) and as runner step 'a' on existing exports:

    python phy_autocurate.py --output_folder <op|sorting_output|phy_export> [--redo] [--n_jobs 4]

Step 'a' skips a phy_export that was already curated (by hand or by this
script) unless --redo / AUTO_CURATE_REDO=1: the current curation is then backed
up to _auto_curation/backup_<time>/ and the curation is redone from the sorter's
units. Needs the recording (params.py dat_path / processed_binary/).

numpy / scipy / pandas / scikit-learn only (no phy, no spikeinterface).
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
import zlib
from pathlib import Path, PureWindowsPath

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cell_type_metrics as ctm  # noqa: E402

VERSION = "1.0"
DATA_DIR = Path(__file__).resolve().parent / "phy_autocurate_data"
CURATION_DIR = "_auto_curation"

# ---- reclustering: the phy_genzel ReclusterPlugin (recluster_n) -------------
GMM_MAX_SPIKES_FIT = 20000   # the mixture is fit on at most this many spikes
GMM_MAX_DIMS = 10            # PCA dimensions before the mixture
MIN_SPIKES_TO_SPLIT = 40     # smaller units are never reclustered
L2_MIN_SPIKES = 1000         # smaller pieces are never reclustered again
# ---- features ---------------------------------------------------------------
N_WAVEFORM_SPIKES = 100      # raw spikes averaged per cluster (as the fork's cell type)
N_ISOLATION_SPIKES = 500     # spikes per cluster in the isolation distance (as SI)
N_NOISE_CHUNKS = 40          # 1-s chunks for the per-channel noise (MAD)
ISI_THRESHOLD_MS = 1.5
DUPLICATE_MS = 0.3
N_TIME_BINS = 60
# ---- decisions ----------------------------------------------------------------
AMP_CUT_SD = 4.0             # amplitude cut: |amp| > mean + 4 SD -> noise
GOOD_MIN_CONF = 0.4          # a 'good' call below this confidence -> unsorted
GOOD_BOOST_NARROW = 1.5      # P(good) weight of narrow-waveform (interneuron) clusters
N_TREES = 500
CLASSES_A = ("good", "mua", "noise", "split2", "split4")
CLASSES_B = ("good", "mua", "noise", "split")
LABELS = ("good", "mua", "noise")

UNIT_FEATS = [
    "snr_wf", "amp_iqr_rel", "amp_p90_rel", "log_n", "log_fr", "log_isr", "log_isf",
    "log_iso", "log_lr", "trough_to_peak", "acg_tau_rise", "presence", "rate_cv",
    "top10_bins_frac", "isi_lt2", "isi_2_4", "isi_4_10", "isi_10_50", "refrac_contrast",
    "wf_ptp_ratio2", "wf_pos_neg_ratio", "wf_flatness", "tet_max_snr", "tet_n_snr15",
    "amp_rel_tetmax",
]
SUB_FEATS = UNIT_FEATS + [
    "n_frac", "amp_rel_unit", "amp_rank", "amp_rel_maxsib", "n_rank", "n_siblings",
    "u_snr", "u_log_isf", "u_log_fr", "u_log_n", "rel_isf", "rel_iso", "rel_snr",
    "rel_presence", "rel_lt2", "level",
]
# Settings that can be changed per run (env / config file / CLI); defaults here.
DEFAULT_PARAMS = {
    "good_min_conf": GOOD_MIN_CONF,
    "amp_cut_sd": AMP_CUT_SD,
    "level2": True,             # recluster pieces predicted 'split'
    "mua_refine": True,         # mua-vs-noise model on pieces called noise
    "cell_type_models": False,  # separate models for narrow vs wide waveforms
    "augment_noise": False,     # pieces of whole-noise units as extra noise rows
    "good_min_conf_narrow": None,  # if set: the unsorted threshold for narrow waveforms
    "split_bias": 1.0,          # > 1 reclusters more readily, < 1 less
    "split_bias_narrow": 1.0,   # extra split_bias for narrow-waveform units (interneurons)
    "good_boost_narrow": GOOD_BOOST_NARROW,  # weight on P(good) of narrow-waveform clusters
    "model": "rf",              # rf | et (extra trees) | hgb (gradient boosting)
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

    def __init__(self, phy_dir, cache_dir=None, need_traces=True):
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
        if need_traces:
            dat = resolve_dat(self.dir, p["dat_path"])
            dtype = np.dtype(p.get("dtype", "int16"))
            offset = int(p.get("offset", 0))
            n_samples = (dat.stat().st_size - offset) // (n_ch * dtype.itemsize)
            self.traces = np.memmap(dat, dtype=dtype, mode="r", offset=offset,
                                    shape=(n_samples, n_ch))
            self.duration = n_samples / self.fs
        else:
            self.duration = float(self.spike_samples[-1]) / self.fs
        self._template_info()
        self._unit_spikes(cache_dir)

    def _template_info(self):
        """Per template: channels by decreasing amplitude (phylib's order), the
        peak channel and its tetrode (channel group)."""
        self.unit_channels, self.unit_peak = {}, {}
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


def noise_levels(data, cache_dir=None):
    """Per-channel noise, MAD / 0.6745 of the (filtered) recording, 40 x 1 s."""
    if cache_dir and (Path(cache_dir) / "noise.npy").exists():
        return np.load(Path(cache_dir) / "noise.npy")
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


def gmm_split(data, spikes, unit, n_clusters):
    """phy_genzel recluster_n: PCA (<= 10 dims, not whitened) of the PC
    features on the unit's channels, then a full-covariance Gaussian mixture
    fit on <= 20000 spikes (seed 0) and applied to all. Labels 0..n-1."""
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    x = pc_features(data, spikes, data.unit_channels[unit])
    x = PCA(n_components=min(GMM_MAX_DIMS, x.shape[1], x.shape[0]), whiten=False,
            random_state=0).fit_transform(x)
    x_fit = x
    if len(x) > GMM_MAX_SPIKES_FIT:
        rng = np.random.default_rng(0)
        x_fit = x[rng.choice(len(x), GMM_MAX_SPIKES_FIT, replace=False)]
    try:
        gmm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                              random_state=0).fit(x_fit)
    except ValueError:
        # degenerate features (e.g. a few identical spikes): the Phy action
        # would fail too -> the cluster cannot be reclustered
        return np.zeros(len(spikes), np.int8)
    lab = gmm.predict(x)
    # Number the pieces by their first spike (a component the mixture left
    # empty is dropped), so piece k is the same set of spikes on every run.
    used, first = np.unique(lab, return_index=True)
    remap = np.full(n_clusters, -1, np.int64)
    remap[used[np.argsort(first)]] = np.arange(len(used))
    return remap[lab].astype(np.int8)


def mean_waveform(data, spikes, channels, seed):
    """Mean raw waveform (4 ms, the cluster's channels) of <= 100 spikes."""
    if data.traces is None or len(spikes) == 0:
        return None
    half = int(round(ctm.WAVEFORM_WINDOW_S * data.fs / 2))
    s = np.asarray(data.spike_samples[np.asarray(spikes)], dtype=np.int64)
    s = s[(s >= half) & (s + half <= data.traces.shape[0])]
    if len(s) == 0:
        return None
    s = np.sort(sample_spikes(s, N_WAVEFORM_SPIKES, seed))
    acc = np.zeros((2 * half, len(channels)))
    for t in s:
        acc += np.asarray(data.traces[t - half:t + half], dtype=np.float64)[:, channels]
    return acc / len(s)


INTRINSIC_KEYS = (
    "n_spikes", "num_spikes", "fr", "isi_violations_ratio", "isi_violations_count",
    "isi_violations_fraction", "presence", "rate_cv", "top10_bins_frac", "isi_lt2", "isi_2_4",
    "isi_4_10", "isi_10_50", "isi_median_ms", "refrac_contrast", "amp_med", "amp_iqr_rel",
    "amp_p90_rel", "trough_to_peak", "peak_channel", "wf_ptp", "wf_ptp_ratio2", "wf_trough",
    "wf_peak_pos", "wf_pos_neg_ratio", "wf_flatness", "snr_wf", "acg_tau_rise",
    "isolation_distance", "l_ratio")


def intrinsic_features(data, spikes, unit, neighbors, key, noise):
    """Features of one spike set that do not depend on its siblings.
    `neighbors`: spike arrays of the clusters it is compared with (isolation)."""
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
    wf = mean_waveform(data, spikes, channels, seed_of(key, "wf"))
    if wf is not None:
        half = wf.shape[0] // 2
        f["trough_to_peak"] = ctm.waveform_trough_to_peak(wf, data.fs, spike_index=half)
        w = ctm.bandpass_filter(wf, data.fs, axis=0)
        ptp = w.max(0) - w.min(0)
        b = int(np.argmax(ptp))
        f["peak_channel"] = int(channels[b])
        f["wf_ptp"] = float(ptp[b])
        f["wf_ptp_ratio2"] = float(np.sort(ptp)[-2] / max(ptp[b], 1e-9)) if len(ptp) > 1 else 0.0
        f["wf_trough"] = float(w[half - 8:half + 8, b].min())
        f["wf_peak_pos"] = float(w[:, b].max())
        f["wf_pos_neg_ratio"] = float(w[:, b].max() / max(-w[:, b].min(), 1e-9))
        edge = np.r_[w[:max(half - 25, 1), b], w[half + 30:, b]]
        f["wf_flatness"] = float(np.std(edge) / max(ptp[b], 1e-9))
        nz = noise[int(channels[b])] if noise is not None else np.nan
        f["snr_wf"] = float(ptp[b] / nz) if np.isfinite(nz) and nz > 0 else np.nan
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


def add_context(df, units_df=None):
    """Derived and context features: logs, siblings (same parent), unit, tetrode.
    `df` rows: key, unit, parent (the spike set they were split from), tetrode,
    level (0 = whole unit, 1 = piece, 2 = piece of a piece). `units_df` = the
    whole-unit rows (for the unit and tetrode context)."""
    import pandas as pd
    df = df.copy()
    df["log_n"] = np.log10(df.num_spikes.clip(lower=1))
    df["log_fr"] = np.log10(df.fr.clip(lower=1e-4))
    df["log_isr"] = np.log10(df.isi_violations_ratio.fillna(0) + 1e-3)
    df["log_isf"] = np.log10(df.isi_violations_fraction.fillna(0) + 1e-5)
    df["log_iso"] = np.log10(df.isolation_distance.clip(lower=0.1))
    df["log_lr"] = np.log10(df.l_ratio.clip(lower=1e-4))
    df["is_narrow"] = (df.trough_to_peak <= ctm.TROUGH_TO_PEAK_BOUNDARY_MS).astype(float)
    U = df[df.level == 0] if units_df is None else units_df
    U = U.set_index("unit")
    tet = U.groupby("tetrode").snr_wf
    tet_max = tet.max()
    tet_n = tet.apply(lambda s: float((s > 15).sum()))
    df["tet_max_snr"] = df.tetrode.map(tet_max)
    df["tet_n_snr15"] = df.tetrode.map(tet_n)
    df["amp_rel_tetmax"] = df.snr_wf / df.tet_max_snr
    df["unit_n"] = df.unit.map(U.n_spikes)
    df["n_frac"] = df.n_spikes / df.unit_n
    df["amp_rel_unit"] = df.amp_med / df.unit.map(U.amp_med)
    u_log = lambda col: df.unit.map(U[col])
    df["u_snr"] = df.unit.map(U.snr_wf)
    df["u_log_isf"] = np.log10(u_log("isi_violations_fraction").fillna(0) + 1e-5)
    df["u_log_fr"] = np.log10(u_log("fr").clip(lower=1e-4))
    df["u_log_n"] = np.log10(u_log("num_spikes").clip(lower=1))
    g = df.groupby("parent")
    df["n_siblings"] = g.key.transform("size")
    df["amp_rank"] = g.amp_med.rank(ascending=False, method="first") - 1
    df["n_rank"] = g.n_spikes.rank(ascending=False, method="first") - 1
    df["amp_rel_maxsib"] = df.amp_med / g.amp_med.transform("max")
    df["rel_isf"] = df.log_isf - g.log_isf.transform("min")
    df["rel_iso"] = df.log_iso - g.log_iso.transform("max")
    df["rel_snr"] = df.snr_wf / g.snr_wf.transform("max")
    df["rel_presence"] = df.presence - g.presence.transform("max")
    df["rel_lt2"] = df.isi_lt2 - g.isi_lt2.transform("min")
    return df


# =============================================================================
# Candidate clusters of a unit: whole, recluster_n 2 / 4, and their pieces split again
# =============================================================================
def _unit_task(phy_dir, cache_dir, unit, todo):
    """Worker: features (and GMM labels) of one unit's candidates.

    todo: dict with
      'whole': bool                         features of the whole unit
      'splits': [2, 4]                      partitions to compute (+ piece features)
      'l2': {(n, k): None or labels}        pieces to recluster (n=4) + child features
      'l2_all_min': int or None             recluster every piece >= this many spikes
    Returns dict(rows=[...], labels={n: int8 array}, l2_labels={(n, k): int8 array})."""
    data = get_data(phy_dir, cache_dir)
    noise = noise_levels(data, cache_dir)
    spikes = data.spikes_of(unit)
    tet = data.unit_tetrode[unit]
    others = [data.spikes_of(v) for v in data.units_on_tetrode(tet) if v != unit]
    out = {"rows": [], "labels": {}, "l2_labels": {}}

    def row(key, sp, nbs, level, parent):
        f = intrinsic_features(data, sp, unit, nbs, key, noise)
        f.update(key=key, unit=unit, tetrode=tet, level=level, parent=parent)
        out["rows"].append(f)

    if todo.get("whole"):
        row(f"u{unit}", spikes, others, 0, f"u{unit}")
    if len(spikes) < MIN_SPIKES_TO_SPLIT:
        return out
    l2_min = todo.get("l2_all_min") or L2_MIN_SPIKES
    for n in todo.get("splits", []):
        if len(spikes) < 10 * n:
            continue
        lab = todo.get("given_labels", {}).get(n)
        if lab is None:
            lab = gmm_split(data, spikes, unit, n)
        out["labels"][n] = lab
        m = int(lab.max()) + 1
        if m < 2:
            continue
        pieces = [spikes[lab == k] for k in range(m)]
        if not todo.get("skip_piece_features"):
            for k in range(m):
                sib = [pieces[j] for j in range(m) if j != k]
                row(f"u{unit}/s{n}k{k}", pieces[k], sib + others, 1, f"u{unit}/s{n}")
        for k in range(m):
            asked = (n, k) in todo.get("l2", {})
            if not (asked or (todo.get("l2_all_min") and len(pieces[k]) >= l2_min)):
                continue
            if len(pieces[k]) < l2_min:
                continue
            lab2 = gmm_split(data, pieces[k], unit, 4)
            m2 = int(lab2.max()) + 1
            if m2 < 2:
                continue
            out["l2_labels"][(n, k)] = lab2
            kids = [pieces[k][lab2 == c] for c in range(m2)]
            sib1 = [pieces[j] for j in range(m) if j != k]
            for c in range(m2):
                sib2 = [kids[j] for j in range(m2) if j != c]
                row(f"u{unit}/s{n}k{k}/c{c}", kids[c], sib2 + sib1 + others, 2,
                    f"u{unit}/s{n}k{k}")
    return out


def run_tasks(phy_dir, cache_dir, todos, n_jobs=1):
    """Run _unit_task for {unit: todo}; largest units first. Returns {unit: result}."""
    units = sorted(todos, key=lambda u: -len(get_data(phy_dir, cache_dir).spikes_of(u)))
    if n_jobs and n_jobs > 1 and len(units) > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_unit_task)(str(phy_dir), str(cache_dir), u, todos[u]) for u in units)
    else:
        res = [_unit_task(str(phy_dir), str(cache_dir), u, todos[u]) for u in units]
    return dict(zip(units, res))


def build_bank(phy_dir, cache_dir, n_jobs=1, l2_min=L2_MIN_SPIKES):
    """Every candidate of every unit, for training and evaluation: the whole
    unit, its recluster_n 2 and 4 pieces, and every piece >= l2_min spikes
    reclustered again (n=4). Saved in cache_dir (bank.tsv + per-spike labels)."""
    import pandas as pd
    cache = Path(cache_dir)
    if (cache / "bank.tsv").exists():
        return load_bank(cache)
    cache.mkdir(parents=True, exist_ok=True)
    data = get_data(phy_dir, cache)
    noise_levels(data, cache)
    t0 = time.time()
    todos = {u: {"whole": True, "splits": [2, 4], "l2_all_min": l2_min} for u in data.units}
    res = run_tasks(phy_dir, cache, todos, n_jobs)
    n = data.n_spikes
    lab = {2: np.full(n, -1, np.int8), 4: np.full(n, -1, np.int8)}
    l2 = {2: np.full(n, -1, np.int8), 4: np.full(n, -1, np.int8)}
    rows = []
    for u, r in res.items():
        rows += r["rows"]
        sp = data.spikes_of(u)
        for nn, labels in r["labels"].items():
            lab[nn][sp] = labels
        for (nn, k), labels in r["l2_labels"].items():
            l2[nn][sp[r["labels"][nn] == k]] = labels
    df = add_context(pd.DataFrame(rows))
    df.to_csv(cache / "bank.tsv", sep="\t", index=False)
    for nn in (2, 4):
        np.save(cache / f"s{nn}_labels.npy", lab[nn])
        np.save(cache / f"s{nn}_l2_labels.npy", l2[nn])
    log(f"bank of {phy_dir}: {len(df)} candidates in {time.time() - t0:.0f} s")
    return load_bank(cache)


def load_bank(cache):
    import pandas as pd
    cache = Path(cache)
    b = {"df": pd.read_csv(cache / "bank.tsv", sep="\t")}
    for nn in (2, 4):
        b[f"s{nn}"] = np.load(cache / f"s{nn}_labels.npy")
        b[f"l2_{nn}"] = np.load(cache / f"s{nn}_l2_labels.npy")
    b["units"] = np.load(cache / "unit_order.npy", mmap_mode="r")
    return b


# =============================================================================
# Training targets from a finished Phy curation (spike overlap)
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


def _composition(final_ids, min_frac=0.03, min_spikes=20):
    """Counts of the final clusters in a spike set, without small fragments
    (amplitude-cut artefacts, lasso cuts of a few spikes)."""
    ids, cnt = np.unique(final_ids, return_counts=True)
    keep = cnt >= max(min_spikes, min_frac * cnt.sum())
    if not keep.any():
        keep = cnt == cnt.max()
    return ids[keep], cnt[keep]


def derive_targets(bank, phy_dir, l2_min=L2_MIN_SPIKES):
    """The curator's decision for every unit and the label of every candidate
    piece, from the final curation of `phy_dir`.

    unit: whole (good / mua / noise / unsorted) when >= 95 % of its spikes
    (fragments aside) are in one final cluster, else split2 / split4, whichever
    partition matches the final clusters better (adjusted Rand index).
    piece of the chosen partition: the label of the final cluster holding
    >= 85 % of it, else 'split' (and its own pieces get the majority label)."""
    from sklearn.metrics import adjusted_rand_score
    sc, grp = curated_labels(phy_dir)
    df = bank["df"].set_index("key")
    st = np.load(Path(phy_dir) / "spike_templates.npy").reshape(-1)
    order = np.argsort(st, kind="stable")
    bounds = np.searchsorted(st[order], np.arange(int(st.max()) + 2))
    lab = lambda c: grp.get(int(c), "unsorted")
    targets = {}
    rng = np.random.default_rng(0)
    for u in sorted(df[df.level == 0].unit.astype(int)):
        sp = order[bounds[u]:bounds[u + 1]]
        fin = sc[sp]
        ids, cnt = _composition(fin)
        if cnt.max() >= 0.95 * cnt.sum() or f"u{u}/s2k0" not in df.index:
            targets[f"u{u}"] = lab(ids[np.argmax(cnt)])
            continue
        keep = np.isin(fin, ids)
        sub = np.nonzero(keep)[0]
        if len(sub) > 100000:
            sub = np.sort(rng.choice(sub, 100000, replace=False))
        scores = {}
        for n in (2, 4):
            if f"u{u}/s{n}k0" in df.index:
                scores[n] = adjusted_rand_score(fin[sub], bank[f"s{n}"][sp][sub])
        n = max(scores, key=scores.get)
        targets[f"u{u}"] = f"split{n}"
        pl = bank[f"s{n}"][sp]
        for k in range(n):
            pk = sp[pl == k]
            if f"u{u}/s{n}k{k}" not in df.index or len(pk) == 0:
                continue
            ids_k, cnt_k = _composition(sc[pk], min_frac=0.05)
            main = ids_k[np.argmax(cnt_k)]
            if cnt_k.max() >= 0.85 * cnt_k.sum() or len(pk) < l2_min:
                targets[f"u{u}/s{n}k{k}"] = lab(main)
                continue
            targets[f"u{u}/s{n}k{k}"] = "split"
            l2 = bank[f"l2_{n}"][pk]
            for c in range(4):
                pc = pk[l2 == c]
                if len(pc) == 0 or f"u{u}/s{n}k{k}/c{c}" not in df.index:
                    continue
                ids_c, cnt_c = np.unique(sc[pc], return_counts=True)
                targets[f"u{u}/s{n}k{k}/c{c}"] = lab(ids_c[np.argmax(cnt_c)])
    return targets


def training_tables(bank, targets, dataset):
    """(units, pieces) training rows: features + 'target' + 'dataset'."""
    df = bank["df"].copy()
    df["target"] = df.key.map(targets)
    df = df[df.target.notna()]
    df["dataset"] = dataset
    units = df[df.level == 0]
    pieces = df[df.level > 0]
    return units, pieces


# =============================================================================
# Models
# =============================================================================
class Models:
    """Random forests fit on the training tables: A (unit decision), B (piece
    label or split again) and M (mua vs noise, for pieces called noise)."""

    def __init__(self, units, pieces, params=None, n_jobs=1):
        from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                                      RandomForestClassifier)
        self.params = dict(DEFAULT_PARAMS, **(params or {}))
        self.n_jobs = n_jobs
        units = units[units.target.isin(CLASSES_A)]
        pieces = pieces[pieces.target.isin(CLASSES_B)]
        kind = self.params.get("model", "rf")

        def rf():
            if kind == "hgb":
                return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                                      class_weight="balanced", random_state=0)
            cls = ExtraTreesClassifier if kind == "et" else RandomForestClassifier
            return cls(n_estimators=N_TREES, min_samples_leaf=2, class_weight="balanced_subsample",
                       random_state=0, n_jobs=n_jobs)
        groups = [None]
        if self.params["cell_type_models"]:
            groups = [0.0, 1.0]
        self.A, self.B, self.M = {}, {}, {}
        for gval in groups:
            u = units if gval is None else units[units.is_narrow.fillna(0) == gval]
            p = pieces if gval is None else pieces[pieces.is_narrow.fillna(0) == gval]
            if gval is not None and (len(u) < 30 or len(p) < 50):
                u, p = units, pieces
            self.A[gval] = rf().fit(_X(u, UNIT_FEATS), u.target.values)
            self.B[gval] = rf().fit(_X(p, SUB_FEATS), p.target.values)
            pm = p[p.target.isin(["mua", "noise"])]
            self.M[gval] = rf().fit(_X(pm, SUB_FEATS), (pm.target == "mua").astype(int).values) \
                if (pm.target == "mua").sum() >= 5 else None

    def _group(self, rows):
        if None in self.A:
            return {None: np.ones(len(rows), bool)}
        nar = rows.is_narrow.fillna(0).values
        return {0.0: nar == 0, 1.0: nar == 1}

    def _proba(self, models, rows, feats, classes):
        out = np.zeros((len(rows), len(classes)))
        for g, m in self._group(rows).items():
            if not m.any():
                continue
            clf = models[g]
            p = clf.predict_proba(_X(rows[m], feats))
            for j, c in enumerate(clf.classes_):
                if c in classes:
                    out[m, classes.index(c)] = p[:, j]
        return out

    def unit_proba(self, rows):
        return self._proba(self.A, rows, UNIT_FEATS, list(CLASSES_A))

    def piece_proba(self, rows):
        return self._proba(self.B, rows, SUB_FEATS, list(CLASSES_B))

    def mua_proba(self, rows):
        out = np.zeros(len(rows))
        for g, m in self._group(rows).items():
            if m.any() and self.M.get(g) is not None:
                out[m] = self.M[g].predict_proba(_X(rows[m], SUB_FEATS))[:, 1]
        return out


def _X(rows, feats):
    x = rows.reindex(columns=feats).astype(float).values
    x[~np.isfinite(x)] = -1
    return x


def load_training_tables(data_dir=DATA_DIR, exclude=()):
    import pandas as pd
    data_dir = Path(data_dir)
    # round_trip: exact floats, so `train` rewrites the rows it keeps unchanged
    u = pd.read_csv(data_dir / "training_units.tsv", sep="\t", float_precision="round_trip")
    p = pd.read_csv(data_dir / "training_pieces.tsv", sep="\t", float_precision="round_trip")
    if exclude:
        u, p = u[~u.dataset.isin(exclude)], p[~p.dataset.isin(exclude)]
    return u, p


# =============================================================================
# Decisions: from candidate features to the final clusters
# =============================================================================
def decide(models, units_df, get_split, get_l2, params=None):
    """The curation of one recording.

    units_df: whole-unit rows (with context features).
    get_split(unit, n) -> piece rows (context features), or None if it cannot split.
    get_l2(unit, n, k) -> child rows of piece k, or None.
    Returns a list of leaves: dict(unit, n, k, c, label, conf, step), where
    n = 0 (whole) / 2 / 4, k = piece, c = child (-1 = none)."""
    params = dict(DEFAULT_PARAMS, **(params or {}))
    leaves = []
    pa = models.unit_proba(units_df)
    for i, (_, u) in enumerate(units_df.iterrows()):
        unit = int(u.unit)
        narrow = bool(u.get("is_narrow", 0) == 1)
        p_raw = dict(zip(CLASSES_A, pa[i]))
        p = _boost(p_raw, narrow, params)
        whole = max(LABELS, key=lambda c: p[c])
        if not wants_split(u, p_raw, params):
            leaves.append(_leaf(unit, 0, -1, -1, whole, p[whole], "asis", params, narrow))
            continue
        n = 2 if p["split2"] > p["split4"] else 4
        pieces = get_split(unit, n)
        if pieces is None or len(pieces) < 2:
            leaves.append(_leaf(unit, 0, -1, -1, whole, p[whole], "asis", params, narrow))
            continue
        pb = models.piece_proba(pieces)
        pm = models.mua_proba(pieces) if params["mua_refine"] else np.zeros(len(pieces))
        unit_leaves = []
        for j, (_, r) in enumerate(pieces.iterrows()):
            k = int(r.key.split("k")[-1].split("/")[0])
            q = _boost(dict(zip(CLASSES_B, pb[j])), r.get("is_narrow", 0) == 1, params)
            best = max(CLASSES_B, key=lambda c: q[c])
            kids = None
            if best == "split" and params["level2"] and r.n_spikes >= L2_MIN_SPIKES:
                kids = get_l2(unit, n, k)
            if kids is not None and len(kids) >= 2:
                pk = models.piece_proba(kids)
                pmk = models.mua_proba(kids) if params["mua_refine"] else np.zeros(len(kids))
                kid_leaves = []
                for jj, (_, rr) in enumerate(kids.iterrows()):
                    c = int(rr.key.split("/c")[-1])
                    qq = _boost(dict(zip(CLASSES_B, pk[jj])), rr.get("is_narrow", 0) == 1, params)
                    lab = max(LABELS, key=lambda x: qq[x])
                    kid_leaves.append(_piece_leaf(unit, n, k, c, lab, qq, pmk[jj], "L2-n4", params,
                                                  rr.get("is_narrow", 0) == 1))
                if any(l["label"] != "noise" for l in kid_leaves):
                    unit_leaves += kid_leaves
                    continue
            lab = max(LABELS, key=lambda x: q[x])
            unit_leaves.append(_piece_leaf(unit, n, k, -1, lab, q, pm[j], f"L1-n{n}", params,
                                           r.get("is_narrow", 0) == 1))
        labs = [l["label"] for l in unit_leaves]
        if all(l == "noise" for l in labs):
            conf = float(np.mean([l["p_noise"] for l in unit_leaves]))
            leaves.append(_leaf(unit, 0, -1, -1, "noise", conf, f"asis(split{n}-reverted)", params,
                                narrow))
        elif n == 2 and all(l in ("good", "unsorted") for l in labs) and "good" in labs:
            conf = float(np.mean([l["p_good"] for l in unit_leaves]))
            leaves.append(_leaf(unit, 0, -1, -1, "good", conf, "asis(split2-reverted)", params,
                                narrow))
        else:
            leaves += unit_leaves
    return leaves


def _boost(q, narrow, params):
    """Probabilities with P(good) weighted for a narrow-waveform cluster
    (good_boost_narrow), renormalised; unchanged otherwise."""
    b = params.get("good_boost_narrow", 1.0)
    if not narrow or b == 1.0:
        return q
    q = dict(q)
    q["good"] = q["good"] * b
    tot = sum(q.values())
    return {c: v / tot for c, v in q.items()} if tot > 0 else q


def wants_split(unit_row, p, params):
    """Whether the unit model asks for a recluster: split2 + split4 (times
    split_bias, and split_bias_narrow for a narrow waveform) beats the best
    whole-unit label, and the unit is big enough."""
    narrow = bool(unit_row.get("is_narrow", 0) == 1)
    p = _boost(p, narrow, params)
    bias = params["split_bias"] * (params.get("split_bias_narrow", 1.0) if narrow else 1.0)
    p_split = (p["split2"] + p["split4"]) * bias
    return unit_row.n_spikes >= MIN_SPIKES_TO_SPLIT and p_split > max(p[c] for c in LABELS)


def _leaf(unit, n, k, c, label, conf, step, params, narrow=False):
    thr = params["good_min_conf"]
    if narrow and params.get("good_min_conf_narrow") is not None:
        thr = params["good_min_conf_narrow"]
    if label == "good" and conf < thr:
        label = "unsorted"
    return dict(unit=unit, n=n, k=k, c=c, label=label, conf=float(conf), step=step,
                p_good=np.nan, p_noise=np.nan)


def _piece_leaf(unit, n, k, c, label, q, p_mua, step, params, narrow=False):
    conf = q[label]
    if label == "noise" and params["mua_refine"] and p_mua >= 0.5:
        label, conf = "mua", p_mua
    leaf = _leaf(unit, n, k, c, label, conf, step, params, narrow)
    leaf.update(p_good=q["good"], p_noise=q["noise"])
    return leaf


def amplitude_cut(amplitudes, spikes, k_sd=AMP_CUT_SD):
    """Spikes of a cluster whose |amplitude| exceeds mean + k SD."""
    a = np.abs(np.asarray(amplitudes[spikes], dtype=np.float64))
    if len(a) < 3:
        return np.zeros(len(spikes), bool)
    return a > a.mean() + k_sd * a.std()


# =============================================================================
# Evaluation against a curation (leave-one-recording-out)
# =============================================================================
def bank_providers(bank):
    """get_split / get_l2 that read the candidates from a bank."""
    df = bank["df"].set_index("key", drop=False)
    return df[df.level == 0].sort_values("unit"), *_providers(df)


def _providers(ix):
    """get_split / get_l2 over a key-indexed candidate table (pieces the
    mixture left empty are simply absent)."""
    def pick(keys):
        keys = [k for k in keys if k in ix.index]
        return ix.loc[keys] if len(keys) >= 2 else None

    def get_split(unit, n):
        return pick([f"u{unit}/s{n}k{k}" for k in range(n)])

    def get_l2(unit, n, k):
        return pick([f"u{unit}/s{n}k{k}/c{c}" for c in range(4)])

    return get_split, get_l2


def leaves_to_spike_ids(leaves, bank, st, amplitudes=None, amp_cut_sd=None):
    """Per-spike predicted cluster (int) and {cluster: label}."""
    order = np.argsort(st, kind="stable")
    bounds = np.searchsorted(st[order], np.arange(int(st.max()) + 2))
    pred = np.full(len(st), -1, np.int64)
    labels = {}
    cid = 0
    for lf in leaves:
        u = lf["unit"]
        sp = order[bounds[u]:bounds[u + 1]]
        if lf["n"]:
            sp = sp[bank[f"s{lf['n']}"][sp] == lf["k"]]
            if lf["c"] >= 0:
                sp = sp[bank[f"l2_{lf['n']}"][sp] == lf["c"]]
        lab = lf["label"]
        if lab == "good" and amp_cut_sd and amplitudes is not None and len(sp):
            cut = amplitude_cut(amplitudes, sp, amp_cut_sd)
            if cut.any():
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


def _backup(phy, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for f in ["spike_clusters.npy", "phy.log", "cluster_info.tsv"]:
        if (phy / f).exists():
            shutil.copy2(phy / f, dest / f)
    for f in phy.glob("cluster_*.tsv"):
        shutil.copy2(f, dest / f.name)
    # .phy is a cache Phy rebuilds (it can be GBs): keep only the GUI state files
    for f in ("state.json", "new_cluster_id.json"):
        if (phy / ".phy" / f).exists():
            (dest / "dot_phy").mkdir(exist_ok=True)
            shutil.copy2(phy / ".phy" / f, dest / "dot_phy" / f)


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


def write_curation(phy_dir, data, leaves, unit_rows, leaf_spikes, params, info):
    """Write spike_clusters.npy, cluster_group.tsv, inherited cluster_<metric>.tsv,
    the review columns, the logs; clear Phy's cache. Cluster ids follow Phy's
    rule: the sorter's id for a unit kept whole, new ids (max + 1, ...) for pieces
    in the order Phy would create them (first-level splits, second-level splits,
    amplitude cuts)."""
    import pandas as pd
    phy = Path(phy_dir)
    cur = phy / CURATION_DIR
    cur.mkdir(exist_ok=True)
    orig = _original_tsvs(phy, cur)
    st = np.asarray(data.spike_templates, dtype=np.int64)
    sc = st.copy()
    next_id = int(st.max()) + 1
    # ids: first-level pieces (unit order), then second-level, then amplitude cuts
    ids = {}
    for phase in (1, 2):
        for i, lf in enumerate(leaves):
            if lf["n"] and ((phase == 1 and lf["c"] < 0) or (phase == 2 and lf["c"] >= 0)):
                ids[i] = next_id
                next_id += 1
    for i, lf in enumerate(leaves):
        if not lf["n"]:
            ids[i] = lf["unit"]
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
    _write_logs(cur, leaves, ids, labels, conf, step, parent, unit_rows, cuts, sc, info, same)
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


def _write_logs(cur, leaves, ids, labels, conf, step, parent, unit_rows, cuts, sc, info, same=()):
    import pandas as pd
    lines = [f"Phy auto-curation v{VERSION} - {info['time']}",
             f"Models trained on: {', '.join(info['datasets'])}",
             f"Settings: {info['params']}", "",
             "Each unit, in id order: the decision with its probabilities, then the clusters.",
             "Numbers after a label are the model confidence.", ""]
    rows = []
    by_unit = {}
    cut_to = {c["cluster"]: c["good_id"] for c in cuts}
    for i, lf in enumerate(leaves):
        by_unit.setdefault(lf["unit"], []).append(i)
    for _, u in unit_rows.sort_values("unit").iterrows():
        unit = int(u.unit)
        p = "/".join(f"{u.get('p_' + c, np.nan):.2f}" for c in CLASSES_A)
        lines.append(f"unit {unit:4d}  {int(u.n_spikes):>9,} sp  snr~{u.snr_wf:6.2f}  "
                     f"t2p {u.trough_to_peak:.2f} ms  P(good/mua/noise/split2/split4)={p}")
        for i in by_unit.get(unit, []):
            lf = leaves[i]
            cid = ids[i]
            where = "kept whole" if not lf["n"] else (f"recluster_n {lf['n']} piece {lf['k']}" +
                                                       (f" -> recluster_n 4 child {lf['c']}" if lf["c"] >= 0 else ""))
            final = cut_to.get(cid, cid)
            moved = f" -> {final} after the amplitude cut" if final != cid else ""
            lines.append(f"    {where}: cluster {cid} {lf['label']} {lf['conf']:.2f} ({lf['step']}){moved}")
            rows.append(dict(unit=unit, n=lf["n"], piece=lf["k"], child=lf["c"], cluster=cid,
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
    pd.DataFrame(rows).to_csv(cur / "cutting_log.tsv", sep="\t", index=False)
    if cuts:
        pd.DataFrame(cuts).to_csv(cur / "amplitude_cuts.tsv", sep="\t", index=False)
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
                          reason="step a"):
    """Auto-curate one phy_export. Returns a short status string; never raises
    for a folder it skips."""
    import pandas as pd
    phy = Path(phy_dir)
    params = dict(DEFAULT_PARAMS, **(params or {}))
    if not (phy / "params.py").exists():
        return "not a phy folder"
    state = folder_state(phy)
    cur = phy / CURATION_DIR
    if state != "fresh" and not redo:
        log(f"{phy}: already curated ({state}) - skipped (use --redo / AUTO_CURATE_REDO=1 "
            f"to back it up and curate it again from the sorter's units)")
        return f"skipped ({state})"
    if state != "fresh":
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = cur / f"backup_{stamp}"
        _backup(phy, dest)
        log(f"{phy}: current curation ({state}) backed up to {dest}")
        orig = cur / "original"
        if orig.exists():
            for f in phy.glob("cluster_*.tsv"):
                f.unlink()
            for f in orig.glob("cluster_*.tsv"):
                shutil.copy2(f, phy / f.name)
        (cur / "curation.json").unlink(missing_ok=True)
    t0 = time.time()
    units_tab, pieces_tab = load_training_tables(data_dir)
    models = Models(units_tab, pieces_tab, params, n_jobs=n_jobs)
    datasets = sorted(set(units_tab.dataset))
    log(f"{phy}: models fit on {len(units_tab)} units / {len(pieces_tab)} pieces of {datasets}")
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="autocurate_"))  # per-unit spike lists for the workers
    data = get_data(phy, tmp)
    noise_levels(data, tmp)
    # 1. whole units
    res = run_tasks(phy, tmp, {u: {"whole": True} for u in data.units}, n_jobs)
    rows = [r for u in data.units for r in res[u]["rows"]]
    units_df = add_context(pd.DataFrame(rows))
    units_df = units_df.sort_values("unit").reset_index(drop=True)
    pa = models.unit_proba(units_df)
    for j, c in enumerate(CLASSES_A):
        units_df[f"p_{c}"] = pa[:, j]
    # 2. the splits the unit model asks for
    want = {}
    for i, u in units_df.iterrows():
        p = {c: u[f"p_{c}"] for c in CLASSES_A}
        if wants_split(u, p, params):
            want[int(u.unit)] = 2 if p["split2"] > p["split4"] else 4
    log(f"{len(units_df)} units: {len(want)} to recluster")
    res2 = run_tasks(phy, tmp, {u: {"splits": [n]} for u, n in want.items()}, n_jobs)
    piece_rows = [r for u in want for r in res2[u]["rows"]]
    labels_of = {u: res2[u]["labels"].get(n) for u, n in want.items()}
    pieces_df = add_context(pd.DataFrame(piece_rows), units_df) if piece_rows else pd.DataFrame()
    # 3. second-level splits for pieces predicted 'split'
    l2_want = {}
    if len(pieces_df) and params["level2"]:
        pb = models.piece_proba(pieces_df)
        for j, (_, r) in enumerate(pieces_df.iterrows()):
            q = _boost(dict(zip(CLASSES_B, pb[j])), r.get("is_narrow", 0) == 1, params)  # as decide()
            if max(CLASSES_B, key=lambda c: q[c]) == "split" and r.n_spikes >= L2_MIN_SPIKES:
                u = int(r.unit)
                n = want[u]
                k = int(r.key.split("k")[-1])
                l2_want.setdefault(u, {})[(n, k)] = None
    res3 = run_tasks(phy, tmp, {u: {"splits": [want[u]], "given_labels": {want[u]: labels_of[u]},
                                     "l2": l2, "skip_piece_features": True}
                                 for u, l2 in l2_want.items()}, n_jobs) if l2_want else {}
    kid_rows = [r for u in res3 for r in res3[u]["rows"] if r["level"] == 2]
    l2_labels = {(u, nk): lab for u in res3 for nk, lab in res3[u]["l2_labels"].items()}
    kids_df = add_context(pd.DataFrame(kid_rows), units_df) if kid_rows else pd.DataFrame()
    log(f"{sum(len(v) for v in l2_want.values())} piece(s) reclustered again")
    import pandas as pd
    cand = pd.concat([d for d in (pieces_df, kids_df) if len(d)]) if (len(pieces_df) or len(kids_df)) \
        else pd.DataFrame(columns=["key"])
    get_split, get_l2 = _providers(cand.set_index("key", drop=False))
    leaves = decide(models, units_df, get_split, get_l2, params)
    leaf_spikes = []
    for lf in leaves:
        sp = data.spikes_of(lf["unit"])
        if lf["n"]:
            sp = sp[labels_of[lf["unit"]] == lf["k"]]
            if lf["c"] >= 0:
                sp = sp[l2_labels[(lf["unit"], (lf["n"], lf["k"]))] == lf["c"]]
        leaf_spikes.append(sp)
    info = dict(version=VERSION, time=_dt.datetime.now().isoformat(timespec="seconds"),
                reason=reason, params=params, datasets=datasets,
                n_units=len(units_df), runtime_s=round(time.time() - t0))
    counts, n_clusters = write_curation(phy, data, leaves, units_df, leaf_spikes, params, info)
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
# Training and evaluation on curated folders
# =============================================================================
def cmd_train(curated, names, cache_root, n_jobs, data_dir=DATA_DIR, replace=False):
    """Add finished Phy curations to the training tables (a dataset already in
    the tables is replaced; --replace starts the tables afresh)."""
    import pandas as pd
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    us, ps, meta = [], [], []
    if not replace and (data_dir / "training_units.tsv").exists():
        old_u, old_p = load_training_tables(data_dir)
        us.append(old_u[~old_u.dataset.isin(names)])
        ps.append(old_p[~old_p.dataset.isin(names)])
        if (data_dir / "training_datasets.tsv").exists():
            old_m = pd.read_csv(data_dir / "training_datasets.tsv", sep="\t")
            meta += old_m[~old_m.dataset.isin(names)].to_dict("records")
    for phy, name in zip(curated, names):
        cache = Path(cache_root) / name
        bank = build_bank(phy, cache, n_jobs)
        targets = derive_targets(bank, phy)
        u, p = training_tables(bank, targets, name)
        us.append(u)
        ps.append(p)
        meta.append(dict(dataset=name, phy_export=str(phy), n_units=len(u), n_pieces=len(p),
                         built=_dt.date.today().isoformat()))
        log(f"{name}: {len(u)} units {u.target.value_counts().to_dict()}, "
            f"{len(p)} pieces {p.target.value_counts().to_dict()}")
    pd.concat(us).to_csv(data_dir / "training_units.tsv", sep="\t", index=False)
    pd.concat(ps).to_csv(data_dir / "training_pieces.tsv", sep="\t", index=False)
    pd.DataFrame(meta).to_csv(data_dir / "training_datasets.tsv", sep="\t", index=False)
    log(f"training tables written to {data_dir}")


def evaluate_lodo(curated, names, cache_root, variants, n_jobs=1):
    """Leave-one-recording-out: fit on the others, curate the held-out
    recording from its candidates, compare with its manual curation."""
    import pandas as pd
    banks, tabs, users, sts, amps = {}, {}, {}, {}, {}
    for phy, name in zip(curated, names):
        banks[name] = build_bank(phy, Path(cache_root) / name, n_jobs)
        t = derive_targets(banks[name], phy)
        tabs[name] = training_tables(banks[name], t, name)
        users[name] = curated_labels(phy)
        sts[name] = np.load(Path(phy) / "spike_templates.npy").reshape(-1)
        amps[name] = np.load(Path(phy) / "amplitudes.npy", mmap_mode="r").reshape(-1)
    out = []
    for vname, params in variants.items():
        for test in names:
            import pandas as pd
            u = pd.concat([tabs[n][0] for n in names if n != test])
            p = pd.concat([tabs[n][1] for n in names if n != test])
            if params.get("augment_noise"):
                p = pd.concat([p] + [_aug_rows(banks[n], tabs[n][0]) for n in names if n != test])
            models = Models(u, p, params, n_jobs=n_jobs)
            units, gs, gl = bank_providers(banks[test])
            leaves = decide(models, units, gs, gl, params)
            pred, plab = leaves_to_spike_ids(leaves, banks[test], sts[test], amps[test],
                                             params.get("amp_cut_sd", AMP_CUT_SD))
            sc, grp = users[test]
            r = compare_curations(pred, plab, sc, grp)
            r.update(variant=vname, test=test)
            out.append(r)
            log(f"{vname:24s} test={test:10s} {r}")
    return pd.DataFrame(out)


def _aug_rows(bank, units_tab):
    """Pieces (n=4) of units curated noise whole, as extra noise examples."""
    df = bank["df"]
    noise_units = set(units_tab[units_tab.target == "noise"].unit.astype(int))
    rows = df[(df.level == 1) & df.key.str.contains("/s4k") & df.unit.isin(noise_units)].copy()
    rows["target"] = "noise"
    rows["dataset"] = units_tab.dataset.iloc[0] if len(units_tab) else ""
    return rows


# =============================================================================
# CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--output_folder", help="op folder (or *_sorting_output / phy_export)")
    ap.add_argument("--n_jobs", type=int, default=int(os.environ.get("AUTO_CURATE_JOBS", "4") or 4))
    ap.add_argument("--redo", action="store_true",
                    help="back up an existing curation and curate again from the sorter's units")
    ap.add_argument("--config", default=None, help=argparse.SUPPRESS)
    tr = sub.add_parser("train", help="build the training tables from curated phy folders")
    tr.add_argument("--curated", nargs="+", required=True)
    tr.add_argument("--names", nargs="+", required=True)
    tr.add_argument("--cache", required=True, help="folder for the candidate banks")
    tr.add_argument("--n_jobs", type=int, default=4)
    tr.add_argument("--data_dir", default=str(DATA_DIR))
    tr.add_argument("--replace", action="store_true",
                    help="start the training tables afresh instead of adding to them")
    ev = sub.add_parser("evaluate", help="leave-one-recording-out agreement with curated folders")
    ev.add_argument("--curated", nargs="+", required=True)
    ev.add_argument("--names", nargs="+", required=True)
    ev.add_argument("--cache", required=True)
    ev.add_argument("--n_jobs", type=int, default=4)
    ev.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "train":
        if len(args.curated) != len(args.names):
            ap.error("give one --names entry per --curated folder")
        cmd_train(args.curated, args.names, args.cache, args.n_jobs, args.data_dir, args.replace)
        return
    if args.cmd == "evaluate":
        res = evaluate_lodo(args.curated, args.names, args.cache, {"default": {}}, args.n_jobs)
        print(res.to_string(index=False))
        if args.out:
            res.to_csv(args.out, sep="\t", index=False)
        return
    if not args.output_folder:
        ap.error("--output_folder is required")
    redo = args.redo or flag("AUTO_CURATE_REDO")
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
