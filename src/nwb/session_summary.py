"""
Runner step [s]: cross-session summary per animal.

Scans a folder tree for session NWBs (written by steps w/u), groups them by animal
(NWB subject_id), and for each animal plots — as a function of session date (labelled
with Repeat & Session) —:
  - number of GOOD and MUA units
  - among GOOD units, number of pyramidal vs interneuron
  - pyramidal spatial information (Skaggs, bits/spike)
  - pyramidal selectivity (peak/mean rate)
  - pyramidal number of place fields
  - pyramidal mean place-field distance to the goal node (m)

Cell type + waveform metrics are read from the NWB when step u stored them, else
recomputed (spike_metrics). Place-field metrics are computed here from the NWB
position + units (reusing visualize_nwb), using the session's dominant goal node.

Usage:
    python session_summary.py --root <folder> [--bin_cm 5] [--speed 0.05]
"""

import re
import sys
import argparse
import traceback
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import f_oneway, ttest_ind
from pynwb import NWBHDF5IO

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:                      # graceful fallback if tqdm is absent
    _HAS_TQDM = False
    def tqdm(it, **k):
        return it


def _log(msg):
    (tqdm.write if _HAS_TQDM else print)(msg)


sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_nwb as V          # load_position, place_field_metrics, load_nodes, ...
import spike_metrics as SM         # waveform_metrics, acg_tau_rise, classify_cell_type

_PF_KEYS = ("spatial_info", "selectivity", "n_fields", "field_goal_m",
            "field_goal_largest_m", "field_goal_2ndlargest_m", "field_goal_smallest_m")
_PF_PLOT = [("spatial_info", "pyramidal spatial information", "bits/spike"),
            ("selectivity", "pyramidal selectivity", "peak/mean"),
            ("n_fields", "pyramidal # place fields", "n fields"),
            ("field_goal_m", "field-to-goal (mean of all fields)", "metres"),
            ("field_goal_largest_m", "field-to-goal (largest field)", "metres"),
            ("field_goal_2ndlargest_m", "field-to-goal (2nd-largest field)", "metres"),
            ("field_goal_smallest_m", "field-to-goal (smallest field)", "metres")]
# 3-way subtype colours for the cell-type scatter
SUBTYPE_COLORS = {"pyramidal": "#2166ac",            # blue
                  "narrow interneuron": "#d62728",   # red
                  "wide interneuron": "#2ca02c"}     # green


def _subtype(cell_type, t2p_s):
    """pyramidal / narrow interneuron (t2p<=0.425ms) / wide interneuron."""
    if cell_type == "pyramidal":
        return "pyramidal"
    if np.isfinite(t2p_s) and t2p_s <= SM.TROUGH_PEAK_THRESH_S:
        return "narrow interneuron"
    return "wide interneuron"


# ------------------------------------------------------------
#                 statistics (ANOVA + post-hoc)
# ------------------------------------------------------------
def _groups_by(units, group_col, value_col, order=None):
    """{group -> finite values} for a metric, using the whole/before epoch rows
    (one value per neuron per session, matching the plotted line)."""
    d = units[units["epoch"].isin(["whole", "before"])]
    out = {}
    for g, sub in d.groupby(group_col):
        v = pd.to_numeric(sub[value_col], errors="coerce").to_numpy()
        out[str(g)] = v[np.isfinite(v)]
    if order is not None:
        out = {str(k): out.get(str(k), np.array([])) for k in order}
    return out


def _oneway_anova(groups):
    """One-way ANOVA across groups (dict or list). Returns (F, p, k, N)."""
    gs = [g for g in (groups.values() if isinstance(groups, dict) else groups) if len(g) >= 2]
    if len(gs) < 2:
        return np.nan, np.nan, len(gs), int(sum(len(g) for g in gs))
    F, p = f_oneway(*gs)
    return float(F), float(p), len(gs), int(sum(len(g) for g in gs))


def _holm(pvals):
    """Holm–Bonferroni adjusted p-values."""
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(np.argsort(pvals)):
        prev = max(prev, min((m - rank) * pvals[idx], 1.0))
        adj[idx] = prev
    return adj


def _posthoc(groups, scope, metric):
    """Pairwise Welch t-tests between session groups with Holm correction."""
    keys = [k for k, v in groups.items() if len(v) >= 2]
    pairs, praw = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = groups[keys[i]], groups[keys[j]]
            tv, p = ttest_ind(a, b, equal_var=False)
            pairs.append((keys[i], keys[j], len(a), len(b), float(np.mean(a)),
                          float(np.mean(b)), float(tv), float(p)))
            praw.append(p)
    padj = _holm(praw) if praw else []
    rows = []
    for (l1, l2, n1, n2, m1, m2, tv, p), pa in zip(pairs, padj):
        rows.append({"scope": scope, "metric": metric, "group1": l1, "group2": l2,
                     "n1": n1, "n2": n2, "mean1": m1, "mean2": m2, "t": tv,
                     "p_raw": p, "p_holm": float(pa), "sig": "*" if pa < 0.05 else ""})
    return rows


def _unit_metrics(nwb, udf, fs=30000.0):
    """Per-unit DataFrame with firing_rate_hz, trough_to_peak_s, acg_tau_rise_ms,
    cell_type and subtype — read from the columns step u stored, else recomputed."""
    n = len(udf)
    dur = max((np.asarray(udf.iloc[i]["spike_times"]).max()
               for i in range(n) if len(udf.iloc[i]["spike_times"])), default=1.0) or 1.0
    has = lambda c: c in udf.columns
    fr = udf["firing_rate_hz"].astype(float) if has("firing_rate_hz") else pd.Series(
        [len(np.asarray(udf.iloc[i]["spike_times"])) / dur for i in range(n)], index=udf.index)
    if has("cell_type") and has("trough_to_peak_s") and has("acg_tau_rise_ms"):
        t2p = udf["trough_to_peak_s"].astype(float)
        tau = udf["acg_tau_rise_ms"].astype(float)
        ct = udf["cell_type"].astype(str)
    else:                                   # recompute (NWB predates step u)
        has_wf = has("waveform_mean")
        ql = udf["quality_label"].astype(str) if has("quality_label") else pd.Series("good", index=udf.index)
        t2p_l, tau_l, ct_l = [], [], []
        for i in range(n):
            r = udf.iloc[i]; st = np.asarray(r["spike_times"], dtype=float)
            v = SM.waveform_metrics(r["waveform_mean"], fs).get("peak_to_trough_s", np.nan) if has_wf else np.nan
            g = SM.acg_tau_rise(st) if ql.iloc[i] == "good" else np.nan
            t2p_l.append(v); tau_l.append(g); ct_l.append(SM.classify_cell_type(fr.iloc[i], v, g))
        t2p = pd.Series(t2p_l, index=udf.index); tau = pd.Series(tau_l, index=udf.index)
        ct = pd.Series(ct_l, index=udf.index)
    out = pd.DataFrame({"firing_rate_hz": fr, "trough_to_peak_s": t2p,
                        "acg_tau_rise_ms": tau, "cell_type": ct}, index=udf.index)
    out["subtype"] = [_subtype(c, v) for c, v in zip(ct, t2p)]
    return out


_DEC_QUALS = ("good", "good_mua")           # file tags
_DEC_LEADS = (0.0, 1.0, 3.0)                 # prediction leads (s) shown in summary


def _dec_key(qtag, lead):
    return f"dec_err_{qtag}_lead{lead:g}"


def _decode_accuracy(session_dir):
    """Median decoding error (m) per unit set AND per prediction lead, read from the
    decoder's (step b) leads_summary_*.npz (falls back to decoded_*.npz for lead 0).
    Returns {} when a session has no decoding output."""
    d = Path(session_dir) / "decoding"
    out = {}
    for qtag in _DEC_QUALS:
        f = d / f"leads_summary_{qtag}.npz"
        if f.exists():
            try:
                z = np.load(f, allow_pickle=True)
                for L, m in zip(np.asarray(z["leads"], float), np.asarray(z["median_err"], float)):
                    out[_dec_key(qtag, float(L))] = float(m)
                continue
            except Exception:
                pass
        g = d / f"decoded_{qtag}.npz"          # fallback: lead-0 only
        if g.exists():
            try:
                z = np.load(g, allow_pickle=True)
                if "err" in z and len(z["err"]):
                    out[_dec_key(qtag, 0.0)] = float(np.median(z["err"]))
            except Exception:
                pass
    return out


def _session_goal(nwb, udf):
    """Dominant goal node id for the session: most common in the Trials_Data table,
    else parsed from session_description ('Goal_Node G'). None if unavailable."""
    try:
        tr = nwb.processing["Behavior"]["Trials_Data"].to_dataframe()
        if "Goal_node" in tr.columns:
            vals = [int(v.decode() if isinstance(v, bytes) else v) for v in tr["Goal_node"]]
            if vals:
                return Counter(vals).most_common(1)[0][0]
    except Exception:
        pass
    m = re.search(r"Goal_Node\s+(\d+)", str(nwb.session_description))
    return int(m.group(1)) if m else None


def collect_session(nwb_path, bin_cm=5.0, sigma=2.0, speed=0.05):
    """Return a dict of per-session summary stats, or None on failure."""
    io = NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True)
    try:
        nwb = io.read()
        subj = nwb.subject.subject_id if nwb.subject is not None else "?"
        desc = str(nwb.session_description)
        # session date: prefer the parent folder name (session folders are named
        # YYYYMMDD), else the NWB session_id, else an 8-digit token in the filename.
        folder = nwb_path.parent.name
        sid = str(nwb.session_id) if nwb.session_id else ""
        if len(folder) == 8 and folder.isdigit():
            date = folder
        elif len(sid) == 8 and sid.isdigit():
            date = sid
        else:
            m = re.search(r"(\d{8})", nwb_path.stem)
            date = m.group(1) if m else (sid or nwb_path.stem)
        rep = re.search(r"Repeat\s+(\d+)", desc)
        ses = re.search(r"Session\s+(\d+)", desc)
        repeat = int(rep.group(1)) if rep else None
        session = int(ses.group(1)) if ses else None
        out = {"animal": f"Rat{int(subj)}" if str(subj).isdigit() else str(subj),
               "date": date, "repeat": repeat, "session": session, "split": False,
               "n_good": 0, "n_mua": 0, "n_pyr": 0, "n_int": 0}
        for qtag in _DEC_QUALS:                        # decoding accuracy per lead
            for L in _DEC_LEADS:
                out[_dec_key(qtag, L)] = np.nan
        for k in _PF_KEYS:
            out[k] = np.nan; out[k + "_post"] = np.nan
        out.update(_decode_accuracy(nwb_path.parent))   # decoding accuracy (step b)
        if nwb.units is None or len(nwb.units.id) == 0:
            return out
        udf = nwb.units.to_dataframe()
        ql = udf["quality_label"].astype(str) if "quality_label" in udf else pd.Series("good", index=udf.index)
        out["n_good"] = int((ql == "good").sum())
        out["n_mua"] = int((ql == "mua").sum())
        um = _unit_metrics(nwb, udf)
        udf["cell_type"] = um["cell_type"]
        good = udf[ql == "good"]
        out["n_pyr"] = int((good["cell_type"] == "pyramidal").sum())
        out["n_int"] = int((good["cell_type"] == "interneuron").sum())
        # per-unit metrics (good units) for the cross-session cell-type scatter
        gm = um.loc[good.index].copy()
        gm.insert(0, "date", out["date"]); gm.insert(0, "animal", out["animal"])
        out["units"] = gm

        pos = V.load_position(nwb)
        pyr = good[good["cell_type"] == "pyramidal"]
        if pos is None or not len(pyr):
            return out
        x = pos[0] / V.SCALE_X; y = pos[1] / V.SCALE_Y; t = pos[2]
        dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0 / 30
        ext = V.MAZE_EXTENT
        bm = bin_cm / 100.0
        bins = (max(5, int(round((ext[1] - ext[0]) / bm))),
                max(5, int(round((ext[3] - ext[2]) / bm))))
        nodes = V.load_nodes()

        # Per-unit datapoints (for statistics) + the session-level means (for the
        # plots). A base row carries the unit's identity + waveform/ACG metrics.
        pyr_idx = list(good.index[good["cell_type"] == "pyramidal"])
        unit_rows = []

        def _base(idx):
            r = um.loc[idx]
            uid = int(udf.loc[idx, "phy_cluster_id"]) if "phy_cluster_id" in udf.columns else int(idx)
            return {"animal": out["animal"], "date": date, "repeat": repeat,
                    "session": session, "unit_id": uid, "quality_label": "good",
                    "cell_type": r["cell_type"], "subtype": r["subtype"],
                    "firing_rate_hz": r["firing_rate_hz"],
                    "trough_to_peak_s": r["trough_to_peak_s"],
                    "acg_tau_rise_ms": r["acg_tau_rise_ms"]}

        def _pyr_epoch(epoch, goal_node, t0, t1):
            """Compute per-pyramidal-unit place-field metrics for one epoch, append
            each unit's datapoint row, and return the session mean per metric."""
            gxy = nodes.get(goal_node)
            means = {k: [] for k in _PF_KEYS}
            for idx in pyr_idx:
                st = np.asarray(udf.loc[idx, "spike_times"], dtype=float)
                m, _, _ = V.place_field_metrics(x, y, t, st, ext, bins, dt, sigma, speed,
                                                goal_xy=gxy, t0=t0, t1=t1)
                d = _base(idx); d["epoch"] = epoch; d["goal_node"] = goal_node
                for k in _PF_KEYS:
                    d[k] = m[k]; means[k].append(m[k])
                unit_rows.append(d)
            return {k: (float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan)
                    for k, v in means.items()}

        # Split into before/after the type-5 (goal-switch) trial ONLY for RxS1
        # sessions with repeat > 1 (R1S1 and all S>1 stay whole-session).
        type5 = None
        if session == 1 and repeat and repeat > 1:
            raw = V.read_trials_raw(nwb_path.parent)
            trials = V.align_trials(raw, nwb.session_start_time,
                                    float(t.min()), float(t.max())) if raw else []
            type5 = next((tr for tr in trials if tr[0] == 5), None)
            if type5 is not None:
                _tt, g5, _sn, t50, t51 = type5
                gb = [g for (tt, g, sn, a, b) in trials if b <= t50 and g is not None]
                goal_before = Counter(gb).most_common(1)[0][0] if gb else None
                pre = _pyr_epoch("before", goal_before, float(t.min()), t50)
                post = _pyr_epoch("after", g5, t51, float(t.max()))
                out["split"] = True
                for k in _PF_KEYS:
                    out[k] = pre[k]; out[k + "_post"] = post[k]
        if type5 is None:
            whole = _pyr_epoch("whole", _session_goal(nwb, udf), None, None)
            for k in _PF_KEYS:
                out[k] = whole[k]

        # interneuron good units: one datapoint row each (no place fields)
        for idx in good.index[good["cell_type"] != "pyramidal"]:
            d = _base(idx); d["epoch"] = "whole"; d["goal_node"] = None
            for k in _PF_KEYS:
                d[k] = np.nan
            unit_rows.append(d)
        out["unit_rows"] = pd.DataFrame(unit_rows)
        return out
    finally:
        io.close()


def _plot_animal(pdf, animal, sessions, units_df=None):
    sessions = sorted(sessions, key=lambda s: s["date"])
    dates = [s["date"] for s in sessions]
    x = np.arange(len(sessions))
    labels = [f"{s['date']}\nR{s['repeat']}·S{s['session']}" for s in sessions]

    def col(key):
        return np.array([s[key] if s[key] is not None else np.nan for s in sessions], dtype=float)

    fig, axes = plt.subplots(5, 2, figsize=(11, 18))
    # units: good vs mua
    ax = axes[0, 0]; w = 0.4
    ax.bar(x - w / 2, col("n_good"), w, label="good", color="#2166ac")
    ax.bar(x + w / 2, col("n_mua"), w, label="mua", color="#b2182b")
    ax.set_title("units: good vs mua"); ax.set_ylabel("count"); ax.legend(fontsize=8)
    # good composition: pyr vs int
    ax = axes[0, 1]
    ax.bar(x - w / 2, col("n_pyr"), w, label="pyramidal", color="#2166ac")
    ax.bar(x + w / 2, col("n_int"), w, label="interneuron", color="#f4a582")
    ax.set_title("good units: pyramidal vs interneuron"); ax.set_ylabel("count"); ax.legend(fontsize=8)
    # pyramidal metrics: "pre / whole" line + "after type5" markers (RxS1 splits)
    any_split = any(s.get("split") for s in sessions)
    metric_axes = [axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1],
                   axes[3, 0], axes[3, 1], axes[4, 0]]
    # decoding accuracy across sessions (median error, step b) at every lead:
    # colour = unit set (good/good+mua), line style = prediction lead (0/1/3 s).
    axd = axes[4, 1]
    _qcol = {"good": "#2166ac", "good_mua": "#b2182b"}
    _qlab = {"good": "good", "good_mua": "good+mua"}
    _lsty = {0.0: "-", 1.0: "--", 3.0: ":"}
    any_dec = False
    for qtag in _DEC_QUALS:
        for L in _DEC_LEADS:
            vals = col(_dec_key(qtag, L))
            if np.isfinite(vals).any():
                any_dec = True
                axd.plot(x, vals, marker="o", ms=4, color=_qcol[qtag],
                         ls=_lsty.get(L, "-"), label=f"{_qlab[qtag]} +{L:g}s")
    if any_dec:
        axd.set_title("decoding accuracy (median error)"); axd.set_ylabel("error (m)")
        axd.legend(fontsize=6, ncol=2)
        axd.set_xticks(x); axd.set_xticklabels(labels, fontsize=6)
        axd.spines["top"].set_visible(False); axd.spines["right"].set_visible(False)
        axd.set_ylim(bottom=0)
    else:
        axd.axis("off")
    for ax, (key, title, ylab) in zip(metric_axes, _PF_PLOT):
        ax.plot(x, col(key), "o-", color="#2166ac",
                label="whole / before type5" if any_split else None)
        if any_split:
            ax.plot(x, col(key + "_post"), "s", color="#d62728", label="after type5")
            ax.legend(fontsize=6)
        ax.set_title(title); ax.set_ylabel(ylab)
        if units_df is not None:        # per-animal one-way ANOVA across sessions
            _F, p, k, N = _oneway_anova(_groups_by(units_df, "date", key, order=dates))
            if np.isfinite(p):
                ax.text(0.98, 0.03, f"1-way ANOVA p={p:.3g} (k={k})", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7,
                        color="#b2182b" if p < 0.05 else "0.3")
    for ax in [axes[0, 0], axes[0, 1]] + metric_axes:
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)          # all summary axes start from 0
    fig.suptitle(f"{animal} — cross-session summary ({len(sessions)} sessions)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pdf.savefig(fig); plt.close(fig)


def _plot_scatter(pdf, animal, units):
    """Scatter of ALL good units across the animal's sessions, coloured by subtype
    (pyramidal blue, narrow interneuron red, wide interneuron green): trough-to-peak
    vs ACG tau_rise, and trough-to-peak vs firing rate."""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 5))
    for sub, c in SUBTYPE_COLORS.items():
        d = units[units["subtype"] == sub]
        axa.scatter(d["trough_to_peak_s"] * 1e3, d["acg_tau_rise_ms"], s=14, alpha=0.6,
                    c=c, label=f"{sub} (n={len(d)})")
        axb.scatter(d["trough_to_peak_s"] * 1e3, d["firing_rate_hz"], s=14, alpha=0.6, c=c)
    axa.axvline(SM.TROUGH_PEAK_THRESH_S * 1e3, ls="--", c="grey", lw=1)
    axa.axhline(SM.ACG_TAU_RISE_THRESH_MS, ls="--", c="grey", lw=1)
    axa.set_xlabel("trough-to-peak (ms)"); axa.set_ylabel("ACG tau_rise (ms)")
    axa.legend(fontsize=7); axa.set_xlim(left=0); axa.set_ylim(bottom=0)
    axb.axvline(SM.TROUGH_PEAK_THRESH_S * 1e3, ls="--", c="grey", lw=1)
    axb.axhline(SM.RATE_THRESH_HZ, ls="--", c="grey", lw=1)
    axb.set_xlabel("trough-to-peak (ms)"); axb.set_ylabel("firing rate (Hz)")
    axb.set_yscale("log"); axb.set_xlim(left=0)
    for ax in (axa, axb):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle(f"{animal} — all good units ({len(units)}) across sessions", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def _label_order(labels):
    def key(s):
        r = re.search(r"R(\d+)", s); ss = re.search(r"S(\d+)", s)
        return (int(r.group(1)) if r else 0, int(ss.group(1)) if ss else 0)
    return sorted(labels, key=key)


def _plot_combined(pdf, units_all):
    """All animals pooled: pyramidal place-field metrics per session (grouped by
    R{repeat}S{session} label), box + mean, annotated with the pooled one-way ANOVA."""
    d = units_all[units_all["epoch"].isin(["whole", "before"])].copy()
    fig, axes = plt.subplots(4, 2, figsize=(11, 17))
    axes.ravel()[-1].axis("off")
    for ax, (key, title, ylab) in zip(axes.ravel(), _PF_PLOT):
        groups = _groups_by(d, "session_label", key)
        labs = [l for l in _label_order(groups) if len(groups[l])]
        data = [groups[l] for l in labs]
        xx = np.arange(len(labs))
        if data:
            ax.boxplot(data, positions=xx, widths=0.6, showfliers=False)
            ax.plot(xx, [np.mean(g) for g in data], "o-", color="#2166ac")
        _F, p, k, N = _oneway_anova(groups)
        if np.isfinite(p):
            ax.text(0.98, 0.97, f"1-way ANOVA p={p:.3g} (k={k}, N={N})", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="#b2182b" if p < 0.05 else "0.3")
        ax.set_title(title); ax.set_ylabel(ylab)
        ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=7)
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    n_an = units_all["animal"].nunique()
    fig.suptitle(f"All animals combined ({n_an}) — pyramidal metrics by session", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


def _stats_tables(animal_units, units_all):
    """Per-animal (grouped by date) + combined (grouped by session label) one-way
    ANOVA and post-hoc pairwise tables for each pyramidal metric."""
    anova, posthoc = [], []
    for animal in sorted(animal_units):
        u = animal_units[animal]
        dates = sorted(u["date"].unique())
        for key in _PF_KEYS:
            g = _groups_by(u, "date", key, order=dates)
            F, p, k, N = _oneway_anova(g)
            anova.append({"scope": animal, "metric": key, "F": F, "p": p, "k_groups": k, "N": N})
            posthoc += _posthoc(g, animal, key)
    for key in _PF_KEYS:
        g = _groups_by(units_all, "session_label", key)
        g = {l: g[l] for l in _label_order(g)}
        F, p, k, N = _oneway_anova(g)
        anova.append({"scope": "ALL", "metric": key, "F": F, "p": p, "k_groups": k, "N": N})
        posthoc += _posthoc(g, "ALL", key)
    return pd.DataFrame(anova), pd.DataFrame(posthoc)


def run(root, bin_cm=5.0, sigma=2.0, speed=0.05):
    root = Path(root)
    # Cross-session summary uses only the session NWBs, which sit at most a few
    # levels down (root[/animal]/session/RatX_date.nwb). Use BOUNDED-DEPTH globs
    # instead of '**' so we never descend into the huge *_sorting_output/phy_export
    # folders (thousands of files + recording.dat) — that recursion is what makes
    # this crawl for minutes on the SMB mount. ip* folders never hold NWBs anyway.
    patterns = ["*.nwb", "*/*.nwb", "*/*/*.nwb", "*/*/*/*.nwb"]
    nwbs = sorted({p for pat in patterns for p in root.glob(pat)
                   if not p.name.endswith(".tmp.nwb")
                   and not any(re.fullmatch(r"ip\d+", part, re.I) for part in p.parts)})
    if not nwbs:
        print(f"No .nwb files found under {root}.")
        return
    print(f"Found {len(nwbs)} NWB file(s).")
    by_animal = defaultdict(list)
    for p in tqdm(nwbs, desc="sessions", unit="nwb"):
        try:
            s = collect_session(p, bin_cm=bin_cm, sigma=sigma, speed=speed)
            if s is not None:
                by_animal[s["animal"]].append(s)
                _log(f"  {p.parent.name}/{p.name}: {s['animal']} {s['date']} "
                     f"good={s['n_good']} mua={s['n_mua']} pyr={s['n_pyr']} int={s['n_int']}")
        except Exception as e:
            print(f"  Failed on {p}: {e}")
            traceback.print_exc()

    # per-animal per-neuron datapoints (with a session_label for pooled grouping)
    animal_units = {}
    for animal in sorted(by_animal):
        urows = [s["unit_rows"] for s in by_animal[animal]
                 if isinstance(s.get("unit_rows"), pd.DataFrame) and not s["unit_rows"].empty]
        if urows:
            u = pd.concat(urows, ignore_index=True)
            u["session_label"] = ("R" + u["repeat"].astype("Int64").astype(str)
                                  + "S" + u["session"].astype("Int64").astype(str))
            animal_units[animal] = u
    units_all = pd.concat(animal_units.values(), ignore_index=True) if animal_units else pd.DataFrame()

    out = root / "session_summary.pdf"
    with PdfPages(str(out)) as pdf:
        for animal in sorted(by_animal):
            _plot_animal(pdf, animal, by_animal[animal], animal_units.get(animal))
            units = [s["units"] for s in by_animal[animal] if s.get("units") is not None]
            if units:
                _plot_scatter(pdf, animal, pd.concat(units, ignore_index=True))
        if not units_all.empty and units_all["animal"].nunique() > 1:
            _plot_combined(pdf, units_all)       # all animals pooled
    print(f"\nWrote {out} ({len(by_animal)} animal(s)).")

    # per-session summary table (before/after type5 columns for splits)
    cols = (["animal", "date", "repeat", "session", "split",
             "n_good", "n_mua", "n_pyr", "n_int"]
            + [_dec_key(q, L) for q in _DEC_QUALS for L in _DEC_LEADS]
            + [k for key in _PF_KEYS for k in (key, key + "_post")])
    df = pd.DataFrame([{c: s.get(c) for c in cols}
                       for animal in sorted(by_animal)
                       for s in sorted(by_animal[animal], key=lambda z: z["date"])], columns=cols)
    ucols = ["animal", "date", "repeat", "session", "session_label", "epoch", "unit_id",
             "quality_label", "cell_type", "subtype", "firing_rate_hz", "trough_to_peak_s",
             "acg_tau_rise_ms", "goal_node"] + list(_PF_KEYS)
    units_dp = units_all.reindex(columns=ucols) if not units_all.empty else pd.DataFrame()
    anova_df, posthoc_df = _stats_tables(animal_units, units_all) if animal_units else (pd.DataFrame(), pd.DataFrame())

    xlsx = root / "session_summary.xlsx"
    try:
        with pd.ExcelWriter(xlsx) as xw:
            df.to_excel(xw, index=False, sheet_name="sessions")
            if not units_dp.empty:
                units_dp.to_excel(xw, index=False, sheet_name="units")
            if not anova_df.empty:
                anova_df.to_excel(xw, index=False, sheet_name="anova")
            if not posthoc_df.empty:
                posthoc_df.to_excel(xw, index=False, sheet_name="posthoc")
        print(f"Wrote {xlsx} (sessions: {len(df)}, unit datapoints: {len(units_dp)}, "
              f"anova: {len(anova_df)}, posthoc: {len(posthoc_df)} rows)")
    except Exception as e:
        for name, d in [("", df), ("_units", units_dp), ("_anova", anova_df), ("_posthoc", posthoc_df)]:
            if not d.empty:
                d.to_csv(root / f"session_summary{name}.csv", index=False)
        print(f"Could not write xlsx ({e}); wrote CSVs instead.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-session per-animal summary plots from NWBs.")
    ap.add_argument("--root", required=True, help="folder to scan recursively for session NWBs.")
    ap.add_argument("--config", required=False, default=None, help="Accepted for runner consistency (unused).")
    ap.add_argument("--bin_cm", type=float, default=5.0)
    ap.add_argument("--speed", type=float, default=0.05)
    ap.add_argument("--smooth", type=float, default=2.0)
    args = ap.parse_args()
    try:
        run(args.root, bin_cm=args.bin_cm, sigma=args.smooth, speed=args.speed)
    except Exception as e:
        print(f"[session-summary] Failed: {e}")
        traceback.print_exc()
        sys.exit(1)
