"""
Runner step [v]: visualize the curated Units in a session NWB.

Runs AFTER step [u] (the NWB must already carry the curated Units table, and
ideally waveform_mean). For each op folder it reads:
  - the session NWB               <op>/Rat*_*.nwb      (units, position, metrics)
  - the phy export                <op>/*_sorting_output/phy_export
                                    (amplitudes.npy for the amplitude-vs-time plot)
  - RecordingMeta.xlsx            <op>/RecordingMeta.xlsx   (per-trial Trial_Type + times)

and writes PDFs into <op>/nwb_viz/:

  summary.pdf
    - scatter of peak-to-trough duration vs firing rate (good units, coloured by
      putative cell type) with the classification thresholds drawn
    - distribution (histogram) of every quality metric
    - place fields of ALL putative pyramidal cells overlaid, one colour per cell

  Unit_<clusterid>.pdf   (one per GOOD unit)
    page 1  - spike amplitude vs time, mean waveform template, and a text panel
              labelling the unit pyramidal / interneuron with its key stats
    page 2+ - rate map (full duration); rate map of each of the 24 trials; rate
              map of each type-4 trial; rate maps before and after the 2nd
              special (type-4/5) trial

Cell-type rule (good units only):
    firing rate > 10 Hz                        -> interneuron
    else, peak half-width > 0.425 ms           -> (putative) pyramidal
    else                                       -> interneuron

Usage:
    python visualize_nwb.py --output_folder <op_folder> [--bins 40x25] [--smooth 1.5]
"""

import sys
import argparse
import traceback
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter, gaussian_filter1d, label

from pynwb import NWBHDF5IO

# --- cell-type thresholds ---
# Rule (firing-rate first): FR > 10 Hz => interneuron; else pyramidal only if the
# spike is wide (peak half-width > 0.425 ms), otherwise interneuron.
PHW_THRESH_S = 0.425e-3   # 0.425 ms peak half-width
RATE_THRESH_HZ = 10.0     # firing rate

# --- spatial frame (plot_trials.py convention) ---
# Pixel -> metre scaling and the fixed maze frame, copied from plot_trials.py so
# rate maps share its coordinate system. Positions divided by these give metres;
# the maze then lives in the fixed MAZE_EXTENT box (metres), identical for every
# session/trial so all spatial panels are directly comparable.
SCALE_X = 2352 / 2 / 9
SCALE_Y = 1424 / 2 / 5
MAZE_EXTENT = (0.0, 9.0, 0.0, 5.0)


# ------------------------------------------------------------
#                 locating inputs
# ------------------------------------------------------------
def find_nwb_file(output_folder):
    op = Path(output_folder)
    cands = [p for p in sorted(op.glob("*.nwb")) if not p.name.endswith(".tmp.nwb")]
    if cands:
        return cands[0]
    cands = [p for p in sorted(op.glob("**/*.nwb")) if not p.name.endswith(".tmp.nwb")]
    return cands[0] if cands else None


def find_phy_folders(output_folder):
    op = Path(output_folder)
    if (op / "params.py").exists():
        return [op]
    phys = sorted(op.glob("*_sorting_output/phy_export"))
    if not phys:
        phys = sorted(op.glob("*/phy_export"))
    if not phys and (op / "phy_export").exists():
        phys = [op / "phy_export"]
    return phys


def _read_phy_params(phy_folder):
    ns = {}
    exec((Path(phy_folder) / "params.py").read_text(), {}, ns)
    return {k: v for k, v in ns.items() if not k.startswith("__")}


# ------------------------------------------------------------
#                 loading NWB content
# ------------------------------------------------------------
def load_position(nwb):
    """Return (x, y, t) for the animal in session-relative seconds, or None."""
    try:
        pos = nwb.processing["Behavior"]["Position"]
    except Exception:
        return None
    ss = pos.spatial_series
    # Prefer a series literally named 'Rat'; else the first one.
    key = "Rat" if "Rat" in ss else next(iter(ss), None)
    if key is None:
        return None
    s = ss[key]
    xy = np.asarray(s.data[:], dtype=float)
    t = np.asarray(s.timestamps[:], dtype=float)
    x, y = xy[:, 0], xy[:, 1]
    order = np.argsort(t)
    return x[order], y[order], t[order]


def read_trials_raw(op_folder):
    """Per-trial (type, goal_node, start_unix, end_unix) from RecordingMeta.xlsx,
    in the raw unix clock. Returns [] if unavailable."""
    metas = list(Path(op_folder).glob("RecordingMeta.xlsx"))
    if not metas:
        return []
    try:
        df = pd.read_excel(metas[0], sheet_name=0)
    except Exception as e:
        print(f"  Could not read RecordingMeta.xlsx ({e}).")
        return []
    need = {"Trial_Type", "trial_start_time", "trial_end_time"}
    if not need <= set(df.columns):
        print("  RecordingMeta.xlsx missing Trial_Type/trial_start_time/trial_end_time.")
        return []
    trials = []
    for _, r in df.iterrows():
        if pd.isna(r["trial_start_time"]) or pd.isna(r["trial_end_time"]):
            continue
        try:
            ttype = int(r["Trial_Type"])
        except (TypeError, ValueError):
            ttype = -1
        goal = int(r["Goal_Node"]) if "Goal_Node" in df.columns and pd.notna(r["Goal_Node"]) else None
        trials.append((ttype, goal, float(r["trial_start_time"]), float(r["trial_end_time"])))
    return trials


def align_trials(raw_trials, nwb_start, t_min, t_max):
    """Convert raw-unix trials to session-relative seconds, auto-picking the
    relative-zero unix. New NWBs store a correct session_start_time (fixed
    create_nwb), but older ones are off by the tz offset (session_start_time was
    tz-stamped without converting). We try both readings and keep whichever lands
    the most trials inside the recording's position coverage [t_min, t_max].
    Returns [(type, goal_node, t0_rel, t1_rel), ...]."""
    if not raw_trials:
        return []
    cands = []
    try:
        cands.append(nwb_start.timestamp())                                  # fixed NWBs
    except Exception:
        pass
    try:
        cands.append(nwb_start.replace(tzinfo=timezone.utc).timestamp())     # legacy NWBs
    except Exception:
        pass
    if not cands:
        return []
    in_range = lambda off: sum(1 for (_tt, _g, s, e) in raw_trials if t_min <= s - off <= t_max)
    off = max(cands, key=in_range)
    return [(tt, g, s - off, e - off) for (tt, g, s, e) in raw_trials]


def load_nodes():
    """{node_id: (x_m, y_m)} maze-node coordinates (metres, plot_trials frame),
    from src/tools/node_list_new.csv. Used to mark the goal node on trial pages."""
    p = Path(__file__).resolve().parent.parent / "tools" / "node_list_new.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, header=None, names=["id", "x", "y"])
        return {int(r.id): (r.x / SCALE_X, r.y / SCALE_Y) for r in df.itertuples()}
    except Exception:
        return {}


def place_field_mask(rate, frac=0.5):
    """Boolean mask of a cell's main place field: the connected region of bins
    >= frac*peak that contains the peak bin. None if no field."""
    if rate is None or not rate.count():
        return None
    peak = float(np.ma.max(rate))
    if peak <= 0:
        return None
    binary = rate.filled(0) >= frac * peak
    lab, n = label(binary)
    if n == 0:
        return None
    iy, ix = np.unravel_index(np.ma.argmax(rate), rate.shape)
    pk = lab[iy, ix]
    if pk == 0:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        pk = int(np.argmax(sizes))
    return lab == pk


# ------------------------------------------------------------
#                 speed – firing-rate correlation
# ------------------------------------------------------------
def speed_and_fr(x, y, t, spike_times, dt, t0=None, t1=None,
                 fr_sigma_s=0.25, speed_cap=0.6):
    """Per-position-sample running speed (m/s) and smoothed instantaneous firing
    rate (Hz), aligned, over [t0,t1]. Both Gaussian-smoothed (~fr_sigma_s). Speed
    is clipped to speed_cap m/s (>0.6 m/s treated as 0.6). Returns (speed, fr)."""
    if t0 is not None:
        m = (t >= t0) & (t <= t1)
        x, y, t = x[m], y[m], t[m]
        spike_times = spike_times[(spike_times >= t0) & (spike_times <= t1)]
    if t.size < 5:
        return None, None
    speed = np.zeros_like(x, dtype=float)
    dts = np.diff(t)
    speed[1:] = np.hypot(np.diff(x), np.diff(y)) / np.where(dts > 0, dts, np.inf)
    speed = np.clip(speed, 0, speed_cap)
    counts, _ = np.histogram(spike_times, bins=np.append(t, t[-1] + dt))
    fr = counts.astype(float) / dt
    sig = max(1.0, fr_sigma_s / dt)
    fr = gaussian_filter1d(fr, sig)
    speed = gaussian_filter1d(speed, sig)
    ok = np.isfinite(speed) & np.isfinite(fr)
    return speed[ok], fr[ok]


def speed_score(speed, fr):
    """Speed score = Pearson correlation between running speed and firing rate."""
    if speed is None or len(speed) < 5 or np.std(speed) == 0 or np.std(fr) == 0:
        return np.nan
    return float(np.corrcoef(speed, fr)[0, 1])


def _plot_speed_tuning(ax, speed, fr, title, color="tab:cyan"):
    """Scatter of speed vs firing rate (10% subsample) + sliding-window 25/50/75
    percentile lines, annotated with the speed score r."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if speed is None or len(speed) < 5:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return
    vmax = float(np.nanmax(speed))
    bins_c = np.arange(0, vmax, 0.04) if vmax > 0.04 else np.array([0.0])
    win = 0.04 * 1.5
    p25, p50, p75 = [], [], []
    for c in bins_c:
        s = fr[(speed >= c - win) & (speed <= c + win)]
        q = np.quantile(s, [0.25, 0.5, 0.75]) if s.size else (np.nan, np.nan, np.nan)
        p25.append(q[0]); p50.append(q[1]); p75.append(q[2])
    idx = np.random.randint(0, len(speed), size=int(np.ceil(len(speed) * 0.1)))
    ax.scatter(speed[idx], fr[idx], s=2, edgecolors="white", facecolors=color,
               linewidths=0.1, alpha=0.15)
    ax.plot(bins_c, p50, "k.", markersize=2)
    ax.plot(bins_c, p25, "k-", lw=0.8)
    ax.plot(bins_c, p75, "k-", lw=0.8)
    ax.text(0.95, 0.95, f"r = {speed_score(speed, fr):.2f}", ha="right", va="top",
            transform=ax.transAxes, fontsize=8)
    ax.set_xlabel("Speed (m/s)", fontsize=7)
    ax.set_ylabel("Firing rate (Hz)", fontsize=7)
    ax.set_title(title, fontsize=8)


def _speed_page(pdf, x, y, t, spike_times, dt, windows, cid, color):
    """One page of speed–firing-rate tuning panels, one per (label, t0, t1)."""
    ncols = 3
    nrows = int(np.ceil(len(windows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.69, 3.4 * nrows + 0.6),
                             squeeze=False)
    for k, (label, t0, t1) in enumerate(windows):
        sp, fr = speed_and_fr(x, y, t, spike_times, dt, t0=t0, t1=t1)
        _plot_speed_tuning(axes[k // ncols][k % ncols], sp, fr, label, color=color)
    for k in range(len(windows), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"Unit {cid} — speed–firing-rate correlation", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


def load_amplitudes(phy, cluster_id):
    """Per-spike (time_sec, amplitude) for one curated cluster, from the phy
    amplitudes.npy. Returns (None, None) if unavailable."""
    phy = Path(phy)
    amp_f = phy / "amplitudes.npy"
    st_f = phy / "spike_times.npy"
    sc_f = phy / "spike_clusters.npy"
    if not (amp_f.exists() and st_f.exists() and sc_f.exists()):
        return None, None
    try:
        fs = float(_read_phy_params(phy)["sample_rate"])
        clusters = np.load(sc_f).astype("int64").flatten()
        mask = clusters == int(cluster_id)
        if not mask.any():
            return None, None
        amps = np.load(amp_f).astype(float).flatten()[mask]
        times = np.load(st_f).astype("int64").flatten()[mask] / fs
        order = np.argsort(times)
        return times[order], amps[order]
    except Exception:
        return None, None


# ------------------------------------------------------------
#                 rate maps
# ------------------------------------------------------------
def make_rate_map(x, y, t, spike_times, extent, bins, dt, sigma,
                  t0=None, t1=None, speed_thresh=0.0):
    """Firing-rate (place-field) map, Hz: binned spike counts / occupancy, EXACTLY
    the plot_trials.py occupancy convention (count * dt seconds per bin), and
    masked to bins the animal actually visited (occupancy > 0) so nothing is drawn
    off the maze path. Optional speed gating (only samples/spikes with speed >
    speed_thresh, in position-units/s) mirrors plot_trials' "Speed > N" maps.

    x,y,t   : animal position (session-relative seconds), t ascending
    extent  : (xmin, xmax, ymin, ymax) in position (pixel) units
    bins    : (nx, ny)
    returns : (rate 2D [ny, nx] masked to visited bins, extent) or (None, extent)
    """
    xmin, xmax, ymin, ymax = extent
    nx, ny = bins
    rng = [[xmin, xmax], [ymin, ymax]]

    if t0 is not None:
        m = (t >= t0) & (t <= t1)
        x, y, t = x[m], y[m], t[m]
        spike_times = spike_times[(spike_times >= t0) & (spike_times <= t1)]
    good = np.isfinite(x) & np.isfinite(y)
    x, y, t = x[good], y[good], t[good]
    if x.size < 2:
        return None, extent

    # speed (position units / s), aligned to each position sample
    speed = np.zeros_like(x)
    if x.size > 1:
        d = np.hypot(np.diff(x), np.diff(y))
        dts = np.diff(t)
        speed[1:] = d / np.where(dts > 0, dts, np.inf)
    move = speed > speed_thresh if speed_thresh > 0 else np.ones_like(x, dtype=bool)

    # occupancy (seconds per bin) from moving samples — plot_trials convention
    occ, _, _ = np.histogram2d(x[move], y[move], bins=[nx, ny], range=rng)
    occ = occ.T * dt

    # spike positions (interpolated onto the trajectory), speed-gated the same way
    if spike_times.size:
        sx = np.interp(spike_times, t, x, left=np.nan, right=np.nan)
        sy = np.interp(spike_times, t, y, left=np.nan, right=np.nan)
        sv = np.interp(spike_times, t, speed, left=0.0, right=0.0)
        ok = np.isfinite(sx) & np.isfinite(sy)
        if speed_thresh > 0:
            ok &= sv > speed_thresh
        spk, _, _ = np.histogram2d(sx[ok], sy[ok], bins=[nx, ny], range=rng)
        spk = spk.T
    else:
        spk = np.zeros_like(occ)

    visited = occ > 0            # bins the animal actually entered
    if sigma and sigma > 0:      # optional light smoothing, kept ON the path only
        occ = gaussian_filter(occ, sigma)
        spk = gaussian_filter(spk, sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(occ > 0, spk / occ, 0.0)
    rate = np.ma.masked_where(~visited, rate)   # draw ONLY on visited bins
    return rate, extent


def _draw_map(ax, rate, extent, title, vmax=None):
    """Draw a rate map. `vmax` sets the colour-scale ceiling; when None it falls
    back to 75% of this map's own peak. Callers pass a shared vmax so all of one
    unit's maps use the same colour scale."""
    ax.set_xticks([]); ax.set_yticks([])
    if rate is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return
    peak = float(np.ma.max(rate)) if rate.count() else 0.0
    if vmax is None:
        vmax = 0.75 * peak if peak > 0 else None
    # Draw the array in true data coords (origin='lower': row 0 = ymin) then invert
    # the y-axis exactly like the spike-on-path panels (camera y grows downward),
    # so the maze AND any goal-node marker line up between the two panels.
    im = ax.imshow(rate, origin="lower", extent=extent, aspect="equal",
                   cmap="jet", interpolation="nearest", vmin=0, vmax=vmax)
    ax.set_ylim(extent[3], extent[2])
    ax.set_title(f"{title}\npeak {peak:.1f} Hz", fontsize=8)
    return im


def plot_spike_path(ax, x, y, t, spike_times, title, extent, t0=None, t1=None, goal_xy=None):
    """Classic spike-on-trajectory plot: grey path + red spikes at the animal's
    position when each spike occurred. Restricted to [t0,t1] when given. All
    panels share `extent` (the full maze) so trials are directly comparable.
    `goal_xy` (metres) marks the goal node with a gold star."""
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2])   # inverted: camera y grows downward
    if t0 is not None:
        m = (t >= t0) & (t <= t1)
        x, y, t = x[m], y[m], t[m]
        spike_times = spike_times[(spike_times >= t0) & (spike_times <= t1)]
    if x.size < 2:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        if goal_xy is not None:
            ax.scatter(*goal_xy, marker="*", s=260, c="gold", edgecolors="k",
                       linewidths=0.8, zorder=5)
        return
    ax.plot(x, y, color="0.8", lw=0.4, zorder=1)
    if spike_times.size:
        sx = np.interp(spike_times, t, x, left=np.nan, right=np.nan)
        sy = np.interp(spike_times, t, y, left=np.nan, right=np.nan)
        ax.scatter(sx, sy, s=4, c="red", alpha=0.5, edgecolors="none", zorder=2)
    if goal_xy is not None:
        ax.scatter(*goal_xy, marker="*", s=260, c="gold", edgecolors="k",
                   linewidths=0.8, zorder=5)
    ax.set_title(f"{title}\n{int(spike_times.size)} spk", fontsize=8)


# ------------------------------------------------------------
#                 the visualiser
# ------------------------------------------------------------
def visualize(output_folder, bin_cm=5.0, sigma=2.0, speed=0.05):
    nwb_path = find_nwb_file(output_folder)
    if nwb_path is None:
        print(f"No .nwb under '{output_folder}' (run steps w/u first). Skipping.")
        return
    phys = find_phy_folders(output_folder)
    phy = phys[0] if phys else None

    out_dir = Path(output_folder) / "nwb_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"NWB: {nwb_path}")
    io = NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True)
    try:
        nwb = io.read()
        if nwb.units is None or len(nwb.units.id) == 0:
            print("NWB has no Units table — nothing to visualize.")
            return

        udf = nwb.units.to_dataframe()
        has_wf = "waveform_mean" in udf.columns
        pos = load_position(nwb)
        if pos is None:
            print("  WARNING: no Position data — rate maps/place fields will be skipped.")
        else:
            # to metres in the plot_trials frame (px / SCALE), so bins are physical
            _px, _py, _t = pos
            pos = (_px / SCALE_X, _py / SCALE_Y, _t)
        # firing rate + classification for every unit
        extent = MAZE_EXTENT                       # fixed metre frame (0..9, 0..5)
        bin_m = bin_cm / 100.0
        bins = (max(5, int(round((extent[1] - extent[0]) / bin_m))),   # nx
                max(5, int(round((extent[3] - extent[2]) / bin_m))))   # ny
        if pos is not None:
            x, y, t = pos
            duration = float(t.max() - t.min()) or 1.0
            dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0 / 30
        else:
            duration, dt = 1.0, 1.0 / 30
        print(f"  spatial: {bin_cm:g} cm bins ({bins[0]}x{bins[1]}), smooth {sigma:g} bin, "
              f"speed gate {speed:g} m/s")

        nspk = udf["num_spikes"].astype(float) if "num_spikes" in udf else \
            pd.Series([len(nwb.units["spike_times"][i]) for i in range(len(udf))], index=udf.index)
        udf["firing_rate_hz"] = nspk / duration
        phw = udf["peak_half_width"].astype(float) if "peak_half_width" in udf \
            else pd.Series(np.nan, index=udf.index)
        # pyramidal only if slow AND wide-spiked (peak half-width); else interneuron
        is_pyr = (udf["firing_rate_hz"] <= RATE_THRESH_HZ) & (phw > PHW_THRESH_S)
        udf["cell_type"] = np.where(is_pyr, "pyramidal", "interneuron")

        ql = udf["quality_label"].astype(str) if "quality_label" in udf else \
            pd.Series("good", index=udf.index)
        good = udf[ql == "good"]
        print(f"  {len(udf)} units | good={len(good)} | "
              f"pyr={int((good['cell_type']=='pyramidal').sum())} "
              f"int={int((good['cell_type']=='interneuron').sum())} | waveforms={has_wf}")

        # trials aligned to the position clock (auto-picks the offset so both the
        # fixed and legacy-tz NWBs work — see align_trials).
        if pos is not None:
            trials = align_trials(read_trials_raw(output_folder),
                                  nwb.session_start_time, float(t.min()), float(t.max()))
        else:
            trials = []
        nodes = load_nodes()   # maze-node coords (metres) to mark the goal node

        # ---- summary.pdf ----
        _write_summary(out_dir / "summary.pdf", udf, good, pos, extent, bins, dt, sigma, speed, nwb)

        # ---- one PDF per good unit ----
        for uid, row in good.iterrows():
            cid = int(row["phy_cluster_id"]) if "phy_cluster_id" in row else int(uid)
            spike_times = np.asarray(row["spike_times"], dtype=float)
            wf = np.asarray(row["waveform_mean"]) if has_wf else None
            amp_t, amp_v = load_amplitudes(phy, cid) if phy is not None else (None, None)
            _write_unit_pdf(out_dir / f"Unit_{cid}.pdf", row, cid, spike_times, wf,
                            amp_t, amp_v, pos, extent, bins, dt, sigma, speed, trials, nodes)
        print(f"  Wrote summary.pdf + {len(good)} unit PDF(s) to {out_dir}")
    finally:
        io.close()


def _write_summary(path, udf, good, pos, extent, bins, dt, sigma, speed, nwb):
    metric_cols = [c for c in udf.columns if c not in (
        "phy_cluster_id", "sorting_group", "quality_label", "auto_quality_label",
        "spike_times", "waveform_mean", "cell_type", "electrodes")]
    with PdfPages(str(path)) as pdf:
        # Page 1: scatter peak half-width vs firing rate
        fig, ax = plt.subplots(figsize=(8.27, 5.5))
        for ct, col in (("pyramidal", "#2166ac"), ("interneuron", "#b2182b")):
            sub = good[good["cell_type"] == ct]
            if "peak_half_width" in sub:
                ax.scatter(sub["peak_half_width"] * 1e3, sub["firing_rate_hz"],
                           s=22, alpha=0.8, c=col, label=f"{ct} (n={len(sub)})")
        ax.axvline(PHW_THRESH_S * 1e3, ls="--", c="grey", lw=1)
        ax.axhline(RATE_THRESH_HZ, ls="--", c="grey", lw=1)
        ax.set_xlabel("peak half-width (ms)")
        ax.set_ylabel("firing rate (Hz)")
        ax.set_yscale("log")
        ax.set_title(f"{nwb.session_description}\nGood units: peak-to-trough vs firing rate")
        ax.legend(fontsize=8)
        pdf.savefig(fig); plt.close(fig)

        # Page 2+: quality-metric distributions (good units)
        if metric_cols:
            per = 9
            for start in range(0, len(metric_cols), per):
                chunk = metric_cols[start:start + per]
                fig, axes = plt.subplots(3, 3, figsize=(8.27, 8.0))
                for ax, col in zip(axes.ravel(), chunk):
                    vals = pd.to_numeric(good[col], errors="coerce").dropna()
                    if len(vals):
                        ax.hist(vals, bins=30, color="#4393c3", edgecolor="white")
                    ax.set_title(col, fontsize=8)
                    ax.tick_params(labelsize=6)
                for ax in axes.ravel()[len(chunk):]:
                    ax.axis("off")
                fig.suptitle("Quality-metric distributions (good units)", fontsize=11)
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig); plt.close(fig)

        # Last page: actual PLACE-FIELD AREAS of all pyramidal good cells, one
        # colour each — each cell's main field (connected region >=50% of its peak
        # rate) drawn as a filled patch over the grey trajectory.
        pyr = good[good["cell_type"] == "pyramidal"]
        fig, ax = plt.subplots(figsize=(8.27, 6.0))
        if pos is not None and len(pyr):
            x, y, t = pos
            ax.plot(x, y, color="0.88", lw=0.3, zorder=0)
            cmap = plt.get_cmap("gist_rainbow")
            n = len(pyr)
            drawn = 0
            for i, (uid, row) in enumerate(pyr.iterrows()):
                st = np.asarray(row["spike_times"], dtype=float)
                rate, ext = make_rate_map(x, y, t, st, extent, bins, dt, sigma, speed_thresh=speed)
                mask = place_field_mask(rate, frac=0.5)
                if mask is None or not mask.any():
                    continue
                # fill the field area; contourf treats mask row 0 as ymin, matching
                # the inverted y-limits set below.
                try:
                    ax.contourf(mask.astype(float), levels=[0.5, 1.5], extent=ext,
                                colors=[cmap(i / max(n - 1, 1))], alpha=0.4, zorder=1)
                except Exception:
                    continue
                drawn += 1
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[3], extent[2])
            ax.set_aspect("equal", "box")
            ax.set_title(f"Place-field areas of {drawn} putative pyramidal cells "
                         f"(main field >=50% peak, one colour per cell)")
        else:
            ax.text(0.5, 0.5, "no position data / no pyramidal cells",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        pdf.savefig(fig); plt.close(fig)


def _write_unit_pdf(path, row, cid, spike_times, wf, amp_t, amp_v,
                    pos, extent, bins, dt, sigma, speed, trials, nodes):
    with PdfPages(str(path)) as pdf:
        # ---- Page 1: summary info ----
        fig = plt.figure(figsize=(8.27, 6.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
        ax_amp = fig.add_subplot(gs[0, :])
        if amp_t is not None and len(amp_t):
            ax_amp.scatter(amp_t, amp_v, s=2, alpha=0.3, c="#333333", edgecolors="none")
            ax_amp.set_ylabel("amplitude")
        else:
            ax_amp.text(0.5, 0.5, "no phy amplitudes.npy", ha="center", va="center",
                        transform=ax_amp.transAxes)
        ax_amp.set_xlabel("time (s)")
        ax_amp.set_title(f"Unit {cid} — spike amplitude vs time")

        ax_wf = fig.add_subplot(gs[1, 0])
        if wf is not None and np.size(wf):
            w = np.asarray(wf, dtype=float)
            if w.ndim == 1:
                w = w[:, None]
            # stored as (n_samples, n_channels)
            for ch in range(w.shape[1]):
                ax_wf.plot(w[:, ch], lw=1, label=f"ch{ch}")
            ax_wf.set_title("mean waveform template")
            ax_wf.set_xlabel("sample")
            if w.shape[1] <= 6:
                ax_wf.legend(fontsize=6, ncol=2)
        else:
            ax_wf.text(0.5, 0.5, "no waveform_mean\n(regenerate NWB via r→w→u)",
                       ha="center", va="center", transform=ax_wf.transAxes)
            ax_wf.set_title("mean waveform template")

        ax_txt = fig.add_subplot(gs[1, 1]); ax_txt.axis("off")
        phw_ms = float(row.get("peak_half_width", np.nan)) * 1e3
        lines = [
            f"cluster id: {cid}",
            f"cell type: {row['cell_type'].upper()}",
            f"quality: {row.get('quality_label', '?')}",
            f"firing rate: {row['firing_rate_hz']:.2f} Hz",
            f"peak half-width: {phw_ms:.3f} ms",
            f"n spikes: {int(row.get('num_spikes', len(spike_times)))}",
            f"SNR: {float(row.get('snr', np.nan)):.2f}",
            "",
            f"rule: pyramidal if FR<={RATE_THRESH_HZ:.0f}Hz & phw>{PHW_THRESH_S*1e3:.3f}ms",
            "       else interneuron",
        ]
        ax_txt.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=9, transform=ax_txt.transAxes)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        if pos is None:
            return
        x, y, t = pos
        args = (x, y, t, spike_times, extent, bins, dt, sigma, speed)

        # Shared colour scale for EVERY rate map of this unit: 75% of the
        # full-duration peak. Computed once here so per-trial maps are directly
        # comparable in colour to the full-duration map.
        full_rate, _ = make_rate_map(x, y, t, spike_times, extent, bins, dt, sigma,
                                     speed_thresh=speed)
        full_peak = float(np.ma.max(full_rate)) if (full_rate is not None and full_rate.count()) else 0.0
        unit_vmax = 0.75 * full_peak if full_peak > 0 else None

        # ---- one page per window: spike-on-path + its rate map (2 plots) ----
        _spatial_page(pdf, *args, f"Unit {cid} — full duration", vmax=unit_vmax)

        for i, (tt, gn, t0, t1) in enumerate(trials):
            _spatial_page(pdf, *args, f"Unit {cid} — trial {i+1}  (type {tt}, goal {gn})",
                          t0=t0, t1=t1, vmax=unit_vmax, goal_xy=nodes.get(gn))

        # free-roaming (special) trials: the long ~10-min type-4 / type-5 trials.
        # Shown at the end, one page each, named freeroaming 1/2/3 with their type.
        specials = [(tt, gn, t0, t1) for (tt, gn, t0, t1) in trials if tt in (4, 5)]
        for i, (tt, gn, t0, t1) in enumerate(specials):
            _spatial_page(pdf, *args,
                          f"Unit {cid} — freeroaming {i+1} (type {tt}, goal {gn})",
                          t0=t0, t1=t1, vmax=unit_vmax, goal_xy=nodes.get(gn))

        # rate map before / after the 2nd free-roaming trial (spatial remapping)
        if len(specials) >= 2:
            _tt2, g2, s0, s1 = specials[1]
            _spatial_page(pdf, *args, f"Unit {cid} — before freeroaming 2 (goal {g2})",
                          t0=float(t.min()), t1=s0, vmax=unit_vmax, goal_xy=nodes.get(g2))
            _spatial_page(pdf, *args, f"Unit {cid} — after freeroaming 2 (goal {g2})",
                          t0=s1, t1=float(t.max()), vmax=unit_vmax, goal_xy=nodes.get(g2))

        # ---- speed–firing-rate correlation page (whole length + 3 free-roaming
        #      trials + before/after the 2nd free-roaming trial) ----
        speed_windows = [("whole length", None, None)]
        for i, (tt, gn, s0, s1) in enumerate(specials):
            speed_windows.append((f"freeroaming {i+1} (type {tt})", s0, s1))
        if len(specials) >= 2:
            _tt2, g2, s0, s1 = specials[1]
            speed_windows.append(("before freeroaming 2", float(t.min()), s0))
            speed_windows.append(("after freeroaming 2", s1, float(t.max())))
        color = "tab:blue" if row.get("cell_type") == "pyramidal" else "tab:cyan"
        _speed_page(pdf, x, y, t, spike_times, dt, speed_windows, cid, color)


def _spatial_page(pdf, x, y, t, spike_times, extent, bins, dt, sigma, speed,
                  title, t0=None, t1=None, vmax=None, goal_xy=None):
    """One portrait PDF page holding exactly two stacked plots: spike-on-path on
    top and its rate map below. `vmax` is the shared per-unit colour ceiling;
    `goal_xy` (metres) marks the trial's goal node on both panels."""
    fig, (axp, axr) = plt.subplots(2, 1, figsize=(8.27, 11.69))   # A4 portrait
    plot_spike_path(axp, x, y, t, spike_times, "spikes on path", extent,
                    t0=t0, t1=t1, goal_xy=goal_xy)
    rate, ext = make_rate_map(x, y, t, spike_times, extent, bins, dt, sigma,
                              t0=t0, t1=t1, speed_thresh=speed)
    im = _draw_map(axr, rate, ext, "rate map", vmax=vmax)
    if goal_xy is not None:
        axr.scatter(*goal_xy, marker="*", s=260, facecolor="none",
                    edgecolors="k", linewidths=1.4, zorder=5)
    if im is not None:
        lbl = f"Hz (scale 0–{vmax:.1f})" if vmax else "Hz"
        fig.colorbar(im, ax=axr, fraction=0.046, label=lbl)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig); plt.close(fig)


def _parse_bins(s):
    try:
        a, b = s.lower().split("x")
        return int(a), int(b)
    except Exception:
        return 40, 25


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize curated Units in the session NWB (summary + per-unit PDFs).")
    parser.add_argument("--output_folder", required=True,
                        help="op folder with the NWB, *_sorting_output/phy_export and RecordingMeta.xlsx.")
    parser.add_argument("--config", required=False, default=None,
                        help="Accepted for runner consistency (unused).")
    parser.add_argument("--bin_cm", type=float, default=5.0,
                        help="Spatial bin size in cm (square) in the plot_trials metre frame.")
    parser.add_argument("--smooth", type=float, default=2.0,
                        help="Gaussian smoothing sigma in bins (0 = none).")
    parser.add_argument("--speed", type=float, default=0.05,
                        help="Speed gate in m/s: only samples/spikes above this count toward "
                             "occupancy/rate maps (0 = no gating).")
    args = parser.parse_args()

    try:
        visualize(args.output_folder, bin_cm=args.bin_cm, sigma=args.smooth, speed=args.speed)
    except Exception as e:
        print(f"[viz] Failed: {e}")
        traceback.print_exc()
        sys.exit(1)
