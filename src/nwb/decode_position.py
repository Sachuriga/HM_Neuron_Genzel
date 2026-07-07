"""
Runner step [b]: decode the animal's position from the neurons, per session.

Population Bayesian decoding (Poisson / flat prior) — the same maths as
pynapple.process.decoding.decode_2d (https://pynapple.org), implemented here so
the pipeline needs no extra dependency:

    log P(x | n) = sum_i [ n_i * log(TC_i(x))  -  dt * TC_i(x) ]  + const

where TC_i(x) is unit i's 2D tuning curve (rate map, Hz) and n_i the unit's spike
count in a dt-second time bin. The decoded position is the argmax bin.

By default the tuning curves are trained on ALL of the session (--folds 1) and
used to decode the whole session. Pass --folds N (>1) for N-fold cross-validation
(the session is cut into contiguous chunks assigned round-robin to folds; each bin
is decoded with tuning curves trained on the OTHER folds), which gives an honest,
slightly larger out-of-sample error.

By default only GOOD units are used; pass --quality good mua to add MUA.

Usage:
    python decode_position.py --output_folder <op> [--quality good|mua ...]
                              [--bin_cm 10] [--time_bin 0.5] [--folds 1]
"""

import sys
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter
from pynwb import NWBHDF5IO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_nwb as V     # load_position, load_nodes, SCALE_X/Y, MAZE_EXTENT, find_nwb_file


def _select_units(nwb, qualities):
    """[(unit_id, spike_times_seconds), ...] for units whose quality_label is in
    `qualities` (e.g. {'good'} or {'good','mua'})."""
    udf = nwb.units.to_dataframe()
    ql = udf["quality_label"].astype(str) if "quality_label" in udf.columns \
        else pd.Series("good", index=udf.index)
    sel = udf[ql.isin(set(qualities))]
    out = []
    for i, r in sel.iterrows():
        uid = int(r["phy_cluster_id"]) if "phy_cluster_id" in sel.columns else int(i)
        out.append((uid, np.asarray(r["spike_times"], dtype=float)))
    return out


def _smooth2d(flat, nx, ny, sigma):
    return gaussian_filter(flat.reshape(ny, nx), sigma).ravel() if sigma else flat


def decode_session(nwb_path, qualities, bin_cm=10.0, time_bin=0.5, folds=1,
                   sigma=1.0, n_chunks=100, lead_s=0.0, save=True, plot=True):
    """Cross-validated Bayesian decoding for one session NWB. Returns a dict of
    results (or None) and writes a PDF next to it.

    lead_s > 0 turns the decoder into a PREDICTOR of the future position: each
    spike is associated with where the animal will be `lead_s` seconds later, so
    decoding a spike-count bin yields the predicted position `lead_s` seconds
    ahead, compared against the true future position."""
    io = NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True)
    try:
        nwb = io.read()
        if nwb.units is None or len(nwb.units.id) == 0:
            print("  no units — skipping."); return None
        units = _select_units(nwb, qualities)
        pos = V.load_position(nwb)
        if pos is None or len(units) < 2:
            print(f"  need position + >=2 units (have {len(units)} units, "
                  f"pos={pos is not None}) — skipping."); return None
        x = pos[0] / V.SCALE_X; y = pos[1] / V.SCALE_Y; t = pos[2]
        dt_pos = float(np.median(np.diff(t))) if t.size > 1 else 1.0 / 30
        nU = len(units)

        # spatial grid (metres) restricted later to visited bins
        xmin, xmax, ymin, ymax = V.MAZE_EXTENT
        bm = bin_cm / 100.0
        nx = max(5, int(round((xmax - xmin) / bm)))
        ny = max(5, int(round((ymax - ymin) / bm)))
        xe = np.linspace(xmin, xmax, nx + 1); ye = np.linspace(ymin, ymax, ny + 1)
        xc = (xe[:-1] + xe[1:]) / 2; yc = (ye[:-1] + ye[1:]) / 2
        cx = np.repeat(xc[None, :], ny, 0).ravel()      # bin-centre x (flat, ny*nx)
        cy = np.repeat(yc[:, None], nx, 1).ravel()

        good = np.isfinite(x) & np.isfinite(y)
        pix = np.clip(np.digitize(x, xe) - 1, 0, nx - 1)
        piy = np.clip(np.digitize(y, ye) - 1, 0, ny - 1)
        pbid = piy * nx + pix                            # spatial bin per pos sample

        # decode time bins (spike counts + actual position per bin)
        edges = np.arange(t[0], t[-1] + time_bin, time_bin)
        nb = len(edges) - 1
        if nb < 10:
            print("  session too short — skipping."); return None

        # RANDOM k-fold cross-validation: assign every decode time-bin to a random
        # fold; position samples and spikes inherit the fold of the bin they fall in.
        # Random (not contiguous-in-time) folds keep every visited maze region in the
        # training set, so no test bin lands in a place the model never saw — at the
        # cost of some leakage between temporally adjacent bins.
        rng = np.random.default_rng(0)
        bin_fold = (rng.integers(0, folds, size=nb) if folds > 1
                    else np.zeros(nb, dtype=int))

        def _fold_of(times):
            return bin_fold[np.clip(np.digitize(times, edges) - 1, 0, nb - 1)]
        samp_fold = _fold_of(t)

        # per-spike spatial bin + fold. For prediction (lead_s>0) each spike is tied
        # to the position lead_s seconds LATER, so tuning curves map spikes to the
        # animal's future location.
        spk = []
        for uid, st in units:
            sx = np.interp(st + lead_s, t, x, left=np.nan, right=np.nan)
            sy = np.interp(st + lead_s, t, y, left=np.nan, right=np.nan)
            ok = np.isfinite(sx) & np.isfinite(sy)
            sbid = (np.clip(np.digitize(sy[ok], ye) - 1, 0, ny - 1) * nx
                    + np.clip(np.digitize(sx[ok], xe) - 1, 0, nx - 1))
            spk.append((sbid, _fold_of(st[ok]), st[ok]))

        counts = np.zeros((nU, nb))
        for u, (uid, st) in enumerate(units):
            counts[u], _ = np.histogram(st, bins=edges)
        sb = np.clip(np.digitize(t, edges) - 1, 0, nb - 1)
        vs = good
        cnt = np.bincount(sb[vs], minlength=nb).astype(float)
        occupied = cnt > 0
        tc_mid = (edges[:-1] + edges[1:]) / 2
        if lead_s:
            # target = the TRUE position lead_s seconds after each bin centre
            ax_ = np.interp(tc_mid + lead_s, t, x, left=np.nan, right=np.nan)
            ay_ = np.interp(tc_mid + lead_s, t, y, left=np.nan, right=np.nan)
            occupied = occupied & np.isfinite(ax_) & np.isfinite(ay_)
            ax_ = np.nan_to_num(ax_); ay_ = np.nan_to_num(ay_)
        else:
            ax_ = np.bincount(sb[vs], weights=x[vs], minlength=nb) / np.where(cnt > 0, cnt, 1)
            ay_ = np.bincount(sb[vs], weights=y[vs], minlength=nb) / np.where(cnt > 0, cnt, 1)

        decoded_x = np.full(nb, np.nan); decoded_y = np.full(nb, np.nan)
        min_occ = 0.3                        # only decode over well-sampled bins (s)
        tc_max = 50.0                        # clip tuning curves (Hz) — avoids spurious

        def _fit_and_decode(train_samp, spike_mask_fn, test_bins):
            """Fit tuning curves on `train_samp` position samples (+ each unit's
            spikes kept by `spike_mask_fn(sfold)`) and decode `test_bins`."""
            occ = np.bincount(pbid[train_samp], minlength=nx * ny).astype(float) * dt_pos
            occ_s = _smooth2d(occ, nx, ny, sigma)
            TC = np.zeros((nU, nx * ny))
            for u, (sbid, sfold, _st) in enumerate(spk):
                c = np.bincount(sbid[spike_mask_fn(sfold)], minlength=nx * ny).astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    TC[u] = np.where(occ_s > 1e-3, _smooth2d(c, nx, ny, sigma) / occ_s, 0.0)
            TC = np.clip(np.nan_to_num(TC, nan=0.0, posinf=tc_max), 0.0, tc_max)
            visited = occ >= min_occ
            if visited.sum() < 3 or not len(test_bins):
                return
            vidx = np.where(visited)[0]
            logTC = np.log(TC[:, vidx] + 1e-3)           # (nU, nV)
            sumrate = TC[:, vidx].sum(0)                  # (nV,)
            with np.errstate(all="ignore"):              # flat-prior log-posterior
                logpost = counts[:, test_bins].T @ logTC - time_bin * sumrate[None, :]
            logpost[~np.isfinite(logpost)] = -np.inf
            best = vidx[np.argmax(logpost, axis=1)]
            decoded_x[test_bins] = cx[best]; decoded_y[test_bins] = cy[best]

        if folds <= 1:                       # train on ALL data, decode everything
            _fit_and_decode(good, lambda sf: np.ones(sf.shape, bool), np.where(occupied)[0])
        else:                                # cross-validated (train on other folds)
            for f in range(folds):
                _fit_and_decode((samp_fold != f) & good, (lambda ff: (lambda sf: sf != ff))(f),
                                np.where((bin_fold == f) & occupied)[0])

        dec = np.isfinite(decoded_x) & occupied
        err = np.hypot(decoded_x[dec] - ax_[dec], decoded_y[dec] - ay_[dec])
        # chance: distance from the actual position to a random visited-bin centre
        rng = np.random.default_rng(0)
        allvis = np.where(np.bincount(pbid[good], minlength=nx * ny) > 0)[0]
        rc = rng.choice(allvis, size=dec.sum())
        chance = np.hypot(cx[rc] - ax_[dec], cy[rc] - ay_[dec])
        res = {"nwb": nwb_path.name, "quality": "+".join(sorted(qualities)),
               "mode": ("full-data" if folds <= 1 else f"{folds}-fold CV"),
               "lead_s": float(lead_s),
               "n_units": nU, "n_bins": int(dec.sum()),
               "median_err_m": float(np.median(err)), "mean_err_m": float(np.mean(err)),
               "median_chance_m": float(np.median(chance))}

        # trials on the position/spike seconds clock (same source as plot_trials /
        # step v — see visualize_nwb.build_trials): coordinate Trial_Num blocks
        # mapped via stitched_framewise_seconds.csv. NOT the old RecordingMeta-unix
        # windows, which sit on the behavioural-sync clock and drift vs the video.
        trials = V.build_trials(nwb_path.parent, nwb.session_start_time,
                                float(t.min()), float(t.max()))
        # carry the decoded arrays on the result so callers (e.g. the multi-lead
        # comparison) can reuse them without re-reading the NWB.
        res.update({"t": tc_mid[dec], "actual_x": ax_[dec], "actual_y": ay_[dec],
                    "decoded_x": decoded_x[dec], "decoded_y": decoded_y[dec],
                    "err": err, "chance": chance, "trials": trials})

        out_dir = nwb_path.parent / "decoding"
        tag = "_".join(sorted(qualities))
        # prediction runs get a distinct 'predicted_..._lead{n}s' name so they do NOT
        # clobber the 0-lag decoded_*.npz that plot_trials overlays per trial.
        stem = f"decoded_{tag}" if not lead_s else f"predicted_{tag}_lead{lead_s:g}s"
        if save:
            # persist the decoded track so step 5 (plot_trials) can overlay it per
            # trial. Time is t = the stitched-seconds clock (identical to the NWB
            # position/spike timestamps), so no unix conversion is needed anywhere.
            # Coords = scaled metres.
            out_dir.mkdir(exist_ok=True)
            np.savez(out_dir / f"{stem}.npz",
                     t=tc_mid[dec],
                     actual_x=ax_[dec], actual_y=ay_[dec],
                     decoded_x=decoded_x[dec], decoded_y=decoded_y[dec], err=err,
                     quality="+".join(sorted(qualities)), lead_s=float(lead_s),
                     scale_x=V.SCALE_X, scale_y=V.SCALE_Y)
            # per-(trial, unit) place-coding + task-performance table (lead-0 only):
            # -> CSV + PDF here; the NWB write happens after this read handle closes.
            if not lead_s:
                tu = build_trial_unit_metrics(units, x, y, t, dt_pos, trials, (nx, ny),
                                              sigma, tc_mid[dec], err, V.load_nodes(),
                                              nwb_path.parent)
                res["trial_unit_metrics"] = tu
                if tu is not None and len(tu):
                    tu.to_csv(out_dir / f"trial_unit_metrics_{tag}.csv", index=False)
                    plot_trial_unit_metrics(out_dir / f"trial_unit_metrics_{tag}.pdf", tu, tag)
                    print(f"  trial/unit metrics: {len(tu)} rows "
                          f"({tu['trial'].nunique()} trials x {tu['unit'].nunique()} units).")
        if plot:
            _plot(nwb_path, res, tc_mid[dec], ax_[dec], ay_[dec],
                  decoded_x[dec], decoded_y[dec], err, chance, V.load_nodes(), qualities,
                  trials, stem=stem)
        return res
    finally:
        io.close()


# ------------------------------------------------------------
#        per-(trial, unit) metrics table + performance
# ------------------------------------------------------------
_MAZE_G = None


def _maze_graph():
    """Hex-maze networkx graph (raw node coords) for shortest-path performance,
    reusing plot_trials.build_hexmaze_graph. Cached; None if unavailable."""
    global _MAZE_G
    if _MAZE_G is not None:
        return _MAZE_G or None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tracker"))
        import plot_trials as PT
        nd = pd.read_csv(Path(__file__).resolve().parent.parent / "tools" / "node_list_new.csv",
                         header=None, names=["id", "x", "y"])
        _MAZE_G = PT.build_hexmaze_graph(nd.copy())
    except Exception as e:
        print(f"  maze graph unavailable ({e}); performance will be NaN.")
        _MAZE_G = False
    return _MAZE_G or None


def _read_trial_extra(op_folder):
    """{trial_num(1-based): dict(paths, start, goal, trial_type, between_node_speed)}
    from RecordingMeta.xlsx, in row order (== the coordinate Trial_Num)."""
    meta = V._pick_file(op_folder, "RecordingMeta.xlsx") or V._pick_file(op_folder, "*RecordingMeta.xlsx")
    if meta is None:
        return {}
    try:
        df = pd.read_excel(meta, sheet_name=0)
    except Exception:
        return {}
    out = {}
    for i, r in enumerate(df.itertuples(index=False), start=1):
        d = r._asdict()
        out[i] = {
            "paths": str(d["paths"]) if pd.notna(d.get("paths")) else None,
            "start": V._first_int(d.get("Start_Nodes")) if pd.notna(d.get("Start_Nodes")) else None,
            "goal": int(d["Goal_Node"]) if pd.notna(d.get("Goal_Node")) else None,
            "trial_type": int(d["Trial_Type"]) if pd.notna(d.get("Trial_Type")) else -1,
            "between_node_speed": (float(d["avg_between_node_speed"])
                                   if pd.notna(d.get("avg_between_node_speed")) else np.nan),
        }
    return out


def _performance_log10(info, G):
    """Trial performance = log10(shortest_hops / actual_hops): 0 = optimal,
    negative = detour. shortest = graph hops start->goal; actual = steps to the
    first goal visit in the trial's node sequence ('paths')."""
    if info is None or G is None:
        return np.nan
    start, goal, paths = info.get("start"), info.get("goal"), info.get("paths")
    if start is None or goal is None or not paths:
        return np.nan
    pn = [int(v) for v in paths.split(",") if v.strip().lstrip("-").isdigit()]
    actual = pn.index(goal) if goal in pn else (len(pn) - 1)
    try:
        shortest = nx.shortest_path_length(G, str(start), str(goal))
    except Exception:
        return np.nan
    return float(np.log10(shortest / actual)) if (shortest > 0 and actual > 0) else np.nan


def build_trial_unit_metrics(units, x, y, t, dt, trials, bins, sigma,
                             tc_mid, err, nodes, op_folder):
    """Per-(trial, unit) table. Per-unit place-coding metrics (spatial information,
    field size, selectivity = peak/mean, firing rate, peak rate) plus per-trial
    task metrics that repeat across the trial's units (population decoding error,
    performance = log10(shortest/actual path), average between-node speed)."""
    if not trials or not units:
        return pd.DataFrame()
    nx_, ny_ = bins
    xmin, xmax, ymin, ymax = V.MAZE_EXTENT
    bin_area = ((xmax - xmin) / nx_) * ((ymax - ymin) / ny_)      # m^2 per bin
    G = _maze_graph()
    extra = _read_trial_extra(op_folder)
    rows = []
    for k, (ttype, goal, start, t0, t1) in enumerate(trials, start=1):
        info = extra.get(k, {"start": start, "goal": goal, "trial_type": ttype,
                             "paths": None, "between_node_speed": np.nan})
        perf = _performance_log10(info, G)
        bspeed = info.get("between_node_speed", np.nan)
        goal_xy = nodes.get(goal) if (nodes and goal is not None) else None
        dm = (tc_mid >= t0) & (tc_mid <= t1)                     # population decoding error
        dec_err = float(np.median(err[dm])) if dm.any() else np.nan
        dur = max(float(t1 - t0), 1e-9)
        for uid, st in units:
            m, rate, _ = V.place_field_metrics(x, y, t, st, V.MAZE_EXTENT, (nx_, ny_),
                                               dt, sigma, 0.05, goal_xy=goal_xy, t0=t0, t1=t1)
            fields = V.place_fields(rate) if (rate is not None and rate.count()) else []
            field_size = float(sum(int(f.sum()) for f in fields) * bin_area)
            n_spk = int(np.count_nonzero((st >= t0) & (st <= t1)))
            rows.append({
                "trial": k, "trial_type": int(info.get("trial_type", ttype)), "unit": int(uid),
                "spatial_info": m.get("spatial_info", np.nan),
                "field_size_m2": field_size,
                "selectivity": m.get("selectivity", np.nan),
                "firing_rate_hz": n_spk / dur,
                "peak_rate_hz": m.get("peak", np.nan),
                "decoding_error_m": dec_err,
                "performance": perf,
                "between_node_speed": bspeed,
            })
    return pd.DataFrame(rows)


# metrics correlated against performance (label, column)
_PERF_MEASURES = [
    ("spatial_info", "Spatial info (bits/spk)"),
    ("field_size_m2", "Field size (m$^2$)"),
    ("selectivity", "Selectivity (peak/mean)"),
    ("firing_rate_hz", "Firing rate (Hz)"),
    ("decoding_error_m", "Decoding error (m)"),
    ("between_node_speed", "Between-node speed (m/s)"),
]


def _corr_grid(pdf, data, title, annotate_trial=False):
    """2x3 grid of scatter(metric vs performance) with a least-squares line and
    Pearson r/p/n in each panel."""
    from scipy.stats import pearsonr
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    for ax, (col, lab) in zip(axes.ravel(), _PERF_MEASURES):
        cols = [col, "performance"] + (["trial"] if annotate_trial else [])
        d = (data[cols].replace([np.inf, -np.inf], np.nan).dropna()
             if col in data.columns else pd.DataFrame())
        if len(d) >= 3 and d[col].std() > 0 and d["performance"].std() > 0:
            r, p = pearsonr(d[col], d["performance"])
            ax.scatter(d[col], d["performance"], s=18, alpha=0.5, edgecolor="none")
            b, a = np.polyfit(d[col], d["performance"], 1)
            xs = np.linspace(d[col].min(), d[col].max(), 50)
            ax.plot(xs, b * xs + a, color="crimson", lw=1.2)
            ax.set_title(f"{lab}\nr={r:.2f}, p={p:.3g}, n={len(d)}", fontsize=9)
            if annotate_trial:
                for _, rr in d.iterrows():
                    ax.annotate(int(rr["trial"]), (rr[col], rr["performance"]),
                                fontsize=6, alpha=0.6)
        else:
            ax.text(0.5, 0.5, f"{lab}\n(insufficient data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
        ax.set_xlabel(lab, fontsize=8)
        ax.set_ylabel("Performance  log10(short/actual)", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def plot_trial_unit_metrics(pdf_path, df, tag):
    """Per-session PDF: (1) performance vs each metric over all (trial,unit) points;
    (2) performance vs each metric after averaging all units within a trial (one
    point per trial)."""
    if df is None or not len(df):
        return
    units_lab = tag.replace("_", "+")
    agg = (df.groupby("trial")
             .agg({**{c: "mean" for c, _ in _PERF_MEASURES}, "performance": "first"})
             .reset_index())
    with PdfPages(str(pdf_path)) as pdf:
        _corr_grid(pdf, df,
                   f"Performance vs metrics — per (trial, unit) — units {units_lab}")
        _corr_grid(pdf, agg,
                   f"Performance vs unit-averaged metrics — per trial — units {units_lab}",
                   annotate_trial=True)


def save_metrics_nwb(nwb_path, df, tag):
    """Append the trial/unit metrics table to the NWB as a scratch DynamicTable
    named 'trial_unit_metrics_<tag>'. Skips (leaving the CSV/PDF) if it already
    exists or the write fails, so re-runs never crash the pipeline."""
    if df is None or not len(df):
        return
    name = f"trial_unit_metrics_{tag}"
    desc = ("Per-(trial,unit) place-coding metrics (spatial_info, field_size_m2, "
            "selectivity, firing_rate_hz, peak_rate_hz) + per-trial decoding_error_m, "
            f"performance=log10(shortest/actual), between_node_speed. Units: {tag.replace('_', '+')}.")
    try:
        with NWBHDF5IO(str(nwb_path), mode="a", load_namespaces=True) as io:
            nwb = io.read()
            if nwb.scratch is not None and name in nwb.scratch:
                print(f"  NWB already has scratch '{name}'; leaving CSV/PDF only.")
                return
            try:
                nwb.add_scratch(df, name=name, table_description=desc)
            except TypeError:
                nwb.add_scratch(df, name=name, description=desc)
            io.write(nwb)
        print(f"  wrote NWB scratch '{name}' ({len(df)} rows).")
    except Exception as e:
        print(f"  could not write metrics to NWB ({e}); CSV/PDF still written.")


def _plot(nwb_path, res, tt, axr, ayr, dxr, dyr, err, chance, nodes, qualities,
          trials=None, stem=None):
    out_dir = nwb_path.parent / "decoding"
    out_dir.mkdir(exist_ok=True)
    stem = stem or f"decoded_{'_'.join(sorted(qualities))}"
    # decoded_x -> decode_x.pdf ; predicted_..._lead1s -> predict_..._lead1s.pdf
    out = out_dir / (stem.replace("decoded_", "decode_", 1)
                     .replace("predicted_", "predict_", 1) + ".pdf")
    with PdfPages(str(out)) as pdf:
        fig = plt.figure(figsize=(11, 8))
        gs = fig.add_gridspec(2, 2)
        # actual vs decoded trajectory (first ~60 s of decoded bins)
        ax = fig.add_subplot(gs[0, 0])
        if nodes:
            V.draw_maze(ax, nodes)
        w = tt <= tt[0] + 60 if len(tt) else slice(None)
        ax.plot(axr[w], ayr[w], "-", color="0.5", lw=1, label="actual")
        ax.plot(dxr[w], dyr[w], ".", color="#d62728", ms=3, label="decoded")
        ax.set_aspect("equal"); ax.set_ylim(V.MAZE_EXTENT[3], V.MAZE_EXTENT[2])
        ax.set_xlim(V.MAZE_EXTENT[0], V.MAZE_EXTENT[1]); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("actual vs decoded (first 60 s)"); ax.legend(fontsize=7)
        # error over time
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(tt, err, lw=0.4, color="#2166ac"); ax.set_xlabel("time (s)")
        ax.set_ylabel("decoding error (m)"); ax.set_title("error over time")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        # error histogram vs chance
        ax = fig.add_subplot(gs[1, 0])
        ax.hist(err, bins=40, color="#2166ac", alpha=0.8, label="decoder", density=True)
        ax.hist(chance, bins=40, color="0.6", alpha=0.5, label="chance", density=True)
        ax.axvline(res["median_err_m"], color="#2166ac", ls="--")
        ax.set_xlabel("error (m)"); ax.set_ylabel("density"); ax.legend(fontsize=8)
        ax.set_title("error distribution"); ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # text summary
        ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
        ax.text(0.02, 0.98, "\n".join([
            f"session: {res['nwb']}",
            f"units: {res['quality']} (n={res['n_units']})",
            f"training: {res['mode']}",
            f"prediction lead: {res.get('lead_s', 0.0):g} s",
            f"decoded bins: {res['n_bins']}",
            f"median error: {res['median_err_m']:.2f} m",
            f"mean error:   {res['mean_err_m']:.2f} m",
            f"median chance: {res['median_chance_m']:.2f} m",
        ]), va="top", family="monospace", fontsize=10, transform=ax.transAxes)
        _lead = res.get("lead_s", 0.0)
        _kind = f"Position prediction (+{_lead:g}s)" if _lead else "Position decoding"
        fig.suptitle(f"{_kind} — {nwb_path.stem}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        pdf.savefig(fig); plt.close(fig)

        # ---- per-trial decoded vs actual position (grid of small panels) ----
        ext = V.MAZE_EXTENT
        ncols, rows_pp = 4, 6
        per = ncols * rows_pp
        for start in range(0, len(trials or []), per):
            chunk = list(enumerate(trials))[start:start + per]
            nrows = int(np.ceil(len(chunk) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows + 0.5), squeeze=False)
            for ax, (i, (tt_, gn, sn, t0, t1)) in zip(axes.ravel(), chunk):
                if nodes:
                    V.draw_maze(ax, nodes)
                m = (tt >= t0) & (tt <= t1)
                ax.plot(axr[m], ayr[m], "-", color="0.5", lw=0.8, zorder=1)
                ax.plot(dxr[m], dyr[m], ".", color="#d62728", ms=3, zorder=2)
                if nodes.get(gn) is not None:
                    ax.scatter(*nodes[gn], marker="*", s=80, c="gold", edgecolors="k", lw=0.5, zorder=5)
                if nodes.get(sn) is not None:
                    ax.scatter(*nodes[sn], marker="o", s=45, facecolor="none",
                               edgecolors="lime", lw=1.4, zorder=5)
                ax.set_aspect("equal"); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])
                ax.set_xticks([]); ax.set_yticks([])
                e = float(np.median(err[m])) if m.any() else np.nan
                ax.set_title(f"trial {i+1} (type {tt_}, goal {gn})\nerr {e:.2f} m", fontsize=7)
            for ax in axes.ravel()[len(chunk):]:
                ax.axis("off")
            fig.suptitle(f"decoded (red) vs actual (grey) per trial — {nwb_path.stem}", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig); plt.close(fig)
    print(f"  wrote {out}  (median error {res['median_err_m']:.2f} m, "
          f"chance {res['median_chance_m']:.2f} m)")


def decode_leads(nwb_path, qualities, leads=(0.0, 1.0, 3.0), cv_folds=5, viz_folds=1, **kw):
    """Decode at several prediction leads (seconds ahead) and write ONE combined
    comparison PDF. lead 0 = decode the current position; lead>0 = predict the
    future position.

    ACCURACY (the comparison PDF + leads_summary that feeds step s) is
    CROSS-VALIDATED (cv_folds): tuning curves are fit on held-out data and tested
    on unseen bins, so the reported error reflects generalisation, not in-sample
    fit. The continuous VISUALISATION track that plot_trials overlays per trial is
    written separately from a full-data (viz_folds, default 1) lead-0 decode."""
    kw.pop("lead_s", None); kw.pop("folds", None)
    results = []
    for L in leads:
        print(f"  lead {L:g}s (CV {cv_folds}-fold) ...", end=" ", flush=True)
        r = decode_session(nwb_path, qualities, lead_s=float(L), folds=cv_folds,
                           save=False, plot=False, **kw)
        if r is None:
            print("skipped."); continue
        print(f"median err {r['median_err_m']:.2f} m")
        results.append(r)
    if not results:
        return results
    # visualisation track for plot_trials: full-data (in-sample) lead-0 decode, saved
    # as decoded_<tag>.npz — smooth continuous trace for the per-trial overlay only.
    viz = decode_session(nwb_path, qualities, lead_s=0.0, folds=viz_folds,
                         save=True, plot=False, **kw)
    # persist the per-(trial,unit) metrics table into the NWB (CSV/PDF already saved)
    if viz is not None and viz.get("trial_unit_metrics") is not None:
        save_metrics_nwb(nwb_path, viz["trial_unit_metrics"], "_".join(sorted(qualities)))
    # small per-lead accuracy summary so the cross-session summary (step s) can plot
    # decoding accuracy at every lead without re-decoding (cross-validated values).
    out_dir = nwb_path.parent / "decoding"; out_dir.mkdir(exist_ok=True)
    tag = "_".join(sorted(qualities))
    np.savez(out_dir / f"leads_summary_{tag}.npz",
             leads=np.array([r["lead_s"] for r in results], float),
             median_err=np.array([r["median_err_m"] for r in results], float),
             mean_err=np.array([r["mean_err_m"] for r in results], float),
             chance=float(np.median(results[0]["chance"])), cv_folds=int(cv_folds),
             quality=tag)
    _plot_leads(nwb_path, qualities, results)
    return results


def _pick_window(results):
    """Pick a representative trial window (session-relative t0, t1) that has the
    most decoded bins across the lead-0 result; fall back to the first 60 s."""
    r0 = results[0]
    best, best_n = None, 0
    for (_tt, _gn, _sn, t0, t1) in (r0.get("trials") or []):
        n = int(((r0["t"] >= t0) & (r0["t"] <= t1)).sum())
        if n > best_n:
            best, best_n = (t0, t1), n
    if best is not None and best_n >= 10:
        return best
    t = r0["t"]
    return (float(t.min()), float(t.min()) + 60.0) if len(t) else (0.0, 60.0)


def _plot_leads(nwb_path, qualities, results):
    """Multi-page comparison of prediction leads: (1) error/correlation dashboard,
    (2) actual-vs-predicted trajectory per lead over one trial, (3) 'prediction
    reach' arrows from the current position to the predicted future position."""
    import matplotlib.cm as cm
    nodes = V.load_nodes()
    ext = V.MAZE_EXTENT
    leads = [r["lead_s"] for r in results]
    colors = cm.viridis(np.linspace(0.15, 0.85, len(results)))
    tag = "+".join(sorted(qualities))
    out = nwb_path.parent / "decoding" / f"decode_leads_{'_'.join(sorted(qualities))}.pdf"
    out.parent.mkdir(exist_ok=True)
    chance = float(np.median(results[0]["chance"]))

    with PdfPages(str(out)) as pdf:
        # ---- Page 1: dashboard ----
        fig = plt.figure(figsize=(12, 9))
        gs = fig.add_gridspec(2, 2)

        # median error vs lead (+ chance)
        ax = fig.add_subplot(gs[0, 0])
        med = [r["median_err_m"] for r in results]
        ax.plot(leads, med, "-o", color="#2166ac", lw=2)
        for L, m in zip(leads, med):
            ax.annotate(f"{m:.2f}", (L, m), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
        ax.axhline(chance, ls="--", color="0.5", label=f"chance {chance:.2f} m")
        ax.set_xlabel("prediction lead (s)"); ax.set_ylabel("median error (m)")
        ax.set_title("Prediction error vs lead"); ax.set_ylim(bottom=0)
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

        # error distributions per lead
        ax = fig.add_subplot(gs[0, 1])
        for r, c in zip(results, colors):
            ax.hist(r["err"], bins=40, density=True, histtype="step", lw=1.8,
                    color=c, label=f"lead {r['lead_s']:g}s (med {r['median_err_m']:.2f})")
        ax.set_xlabel("error (m)"); ax.set_ylabel("density")
        ax.set_title("Error distribution per lead"); ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # actual-vs-predicted correlation vs lead (x and y)
        ax = fig.add_subplot(gs[1, 0])
        rx = [np.corrcoef(r["actual_x"], r["decoded_x"])[0, 1] for r in results]
        ry = [np.corrcoef(r["actual_y"], r["decoded_y"])[0, 1] for r in results]
        ax.plot(leads, rx, "-o", label="X", color="#1b7837")
        ax.plot(leads, ry, "-s", label="Y", color="#762a83")
        ax.set_xlabel("prediction lead (s)"); ax.set_ylabel("corr(actual, predicted)")
        ax.set_title("Predicted vs actual correlation"); ax.set_ylim(0, 1)
        ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

        # text summary
        ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
        lines = [f"session: {results[0]['nwb']}",
                 f"units: {tag} (n={results[0]['n_units']})",
                 f"training: {results[0]['mode']}",
                 f"chance: {chance:.2f} m", "", "lead    median   mean    corrX  corrY"]
        for r, cxx, cyy in zip(results, rx, ry):
            lines.append(f"{r['lead_s']:>4g}s  {r['median_err_m']:6.2f}  "
                         f"{r['mean_err_m']:6.2f}  {cxx:5.2f}  {cyy:5.2f}")
        ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace",
                fontsize=10, transform=ax.transAxes)
        fig.suptitle(f"Position prediction across leads — {nwb_path.stem} — {tag}",
                     fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96]); pdf.savefig(fig); plt.close(fig)

        # ---- Page 2: trajectory per lead over one representative trial ----
        t0, t1 = _pick_window(results)
        fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 5.4),
                                 squeeze=False)
        for ax, r in zip(axes[0], results):
            if nodes:
                V.draw_maze(ax, nodes)
            m = (r["t"] >= t0) & (r["t"] <= t1)
            ax.plot(r["actual_x"][m], r["actual_y"][m], "-", color="0.55", lw=1.4,
                    zorder=1, label="actual path")
            tt = r["t"][m]
            ct = (tt - tt.min()) / (tt.max() - tt.min()) if m.sum() > 1 and tt.max() > tt.min() else np.zeros(m.sum())
            ax.scatter(r["decoded_x"][m], r["decoded_y"][m], c=ct, cmap="cool", s=16,
                       vmin=0, vmax=1, zorder=3, label="predicted")
            ax.set_aspect("equal"); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"lead {r['lead_s']:g}s — median err {np.median(r['err'][m]) if m.any() else np.nan:.2f} m",
                         fontsize=10)
        axes[0][0].legend(fontsize=8, loc="upper right")
        fig.suptitle(f"Actual (grey) vs predicted (time-coloured) over one trial "
                     f"[{t0:.0f}-{t1:.0f}s] — {tag}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)

        # ---- Page 3: prediction-reach arrows ----
        # At sampled moments draw an arrow from the animal's CURRENT position (lead-0
        # actual) to the predicted position at each lead — longer leads should reach
        # further along the upcoming path. Capped to a short sub-window so the arrows
        # stay legible instead of turning into a hairball.
        r0 = results[0]
        w0, w1 = t0, min(t1, t0 + 45.0)
        fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 5.4),
                                 squeeze=False)
        m0 = (r0["t"] >= w0) & (r0["t"] <= w1)
        # sample ~every 1.5 s within the window on the lead-0 time base
        tsel = r0["t"][m0]
        step = max(1, int(round(1.5 / (np.median(np.diff(tsel)) if len(tsel) > 1 else 0.5))))
        keep_i = np.arange(0, len(tsel), step)
        for ax, r in zip(axes[0], results):
            if nodes:
                V.draw_maze(ax, nodes)
            ax.plot(r0["actual_x"][m0], r0["actual_y"][m0], "-", color="0.7", lw=1.2, zorder=1)
            m = (r["t"] >= w0) & (r["t"] <= w1)
            axs, ays = r0["actual_x"][m0], r0["actual_y"][m0]
            dxs, dys = r["decoded_x"][m], r["decoded_y"][m]
            n = min(len(axs), len(dxs))
            for i in keep_i:
                if i >= n:
                    break
                ax.annotate("", xy=(dxs[i], dys[i]), xytext=(axs[i], ays[i]),
                            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.3, alpha=0.85),
                            zorder=4)
            ax.scatter(axs[keep_i[keep_i < n]], ays[keep_i[keep_i < n]], s=26,
                       color="#1b7837", zorder=5, label="current position")
            ax.set_aspect("equal"); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"lead {r['lead_s']:g}s", fontsize=11)
        axes[0][0].legend(fontsize=8, loc="upper right")
        fig.suptitle("Prediction reach: arrow = current position -> predicted position "
                     f"(sampled ~1.5 s, {w0:.0f}-{w1:.0f}s) — {tag}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)

        # ---- Per-trial pages: actual vs predicted for EVERY trial, one column per
        # lead. grey = actual path, colour = predicted (within-trial time), * = goal.
        trials = results[0].get("trials") or []
        rows_pp, cols = 4, len(results)
        for start in range(0, len(trials), rows_pp):
            chunk = trials[start:start + rows_pp]
            fig, axes = plt.subplots(len(chunk), cols, figsize=(4.6 * cols, 3.4 * len(chunk)),
                                     squeeze=False)
            for ri, (ttype, goal, _snode, tt0, tt1) in enumerate(chunk):
                for ci, r in enumerate(results):
                    ax = axes[ri][ci]
                    if nodes:
                        V.draw_maze(ax, nodes)
                    m = (r["t"] >= tt0) & (r["t"] <= tt1)
                    ax.plot(r["actual_x"][m], r["actual_y"][m], "-", color="0.55",
                            lw=1.2, zorder=1)
                    tt = r["t"][m]
                    ct = (tt - tt.min()) / (tt.max() - tt.min()) if m.sum() > 1 and tt.max() > tt.min() else np.zeros(m.sum())
                    ax.scatter(r["decoded_x"][m], r["decoded_y"][m], c=ct, cmap="cool",
                               s=14, vmin=0, vmax=1, zorder=3)
                    if nodes.get(goal) is not None:
                        ax.scatter(*nodes[goal], marker="*", s=90, c="gold",
                                   edgecolors="k", lw=0.5, zorder=5)
                    ax.set_aspect("equal"); ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])
                    ax.set_xticks([]); ax.set_yticks([])
                    e = float(np.median(r["err"][m])) if m.any() else float("nan")
                    ax.set_title(f"trial {start + ri + 1} (type {ttype}) — "
                                 f"lead {r['lead_s']:g}s, err {e:.2f} m", fontsize=8)
            fig.suptitle(f"Per-trial actual (grey) vs predicted (time-coloured) — {tag}",
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97]); pdf.savefig(fig); plt.close(fig)

    print(f"  wrote {out}  (leads {', '.join(f'{L:g}s' for L in leads)})")


def run(output_folder, qualities, leads=None, **kw):
    nwb_path = V.find_nwb_file(output_folder)
    if nwb_path is None:
        print(f"No .nwb under '{output_folder}' (run steps w/u first). Skipping."); return
    print(f"Decoding {nwb_path} using units: {'+'.join(sorted(qualities))}")
    if leads:
        decode_leads(nwb_path, qualities, leads=leads, **kw)
    else:
        res = decode_session(nwb_path, qualities, **kw)
        if res is not None and res.get("trial_unit_metrics") is not None:
            save_metrics_nwb(nwb_path, res["trial_unit_metrics"], "_".join(sorted(qualities)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bayesian position decoder from neurons, per session.")
    ap.add_argument("--output_folder", required=True, help="op/session folder with the NWB.")
    ap.add_argument("--config", default=None, help="Accepted for runner consistency (unused).")
    ap.add_argument("--quality", nargs="+", default=["good"],
                    choices=["good", "mua", "noise"],
                    help="unit qualities to use (default: good; e.g. --quality good mua).")
    ap.add_argument("--bin_cm", type=float, default=10.0, help="spatial bin size (cm) for tuning curves.")
    ap.add_argument("--time_bin", type=float, default=0.5, help="decoding time bin (s).")
    ap.add_argument("--folds", type=int, default=1,
                    help="1 = train on ALL data (in-sample); >1 = that many CV folds.")
    ap.add_argument("--lead_s", type=float, default=0.0,
                    help="predict the position this many seconds in the FUTURE "
                         "(0 = decode current position; e.g. --lead_s 1.0).")
    ap.add_argument("--leads", type=float, nargs="+", default=None,
                    help="compare several prediction leads in ONE PDF, e.g. "
                         "--leads 0 1 3. Accuracy is cross-validated (--cv_folds); "
                         "the lead-0 visualisation track (decoded_<q>.npz) is full-data.")
    ap.add_argument("--cv_folds", type=int, default=5,
                    help="cross-validation folds for the multi-lead accuracy (default 5).")
    args = ap.parse_args()
    try:
        if args.leads:
            run(args.output_folder, args.quality, leads=args.leads,
                bin_cm=args.bin_cm, time_bin=args.time_bin,
                cv_folds=args.cv_folds, viz_folds=args.folds)
        else:
            run(args.output_folder, args.quality, bin_cm=args.bin_cm,
                time_bin=args.time_bin, folds=args.folds, lead_s=args.lead_s)
    except Exception as e:
        print(f"[decode] Failed: {e}")
        traceback.print_exc()
        sys.exit(1)
