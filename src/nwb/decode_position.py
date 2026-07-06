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
                   sigma=1.0, n_chunks=100):
    """Cross-validated Bayesian decoding for one session NWB. Returns a dict of
    results (or None) and writes a PDF next to it."""
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

        # chunk / fold assignment (contiguous chunks, round-robin folds)
        cedges = np.linspace(t[0], t[-1], n_chunks + 1)
        samp_fold = (np.clip(np.digitize(t, cedges) - 1, 0, n_chunks - 1)) % folds

        # per-spike spatial bin + fold
        spk = []
        for uid, st in units:
            sx = np.interp(st, t, x, left=np.nan, right=np.nan)
            sy = np.interp(st, t, y, left=np.nan, right=np.nan)
            ok = np.isfinite(sx) & np.isfinite(sy)
            sbid = (np.clip(np.digitize(sy[ok], ye) - 1, 0, ny - 1) * nx
                    + np.clip(np.digitize(sx[ok], xe) - 1, 0, nx - 1))
            sfold = (np.clip(np.digitize(st[ok], cedges) - 1, 0, n_chunks - 1)) % folds
            spk.append((sbid, sfold, st[ok]))

        # decode time bins: spike counts + actual position + fold per bin
        edges = np.arange(t[0], t[-1] + time_bin, time_bin)
        nb = len(edges) - 1
        if nb < 10:
            print("  session too short — skipping."); return None
        counts = np.zeros((nU, nb))
        for u, (uid, st) in enumerate(units):
            counts[u], _ = np.histogram(st, bins=edges)
        sb = np.clip(np.digitize(t, edges) - 1, 0, nb - 1)
        vs = good
        cnt = np.bincount(sb[vs], minlength=nb).astype(float)
        ax_ = np.bincount(sb[vs], weights=x[vs], minlength=nb) / np.where(cnt > 0, cnt, 1)
        ay_ = np.bincount(sb[vs], weights=y[vs], minlength=nb) / np.where(cnt > 0, cnt, 1)
        occupied = cnt > 0
        tc_mid = (edges[:-1] + edges[1:]) / 2
        bin_fold = (np.clip(np.digitize(tc_mid, cedges) - 1, 0, n_chunks - 1)) % folds

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
               "n_units": nU, "n_bins": int(dec.sum()),
               "median_err_m": float(np.median(err)), "mean_err_m": float(np.mean(err)),
               "median_chance_m": float(np.median(chance))}

        raw = V.read_trials_raw(nwb_path.parent)
        trials = V.align_trials(raw, nwb.session_start_time, float(t.min()), float(t.max()))
        # offset from session-relative seconds to the UNIX clock (= session_start_unix);
        # taken from the raw-vs-aligned trial times so it matches align_trials exactly.
        off = (raw[0][3] - trials[0][3]) if (raw and trials) else 0.0
        # persist the decoded track so step 5 (plot_trials) can overlay it per trial.
        # Save time in BOTH clocks: t (session-relative s) and t_unix (UNIX s, which
        # is what plot_trials' log 'sys_time' uses). Coordinates are scaled metres.
        out_dir = nwb_path.parent / "decoding"; out_dir.mkdir(exist_ok=True)
        np.savez(out_dir / f"decoded_{'_'.join(sorted(qualities))}.npz",
                 t=tc_mid[dec], t_unix=tc_mid[dec] + off,
                 actual_x=ax_[dec], actual_y=ay_[dec],
                 decoded_x=decoded_x[dec], decoded_y=decoded_y[dec], err=err,
                 quality="+".join(sorted(qualities)), scale_x=V.SCALE_X, scale_y=V.SCALE_Y)
        _plot(nwb_path, res, tc_mid[dec], ax_[dec], ay_[dec],
              decoded_x[dec], decoded_y[dec], err, chance, V.load_nodes(), qualities, trials)
        return res
    finally:
        io.close()


def _plot(nwb_path, res, tt, axr, ayr, dxr, dyr, err, chance, nodes, qualities, trials=None):
    out_dir = nwb_path.parent / "decoding"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"decode_{'_'.join(sorted(qualities))}.pdf"
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
            f"decoded bins: {res['n_bins']}",
            f"median error: {res['median_err_m']:.2f} m",
            f"mean error:   {res['mean_err_m']:.2f} m",
            f"median chance: {res['median_chance_m']:.2f} m",
        ]), va="top", family="monospace", fontsize=10, transform=ax.transAxes)
        fig.suptitle(f"Position decoding — {nwb_path.stem}", fontsize=13)
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


def run(output_folder, qualities, **kw):
    nwb_path = V.find_nwb_file(output_folder)
    if nwb_path is None:
        print(f"No .nwb under '{output_folder}' (run steps w/u first). Skipping."); return
    print(f"Decoding {nwb_path} using units: {'+'.join(sorted(qualities))}")
    decode_session(nwb_path, qualities, **kw)


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
    args = ap.parse_args()
    try:
        run(args.output_folder, args.quality, bin_cm=args.bin_cm,
            time_bin=args.time_bin, folds=args.folds)
    except Exception as e:
        print(f"[decode] Failed: {e}")
        traceback.print_exc()
        sys.exit(1)
