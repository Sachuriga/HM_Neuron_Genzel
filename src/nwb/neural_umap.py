"""
Runner step [m]: UMAP embedding of the neural population activity, per session.

Reproduces the population-activity visualization from Gardner, Hermansen et al.,
"Toroidal topology of population activity in grid cells" (Nature 2022,
s41586-021-04268-7): bin every unit's spikes in short time bins, smooth each unit
with a Gaussian kernel, square-root transform and z-score, then embed the
(time-bin x neuron) matrix into 3D with UMAP (cosine metric). Each embedded point
is one moment in time; colouring it by the animal's position / speed / task
variable reveals the low-dimensional structure of the population code.

Only moving epochs are embedded (like the paper's RUN periods). Two embeddings are
produced per session by the runner: GOOD units only, and GOOD+MUA.

Usage:
    python neural_umap.py --output_folder <op> [--quality good|mua ...]

Outputs, written next to the NWB in <op>/umap/:
    umap_<quality>.npz   embedding + per-bin colour variables
    umap_<quality>.pdf   embedding coloured by position, speed, time, task
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.ndimage import gaussian_filter1d
from pynwb import NWBHDF5IO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_nwb as V     # load_position, load_nodes, read_trials_raw, align_trials, SCALE_*


def _select_units(nwb, qualities):
    """[(unit_id, spike_times_seconds), ...] for units whose quality_label is in
    `qualities` (e.g. {'good'} or {'good','mua'})."""
    udf = nwb.units.to_dataframe()
    ql = (udf["quality_label"].astype(str).str.lower()
          if "quality_label" in udf.columns else None)
    out = []
    for uid, r in udf.iterrows():
        if ql is not None and ql.loc[uid] not in qualities:
            continue
        out.append((uid, np.asarray(r["spike_times"], dtype=float)))
    return out


def population_matrix(units, t0, t1, dt, sigma_s):
    """Smoothed, sqrt-transformed, z-scored (time-bin x neuron) activity matrix.
    Returns (Z, centers) where centers are the bin-centre times (session-relative)."""
    edges = np.arange(t0, t1 + dt, dt)
    centers = edges[:-1] + dt / 2.0
    T, N = len(centers), len(units)
    R = np.zeros((T, N), dtype=float)
    for j, (_uid, st) in enumerate(units):
        st = st[(st >= edges[0]) & (st <= edges[-1])]
        R[:, j] = np.histogram(st, bins=edges)[0]
    # Gaussian smoothing along time (paper smooths each unit's rate)
    R = gaussian_filter1d(R, sigma=max(sigma_s / dt, 1e-6), axis=0)
    rate = R / dt
    Z = np.sqrt(rate)                       # variance-stabilising sqrt transform
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)  # z-score per neuron
    return Z, centers


def _speed(x, y, t):
    """Instantaneous speed (m/s) at each sample, same length as t."""
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.nan
    v = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0])) / dt
    return np.nan_to_num(v, nan=0.0)


def embed_session(nwb_path, qualities, dt=0.1, sigma_s=0.3, speed_thresh=0.05,
                  n_neighbors=50, min_dist=0.1, n_components=3, max_bins=25000,
                  seed=42):
    import umap  # imported lazily so the rest of the runner works without it
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None or len(nwb.units.id) == 0:
            print("  no units — skipping."); return None
        units = _select_units(nwb, qualities)
        pos = V.load_position(nwb)
        if pos is None or len(units) < 3:
            print(f"  need position + >=3 units (have {len(units)} units) — skipping.")
            return None
        x = pos[0] / V.SCALE_X; y = pos[1] / V.SCALE_Y; t = pos[2]
        t0, t1 = float(t.min()), float(t.max())

        Z, centers = population_matrix(units, t0, t1, dt, sigma_s)

        # per-bin behaviour: position, speed (interpolate the tracking onto bins)
        xb = np.interp(centers, t, x)
        yb = np.interp(centers, t, y)
        vb = np.interp(centers, t, _speed(x, y, t))

        # task variables from the trial table (session-relative)
        raw = V.read_trials_raw(nwb_path.parent)
        trials = V.align_trials(raw, nwb.session_start_time, t0, t1)
        nodes = V.load_nodes()
        ttype = np.full(len(centers), np.nan)
        trial_id = np.full(len(centers), np.nan)
        goal_d = np.full(len(centers), np.nan)
        in_trial = np.zeros(len(centers), bool)
        for k, (tp, goal, _start, a, b) in enumerate(trials or [], start=1):
            m = (centers >= a) & (centers <= b)
            if not m.any():
                continue
            in_trial |= m
            ttype[m] = tp
            trial_id[m] = k
            if goal in nodes:
                gx, gy = nodes[goal]
                goal_d[m] = np.hypot(xb[m] - gx, yb[m] - gy)

        # embed only moving bins (paper's RUN epochs)
        keep = vb > speed_thresh
        if keep.sum() < 50:
            print(f"  only {int(keep.sum())} moving bins — skipping."); return None
        idx = np.where(keep)[0]
        if len(idx) > max_bins:                         # subsample huge sessions
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_bins, replace=False))

        Zk = Z[idx]
        nn = int(min(n_neighbors, len(idx) - 1))
        reducer = umap.UMAP(n_components=n_components, n_neighbors=nn,
                            min_dist=min_dist, metric="cosine", random_state=seed)
        emb = reducer.fit_transform(Zk)
        print(f"  embedded {len(idx)} moving bins x {len(units)} units "
              f"-> {n_components}D (n_neighbors={nn}).")

        res = {"emb": emb, "x": xb[idx], "y": yb[idx], "speed": vb[idx],
               "time": centers[idx], "ttype": ttype[idx], "goal_dist": goal_d[idx],
               "trial": trial_id[idx], "in_trial": in_trial[idx],
               "quality": "+".join(sorted(qualities)), "n_units": len(units)}

        out_dir = nwb_path.parent / "umap"
        out_dir.mkdir(exist_ok=True)
        tag = "_".join(sorted(qualities))
        np.savez(out_dir / f"umap_{tag}.npz", **res)
        _plot(out_dir / f"umap_{tag}.pdf", res, nwb_path.name)
        return res


def _scatter3d(ax, emb, c, cmap, label, title):
    p = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], c=c, cmap=cmap, s=4,
                   alpha=0.7, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_zlabel("UMAP 3")
    cb = ax.figure.colorbar(p, ax=ax, fraction=0.03, pad=0.08)
    cb.set_label(label)


def _plot(pdf_path, res, nwb_name):
    from matplotlib.backends.backend_pdf import PdfPages
    emb = res["emb"]
    panels = [
        ("x", res["x"], "viridis", "X position (m)"),
        ("y", res["y"], "plasma", "Y position (m)"),
        ("speed", res["speed"], "hot", "Speed (m/s)"),
        ("time", res["time"], "cool", "Session time (s)"),
    ]
    if np.isfinite(res.get("trial", np.array([np.nan]))).any():
        panels.append(("trial", res["trial"], "gist_rainbow", "Trial number"))
    if np.isfinite(res["goal_dist"]).any():
        panels.append(("goal", res["goal_dist"], "magma", "Distance to goal (m)"))
    if np.isfinite(res["ttype"]).any():
        panels.append(("ttype", res["ttype"], "tab10", "Trial type"))

    # the three 2D planes of the 3D embedding
    proj = [(0, 1, "UMAP 1", "UMAP 2"), (0, 2, "UMAP 1", "UMAP 3"),
            (1, 2, "UMAP 2", "UMAP 3")]

    with PdfPages(pdf_path) as pdf:
        # one 3D page per colour variable
        for _key, c, cmap, label in panels:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            _scatter3d(ax, emb, c, cmap, label,
                       f"Neural population UMAP — {res['quality']} "
                       f"(n={res['n_units']} units)\ncoloured by {label}")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 2D-projection page per colour variable: UMAP1/2, UMAP1/3, UMAP2/3
        for _key, c, cmap, label in panels:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
            for ax, (i, j, xl, yl) in zip(axes, proj):
                p = ax.scatter(emb[:, i], emb[:, j], c=c, cmap=cmap, s=4,
                               alpha=0.7, linewidths=0, rasterized=True)
                ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal", "box")
            fig.colorbar(p, ax=axes, fraction=0.02, pad=0.02).set_label(label)
            fig.suptitle(f"UMAP 2D projections — {res['quality']} — coloured by {label}",
                         fontsize=13)
            pdf.savefig(fig); plt.close(fig)
    print(f"  wrote {pdf_path}")


def run(output_folder, qualities, **kw):
    nwb_path = V.find_nwb_file(output_folder)
    if nwb_path is None:
        print(f"No NWB in {output_folder}."); return
    nwb_path = Path(nwb_path)
    print(f"UMAP embedding {nwb_path} using units: {'+'.join(sorted(qualities))}")
    embed_session(nwb_path, set(qualities), **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="UMAP embedding of population activity, per session.")
    ap.add_argument("--output_folder", required=True, help="op/session folder with the NWB.")
    ap.add_argument("--config", default=None, help="Accepted for runner consistency (unused).")
    ap.add_argument("--quality", nargs="+", default=["good"],
                    help="unit quality labels to include, e.g. --quality good mua")
    ap.add_argument("--dt", type=float, default=0.1, help="time-bin width (s).")
    ap.add_argument("--sigma_s", type=float, default=0.3, help="Gaussian smoothing sigma (s).")
    ap.add_argument("--n_neighbors", type=int, default=50)
    ap.add_argument("--min_dist", type=float, default=0.1)
    a = ap.parse_args()
    run(a.output_folder, set(q.lower() for q in a.quality),
        dt=a.dt, sigma_s=a.sigma_s, n_neighbors=a.n_neighbors, min_dist=a.min_dist)
