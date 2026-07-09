"""
Runner step [b] add-on: is the long-lead "prediction" genuine predictive coding,
or an artifact of behavioural autocorrelation / goal-occupancy bias?

For one session NWB it decodes the future position at several leads (0..3 s) and
compares the NEURAL decoder against controls that need NO neural data, then writes
every figure into a dedicated folder <op>/predictive_coding/:

  predictive_coding_<quality>.pdf
    1. error-vs-lead: median error of the neural decode (cross-validated) vs
       - persistence          (predict future = current position)      [autocorr]
       - constant velocity     (current + velocity x lead)             [kinematics]
       - behaviour-only        (empirical P(future | current bin+heading)) [habit]
       - chance
       plus a SHUFFLE band (spikes circularly shifted -> occupancy-only null).
       GENUINE predictive coding => the neural line drops BELOW behaviour-only and
       BELOW the shuffle band at 2-3 s. If neural ~ behaviour-only / inside the
       shuffle band, the goal-pointing is an artifact.
    2. overshoot: decode's distance-to-goal vs the animal's current distance-to-goal
       (with the TRUE-future distance as reference). Genuine goal look-ahead =>
       the decode reaches the goal EARLIER than the true 2-3 s future does.
    3. density: where decoded positions land on the maze, NEURAL vs SHUFFLE. If both
       pile at the goal identically, the goal-pile is occupancy, not coding.
    4. goal-switch (only if the session has type-5 trials): decode density before vs
       after the switch — genuine goal coding follows the NEW goal.

Reuses decode_position.decode_session (with its new shuffle_seed=... null) and
visualize_nwb for positions / maze / nodes.

Usage:
    python predictive_coding.py --output_folder <op> [--quality good [mua]]
        [--leads 0 1 2 3] [--cv_folds 5] [--n_shuffle 8]
"""

import sys
import argparse
import traceback
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_nwb as V          # noqa: E402
import decode_position as D        # noqa: E402
from pynwb import NWBHDF5IO        # noqa: E402


# ------------------------------------------------------------
#                 positions + behavioural baselines
# ------------------------------------------------------------
def _load_positions(nwb_path):
    """(x, y, t) in metres (plot_trials frame), t ascending. None if absent."""
    io = NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True)
    try:
        pos = V.load_position(io.read())
    finally:
        io.close()
    if pos is None:
        return None
    return pos[0] / V.SCALE_X, pos[1] / V.SCALE_Y, pos[2]


def _grid(bin_cm=10.0):
    xmin, xmax, ymin, ymax = V.MAZE_EXTENT
    bm = bin_cm / 100.0
    nx = max(5, int(round((xmax - xmin) / bm)))
    ny = max(5, int(round((ymax - ymin) / bm)))
    xe = np.linspace(xmin, xmax, nx + 1); ye = np.linspace(ymin, ymax, ny + 1)
    return nx, ny, xe, ye


def _heading_bin(vx, vy, nhead):
    ang = np.nan_to_num(np.arctan2(vy, vx))        # -pi..pi (NaN velocity -> 0)
    hb = np.floor((ang + np.pi) / (2 * np.pi) * nhead).astype(int)
    return np.clip(hb, 0, nhead - 1)


def _behaviour_predict(x, y, t, tc, lead, nx, ny, xe, ye, nhead=8, min_cnt=3):
    """Behaviour-only predictor of the future position at times `tc`: the empirical
    mean position `lead` s later, conditioned on the animal's current bin + heading
    (fallback: current bin, then global). Captures autocorrelation + maze topology +
    habitual goal-approach WITHOUT any neural data. Returns (pred_x, pred_y)."""
    tf = t + lead
    m = (tf <= t[-1] + 1e-9) & np.isfinite(x) & np.isfinite(y)
    fx = np.interp(tf, t, x); fy = np.interp(tf, t, y)
    vx = np.gradient(x, t); vy = np.gradient(y, t)
    bx = np.clip(np.digitize(x, xe) - 1, 0, nx - 1)
    by = np.clip(np.digitize(y, ye) - 1, 0, ny - 1)
    binid = by * nx + bx
    key = binid * nhead + _heading_bin(vx, vy, nhead)
    K = nx * ny * nhead
    cnt = np.bincount(key[m], minlength=K).astype(float)
    sfx = np.bincount(key[m], weights=fx[m], minlength=K)
    sfy = np.bincount(key[m], weights=fy[m], minlength=K)
    cb = np.bincount(binid[m], minlength=nx * ny).astype(float)
    bfx = np.bincount(binid[m], weights=fx[m], minlength=nx * ny)
    bfy = np.bincount(binid[m], weights=fy[m], minlength=nx * ny)
    gfx, gfy = float(fx[m].mean()), float(fy[m].mean())
    # test points
    cx0 = np.interp(tc, t, x); cy0 = np.interp(tc, t, y)
    vxc = np.interp(tc, t, vx); vyc = np.interp(tc, t, vy)
    bxc = np.clip(np.digitize(cx0, xe) - 1, 0, nx - 1)
    byc = np.clip(np.digitize(cy0, ye) - 1, 0, ny - 1)
    bc = byc * nx + bxc
    kc = bc * nhead + _heading_bin(vxc, vyc, nhead)
    px = np.full(tc.shape, gfx); py = np.full(tc.shape, gfy)
    use_k = cnt[kc] >= min_cnt
    px[use_k] = sfx[kc[use_k]] / cnt[kc[use_k]]
    py[use_k] = sfy[kc[use_k]] / cnt[kc[use_k]]
    use_b = (~use_k) & (cb[bc] >= min_cnt)
    px[use_b] = bfx[bc[use_b]] / cb[bc[use_b]]
    py[use_b] = bfy[bc[use_b]] / cb[bc[use_b]]
    return px, py, cx0, cy0, vxc, vyc


# ------------------------------------------------------------
#                 per-lead analysis
# ------------------------------------------------------------
def _analyse_lead(nwb_path, quality, lead, pos, cv_folds, n_shuffle, bin_cm, time_bin,
                  keep_density):
    """Decode one lead (CV) + its baselines + a shuffle null. Returns a dict, or
    None if the decode failed."""
    x, y, t = pos
    res = D.decode_session(nwb_path, quality, bin_cm=bin_cm, time_bin=time_bin,
                           folds=cv_folds, lead_s=float(lead), save=False, plot=False)
    if res is None:
        return None
    tc = np.asarray(res["t"], float)
    futx = np.asarray(res["actual_x"], float); futy = np.asarray(res["actual_y"], float)
    neux = np.asarray(res["decoded_x"], float); neuy = np.asarray(res["decoded_y"], float)
    neu_err = np.asarray(res["err"], float)
    chance = float(np.median(res["chance"]))
    nx, ny, xe, ye = _grid(bin_cm)

    # behaviour-only + kinematic baselines at the decoded bins
    bpx, bpy, cx0, cy0, vxc, vyc = _behaviour_predict(x, y, t, tc, lead, nx, ny, xe, ye)
    persist = np.hypot(cx0 - futx, cy0 - futy)
    cvel = np.hypot((cx0 + vxc * lead) - futx, (cy0 + vyc * lead) - futy)
    behav = np.hypot(bpx - futx, bpy - futy)

    # shuffle null: median error per shuffle (occupancy-only decode)
    shuf_med, shuf_dx, shuf_dy = [], [], []
    for k in range(n_shuffle):
        rs = D.decode_session(nwb_path, quality, bin_cm=bin_cm, time_bin=time_bin,
                              folds=1, lead_s=float(lead), save=False, plot=False,
                              shuffle_seed=1000 + k)
        if rs is None:
            continue
        shuf_med.append(float(np.median(rs["err"])))
        if keep_density:
            shuf_dx.append(np.asarray(rs["decoded_x"], float))
            shuf_dy.append(np.asarray(rs["decoded_y"], float))
    return {
        "lead": float(lead), "t": tc, "futx": futx, "futy": futy,
        "neux": neux, "neuy": neuy, "cx0": cx0, "cy0": cy0,
        "neu_err": neu_err, "persist": persist, "cvel": cvel, "behav": behav,
        "speed": np.hypot(vxc, vyc),
        "chance": chance, "shuf_med": np.array(shuf_med),
        "shuf_dx": np.concatenate(shuf_dx) if shuf_dx else np.empty(0),
        "shuf_dy": np.concatenate(shuf_dy) if shuf_dy else np.empty(0),
        "trials": res.get("trials"),
    }


# ------------------------------------------------------------
#                 figures
# ------------------------------------------------------------
def _goal_xy(per_lead, nodes):
    """Most common goal node's (x, y) in metres, from the trials list."""
    trials = next((d["trials"] for d in per_lead if d and d.get("trials")), None)
    if not trials:
        return None, None
    goals = [g for (_tt, g, _s, _a, _b) in trials if g is not None]
    if not goals:
        return None, None
    g = max(set(goals), key=goals.count)
    return g, nodes.get(g)


def _fig_error_vs_lead(pdf, per_lead, tag, move_thr=0.1):
    # Compare over MOVING bins only (speed > move_thr m/s): during dwelling every
    # baseline is trivially ~0 (the animal doesn't move), which hides the real test.
    leads = [d["lead"] for d in per_lead]
    masks = [d["speed"] > move_thr for d in per_lead]
    nmove = [int(m.sum()) for m in masks]

    def med(key):
        out = []
        for d, m in zip(per_lead, masks):
            v = np.asarray(d[key])[m] if np.ndim(d[key]) else np.array([d[key]])
            out.append(float(np.median(v)) if v.size else np.nan)
        return out
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12, 5))
    ax.plot(leads, med("neu_err"), "-o", color="#d62728", lw=2.4, label="neural (CV)", zorder=5)
    ax.plot(leads, med("behav"), "-s", color="#1f77b4", lw=1.8, label="behaviour-only (pos+heading)")
    ax.plot(leads, med("cvel"), "--^", color="#2ca02c", lw=1.4, label="constant velocity")
    ax.plot(leads, med("persist"), ":D", color="#9467bd", lw=1.4, label="persistence (current pos)")
    ax.plot(leads, [d["chance"] for d in per_lead], "-", color="0.6", lw=1.2, label="chance")
    # shuffle band
    smean = np.array([d["shuf_med"].mean() if d["shuf_med"].size else np.nan for d in per_lead])
    sstd = np.array([d["shuf_med"].std() if d["shuf_med"].size else 0.0 for d in per_lead])
    ax.fill_between(leads, smean - sstd, smean + sstd, color="0.5", alpha=0.25,
                    label="shuffle null (±sd)")
    ax.set_xlabel("prediction lead (s)"); ax.set_ylabel("median error (m)")
    ax.set_title("Neural decode vs behavioural / occupancy controls")
    ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    # right panel: neural minus behaviour-only (negative = neural adds info)
    diff = np.array(med("neu_err")) - np.array(med("behav"))
    axb.axhline(0, color="k", lw=1)
    axb.bar(leads, diff, width=0.35, color=["#2ca02c" if v < 0 else "#d62728" for v in diff])
    axb.set_xlabel("prediction lead (s)")
    axb.set_ylabel("neural - behaviour-only  (m)")
    axb.set_title("Below 0 = neural beats behaviour\n(genuine predictive info)")
    axb.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Predictive-coding test — error vs lead — units {tag}   "
                 f"(moving bins > {move_thr:g} m/s; n≈{min(nmove)}–{max(nmove)})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); pdf.savefig(fig); plt.close(fig)


def _fig_overshoot(pdf, per_lead, goal_xy, tag):
    if goal_xy is None:
        return
    gx, gy = goal_xy
    leads = [d for d in per_lead if d["lead"] > 0]
    if not leads:
        return
    ncol = len(leads)
    fig, axes = plt.subplots(1, ncol, figsize=(4.4 * ncol, 4.6), squeeze=False)
    for ax, d in zip(axes[0], leads):
        adist = np.hypot(d["cx0"] - gx, d["cy0"] - gy)      # animal now
        fdist = np.hypot(d["futx"] - gx, d["futy"] - gy)    # true future
        ddist = np.hypot(d["neux"] - gx, d["neuy"] - gy)    # neural decode
        ax.scatter(adist, ddist, s=8, alpha=0.3, color="#d62728", label="neural decode")
        # binned true-future reference
        order = np.argsort(adist)
        bins = np.linspace(adist.min(), adist.max(), 12)
        idx = np.digitize(adist, bins)
        bx = [adist[idx == b].mean() for b in range(1, len(bins)) if (idx == b).any()]
        bf = [fdist[idx == b].mean() for b in range(1, len(bins)) if (idx == b).any()]
        ax.plot(bx, bf, "-o", color="#1f77b4", lw=1.8, ms=4, label="true future (ref)")
        ax.plot([0, adist.max()], [0, adist.max()], ":", color="0.6", lw=1)
        ax.set_xlabel("animal distance to goal now (m)")
        ax.set_ylabel("distance to goal (m)")
        ax.set_title(f"lead {d['lead']:g}s")
        ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Overshoot: decode closer to goal than the true future = look-ahead "
                 f"(below the blue line) — units {tag}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)


def _fig_density(pdf, per_lead, nodes, goal_xy, tag, bin_cm=10.0):
    dens = [d for d in per_lead if d["shuf_dx"].size]
    if not dens:
        return
    dmax = dens[-1]                                  # largest lead
    d0 = next((d for d in per_lead if d["lead"] == 0.0), None)
    ext = V.MAZE_EXTENT
    nx, ny, xe, ye = _grid(bin_cm)
    panels = []
    if d0 is not None:
        panels.append((d0["neux"], d0["neuy"], f"neural, lead {d0['lead']:g}s"))
    panels.append((dmax["neux"], dmax["neuy"], f"neural, lead {dmax['lead']:g}s"))
    panels.append((dmax["shuf_dx"], dmax["shuf_dy"], f"shuffle null, lead {dmax['lead']:g}s"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.8), squeeze=False)
    for ax, (dx, dy, ttl) in zip(axes[0], panels):
        V.draw_maze(ax, nodes)
        h, _, _ = np.histogram2d(dx, dy, bins=[xe, ye])
        ax.imshow(h.T, origin="lower", extent=ext, aspect="equal", cmap="magma",
                  interpolation="nearest", alpha=0.85)
        if goal_xy is not None:
            ax.scatter(*goal_xy, marker="*", s=240, c="gold", edgecolors="k", zorder=6)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl)
    fig.suptitle("Where decoded positions land — lead-0 (current) vs long-lead vs shuffle null.  "
                 "EXTRA goal-pile at long lead vs lead-0 => prospective; SAME pile already at "
                 f"lead-0 => occupancy/dwelling. Shuffle = uniform-over-visited null. — units {tag}",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)


def _fig_goal_switch(pdf, per_lead, nodes, tag, bin_cm=10.0):
    """Only meaningful with type-5 goal-switch trials; otherwise a text note."""
    d = next((x for x in per_lead if x["lead"] in (2.0, 3.0)), per_lead[-1])
    trials = d.get("trials") or []
    specials = [(tt, g, s, a, b) for (tt, g, s, a, b) in trials if tt == 5]
    fig, ax = plt.subplots(figsize=(8.27, 5.5))
    if len(specials) < 1:
        ax.axis("off")
        ax.text(0.5, 0.5, "No goal-switch (type-5) trials in this session.\n"
                "Goal is fixed, so 'decode -> goal' cannot be dissociated from a\n"
                "fixed spatial/occupancy bias here. Run this on a goal-switch or\n"
                "multi-goal session to test whether the decode follows the NEW goal.",
                ha="center", va="center", fontsize=11)
        pdf.savefig(fig); plt.close(fig); return
    # before/after the (first) switch: decode density on the maze
    _tt, g_after, _s, s0, s1 = specials[0]
    ext = V.MAZE_EXTENT; nx, ny, xe, ye = _grid(bin_cm)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    t = d["t"]
    for ax, (mask, ttl) in zip(axes, [(t < s0, "before switch"), (t >= s0, "after switch")]):
        V.draw_maze(ax, nodes)
        if mask.any():
            h, _, _ = np.histogram2d(d["neux"][mask], d["neuy"][mask], bins=[xe, ye])
            ax.imshow(h.T, origin="lower", extent=ext, aspect="equal", cmap="magma",
                      interpolation="nearest", alpha=0.85)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[3], ext[2])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl)
    fig.suptitle(f"Goal-switch: does the decode follow the new goal? (lead {d['lead']:g}s) "
                 f"— units {tag}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)


# ------------------------------------------------------------
#                 orchestration
# ------------------------------------------------------------
def run(output_folder, quality=("good",), leads=(0.0, 1.0, 2.0, 3.0),
        cv_folds=5, n_shuffle=8, bin_cm=10.0, time_bin=0.5, out_dir=None):
    nwb_path = V.find_nwb_file(output_folder)
    if nwb_path is None:
        print(f"No .nwb under '{output_folder}' (run steps w/u first). Skipping.")
        return None
    quals = set(quality); tag = "+".join(sorted(quals))
    pos = _load_positions(nwb_path)
    if pos is None:
        print("  no Position data — skipping."); return None
    # keep only tracked samples so np.gradient / np.interp / the behaviour map never
    # see NaN (untracked frames); t stays ascending.
    _x, _y, _t = pos
    _ok = np.isfinite(_x) & np.isfinite(_y)
    if _ok.sum() < 10:
        print("  too few tracked position samples — skipping."); return None
    pos = (_x[_ok], _y[_ok], _t[_ok])
    nodes = V.load_nodes()

    print(f"  predictive-coding test on {nwb_path.name}  units {tag}  "
          f"leads {', '.join(f'{L:g}' for L in leads)}  CV {cv_folds}  shuffles {n_shuffle}")
    per_lead = []
    max_lead = max(leads)
    for L in leads:
        print(f"    lead {L:g}s ...", flush=True)
        d = _analyse_lead(nwb_path, quals, L, pos, cv_folds, n_shuffle, bin_cm,
                          time_bin, keep_density=(L == max_lead or L == 0.0))
        if d is not None:
            per_lead.append(d)
    if not per_lead:
        print("  nothing decoded — skipping."); return None

    goal_node, goal_xy = _goal_xy(per_lead, nodes)
    out_dir = Path(out_dir) if out_dir else Path(output_folder) / "predictive_coding"
    out_dir.mkdir(parents=True, exist_ok=True)
    tagf = "_".join(sorted(quals))
    pdf_path = out_dir / f"predictive_coding_{tagf}.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        _fig_error_vs_lead(pdf, per_lead, tag)
        _fig_overshoot(pdf, per_lead, goal_xy, tag)
        _fig_density(pdf, per_lead, nodes, goal_xy, tag, bin_cm)
        _fig_goal_switch(pdf, per_lead, nodes, tag, bin_cm)
    # concise verdict to the log
    m_neu = [float(np.median(d["neu_err"])) for d in per_lead]
    m_beh = [float(np.median(d["behav"])) for d in per_lead]
    verdict = "; ".join(f"{d['lead']:g}s neu {n:.2f} vs behav {b:.2f}m"
                        for d, n, b in zip(per_lead, m_neu, m_beh))
    print(f"  wrote {pdf_path}\n    {verdict}")
    return pdf_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Test whether long-lead position 'prediction' is genuine predictive coding.")
    ap.add_argument("--output_folder", required=True, help="op/session folder with the NWB.")
    ap.add_argument("--config", default=None, help="Accepted for runner consistency (unused).")
    ap.add_argument("--quality", nargs="+", default=["good"],
                    choices=["good", "mua", "noise"], help="unit qualities (default good).")
    ap.add_argument("--leads", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0])
    ap.add_argument("--cv_folds", type=int, default=5, help="CV folds for the neural decode.")
    ap.add_argument("--n_shuffle", type=int, default=8, help="spike-shuffle nulls per lead.")
    ap.add_argument("--bin_cm", type=float, default=10.0)
    ap.add_argument("--time_bin", type=float, default=0.5)
    args = ap.parse_args()
    try:
        r = run(args.output_folder, quality=tuple(args.quality), leads=tuple(args.leads),
                cv_folds=args.cv_folds, n_shuffle=args.n_shuffle,
                bin_cm=args.bin_cm, time_bin=args.time_bin)
        if r is None:
            sys.exit(1)
    except Exception as e:
        print(f"[predictive-coding] Failed: {e}")
        traceback.print_exc()
        sys.exit(1)
