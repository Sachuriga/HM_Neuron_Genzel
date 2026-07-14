"""
Runner step [m]: overlay analysis onto the REAL behaviour video, one mp4 per trial.

Two kinds of video, one file per goal trial (the ~10-min free-roaming type-4/5
"special" trials are EXCLUDED). Each file is the recorded overhead maze video
(<op>/<date>_Rat*.mp4, 1176x712 @30fps — the tracker's annotated output) with our
overlays drawn on top, replaying a single trial from scratch (reset per trial):

  videos/spikes_on_video_<quality>_trialNN_typeT.mp4
     The top-N putative pyramidal cells by whole-session Skaggs spatial information
     (default N=20), all overlaid on ONE view: as the animal moves, each unit's
     spikes are dropped onto the path where the animal was when it fired, one
     COLOUR per unit. Spikes accumulate through the trial.

  videos/decoded_on_video_<quality>_trialNN_typeT.mp4
     The Bayesian-decoded position for each prediction lead (default 0,1,2,3 s
     ahead) overlaid on the same view: one colour per lead, decoded positions
     accumulating as a trail, the current decoded point highlighted with an arrow
     from the animal's current position to it (the "prediction reach").

Alignment (verified on rat5_491390/20260626): the video's frame k == Frame_Index k
in <date>_Rat*_Coordinates_Full_with_frames.csv (Rat_X/Rat_Y are in the video's
1176x712 pixel frame), and Frame_Index -> seconds via stitched_framewise_seconds.csv
is the SAME clock the NWB spike times / decoder use. Spikes and decoded positions
therefore land on the right pixel at the right frame with no scaling guesswork.

Reused pipeline machinery: visualize_nwb (V) for spatial-info ranking, SCALE_X/Y,
trial metadata, find_nwb_file; decode_position (D) for the per-lead decode.

Usage:
    python make_videos.py --output_folder <op> [--which spikes|decoded|both]
        [--n_units 20] [--leads 0 1 2 3] [--quality good [mua]]
        [--exclude_types 4 5] [--trials 3 4 ...] [--stride 1] [--hold_s 0.6]
"""

import sys
import argparse
import traceback
from pathlib import Path

from session_prefix import file_prefix

import numpy as np
import pandas as pd
import cv2

try:
    from tqdm import tqdm            # live per-frame progress bar (with ETA)
except Exception:                    # degrade gracefully if tqdm is absent
    tqdm = None

import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_nwb as V          # noqa: E402  spatial info, SCALE, trial meta
import decode_position as D        # noqa: E402  per-lead Bayesian decoding

from pynwb import NWBHDF5IO        # noqa: E402


# ------------------------------------------------------------
#                 colours (matplotlib RGB -> cv2 BGR)
# ------------------------------------------------------------
def _bgr(rgb):
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


def _unit_colors(n):
    """n BGR colours sampled across the 'jet' colormap for the units (blue->red,
    ordered by the units' spatial-info rank)."""
    cmap = colormaps["jet"]
    return [_bgr(cmap(i / max(n - 1, 1))) for i in range(n)]


def _lead_colors(n):
    """n BGR colours sampled across the 'jet' colormap for the prediction leads."""
    cmap = colormaps["jet"]
    return [_bgr(cmap(i / max(n - 1, 1))) for i in range(n)]


# ------------------------------------------------------------
#                 inputs: video, frame arrays, trials
# ------------------------------------------------------------
def _find_behavior_video(op, want_w=1176, want_h=712):
    """The overhead maze video whose pixel frame matches the coordinates
    (<date>_Rat*.mp4 the tracker wrote). Prefers a *Rat*.mp4 whose width/height
    match `want_w/want_h`; else the closest-sized non-'collected_frames' mp4."""
    op = Path(op)
    cands = [p for p in sorted(op.glob("*.mp4"))
             if not p.name.startswith("._") and "collected_frames" not in p.name
             and not p.name.lower().startswith("eye")]
    best, best_score = None, None
    for p in cands:
        cap = cv2.VideoCapture(str(p))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        score = abs(w - want_w) + abs(h - want_h) - (1000 if "Rat" in p.name else 0)
        if best is None or score < best_score:
            best, best_score = p, score
    return best


def _load_frame_arrays(op):
    """Per-frame arrays indexed by Frame_Index (0..N-1):
    sec[f] (NWB seconds clock), x[f]/y[f] (Rat pixel pos, NaN if untracked),
    trial[f] (Trial_Num, NaN outside trials). None if the CSVs are missing."""
    coords = (V._pick_file(op, "*Coordinates_Full_with_frames.csv")
              or V._pick_file(op, "*Coordinates_Full.csv"))
    if coords is None:
        print("  no Coordinates_Full_with_frames.csv — cannot overlay on video.")
        return None
    cf = V._read_csv_tol(coords)
    need = {"Frame_Index", "Rat_X", "Rat_Y", "Trial_Num"}
    if not need <= set(cf.columns):
        print(f"  {coords.name} missing {need - set(cf.columns)}.")
        return None
    fi = pd.to_numeric(cf["Frame_Index"], errors="coerce")
    n = int(np.nanmax(fi)) + 1
    sec = np.full(n, np.nan); x = np.full(n, np.nan); y = np.full(n, np.nan)
    trial = np.full(n, np.nan)
    idx = fi.to_numpy()
    ok = np.isfinite(idx)
    ii = idx[ok].astype(int)
    x[ii] = pd.to_numeric(cf["Rat_X"], errors="coerce").to_numpy()[ok]
    y[ii] = pd.to_numeric(cf["Rat_Y"], errors="coerce").to_numpy()[ok]
    trial[ii] = pd.to_numeric(cf["Trial_Num"], errors="coerce").to_numpy()[ok]
    f2s = V.frame_to_seconds(op)                # {frame: seconds} from stitched csv
    if f2s:
        for f, s in f2s.items():
            if 0 <= f < n:
                sec[f] = s
    if not np.isfinite(sec).any():
        print("  no frame->seconds map (stitched_framewise_seconds.csv) — aborting.")
        return None
    return sec, x, y, trial, n


def _trial_windows_frames(trial, sec, op, exclude_types, trials_sel, n_video):
    """[(trialnum, type, goal, start, f0, f1, t0, t1)] per Trial_Num block, dropping
    excluded types, clipped to the video's frame count. trialnum == Trial_Num."""
    meta = V.read_trial_meta(op)               # {Trial_Num: (type, goal, start)}
    out = []
    nums = np.unique(trial[np.isfinite(trial)]).astype(int)
    for k in nums:
        if trials_sel and int(k) not in trials_sel:
            continue
        tt, goal, start = meta.get(int(k), (-1, None, None))
        if tt in exclude_types:
            continue
        frames = np.where(trial == k)[0]
        if frames.size == 0:
            continue
        f0, f1 = int(frames.min()), min(int(frames.max()), n_video - 1)
        if f1 <= f0:
            continue
        out.append((int(k), int(tt), goal, start,
                    f0, f1, float(sec[f0]), float(sec[f1])))
    out.sort(key=lambda r: r[4])
    return out


# ------------------------------------------------------------
#                 drawing helpers (on the video frame)
# ------------------------------------------------------------
def _box(frame, x0, y0, x1, y1, alpha=0.5):
    """Darken a rectangle so overlaid text/legend is readable on the busy video."""
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(frame.shape[1], x1); y1 = min(frame.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    sub = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.addWeighted(sub, 1 - alpha, np.zeros_like(sub), alpha, 0)


def _title_bar(frame, text, sub=""):
    """Translucent title strip at the very bottom-left with our label + time."""
    h = frame.shape[0]
    _box(frame, 0, h - 40, 760, h, alpha=0.55)
    cv2.putText(frame, text, (10, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    if sub:
        cv2.putText(frame, sub, (10, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 220, 255), 1, cv2.LINE_AA)


def _legend(frame, items, title):
    """Colour legend in a translucent panel at the top-right. items=[(label,bgr)]."""
    n = len(items)
    ncol = 2 if n > 8 else 1
    per = int(np.ceil(n / ncol))
    row_h, col_w = 16, 96
    W = frame.shape[1]
    x0 = W - col_w * ncol - 12
    y0 = 6
    _box(frame, x0 - 6, y0 - 2, W - 2, y0 + row_h * per + 18, alpha=0.5)
    cv2.putText(frame, title, (x0, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    for i, (lab, bgr) in enumerate(items):
        cx = x0 + (i // per) * col_w
        cy = y0 + 18 + (i % per) * row_h
        cv2.circle(frame, (cx + 5, cy + 5), 5, bgr, -1)
        cv2.putText(frame, lab, (cx + 15, cy + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (235, 235, 235), 1, cv2.LINE_AA)


def _writer(path, w, h, fps):
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


# ------------------------------------------------------------
#                 loading + ranking (unchanged)
# ------------------------------------------------------------
def _open_nwb(output_folder):
    nwb_path = V.find_nwb_file(output_folder)
    if nwb_path is None:
        print(f"No .nwb under '{output_folder}' (run steps w/u first). Skipping.")
        return None, None, None
    io = NWBHDF5IO(str(nwb_path), mode="r", load_namespaces=True)
    return nwb_path, io, io.read()


def _spatial_frame(t):
    ext = V.MAZE_EXTENT
    bins = (max(5, int(round((ext[1] - ext[0]) / 0.05))),
            max(5, int(round((ext[3] - ext[2]) / 0.05))))
    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0 / 30
    return ext, bins, dt


def top_spatial_info_units(nwb, pos, n_units, quality, sigma=2.0, speed=0.05):
    """The n_units GOOD putative-pyramidal cells with the highest whole-session
    Skaggs spatial information. Returns [(cid, spike_times, spatial_info), ...]."""
    udf = nwb.units.to_dataframe()
    ql = udf["quality_label"].astype(str) if "quality_label" in udf.columns \
        else pd.Series("good", index=udf.index)
    ct = udf["cell_type"].astype(str) if "cell_type" in udf.columns \
        else pd.Series("pyramidal", index=udf.index)
    sel = udf[(ql == quality) & (ct == "pyramidal")]
    if not len(sel):
        print("  no good pyramidal cells found.")
        return []
    x, y, t = pos
    ext, bins, dt = _spatial_frame(t)
    ranked = []
    for uid, row in sel.iterrows():
        st = np.asarray(row["spike_times"], dtype=float)
        m, _, _ = V.place_field_metrics(x, y, t, st, ext, bins, dt, sigma, speed)
        si = m.get("spatial_info", np.nan)
        cid = int(row["phy_cluster_id"]) if "phy_cluster_id" in sel.columns else int(uid)
        ranked.append((cid, st, float(si) if np.isfinite(si) else -np.inf))
    ranked.sort(key=lambda r: -r[2])
    top = ranked[:n_units]
    print(f"  {len(sel)} good pyramidal cells; taking top {len(top)} by spatial info "
          f"(SI {top[0][2]:.2f}..{top[-1][2]:.2f} bits/spk).")
    return top


def _spike_pixels(spike_times, sec, x, y):
    """(ts, px, py) for each spike: its time (s) and the animal's pixel position
    then, interpolated onto the tracked trajectory. Spikes off-track are dropped."""
    ok = np.isfinite(sec) & np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.empty(0), np.empty(0), np.empty(0)
    sv, xv, yv = sec[ok], x[ok], y[ok]
    order = np.argsort(sv)
    sv, xv, yv = sv[order], xv[order], yv[order]
    px = np.interp(spike_times, sv, xv, left=np.nan, right=np.nan)
    py = np.interp(spike_times, sv, yv, left=np.nan, right=np.nan)
    good = np.isfinite(px) & np.isfinite(py)
    return spike_times[good], px[good], py[good]


# ------------------------------------------------------------
#                 video 1: spikes on the real video
# ------------------------------------------------------------
def make_spike_path_videos(output_folder, n_units=20, quality="good",
                           exclude_types=(4, 5), hold_s=0.6, stride=1,
                           trials_sel=None, trail=True, jitter=True,
                           fr_offset_px=15.0, jitter_px=6.0, speed_thresh=0.05):
    """One mp4 per goal trial: the real behaviour video with the top-`n_units`
    spatial-information pyramidal cells' spikes overlaid (one colour per unit,
    accumulating), reset per trial."""
    nwb_path, io, nwb = _open_nwb(output_folder)
    if nwb is None:
        return []
    try:
        if nwb.units is None or len(nwb.units.id) == 0:
            print("  NWB has no Units table — skipping."); return []
        pos = V.load_position(nwb)
        if pos is None:
            print("  no Position data — skipping."); return []
        pm = (pos[0] / V.SCALE_X, pos[1] / V.SCALE_Y, pos[2])
        top = top_spatial_info_units(nwb, pm, n_units, quality)
    finally:
        io.close()
    if not top:
        return []

    arrs = _load_frame_arrays(output_folder)
    if arrs is None:
        return []
    sec, x, y, trial, ncoord = arrs
    vid = _find_behavior_video(output_folder)
    if vid is None:
        print("  no behaviour video found in op folder — skipping."); return []
    cap = cv2.VideoCapture(str(vid))
    nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  behaviour video: {vid.name} ({W}x{H}, {nvid} frames @ {fps:g} fps)")

    render = _trial_windows_frames(trial, sec, output_folder, tuple(exclude_types),
                                   trials_sel, min(nvid, ncoord))
    if not render:
        cap.release(); print("  no goal trials to render."); return []

    colors = _unit_colors(len(top))
    # firing-rate-ranked perpendicular OFFSET so the 20 colour-coded units don't all
    # pile onto the 1-D path: highest-FR unit hugs the line (offset 0), lowest-FR unit
    # sits farthest out; a small random jitter spreads a unit's own spikes too.
    dur = float(pos[2].max() - pos[2].min()) or 1.0
    fr = np.array([st.size / dur for (_c, st, _s) in top], float)
    rank = np.argsort(np.argsort(fr))                    # 0=lowest FR .. n-1=highest
    base_off = fr_offset_px * (1.0 - rank / max(len(top) - 1, 1))
    # path direction (px/s) for the perpendicular, from the tracked trajectory
    vok = np.isfinite(sec) & np.isfinite(x) & np.isfinite(y)
    svu, uidx = np.unique(sec[vok], return_index=True)
    xu, yu = x[vok][uidx], y[vok][uidx]
    vx, vy = np.gradient(xu, svu), np.gradient(yu, svu)
    # running speed (m/s) per sample, to speed-gate spikes to RUN epochs (place-cell
    # convention, same 0.05 m/s used by the rate maps / spatial info / decoder).
    speed_m = np.hypot(np.gradient(xu / V.SCALE_X, svu), np.gradient(yu / V.SCALE_Y, svu))
    rng = np.random.default_rng(0)

    # per-unit spike (time, pixel) once for the whole session
    units = []
    for k, (cid, st, si) in enumerate(top):
        ts, sx, sy = _spike_pixels(st, sec, x, y)
        if speed_thresh and speed_thresh > 0 and ts.size:
            keep = np.interp(ts, svu, speed_m) > speed_thresh    # keep run-epoch spikes
            ts, sx, sy = ts[keep], sx[keep], sy[keep]
        if jitter and ts.size:
            vxk = np.interp(ts, svu, vx); vyk = np.interp(ts, svu, vy)
            nrm = np.hypot(vxk, vyk); nrm[nrm == 0] = 1.0
            perpx, perpy = -vyk / nrm, vxk / nrm         # unit normal to the path
            sign = rng.choice([-1.0, 1.0], size=ts.size)
            r = base_off[k] + rng.uniform(0.0, jitter_px, size=ts.size)
            sx = sx + perpx * sign * r
            sy = sy + perpy * sign * r
        units.append({"cid": cid, "si": si, "ts": ts, "sx": sx, "sy": sy,
                      "bgr": colors[k]})
    legend = [(f"u{u['cid']}", u["bgr"]) for u in units]

    out_dir = Path(output_folder) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    hold_n = int(round(hold_s * fps / max(stride, 1)))
    out_fps = fps / max(stride, 1)

    total = sum(len(range(f0, f1 + 1, stride)) + hold_n
                for (_k, _tt, _g, _s, f0, f1, _t0, _t1) in render)
    print(f"  spikes-on-video: {len(units)} units, {len(render)} goal trials "
          f"(excluded {tuple(exclude_types)}), ~{total} frames. Writing to {out_dir} ...")
    pbar = tqdm(total=total, unit="frame", desc="spikes-on-video",
                dynamic_ncols=True, mininterval=0.5, file=sys.stdout) if tqdm else None

    made = []
    for (k, tt, goal, start, f0, f1, t0, t1) in render:
        out_path = out_dir / f"{file_prefix(output_folder)}spikes_on_video_{quality}_trial{k:02d}_type{tt}.mp4"
        vw = _writer(out_path, W, H, out_fps)
        overlay = np.zeros((H, W, 3), np.uint8)
        drawn = np.zeros((H, W), np.uint8)
        ptr = [int(np.searchsorted(u["ts"], t0, "left")) for u in units]
        prev_xy = None
        if pbar is not None:
            pbar.set_description(f"spikes trial {k:02d}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        last = None
        for f in range(f0, f1 + 1):
            ok, frame = cap.read()
            if not ok:
                break
            sec_f = sec[f] if f < len(sec) else np.nan
            # accumulate the within-trial trajectory onto the persistent overlay
            if trail and np.isfinite(x[f]) and np.isfinite(y[f]):
                p = (int(x[f]), int(y[f]))
                if prev_xy is not None:
                    cv2.line(overlay, prev_xy, p, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(drawn, prev_xy, p, 255, 1)
                prev_xy = p
            # add any spikes that have now occurred (<= this frame's time)
            if np.isfinite(sec_f):
                for i, u in enumerate(units):
                    j = ptr[i]
                    while j < u["ts"].size and u["ts"][j] <= sec_f:
                        c = (int(u["sx"][j]), int(u["sy"][j]))
                        cv2.circle(overlay, c, 4, u["bgr"], -1, cv2.LINE_AA)
                        cv2.circle(drawn, c, 4, 255, -1)
                        j += 1
                    ptr[i] = j
            if (f - f0) % stride:
                continue
            m = drawn > 0
            frame[m] = overlay[m]
            _title_bar(frame,
                       f"Spikes on path  |  trial {k}  (type {tt}, goal {goal}, start {start})",
                       f"top {len(units)} pyramidal by spatial info   t = {sec_f - t0:5.1f} s")
            _legend(frame, legend, f"unit (n={len(units)})")
            vw.write(frame)
            last = frame
            if pbar is not None:
                pbar.update(1)
        for _ in range(hold_n):            # freeze the finished trial briefly
            if last is not None:
                vw.write(last)
            if pbar is not None:
                pbar.update(1)
        vw.release()
        made.append(out_path)
    if pbar is not None:
        pbar.close()
    cap.release()
    print(f"  wrote {len(made)} spikes-on-video file(s).")
    return made


# ------------------------------------------------------------
#                 video 2: decoded position on the real video
# ------------------------------------------------------------
def make_decoded_leads_videos(output_folder, leads=(0.0, 1.0, 2.0, 3.0),
                              quality=("good",), exclude_types=(4, 5), hold_s=0.6,
                              stride=1, bin_cm=10.0, time_bin=0.5, trials_sel=None):
    """One mp4 per goal trial: the real behaviour video with the Bayesian-decoded
    position for each prediction lead overlaid (one colour per lead, accumulating
    trail + current point + prediction-reach arrow), reset per trial."""
    nwb_path = V.find_nwb_file(output_folder)
    if nwb_path is None:
        print(f"No .nwb under '{output_folder}' (run steps w/u first). Skipping.")
        return []
    quals = list(quality); tag = "_".join(sorted(quals))

    lead_res = []
    for L in leads:
        print(f"  decoding lead {L:g}s ...", flush=True)
        r = D.decode_session(nwb_path, set(quals), bin_cm=bin_cm, time_bin=time_bin,
                             folds=1, lead_s=float(L), save=False, plot=False)
        if r is None:
            print(f"    lead {L:g}s produced no decode — skipping this lead."); continue
        o = np.argsort(r["t"])
        lead_res.append({"lead": float(L), "t": r["t"][o],
                         "px": r["decoded_x"][o] * V.SCALE_X,     # metres -> pixels
                         "py": r["decoded_y"][o] * V.SCALE_Y})
    if not lead_res:
        print("  no leads decoded — skipping."); return []

    arrs = _load_frame_arrays(output_folder)
    if arrs is None:
        return []
    sec, x, y, trial, ncoord = arrs
    vid = _find_behavior_video(output_folder)
    if vid is None:
        print("  no behaviour video found — skipping."); return []
    cap = cv2.VideoCapture(str(vid))
    nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  behaviour video: {vid.name} ({W}x{H}, {nvid} frames @ {fps:g} fps)")

    render = _trial_windows_frames(trial, sec, output_folder, tuple(exclude_types),
                                   None, min(nvid, ncoord))
    if trials_sel:
        render = [r for r in render if r[0] in trials_sel]
    if not render:
        cap.release(); print("  no goal trials to render."); return []

    colors = _lead_colors(len(lead_res))
    for lr, c in zip(lead_res, colors):
        lr["bgr"] = c
    legend = [(f"lead {lr['lead']:g}s", lr["bgr"]) for lr in lead_res]

    out_dir = Path(output_folder) / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    hold_n = int(round(hold_s * fps / max(stride, 1)))
    out_fps = fps / max(stride, 1)

    total = sum(len(range(f0, f1 + 1, stride)) + hold_n
                for (_k, _tt, _g, _s, f0, f1, _t0, _t1) in render)
    print(f"  decoded-on-video: {len(lead_res)} leads, {len(render)} goal trials "
          f"(excluded {tuple(exclude_types)}), ~{total} frames. Writing to {out_dir} ...")
    pbar = tqdm(total=total, unit="frame", desc="decoded-on-video",
                dynamic_ncols=True, mininterval=0.5, file=sys.stdout) if tqdm else None

    made = []
    for (k, tt, goal, start, f0, f1, t0, t1) in render:
        out_path = out_dir / f"{file_prefix(output_folder)}decoded_on_video_{tag}_trial{k:02d}_type{tt}.mp4"
        vw = _writer(out_path, W, H, out_fps)
        overlay = np.zeros((H, W, 3), np.uint8)         # accumulated decoded trails
        drawn = np.zeros((H, W), np.uint8)
        ptr = [int(np.searchsorted(lr["t"], t0, "left")) for lr in lead_res]
        prev = [None] * len(lead_res)
        if pbar is not None:
            pbar.set_description(f"decoded trial {k:02d}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        last = None
        for f in range(f0, f1 + 1):
            ok, frame = cap.read()
            if not ok:
                break
            sec_f = sec[f] if f < len(sec) else np.nan
            cur = [None] * len(lead_res)
            if np.isfinite(sec_f):
                # accumulate decoded positions as DOTS (no connecting lines: the
                # decode jumps between bins, so lines would crisscross into a mess)
                for li, lr in enumerate(lead_res):
                    j = ptr[li]
                    while j < lr["t"].size and lr["t"][j] <= sec_f:
                        p = (int(lr["px"][j]), int(lr["py"][j]))
                        cv2.circle(overlay, p, 3, lr["bgr"], -1, cv2.LINE_AA)
                        cv2.circle(drawn, p, 3, 255, -1)
                        prev[li] = p; cur[li] = p
                        j += 1
                    ptr[li] = j
                    if cur[li] is None:
                        cur[li] = prev[li]
            if (f - f0) % stride:
                continue
            m = drawn > 0
            frame[m] = overlay[m]
            # per-frame layer: current actual position + reach arrows + current decoded
            act = (int(x[f]), int(y[f])) if np.isfinite(x[f]) and np.isfinite(y[f]) else None
            for li, lr in enumerate(lead_res):
                if cur[li] is not None:
                    if act is not None:
                        cv2.arrowedLine(frame, act, cur[li], lr["bgr"], 2,
                                        cv2.LINE_AA, tipLength=0.15)
                    cv2.circle(frame, cur[li], 8, lr["bgr"], -1, cv2.LINE_AA)
                    cv2.circle(frame, cur[li], 8, (255, 255, 255), 1, cv2.LINE_AA)
            if act is not None:
                cv2.drawMarker(frame, act, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
            _title_bar(frame,
                       f"Decoded position  |  trial {k}  (type {tt}, goal {goal})",
                       f"leads {', '.join(f'{lr['lead']:g}s' for lr in lead_res)}   "
                       f"t = {sec_f - t0:5.1f} s   (X = current position)")
            _legend(frame, legend, "prediction lead")
            vw.write(frame)
            last = frame
            if pbar is not None:
                pbar.update(1)
        for _ in range(hold_n):
            if last is not None:
                vw.write(last)
            if pbar is not None:
                pbar.update(1)
        vw.release()
        made.append(out_path)
    if pbar is not None:
        pbar.close()
    cap.release()
    print(f"  wrote {len(made)} decoded-on-video file(s).")
    return made


# ------------------------------------------------------------
#                 orchestration / CLI
# ------------------------------------------------------------
def make_videos(output_folder, which="both", n_units=20, leads=(0.0, 1.0, 2.0, 3.0),
                quality=("good",), exclude_types=(4, 5), hold_s=0.6, stride=1,
                bin_cm=10.0, time_bin=0.5, trials_sel=None, jitter=True,
                fr_offset_px=15.0, speed_thresh=0.05):
    q0 = list(quality)[0] if quality else "good"
    made = []
    if which in ("spikes", "both"):
        made += make_spike_path_videos(
            output_folder, n_units=n_units, quality=q0, exclude_types=exclude_types,
            hold_s=hold_s, stride=stride, trials_sel=trials_sel, jitter=jitter,
            fr_offset_px=fr_offset_px, speed_thresh=speed_thresh)
    if which in ("decoded", "both"):
        made += make_decoded_leads_videos(
            output_folder, leads=leads, quality=quality, exclude_types=exclude_types,
            hold_s=hold_s, stride=stride, bin_cm=bin_cm, time_bin=time_bin,
            trials_sel=trials_sel)
    return made


def run(output_folder, **kw):
    return make_videos(output_folder, **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Overlay spikes / decoded position on the real behaviour video, one mp4 per trial.")
    ap.add_argument("--output_folder", required=True, help="op/session folder with the NWB + behaviour video.")
    ap.add_argument("--config", default=None, help="Accepted for runner consistency (unused).")
    ap.add_argument("--which", choices=["spikes", "decoded", "both"], default="both")
    ap.add_argument("--n_units", type=int, default=20,
                    help="top spatial-info pyramidal cells for video 1 (default 20).")
    ap.add_argument("--leads", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0],
                    help="prediction leads (s) for video 2 (default 0 1 2 3).")
    ap.add_argument("--quality", nargs="+", default=["good"],
                    choices=["good", "mua", "noise"], help="unit qualities (default good).")
    ap.add_argument("--exclude_types", type=int, nargs="*", default=[4, 5],
                    help="trial types to EXCLUDE (default 4 5 = the free-roaming specials).")
    ap.add_argument("--hold_s", type=float, default=0.6,
                    help="freeze the finished trial this long at the end of its file.")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth video frame (1 = real time; 2 = 2x faster/smaller).")
    ap.add_argument("--bin_cm", type=float, default=10.0, help="decoder tuning-curve bin (cm).")
    ap.add_argument("--time_bin", type=float, default=0.5, help="decoder time bin (s).")
    ap.add_argument("--trials", type=int, nargs="+", default=None,
                    help="only render these (1-based Trial_Num) trials.")
    ap.add_argument("--no_jitter", action="store_true",
                    help="disable the firing-rate perpendicular jitter on spikes-on-video.")
    ap.add_argument("--fr_offset_px", type=float, default=15.0,
                    help="max perpendicular offset (px) for the lowest-FR unit's spikes (default 15).")
    ap.add_argument("--speed_thresh", type=float, default=0.05,
                    help="speed gate (m/s) for spikes-on-video: show only run-epoch spikes "
                         "(0 = no gating; default 0.05).")
    args = ap.parse_args()

    try:
        made = make_videos(
            args.output_folder, which=args.which, n_units=args.n_units,
            leads=tuple(args.leads), quality=tuple(args.quality),
            exclude_types=tuple(args.exclude_types), hold_s=args.hold_s,
            stride=max(1, args.stride), bin_cm=args.bin_cm, time_bin=args.time_bin,
            trials_sel=set(args.trials) if args.trials else None,
            jitter=not args.no_jitter, fr_offset_px=args.fr_offset_px,
            speed_thresh=args.speed_thresh)
        if not made:
            print("[videos] Nothing written."); sys.exit(1)
        print(f"[videos] wrote {len(made)} file(s).")
    except Exception as e:
        print(f"[videos] Failed: {e}")
        traceback.print_exc()
        sys.exit(1)
