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

from session_prefix import file_prefix

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from scipy.ndimage import gaussian_filter1d
from pynwb import NWBHDF5IO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import visualize_nwb as V     # load_position, load_nodes, read_trials_raw, align_trials, SCALE_*


def _select_units(nwb, qualities, cell_type=None):
    """[(unit_id, spike_times_seconds), ...] for units whose quality_label is in
    `qualities` (e.g. {'good'} or {'good','mua'}) and — when `cell_type` is given
    (e.g. 'pyramidal') — whose cell_type matches."""
    udf = V.read_units_table(nwb)   # skips the big per-spike waveform columns
    ql = (udf["quality_label"].astype(str).str.lower()
          if "quality_label" in udf.columns else None)
    ct = (udf["cell_type"].astype(str).str.lower()
          if "cell_type" in udf.columns else None)
    out = []
    for uid, r in udf.iterrows():
        if ql is not None and ql.loc[uid] not in qualities:
            continue
        if cell_type is not None and ct is not None and ct.loc[uid] != cell_type:
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


def _seg_dist(x, y, p1, p2):
    """Distance from each (x, y) point to the segment p1-p2 (all in metres)."""
    ax, ay = float(p1[0]), float(p1[1]); bx, by = float(p2[0]), float(p2[1])
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 0:
        return np.hypot(x - ax, y - ay)
    tt = np.clip(((x - ax) * dx + (y - ay) * dy) / L2, 0.0, 1.0)
    return np.hypot(x - (ax + tt * dx), y - (ay + tt * dy))


_BRIDGE_CACHE = {}


def _bridges(nodes, k_short=1.35):
    """All bridge/corridor segments of the hexmaze: the short honeycomb edges WITHIN
    each island (nodes within k_short x the median spacing) PLUS the long corridors
    BETWEEN islands (the shortest edge joining each still-separate component, added
    until the whole maze is one graph — an MST over components). Unlike
    V._maze_edges (honeycomb only) this includes the inter-island corridors, so a
    point mid-corridor is assigned to a real bridge. Cached; [(p1,p2), ...] metres."""
    key = (len(nodes), round(k_short, 3))
    if key in _BRIDGE_CACHE:
        return _BRIDGE_CACHE[key]
    P = np.array([nodes[i] for i in nodes], dtype=float)
    if len(P) < 2:
        _BRIDGE_CACHE[key] = []
        return []
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    np.fill_diagonal(D, np.inf)
    med = float(np.median(D.min(1)))
    n = len(P)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    thr = k_short * med
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if D[i, j] <= thr]
    for i, j in edges:
        parent[find(i)] = find(j)                       # union short (honeycomb) edges
    # connect remaining components with their shortest cross-component edge
    while len({find(i) for i in range(n)}) > 1:
        best, best_d = None, np.inf
        roots = np.array([find(i) for i in range(n)])
        for i in range(n):
            diff = roots != roots[i]
            if diff.any():
                j = int(np.where(diff, D[i], np.inf).argmin())
                if D[i, j] < best_d:
                    best_d, best = D[i, j], (i, j)
        if best is None:
            break
        parent[find(best[0])] = find(best[1]); edges.append(best)
    out = [(P[i], P[j]) for i, j in edges]
    _BRIDGE_CACHE[key] = out
    return out


def _hex_assign(x, y, nodes, node_frac=0.45):
    """Per-sample hexmaze location: the id of the node the animal is AT (within
    node_frac x the median node spacing), else -1; and the index of the nearest
    bridge/corridor it is ON otherwise, else -1. Nodes and bridges are disjoint
    labels so they can be coloured independently."""
    n = len(x)
    loc_node = np.full(n, -1, dtype=int)
    loc_bridge = np.full(n, -1, dtype=int)
    if not nodes or n == 0:
        return loc_node, loc_bridge
    ids = np.array(list(nodes.keys()))
    P = np.array([nodes[i] for i in ids], dtype=float)                       # (Nn, 2)
    dn = np.hypot(x[:, None] - P[None, :, 0], y[:, None] - P[None, :, 1])    # (n, Nn)
    nn_i = dn.argmin(1); nn_d = dn.min(1)
    dd = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    np.fill_diagonal(dd, np.inf)
    med = float(np.median(dd.min(1))) if len(P) > 1 else 1.0
    at_node = nn_d <= node_frac * med
    loc_node[at_node] = ids[nn_i[at_node]]
    bridges = _bridges(nodes)
    if bridges and (~at_node).any():
        seg_d = np.stack([_seg_dist(x, y, p1, p2) for p1, p2 in bridges], axis=1)  # (n, B)
        nb_i = seg_d.argmin(1)
        loc_bridge[~at_node] = nb_i[~at_node]
    return loc_node, loc_bridge


def _node_islands(nodes, k=1.5, min_size=5):
    """{node_id: island_index} + n_islands. Islands are the maze's separate hexagon
    clusters = connected components of the graph joining nodes within k x the median
    node spacing; tiny components merge into the nearest large one. (This maze has 4.)"""
    ids = list(nodes.keys())
    P = np.array([nodes[i] for i in ids], dtype=float)
    if len(P) < 2:
        return {i: 0 for i in ids}, 1
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    np.fill_diagonal(D, np.inf)
    med = float(np.median(D.min(1)))
    _n, lab = connected_components(csr_matrix(D <= k * med), directed=False)
    sizes = np.bincount(lab)
    large = [c for c in range(len(sizes)) if sizes[c] >= min_size] or list(range(len(sizes)))
    cent = {c: P[lab == c].mean(0) for c in large}
    order = sorted(large, key=lambda c: (round(cent[c][1], 1), cent[c][0]))  # top-left first
    isl_id = {c: i for i, c in enumerate(order)}
    node_island = {}
    for k2, nid in enumerate(ids):
        c = lab[k2]
        node_island[nid] = isl_id[c] if c in isl_id else \
            isl_id[min(large, key=lambda ci: np.hypot(*(P[k2] - cent[ci])))]
    return node_island, len(large)


def _assign_islands(x, y, nodes):
    """Island index per (x, y) sample = the island of its nearest node (-1 if none)."""
    if not nodes:
        return np.full(len(x), -1, dtype=int), 0
    node_island, n_isl = _node_islands(nodes)
    ids = list(nodes.keys())
    P = np.array([nodes[i] for i in ids], dtype=float)
    nn = np.hypot(x[:, None] - P[None, :, 0], y[:, None] - P[None, :, 1]).argmin(1)
    return np.array([node_island[ids[i]] for i in nn], dtype=int), n_isl


def _adjacent_nodes(nodes, goal_id, k=3, k_short=1.35):
    """The k node ids directly adjacent to `goal_id` (nearest neighbours within
    k_short x the median node spacing = one honeycomb step). [] if goal not present."""
    if goal_id not in nodes:
        return []
    ids = list(nodes.keys())
    P = np.array([nodes[i] for i in ids], dtype=float)
    gi = ids.index(goal_id)
    d = np.hypot(P[:, 0] - P[gi, 0], P[:, 1] - P[gi, 1]); d[gi] = np.inf
    D = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
    np.fill_diagonal(D, np.inf)
    med = float(np.median(D.min(1)))
    within = np.where(d <= k_short * med)[0]
    order = within[np.argsort(d[within])][:k]
    return [ids[i] for i in order]


def embed_session(nwb_path, qualities, dt=0.1, sigma_s=0.3, speed_thresh=0.05,
                  n_neighbors=50, min_dist=0.1, n_components=3, max_bins=25000,
                  seed=42, cell_type=None):
    import umap  # imported lazily so the rest of the runner works without it
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None or len(nwb.units.id) == 0:
            print("  no units — skipping."); return None
        units = _select_units(nwb, qualities, cell_type)
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

        # task variables from the trial table (session-relative seconds clock,
        # from the coordinate Trial_Num blocks — see visualize_nwb.build_trials)
        trials = V.build_trials(nwb_path.parent, nwb.session_start_time, t0, t1)
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

        # before / after the 2nd free-roaming (type 4/5) trial: split at its onset.
        specials = [(a, b) for (tp, _g, _s, a, b) in (trials or []) if tp in (4, 5)]
        phase2 = np.full(len(centers), np.nan)
        if len(specials) >= 2:
            phase2 = np.where(centers < specials[1][0], 0.0, 1.0)   # 0=before, 1=after

        # hexmaze location per bin: which node the animal is AT, or which bridge/
        # corridor it is ON (nodes vs bridges coloured independently downstream).
        loc_node, loc_bridge = _hex_assign(xb, yb, nodes)
        # which island (hexagon cluster) the bin's nearest node belongs to.
        loc_island, _n_isl = _assign_islands(xb, yb, nodes)

        # first & second GOAL-RUN (type-1) trials: 0=first, 1=second, NaN elsewhere.
        goal_runs = [(a, b) for (tp, _g, _s, a, b) in (trials or []) if tp == 1]
        goal_run = np.full(len(centers), np.nan)
        for gi, (a, b) in enumerate(goal_runs[:2]):
            goal_run[(centers >= a) & (centers <= b)] = float(gi)
        # goal node = most common goal among the goal-run trials (for goal+adjacent).
        gvals = [g for (tp, g, _s, _a, _b) in (trials or []) if tp == 1 and g is not None]
        goal_node = int(max(set(gvals), key=gvals.count)) if gvals else -1

        # the (up to 3) free-roaming (type 4/5) periods: 0,1,2; NaN elsewhere.
        fr_trials = [(a, b) for (tp, _g, _s, a, b) in (trials or []) if tp in (4, 5)]
        free_roam = np.full(len(centers), np.nan)
        for fi, (a, b) in enumerate(fr_trials[:3]):
            free_roam[(centers >= a) & (centers <= b)] = float(fi)

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
               "phase2": phase2[idx], "loc_node": loc_node[idx], "loc_bridge": loc_bridge[idx],
               "island": loc_island[idx], "goal_run": goal_run[idx], "goal_node": goal_node,
               "free_roam": free_roam[idx],
               "quality": "+".join(sorted(qualities)) + (" pyramidal" if cell_type == "pyramidal"
                                                         else (f" {cell_type}" if cell_type else "")),
               "n_units": len(units)}

        out_dir = nwb_path.parent / "umap"
        out_dir.mkdir(exist_ok=True)
        tag = "_".join(sorted(qualities)) + ("_pyr" if cell_type == "pyramidal"
                                             else (f"_{cell_type[:3]}" if cell_type else ""))
        pfx = file_prefix(nwb_path.parent)               # rat_sessiondate_ prefix
        np.savez(out_dir / f"{pfx}umap_{tag}.npz", **res)
        _plot(out_dir / f"{pfx}umap_{tag}.pdf", res, nwb_path.name)
        return res


def _hex_colors(loc_node, loc_bridge, nodes):
    """(rgba per point, node_color dict, bridge_color dict). Nodes are coloured by a
    rainbow (hsv) per node; bridges by an INDEPENDENT greyscale ramp per bridge, so
    'at a node' reads as a distinct colour and 'on a bridge' as a grey."""
    ids = list(nodes.keys())
    nn = max(len(ids), 1)
    ncm = plt.get_cmap("hsv")
    node_color = {nid: ncm(k / nn) for k, nid in enumerate(ids)}
    bridges = _bridges(nodes)
    nb = max(len(bridges), 1)
    gcm = plt.get_cmap("gray")
    bridge_color = {bi: gcm(0.25 + 0.4 * (bi / max(nb - 1, 1))) for bi in range(len(bridges))}
    rgba = np.tile(np.array([0.85, 0.85, 0.85, 0.35]), (len(loc_node), 1))  # unassigned
    nm = loc_node >= 0
    if nm.any():
        rgba[nm] = np.array([node_color[int(v)] for v in loc_node[nm]])
    bm = loc_bridge >= 0
    if bm.any():
        rgba[bm] = np.array([bridge_color[int(v)] for v in loc_bridge[bm]])
    return rgba, node_color, bridge_color


def _draw_maze_key(ax, nodes, node_color, bridge_color):
    """The hexmaze drawn with the SAME node/bridge colours the UMAP points use, so a
    coloured UMAP point can be matched to a maze location."""
    for bi, (p1, p2) in enumerate(_bridges(nodes)):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=bridge_color[bi], lw=2.5, zorder=1)
    for nid, (nx, ny) in nodes.items():
        ax.scatter(nx, ny, color=node_color[nid], s=70, edgecolors="k", linewidths=0.4, zorder=2)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(V.MAZE_EXTENT[0], V.MAZE_EXTENT[1]); ax.set_ylim(V.MAZE_EXTENT[3], V.MAZE_EXTENT[2])
    ax.set_title("maze key — node colours (rainbow) vs bridge colours (greys)")


def _panel_scatter(ax, xs, ys, panel, zs=None):
    """Scatter one panel onto a 2D or 3D axis. Continuous panels use a cmap (the
    caller adds the colourbar); categorical / rgba panels pass per-point colours."""
    kw = dict(s=4, alpha=0.7, linewidths=0, rasterized=True)
    if panel["kind"] == "cont":
        args = (xs, ys) if zs is None else (xs, ys, zs)
        return ax.scatter(*args, c=panel["c"], cmap=panel["cmap"],
                          vmin=panel.get("vmin"), vmax=panel.get("vmax"), **kw)
    args = (xs, ys) if zs is None else (xs, ys, zs)
    ax.scatter(*args, c=panel["rgba"], **kw)
    return None


def _plot(pdf_path, res, nwb_name):
    from matplotlib.backends.backend_pdf import PdfPages
    emb = res["emb"]

    # ---- continuous colour variables (cmap + colourbar) ----
    panels = [
        dict(kind="cont", c=res["x"], cmap="viridis", label="X position (m)"),
        dict(kind="cont", c=res["y"], cmap="plasma", label="Y position (m)"),
        # speed CAPPED at 0.6 m/s (everything faster coloured as 0.6), jet colormap.
        dict(kind="cont", c=np.minimum(res["speed"], 0.6), cmap="jet",
             vmin=0.0, vmax=0.6, label="Speed (m/s, capped at 0.6)"),
        dict(kind="cont", c=res["time"], cmap="cool", label="Session time (s)"),
    ]
    if np.isfinite(res.get("trial", np.array([np.nan]))).any():
        panels.append(dict(kind="cont", c=res["trial"], cmap="gist_rainbow", label="Trial number"))
    if np.isfinite(res["goal_dist"]).any():
        panels.append(dict(kind="cont", c=res["goal_dist"], cmap="magma", label="Distance to goal (m)"))
    if np.isfinite(res["ttype"]).any():
        panels.append(dict(kind="cont", c=res["ttype"], cmap="tab10", label="Trial type"))

    # ---- before / after the 2nd free-roaming trial (categorical) ----
    ph = res.get("phase2", np.array([np.nan]))
    if np.isfinite(ph).any():
        cols = ["#1f77b4", "#d62728"]                       # before=blue, after=red
        rgba = np.tile(np.array([0.85, 0.85, 0.85, 0.3]), (len(ph), 1))
        rgba[ph == 0] = mcolors.to_rgba(cols[0])
        rgba[ph == 1] = mcolors.to_rgba(cols[1])
        legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cols[0],
                         markersize=8, label="before 2nd free-roaming"),
                  Line2D([0], [0], marker="o", color="w", markerfacecolor=cols[1],
                         markersize=8, label="after 2nd free-roaming")]
        panels.append(dict(kind="cat", rgba=rgba, legend=legend,
                           label="2nd free-roaming (before/after)"))

    # ---- hexmaze location: nodes vs bridges (independent colours) ----
    nodes = V.load_nodes()
    if nodes and (np.any(res.get("loc_node", np.array([-1])) >= 0)
                  or np.any(res.get("loc_bridge", np.array([-1])) >= 0)):
        rgba, node_color, bridge_color = _hex_colors(res["loc_node"], res["loc_bridge"], nodes)
        panels.append(dict(kind="rgba", rgba=rgba, label="Hexmaze location (node vs bridge)",
                           node_color=node_color, bridge_color=bridge_color))

    # ---- island (which hexagon cluster) ----
    isl = res.get("island", np.array([-1]))
    if nodes and np.any(isl >= 0):
        node_island, n_isl = _node_islands(nodes)
        icm = plt.get_cmap("tab10")
        icol = [icm(i) for i in range(max(n_isl, 1))]
        rgba_i = np.tile(np.array([0.85, 0.85, 0.85, 0.3]), (len(isl), 1))
        for i in range(n_isl):
            rgba_i[isl == i] = mcolors.to_rgba(icol[i])
        legend_i = [Line2D([0], [0], marker="o", color="w", markerfacecolor=icol[i],
                           markersize=8, label=f"island {i + 1}") for i in range(n_isl)]
        node_color_i = {nid: icol[node_island[nid]] for nid in nodes}
        bridge_color_i = {bi: (0.6, 0.6, 0.6, 1.0)
                          for bi in range(len(_bridges(nodes)))}
        panels.append(dict(kind="rgba", rgba=rgba_i, legend=legend_i, label="Hexmaze island",
                           node_color=node_color_i, bridge_color=bridge_color_i))

    # ---- first & second goal run (rest grey) ----
    gr = res.get("goal_run", np.array([np.nan]))
    if np.isfinite(gr).any():
        cols = ["#1f77b4", "#d62728"]                       # 1st=blue, 2nd=red
        rgba = np.tile(np.array([0.85, 0.85, 0.85, 0.25]), (len(gr), 1))
        rgba[gr == 0] = mcolors.to_rgba(cols[0])
        rgba[gr == 1] = mcolors.to_rgba(cols[1])
        legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cols[0],
                         markersize=8, label="1st goal run"),
                  Line2D([0], [0], marker="o", color="w", markerfacecolor=cols[1],
                         markersize=8, label="2nd goal run")]
        panels.append(dict(kind="rgba", rgba=rgba, legend=legend,
                           label="1st & 2nd goal run (rest grey)"))

    # ---- the 3 free-roaming periods (rest grey) ----
    frm = res.get("free_roam", np.array([np.nan]))
    if np.isfinite(frm).any():
        fcols = ["#1b9e77", "#d95f02", "#7570b3"]           # FR1 / FR2 / FR3
        rgba = np.tile(np.array([0.85, 0.85, 0.85, 0.25]), (len(frm), 1))
        legend = []
        for i in range(3):
            m = frm == i
            if m.any():
                rgba[m] = mcolors.to_rgba(fcols[i])
                legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=fcols[i],
                                     markersize=8, label=f"free-roaming {i + 1}"))
        panels.append(dict(kind="rgba", rgba=rgba, legend=legend,
                           label="3 free-roaming periods (rest grey)"))

    # ---- goal node + its 3 adjacent nodes (rest grey) ----
    gn = int(res.get("goal_node", -1))
    if nodes and gn in nodes:
        adj = _adjacent_nodes(nodes, gn, 3)
        hi = {gn: "#FFD700"}                                 # goal = gold
        for nid, col in zip(adj, ["#e41a1c", "#377eb8", "#4daf4a"]):
            hi[nid] = col
        ln = res.get("loc_node", np.full(len(gr), -1))
        rgba = np.tile(np.array([0.85, 0.85, 0.85, 0.25]), (len(ln), 1))
        legend = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFD700",
                         markersize=9, label=f"goal {gn}")]
        for nid in [gn] + adj:
            rgba[ln == nid] = mcolors.to_rgba(hi[nid])
        for nid in adj:
            legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=hi[nid],
                                 markersize=8, label=f"adjacent {nid}"))
        node_color_g = {nid: (0.85, 0.85, 0.85, 0.6) for nid in nodes}
        for nid, col in hi.items():
            node_color_g[nid] = mcolors.to_rgba(col)
        bridge_color_g = {bi: (0.85, 0.85, 0.85, 0.5) for bi in range(len(_bridges(nodes)))}
        panels.append(dict(kind="rgba", rgba=rgba, legend=legend,
                           label="Goal + 3 adjacent nodes (rest grey)",
                           node_color=node_color_g, bridge_color=bridge_color_g))

    proj = [(0, 1, "UMAP 1", "UMAP 2"), (0, 2, "UMAP 1", "UMAP 3"),
            (1, 2, "UMAP 2", "UMAP 3")]

    with PdfPages(pdf_path) as pdf:
        # one 3D page per colour variable
        for panel in panels:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            p = _panel_scatter(ax, emb[:, 0], emb[:, 1], panel, zs=emb[:, 2])
            ax.set_title(f"Neural population UMAP — {res['quality']} "
                         f"(n={res['n_units']} units)\ncoloured by {panel['label']}", fontsize=11)
            ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2"); ax.set_zlabel("UMAP 3")
            if p is not None:
                fig.colorbar(p, ax=ax, fraction=0.03, pad=0.08).set_label(panel["label"])
            if panel.get("legend"):
                ax.legend(handles=panel["legend"], fontsize=8, loc="upper right")
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 2D-projection page per colour variable (+ a maze-key column for the hex/island
        # panels so a coloured UMAP point can be matched to a maze location).
        for panel in panels:
            has_key = panel.get("node_color") is not None
            ncol = 4 if has_key else 3
            fig, axes = plt.subplots(1, ncol, figsize=(6 * ncol, 5.5))
            for ax, (i, j, xl, yl) in zip(axes, proj):
                p = _panel_scatter(ax, emb[:, i], emb[:, j], panel)
                ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_aspect("equal", "box")
            if p is not None:
                fig.colorbar(p, ax=axes[:3], fraction=0.02, pad=0.02).set_label(panel["label"])
            elif panel.get("legend"):
                axes[0].legend(handles=panel["legend"], fontsize=8, loc="upper right")
            if has_key:
                _draw_maze_key(axes[3], nodes, panel["node_color"], panel["bridge_color"])
            fig.suptitle(f"UMAP 2D projections — {res['quality']} — coloured by {panel['label']}",
                         fontsize=13)
            pdf.savefig(fig); plt.close(fig)
    print(f"  wrote {pdf_path}")


def run(output_folder, qualities, **kw):
    nwb_path = V.find_nwb_file(output_folder)
    if nwb_path is None:
        print(f"No NWB in {output_folder}."); return
    nwb_path = Path(nwb_path)
    ct = kw.get("cell_type")
    print(f"UMAP embedding {nwb_path} using units: {'+'.join(sorted(qualities))}"
          + (f" ({ct} only)" if ct else ""))
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
    ap.add_argument("--cell_type", default=None, choices=["pyramidal", "interneuron"],
                    help="restrict to this putative cell type (default: all cell types).")
    a = ap.parse_args()
    run(a.output_folder, set(q.lower() for q in a.quality),
        dt=a.dt, sigma_s=a.sigma_s, n_neighbors=a.n_neighbors, min_dist=a.min_dist,
        cell_type=a.cell_type)
