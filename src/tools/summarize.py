"""A one-look summary of the whole dataset: for every rat, grouped by repeat, each
session and whether its four expected files (pre / task / post / video) are there,
the right size, and where they live.

The figure is a status matrix — one small grid per rat, a row per session, a column
per file. Colour carries the state (present / short / missing / not-expected) and
every cell also carries text (the size in GB, ✓, ✗, —), so the status is never
conveyed by colour alone. Sessions are banded by repeat with a per-repeat count, so
"how many sessions in each repeat" is legible at a glance.

Build the data with ``build_summary`` (pure, testable) and draw it with
``render_summary`` (matplotlib). Both take the same inputs the scatter/organize
flow already produces: the roster, the search_all_drives index, and the
assign_videos result.
"""

from __future__ import annotations

from pathlib import Path

# Reserved status palette (validated; see dataviz skill). Never themed, never
# reused for categorical series. Each pairs with text in the cell.
GOOD = "#0ca30c"        # present and correctly sized
WARN = "#fab219"        # present but short, or video only in the dump (needs filing)
CRIT = "#d03b3b"        # missing
NA = "#d9d9d6"          # not expected for this animal (video-only rat's ephys)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
BAND = "#efeee9"        # repeat-band background

FILES = ("pre", "task", "post", "video")

# Short-phase rule shared with the coverage table.
SHORT_FRACTION = 0.6
_MIN_SESSIONS_FOR_MEDIAN = 3


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2


def _agg_phase_gb(hits):
    pb = {p: 0 for p in ("pre", "task", "post")}
    for h in hits:
        for ph, b in (h.get("phase_bytes") or {}).items():
            if ph in pb:
                pb[ph] = max(pb[ph], b)
    return {p: round(pb[p] / 1e9, 1) for p in pb}


def build_summary(roster, found, videos=None) -> dict:
    """Return {rat_no: {"implanted": bool, "repeats": {repeat: [session,...]}, ...}}.

    Each session dict carries, per file, a (state, text, where) triple where state
    is one of good/warn/crit/na. Covers every roster session plus any on-disk
    orphan session for a roster rat (grouped under repeat ``None``)."""
    videos = videos or []
    dump_video = {(v["rat_no"], v["date8"]): v for v in videos if v.get("rat_no")}
    implanted = {e["rat_no"]: e["implanted"] for e in roster}
    roster_rats = {e["rat_no"] for e in roster}

    # per-rat phase medians over every found session, for the short check
    by_rat_phase: dict = {}
    for (rn, _d), hits in found.items():
        if not implanted.get(rn):
            continue
        gb = _agg_phase_gb(hits)
        for p, v in gb.items():
            if v > 0:
                by_rat_phase.setdefault(rn, {}).setdefault(p, []).append(v)
    medians = {rn: {p: _median(v) for p, v in phs.items() if len(v) >= _MIN_SESSIONS_FOR_MEDIAN}
               for rn, phs in by_rat_phase.items()}

    date_window = sorted(e["date8"] for e in roster if e["date8"])
    lo = hi = ""
    if date_window:
        lo = str(int(date_window[0][:4]) - 1) + date_window[0][4:]
        hi = str(int(date_window[-1][:4]) + 1) + date_window[-1][4:]

    def _one(rat_no, date8, day, session, repeat, hits):
        imp = implanted.get(rat_no, False)
        gb = _agg_phase_gb(hits)
        phases = set().union(*(h.get("phases", set()) for h in hits)) if hits else set()
        n_video = sum(h.get("n_video", 0) for h in hits)
        vols = sorted({h["volume"] for h in hits})
        paths = [h["path"] for h in hits]
        med = medians.get(rat_no, {})
        cells = {}
        for p in ("pre", "task", "post"):
            if not imp:
                cells[p] = ("na", "—", "")
            elif p not in phases:
                cells[p] = ("crit", "✗", "")
            else:
                short = med.get(p, 0) > 0 and gb[p] < SHORT_FRACTION * med[p]
                cells[p] = ("warn" if short else "good", f"{gb[p]:.0f}", "")
        # video
        dv = dump_video.get((rat_no, date8))
        if n_video > 0:
            cells["video"] = ("good", "✓", "; ".join(vols))
        elif dv:
            cells["video"] = ("warn", "dump", dv["path"])
        else:
            cells["video"] = ("crit", "✗", "")
        # overall
        states = [c[0] for c in cells.values() if c[0] != "na"]
        overall = ("crit" if "crit" in states else "warn" if "warn" in states else "good")
        # where the data actually is — the session folders (drive + tree), or, if
        # only the video was found in the dump, that.
        where = "; ".join(paths) if paths else (dv["path"] if dv else "—")
        return dict(rat_no=rat_no, date8=date8, day=day, session=session, repeat=repeat,
                    implanted=imp, cells=cells, overall=overall, where=where, paths=paths,
                    video_path=dv["path"] if dv else "")

    out: dict = {}
    seen = set()
    for e in sorted(roster, key=lambda x: (x["rat_no"], x.get("repeat") or 0, x["day"])):
        if not e["date8"]:
            continue
        hits = found.get((e["rat_no"], e["date8"]), [])
        s = _one(e["rat_no"], e["date8"], e["day"], e["session"], e.get("repeat"), hits)
        rat = out.setdefault(e["rat_no"], dict(implanted=e["implanted"], repeats={}, orphans=[]))
        rat["repeats"].setdefault(e.get("repeat"), []).append(s)
        seen.add((e["rat_no"], e["date8"]))

    # orphan sessions (implanted rat, in window, not in roster) — no repeat
    for (rat_no, date8), hits in sorted(found.items()):
        if (rat_no, date8) in seen or rat_no not in roster_rats:
            continue
        if lo and not (lo <= date8 <= hi):
            continue
        s = _one(rat_no, date8, "", "", None, hits)
        out.setdefault(rat_no, dict(implanted=implanted.get(rat_no, False),
                                    repeats={}, orphans=[]))["orphans"].append(s)
    return out


_COLOR = {"good": GOOD, "warn": WARN, "crit": CRIT, "na": NA}


def _short_where(where: str) -> str:
    """Compact the location for the figure: the tree each copy sits under (drive +
    first folder), deduped — e.g. 'F:\\HM_neurons; E:\\GL38'. Keeps it readable
    without the long Rat<N>/<date> tail that is the same for every row."""
    if not where or where == "—":
        return where or "—"
    seen = []
    for part in where.split(";"):
        p = part.strip().rstrip("\\/")
        if not p:
            continue
        segs = Path(p).parts
        tag = "\\".join(segs[:2]) if len(segs) >= 2 else segs[0] if segs else p
        if tag not in seen:
            seen.append(tag)
    out = "; ".join(seen)
    return out if len(out) <= 40 else out[:39] + "…"


def render_summary(summary: dict, out_path: str, title: str = "HexMaze — dataset summary"):
    """Draw the status matrix to `out_path` (PNG). One panel per rat, a row per
    session banded by repeat, four file columns. Returns out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch

    rats = sorted(summary)
    if not rats:
        fig, ax = plt.subplots(figsize=(7, 2))
        fig.patch.set_facecolor(SURFACE)
        ax.axis("off")
        ax.text(0.5, 0.5, "No sessions to summarise.", ha="center", va="center",
                fontsize=13, color=INK2, transform=ax.transAxes)
        fig.savefig(out_path, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        return out_path

    # flatten each rat into ordered (repeat, [sessions]) groups incl. orphans
    def groups(rat):
        gs = []
        for rep in sorted(summary[rat]["repeats"], key=lambda r: (r is None, r)):
            gs.append((rep, summary[rat]["repeats"][rep]))
        if summary[rat]["orphans"]:
            gs.append((None, summary[rat]["orphans"]))
        return gs

    # rows per rat drive the panel heights
    rat_rows = {r: sum(len(s) for _, s in groups(r)) for r in rats}
    ncol = 2 if len(rats) > 1 else 1
    nrow = (len(rats) + ncol - 1) // ncol
    row_h = 0.34
    max_rows = max(rat_rows.values()) if rat_rows else 1
    panel_h = max(3.0, max_rows * row_h + 1.2)
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.8 * ncol, panel_h * nrow),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)

    NX = len(FILES)
    LOC_X = NX + 0.35          # left edge of the "where" column
    RIGHT = NX + 4.2           # panel right edge
    for idx, rat in enumerate(rats):
        ax = axes[idx // ncol][idx % ncol]
        ax.set_facecolor(SURFACE)
        gs = groups(rat)
        n = rat_rows[rat] + len([g for g in gs if g[1]])  # rows + a header gap per group
        ax.set_xlim(-3.4, RIGHT)
        ax.set_ylim(-(n + 0.5), 0.8)
        ax.axis("off")

        imp = summary[rat]["implanted"]
        nsess = rat_rows[rat]
        nrep = len([r for r, s in gs if r is not None and s])
        ax.text(-3.3, 0.4, f"Rat{rat}", fontsize=14, fontweight="bold", color=INK, va="center")
        ax.text(RIGHT, 0.4,
                f"{'implanted' if imp else 'video-only'} · {nsess} sessions · {nrep} repeats",
                fontsize=8.5, color=INK2, va="center", ha="right")
        # column headers
        for j, f in enumerate(FILES):
            ax.text(j + 0.5, -0.15, f, fontsize=9, fontweight="bold", color=INK2,
                    ha="center", va="center")
        ax.text(LOC_X, -0.15, "where", fontsize=9, fontweight="bold", color=INK2,
                ha="left", va="center")

        y = -0.9
        for rep, sess in gs:
            if not sess:
                continue
            top = y
            # repeat band background
            ax.add_patch(Rectangle((-3.4, y - len(sess) + 0.1), RIGHT + 3.4, len(sess) - 0.0,
                                   facecolor=BAND, edgecolor="none", zorder=0))
            for s in sess:
                lbl = (f"d{s['day']} s{s['session']}" if s["repeat"] is not None
                       else "unlogged")
                ax.text(-1.65, y - 0.5, f"{lbl}", fontsize=8, color=INK, va="center", ha="left")
                ax.text(-1.65, y - 0.86, s["date8"], fontsize=6.5, color=INK2, va="center", ha="left")
                for j, f in enumerate(FILES):
                    st, txt, _ = s["cells"][f]
                    ax.add_patch(FancyBboxPatch((j + 0.08, y - 0.92), 0.84, 0.84,
                                 boxstyle="round,pad=0,rounding_size=0.08",
                                 facecolor=_COLOR[st], edgecolor=SURFACE, linewidth=1.5, zorder=2))
                    tc = "#ffffff" if st in ("crit", "good") else INK
                    ax.text(j + 0.5, y - 0.5, txt, fontsize=8.5, color=tc,
                            ha="center", va="center", zorder=3, fontweight="bold")
                # where — its own column, left-aligned, with room
                ax.text(LOC_X, y - 0.5, _short_where(s["where"]), fontsize=6.6, color=INK2,
                        va="center", ha="left")
                y -= 1.0
            # repeat label at the band's left
            rep_txt = f"R{rep}" if rep is not None else "?"
            ax.text(-3.3, (top + y) / 2 + 0.0, rep_txt, fontsize=10, fontweight="bold",
                    color=INK2, va="center", ha="left", rotation=0)
            y -= 0.4

    # hide unused panels
    for k in range(len(rats), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    # legend
    import matplotlib.patches as mp
    handles = [mp.Patch(facecolor=GOOD, label="present, correct size"),
               mp.Patch(facecolor=WARN, label="short, or video in dump"),
               mp.Patch(facecolor=CRIT, label="missing"),
               mp.Patch(facecolor=NA, label="not expected (video-only rat)")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, fontsize=16, fontweight="bold", color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return out_path
