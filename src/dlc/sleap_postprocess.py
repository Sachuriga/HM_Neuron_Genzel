"""Post-process SLEAP predictions: export keypoint coordinates and render labeled videos."""
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import sleap_io as sio

# ---------- Config ----------
PREDICTIONS_DIR = Path(
    r"\\genzelneuron.z.science.ru.nl\genzelneuron\code\HM_Sleap\src\predictions"
)
# Map each .slp prediction file to its source video.
# Keys are .slp file stems (without extension), values are full video paths.
VIDEO_MAP: dict[str, Path] = {
    "collected_frames_Rat1_20200914.predictions": Path(
        r"\\genzelneuron.z.science.ru.nl\genzelneuron\data\HM_sleap\collected_frames_Rat1_20200914.mp4"
    ),
    # add more entries here matching prediction stem -> video path
}
OUTPUT_DIR = Path(
    r"\\genzelneuron.z.science.ru.nl\genzelneuron\data\HM_sleap\TEMP"
)
RENDER_VIDEOS = True   # set False to skip video rendering (fast CSV-only run)
KEYPOINT_RADIUS = 5    # pixel radius for drawn keypoint circles
LINE_THICKNESS = 2     # skeleton edge thickness
# ----------------------------

# Distinct BGR colours, one per keypoint index (cycles if more keypoints than colours)
_PALETTE = [
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (255, 0, 0),    # blue
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 165, 0),  # orange
    (128, 0, 128),  # purple
    (0, 128, 255),  # sky blue
]


def _color(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Coordinate export
# ---------------------------------------------------------------------------

def export_coordinates(slp_path: Path) -> Path:
    """Load a .slp prediction file and write a CSV of keypoint coordinates.

    Columns: frame_idx, instance_idx, node, x, y, score, visible
    """
    labels = sio.load_slp(str(slp_path))
    rows = []
    for lf in labels.labeled_frames:
        for inst_idx, instance in enumerate(lf.instances):
            score = getattr(instance, "score", float("nan"))
            for node, point in zip(instance.skeleton.nodes, instance.points):
                rows.append(
                    {
                        "frame_idx": lf.frame_idx,
                        "instance_idx": inst_idx,
                        "node": node.name,
                        "x": point.x,
                        "y": point.y,
                        "score": score,
                        "visible": point.visible,
                    }
                )

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / f"{slp_path.stem}.coordinates.csv"
    df.to_csv(out_csv, index=False)
    print(f"   Coordinates saved → {out_csv.name}  ({len(df)} rows)")
    return out_csv


# ---------------------------------------------------------------------------
# Labeled video rendering
# ---------------------------------------------------------------------------

def render_labeled_video(slp_path: Path, video_path: Path) -> Path:
    """Overlay keypoints and skeleton edges from a .slp file onto the source video."""
    labels = sio.load_slp(str(slp_path))
    skeleton = labels.skeletons[0] if labels.skeletons else None

    # Build a fast lookup: frame_idx -> list[Instance]
    frame_lookup: dict[int, list] = {}
    for lf in labels.labeled_frames:
        frame_lookup[lf.frame_idx] = lf.instances

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video = OUTPUT_DIR / f"{slp_path.stem}.labeled.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

    node_names = [n.name for n in skeleton.nodes] if skeleton else []

    # Build edge index pairs from skeleton
    edge_pairs: list[tuple[int, int]] = []
    if skeleton:
        for edge in skeleton.edges:
            try:
                src_idx = node_names.index(edge.source.name)
                dst_idx = node_names.index(edge.destination.name)
                edge_pairs.append((src_idx, dst_idx))
            except ValueError:
                pass

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        instances = frame_lookup.get(frame_idx, [])
        for inst_idx, instance in enumerate(instances):
            points = instance.points
            coords = [(p.x, p.y, p.visible) for p in points]

            # Draw skeleton edges first (underneath the keypoints)
            for src_i, dst_i in edge_pairs:
                if src_i >= len(coords) or dst_i >= len(coords):
                    continue
                sx, sy, sv = coords[src_i]
                dx, dy, dv = coords[dst_i]
                if sv and dv and not (np.isnan(sx) or np.isnan(sy) or np.isnan(dx) or np.isnan(dy)):
                    cv2.line(
                        frame,
                        (int(sx), int(sy)),
                        (int(dx), int(dy)),
                        (200, 200, 200),
                        LINE_THICKNESS,
                    )

            # Draw keypoints
            for kp_idx, (x, y, visible) in enumerate(coords):
                if visible and not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(frame, (int(x), int(y)), KEYPOINT_RADIUS, _color(kp_idx), -1)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"   rendered {frame_idx}/{total_frames} frames …")

    cap.release()
    writer.release()
    print(f"   Labeled video saved → {out_video.name}")
    return out_video


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    slp_files = sorted(PREDICTIONS_DIR.glob("*.slp"))
    if not slp_files:
        print(f"No .slp files found in {PREDICTIONS_DIR}")
        return

    for slp_path in slp_files:
        print(f"\n=== {slp_path.name} ===")
        export_coordinates(slp_path)

        if RENDER_VIDEOS:
            video_path = VIDEO_MAP.get(slp_path.stem)
            if video_path is None:
                print(f"   No video mapped for {slp_path.stem!r} — skipping render.")
                print(f"   Add an entry to VIDEO_MAP at the top of this script.")
            elif not video_path.exists():
                print(f"   Video not found: {video_path} — skipping render.")
            else:
                render_labeled_video(slp_path, video_path)

    print(f"\nDone. Outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
