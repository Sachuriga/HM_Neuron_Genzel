"""SLEAP inference pipeline: predict keypoints, export coordinates, render labeled videos."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import cv2
import sleap_io as sio
from sleap_nn.predict import run_inference

# ============================================================
# Config
# ============================================================
MODEL_PATH = Path(
    r"\\genzelneuron.z.science.ru.nl\genzelneuron\code\HM_Sleap\src\models\sachi_test-2"
)
# Source videos to process
VIDEOS = [
    Path(r"\\genzelneuron.z.science.ru.nl\genzelneuron\data\HM_sleap\collected_frames_Rat1_20200914.mp4"),
    # add more videos here
]
# Where .slp prediction files are saved
PREDICTIONS_DIR = Path(
    r"\\genzelneuron.z.science.ru.nl\genzelneuron\code\HM_Sleap\src\predictions"
)
# Where CSVs and labeled videos are saved
OUTPUTS_DIR = Path(
    r"\\genzelneuron.z.science.ru.nl\genzelneuron\data\HM_sleap\TEMP"
)

# --- Inference mode ---
MODE = "sample"   # "full" = entire video | "sample" = random N frames
SAMPLE_N = 3000   # number of frames to randomly sample (only used when MODE="sample")

# --- Hardware ---
DEVICE = "cuda"   # "cuda" / "cuda:0" / "cpu"
BATCH_SIZE = 8    # increase to 16 or 32 if VRAM allows

# --- Detection sensitivity ---
# Default in SLEAP is typically 0.2. Lower this if you get 0 predictions.
# Try 0.05 first; go as low as 0.01 to confirm the model detects anything at all.
PEAK_THRESHOLD = 0.05

# --- Post-processing ---
EXPORT_CSV = True        # write a coordinate CSV for each prediction
RENDER_VIDEOS = True     # overlay keypoints on the source video
KEYPOINT_RADIUS = 5      # pixel radius for drawn keypoint circles
LINE_THICKNESS = 2       # skeleton edge thickness
# ============================================================

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


def _pt(point) -> tuple[float, float, bool]:
    """Return (x, y, visible) from either a sleap_io Point or a numpy.void row."""
    if hasattr(point, "x"):
        return float(point.x), float(point.y), bool(point.visible)
    return float(point["x"]), float(point["y"]), bool(point["visible"])


def _video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ============================================================
# Step 1 — Inference
# ============================================================

def predict_one(video_path: Path) -> tuple[Path, sio.Labels]:
    """Run SLEAP inference on one video.

    Returns (slp_path, labels) where labels is the in-memory Labels object
    from run_inference — used directly by later steps to avoid a reload that
    returns 0 frames due to a sleap_nn/sleap_io format mismatch.
    """
    total_frames = _video_frame_count(video_path)

    if MODE == "sample":
        n = min(SAMPLE_N, total_frames)
        frames = sorted(random.sample(range(total_frames), n))
        suffix = f".sample{n}.predictions"
        print(f"   Mode: SAMPLE — {n}/{total_frames} random frames")
    else:
        frames = None
        suffix = ".predictions"
        print(f"   Mode: FULL — {total_frames} frames")

    out_path = PREDICTIONS_DIR / f"{video_path.stem}{suffix}.slp"
    print(f"\n=== {video_path.name} -> {out_path.name} ===")

    labels = run_inference(
        data_path=str(video_path),
        model_paths=[str(MODEL_PATH)],
        output_path=str(out_path),
        make_labels=True,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        frames=frames,
        peak_threshold=PEAK_THRESHOLD,
    )

    # run_inference may return None in some versions — fall back to disk load
    if labels is None or not hasattr(labels, "labeled_frames"):
        print("   run_inference returned no Labels object — loading from disk")
        labels = sio.load_slp(str(out_path))

    n_predicted = len(labels.labeled_frames)
    expected = len(frames) if frames is not None else total_frames
    pct = 100.0 * n_predicted / expected if expected else 0
    print(f"   Predicted {n_predicted}/{expected} frames ({pct:.1f}%)")
    if n_predicted < 0.5 * expected:
        print("   WARNING: fewer than half the frames have predictions")

    return out_path, labels


# ============================================================
# Step 2 — Coordinate export
# ============================================================

def export_coordinates(labels: sio.Labels, out_stem: str) -> Path:
    """Write a CSV of keypoint coordinates from an in-memory Labels object.

    Columns: frame_idx, instance_idx, node, x, y, score, visible
    """
    rows = []
    for lf in labels.labeled_frames:
        for inst_idx, instance in enumerate(lf.instances):
            score = getattr(instance, "score", float("nan"))
            for node, point in zip(instance.skeleton.nodes, instance.points):
                x, y, visible = _pt(point)
                rows.append(
                    {
                        "frame_idx": lf.frame_idx,
                        "instance_idx": inst_idx,
                        "node": node.name,
                        "x": x,
                        "y": y,
                        "score": score,
                        "visible": visible,
                    }
                )

    df = pd.DataFrame(rows)
    out_csv = OUTPUTS_DIR / f"{out_stem}.coordinates.csv"
    df.to_csv(out_csv, index=False)
    print(f"   Coordinates → {out_csv.name}  ({len(df)} rows)")
    return out_csv


# ============================================================
# Step 3 — Labeled video rendering
# ============================================================

def render_labeled_video(labels: sio.Labels, video_path: Path, out_stem: str) -> Path:
    """Overlay keypoints and skeleton edges onto only the predicted frames."""
    skeleton = labels.skeletons[0] if labels.skeletons else None

    frame_lookup: dict[int, list] = {lf.frame_idx: lf.instances for lf in labels.labeled_frames}
    predicted_frame_indices = sorted(frame_lookup)

    if not predicted_frame_indices:
        print("   No predicted frames to render — skipping.")
        return video_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = OUTPUTS_DIR / f"{out_stem}.labeled.mp4"
    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    node_names = [n.name for n in skeleton.nodes] if skeleton else []
    edge_pairs: list[tuple[int, int]] = []
    if skeleton:
        for edge in skeleton.edges:
            try:
                edge_pairs.append(
                    (node_names.index(edge.source.name), node_names.index(edge.destination.name))
                )
            except ValueError:
                pass

    n_total = len(predicted_frame_indices)
    for i, frame_idx in enumerate(predicted_frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"   WARNING: could not read frame {frame_idx}")
            continue

        for instance in frame_lookup[frame_idx]:
            coords = [_pt(p) for p in instance.points]

            for src_i, dst_i in edge_pairs:
                if src_i >= len(coords) or dst_i >= len(coords):
                    continue
                sx, sy, sv = coords[src_i]
                dx, dy, dv = coords[dst_i]
                if sv and dv and not any(np.isnan(v) for v in (sx, sy, dx, dy)):
                    cv2.line(frame, (int(sx), int(sy)), (int(dx), int(dy)), (200, 200, 200), LINE_THICKNESS)

            for kp_idx, (x, y, visible) in enumerate(coords):
                if visible and not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(frame, (int(x), int(y)), KEYPOINT_RADIUS, _color(kp_idx), -1)

        writer.write(frame)
        if (i + 1) % 500 == 0:
            print(f"   rendered {i + 1}/{n_total} frames …")

    cap.release()
    writer.release()
    print(f"   Labeled video → {out_video.name}")
    return out_video


# ============================================================
# Main
# ============================================================

def main():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for video_path in VIDEOS:
        if not video_path.exists():
            print(f"SKIP (not found): {video_path}")
            continue
        try:
            slp_path, labels = predict_one(video_path)
        except Exception as e:
            print(f"FAILED inference on {video_path.name}: {e}")
            continue

        out_stem = slp_path.stem  # e.g. "video.sample3000.predictions"

        if EXPORT_CSV:
            try:
                export_coordinates(labels, out_stem)
            except Exception as e:
                print(f"FAILED export: {e}")

        if RENDER_VIDEOS:
            try:
                render_labeled_video(labels, video_path, out_stem)
            except Exception as e:
                print(f"FAILED render: {e}")

    print(f"\nDone. Outputs in {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
