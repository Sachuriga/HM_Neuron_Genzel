"""SLEAP inference pipeline: predict keypoints, export coordinates, render labeled videos."""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import cv2
import h5py
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
SAMPLE_N = 3000  # number of frames to randomly sample (only used when MODE="sample")

# --- Hardware ---
DEVICE = "cuda"   # "cuda" / "cuda:0" / "cpu"
BATCH_SIZE = 8    # increase to 16 or 32 if VRAM allows

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


def _video_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ============================================================
# Step 1 — Inference
# ============================================================

def predict_one(video_path: Path) -> Path:
    """Run SLEAP inference on one video. Returns the output .slp path."""
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

    run_inference(
        data_path=str(video_path),
        model_paths=[str(MODEL_PATH)],
        output_path=str(out_path),
        make_labels=True,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        frames=frames,  # None = full video
    )

    # Sanity check — inspect what was actually written to the file
    _diagnose_slp(out_path)
    labels = sio.load_slp(str(out_path))
    n_predicted = len(labels.labeled_frames)
    expected = len(frames) if frames is not None else total_frames
    pct = 100.0 * n_predicted / expected if expected else 0
    print(f"   Predicted {n_predicted}/{expected} frames ({pct:.1f}%)")
    if n_predicted < 0.5 * expected:
        print("   WARNING: fewer than half the frames have predictions")

    return out_path


def _diagnose_slp(slp_path: Path) -> None:
    """Print the raw HDF5 structure of an .slp file to help diagnose empty-prediction issues."""
    file_size_mb = slp_path.stat().st_size / 1_048_576
    print(f"\n   [diag] File size: {file_size_mb:.2f} MB")

    try:
        with h5py.File(str(slp_path), "r") as f:
            def _print_item(name: str, obj) -> None:
                if isinstance(obj, h5py.Dataset):
                    print(f"   [diag]   {name}: shape={obj.shape} dtype={obj.dtype}")
                else:
                    print(f"   [diag]   {name}/")
            f.visititems(_print_item)
    except Exception as e:
        print(f"   [diag] Could not inspect HDF5: {e}")


# ============================================================
# Step 2 — Coordinate export
# ============================================================

def export_coordinates(slp_path: Path) -> Path:
    """Write a CSV of keypoint coordinates from a .slp prediction file.

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
    out_csv = OUTPUTS_DIR / f"{slp_path.stem}.coordinates.csv"
    df.to_csv(out_csv, index=False)
    print(f"   Coordinates → {out_csv.name}  ({len(df)} rows)")
    return out_csv


# ============================================================
# Step 3 — Labeled video rendering
# ============================================================

def render_labeled_video(slp_path: Path, video_path: Path) -> Path:
    """Overlay keypoints and skeleton edges onto only the predicted frames."""
    labels = sio.load_slp(str(slp_path))
    skeleton = labels.skeletons[0] if labels.skeletons else None

    # Only render frames that have predictions, in order
    frame_lookup: dict[int, list] = {lf.frame_idx: lf.instances for lf in labels.labeled_frames}
    predicted_frame_indices = sorted(frame_lookup)

    if not predicted_frame_indices:
        print("   No predicted frames to render — skipping.")
        return slp_path

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = OUTPUTS_DIR / f"{slp_path.stem}.labeled.mp4"
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
            coords = [(p.x, p.y, p.visible) for p in instance.points]

            # Skeleton edges (draw beneath keypoints)
            for src_i, dst_i in edge_pairs:
                if src_i >= len(coords) or dst_i >= len(coords):
                    continue
                sx, sy, sv = coords[src_i]
                dx, dy, dv = coords[dst_i]
                if sv and dv and not any(np.isnan(v) for v in (sx, sy, dx, dy)):
                    cv2.line(frame, (int(sx), int(sy)), (int(dx), int(dy)), (200, 200, 200), LINE_THICKNESS)

            # Keypoint circles
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
            slp_path = predict_one(video_path)
        except Exception as e:
            print(f"FAILED inference on {video_path.name}: {e}")
            continue

        if EXPORT_CSV:
            try:
                export_coordinates(slp_path)
            except Exception as e:
                print(f"FAILED export on {slp_path.name}: {e}")

        if RENDER_VIDEOS:
            try:
                render_labeled_video(slp_path, video_path)
            except Exception as e:
                print(f"FAILED render on {slp_path.name}: {e}")

    print(f"\nDone. Outputs in {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
