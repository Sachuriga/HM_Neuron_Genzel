import os
import re
import sys
import cv2
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict

# src/tools/vcodec.py knows which ffmpeg this computer should run (FFMPEG_CMD).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import vcodec
import super_resolution     # this folder: the network for the super-resolved video

def add_region_column(
    df: pd.DataFrame,
    x_col: str = "Rat_X",
    y_col: str = "Rat_Y",
    img_w: int = 1176,
    img_h: int = 712,
    region_w: int = 196,
    region_h: int = 356,
    start_index: int = 0,
    out_col: str = "region_id",
) -> pd.DataFrame:
    """Calculates region IDs based on X,Y coordinates."""
    df[out_col] = np.nan
    if x_col not in df.columns or y_col not in df.columns:
        return df

    valid_rows = df[x_col].notna() & df[y_col].notna()
    if valid_rows.any():
        n_cols = img_w // region_w
        x = np.clip(df.loc[valid_rows, x_col].to_numpy(), 0, img_w - 1)
        y = np.clip(df.loc[valid_rows, y_col].to_numpy(), 0, img_h - 1)
        col = (x // region_w).astype(int)
        row = (y // region_h).astype(int)
        region_id = (row * n_cols + col + start_index).astype(int)
        df.loc[valid_rows, out_col] = region_id
    return df

# Folders that hold copies of the session's files, never the files to work on:
# clock_qc.py / rewrite_nwb_lfp.py keep the originals they replaced in these.
_BACKUP_DIRS = {"clock_repair_backup", "lfp_fix_backup", "clock_qc"}


def _find_files(folder: Path, pattern: str) -> list:
    """Files matching `pattern` below `folder`, the ones nearest the top first.

    Leaves out macOS "._" sidecars (a Mac writes one next to every file on the
    share; they match the pattern, sort first and are not CSVs / videos) and
    everything inside a backup folder — a bare rglob()[0] could return either."""
    hits = [p for p in folder.rglob(pattern)
            if not p.name.startswith("._")
            and not _BACKUP_DIRS.intersection(p.relative_to(folder).parts[:-1])]
    return sorted(hits, key=lambda p: (len(p.relative_to(folder).parts), str(p).lower()))

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def find_video_for_region(region_idd: int | str, video_dir: Path) -> Optional[Path]:
    region_id = int(region_idd) + 1
    pattern = f"eye0{region_id}*.mp4" if region_id < 10 else f"eye{region_id}*.mp4"
    matches = sorted((p for p in video_dir.rglob(pattern) if not p.name.startswith("._")),
                     key=lambda p: _natural_key(p.name))
    return matches[0] if matches else None

# ---------------------------------------------------------------------------
#  Rat-centred crop
#
#  Rat_X / Rat_Y in the tracking CSV are pixels of the tracker's display frame
#  (1176x712), which is the stitched canvas at half size. To cut a small box
#  around the rat out of ONE eye camera's frame we have to undo the stitching.
#  The numbers below are the ones src/tracker/join_views.py stitches with; if
#  the stitch layout ever changes, change them here too.
# ---------------------------------------------------------------------------
STITCH_COLS = 6            # cameras per row on the canvas (2 rows x 6 columns)
STITCH_CROP_X = 104        # px cut off the left (and right) of every eye frame
STITCH_CROP_Y = 91         # px cut off the top of a bottom-row eye frame (after its flip)
STITCH_TILE_W = 392        # width one camera keeps on the canvas  (600 - 2 * 104)
STITCH_TILE_H = 709        # height one camera keeps on the canvas (800 - 91)
CANVAS_PX_PER_DISPLAY_PX = 2.0   # canvas 2352x1424 -> display 1176x712

# The 2352 px wide canvas shows the 9 m long maze (same convention as
# plot_trials.py), and eye frames are stitched at their own resolution, so one
# centimetre is about 2.6 pixels in an eye frame.
EYE_PX_PER_CM = 2352 / 900.0

# Side of the square cut around the rat. The CSV position is the centre of the
# rat's body, so the box must reach from there to the nose: 40 cm (about 106 px)
# holds a whole rat. 10 cm would be only 26 px, smaller than the rat itself.
DEFAULT_CROP_CM = 40.0

CROPPED_VIDEO_NAME = "collected_frames_crop.mp4"   # plain enlargement, real pixels
SR_VIDEO_NAME = "collected_frames_crop_sr.mp4"     # enlarged by the network

# How many crops the super-resolution network gets at once. More is faster on a
# graphics card but needs more of its memory; 32 crops of 106 px are small.
SR_BATCH_SIZE = 32

# The box is only about 106 px wide. That is every pixel the camera has of the
# rat, so cropping loses nothing, but a picture that small is hard to look at
# and a video encoder throws detail away, which hurts most on a tiny picture.
# So each crop is enlarged before it is saved (4 x 106 = 424 px) and saved with
# almost no compression. The plain video does this with Lanczos, which invents
# nothing; the super-resolved video does it with a network (super_resolution.py).
# 4 is also what that network enlarges by, so keep 4 unless you have a reason.
DEFAULT_CROP_SCALE = 4

# libx264 quality: 0 = lossless, 23 = ffmpeg's default, 28 = what step 6 uses.
# 12 looks the same as the original frames to the eye.
CROP_VIDEO_CRF = "12"


def crop_size_in_pixels(crop_cm: float) -> int:
    """Side of the crop box in eye-frame pixels (0 = no cropping).

    Rounded to an even number because video encoders want even frame sizes."""
    if crop_cm <= 0:
        return 0
    size = int(round(crop_cm * EYE_PX_PER_CM))
    if size % 2 == 1:
        size += 1
    return max(size, 2)


def rat_position_in_eye_frame(rat_x: float, rat_y: float, region_id: int) -> Tuple[float, float]:
    """Where the rat is inside the eye frame of camera `region_id`, in pixels.

    For a bottom-row camera (region 6-11) the answer is for the frame AFTER it
    was rotated 180 degrees, which is how this script writes those frames."""
    column = region_id % STITCH_COLS
    row = region_id // STITCH_COLS

    # display pixels -> canvas pixels
    canvas_x = rat_x * CANVAS_PX_PER_DISPLAY_PX
    canvas_y = rat_y * CANVAS_PX_PER_DISPLAY_PX

    # position inside this camera's tile on the canvas
    tile_x = canvas_x - column * STITCH_TILE_W
    tile_y = canvas_y - row * STITCH_TILE_H

    # add back what the stitch cut off before it pasted the tile. Top-row frames
    # lost nothing at the top; bottom-row frames lost STITCH_CROP_Y rows.
    eye_x = tile_x + STITCH_CROP_X
    eye_y = tile_y + (STITCH_CROP_Y if row >= 1 else 0)
    return eye_x, eye_y


def crop_box_origin(center_x: float, center_y: float, crop_size: int,
                    frame_w: int, frame_h: int) -> Tuple[int, int]:
    """Top-left corner of a crop_size box centred on the rat.

    Near a border the box is pushed back inside the frame instead of being
    filled with black, so every crop has the same size and only real pixels."""
    x0 = int(round(center_x - crop_size / 2))
    y0 = int(round(center_y - crop_size / 2))
    x0 = min(max(x0, 0), max(frame_w - crop_size, 0))
    y0 = min(max(y0, 0), max(frame_h - crop_size, 0))
    return x0, y0


def enlarge_crop(cropped: np.ndarray, scale: int) -> np.ndarray:
    """The crop made `scale` times larger (scale 1 = unchanged).

    Lanczos keeps edges (ears, tail, whisker pad) sharper than the default
    bilinear resize, which blurs them."""
    if scale <= 1:
        return cropped
    h, w = cropped.shape[:2]
    return cv2.resize(cropped, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def crop_pixel_to_eye_pixel(crop_x: float, crop_y: float, crop_x0: float, crop_y0: float,
                            scale: float = 1.0) -> Tuple[float, float]:
    """A point found in the (enlarged) cropped video, back in eye-frame pixels.

    `crop_x0`, `crop_y0` and `scale` are the CSV columns crop_x0, crop_y0 and
    crop_scale of that frame's row."""
    return crop_x / scale + crop_x0, crop_y / scale + crop_y0


def _ffmpeg_can_encode_h264(ffmpeg: str) -> bool:
    """True when this ffmpeg really encodes with libx264 (tried on one tiny frame)."""
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "color=black:size=64x64:rate=1", "-frames:v", "1",
             "-c:v", "libx264", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


class HighQualityVideoWriter:
    """Writes frames to an .mp4 with almost no compression loss.

    OpenCV's own writer (codec 'mp4v') gives no control over quality and smears
    a small picture. Here the frames are handed to ffmpeg instead, which encodes
    them as H.264 at a fixed high quality. If ffmpeg cannot do that on this
    computer, the OpenCV writer is used so the step still produces a video.

    Same two methods as cv2.VideoWriter: write(frame) and release()."""

    def __init__(self, path: Path, fps: float, size: Tuple[int, int]):
        self.path = Path(path)
        self.ffmpeg_process = None
        self.cv2_writer = None
        width, height = size

        ffmpeg = vcodec.ffmpeg_cmd()
        if _ffmpeg_can_encode_h264(ffmpeg):
            command = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                # what comes in: raw frames, exactly as OpenCV holds them
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
                # what goes out: H.264, high quality, playable everywhere (yuv420p)
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", CROP_VIDEO_CRF,
                "-pix_fmt", "yuv420p", str(self.path),
            ]
            self.ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE)
        else:
            print("ffmpeg with libx264 not available: the cropped video is written "
                  "with OpenCV (lower quality).")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.cv2_writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))

    def write(self, frame: np.ndarray):
        if self.ffmpeg_process is not None:
            self.ffmpeg_process.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv2_writer.write(frame)

    def release(self):
        if self.ffmpeg_process is not None:
            # closing the pipe tells ffmpeg there are no more frames; then wait
            # until it has finished writing the file
            self.ffmpeg_process.stdin.close()
            if self.ffmpeg_process.wait() != 0:
                print(f"Warning: ffmpeg reported an error while writing {self.path}")
        else:
            self.cv2_writer.release()


class FastVideoReader:
    """Caches VideoCapture objects to prevent reopening files continuously."""
    MAX_SKIP = 60       # frames skipped by reading on instead of seeking

    def __init__(self):
        self.caps: Dict[Path, cv2.VideoCapture] = {}
        self.current_frame_idx: Dict[Path, int] = {}

    def get_frame(self, video_path: Path, frame_idx: int) -> Tuple[bool, Optional[np.ndarray], Optional[str]]:
        if video_path not in self.caps:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False, None, f"Could not open video: {video_path}"
            self.caps[video_path] = cap
            self.current_frame_idx[video_path] = 0

        cap = self.caps[video_path]
        current_idx = self.current_frame_idx[video_path]

        # Only use cap.set() if we are jumping around.
        # If we just need the very next frame, reading sequentially is much faster.
        # A short step forward (the frames where the rat was not tracked) is also
        # read through: a seek makes the decoder start again from the last key
        # frame, which costs more than skipping a few frames — and it lands on the
        # exact frame, which a seek in an h264 file does not always do.
        gap = frame_idx - current_idx
        if 0 < gap <= self.MAX_SKIP:
            for _ in range(gap):
                if not cap.grab():
                    break
        elif gap != 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ok, frame = cap.read()
        if ok:
            self.current_frame_idx[video_path] = frame_idx + 1
            return True, frame, None
        else:
            return False, None, f"Read error on frame {frame_idx}"

    def release_all(self):
        for cap in self.caps.values():
            cap.release()

def remove_stale_video(video_path: Path):
    """Delete a video an earlier run left behind when this run does not write it:
    its frames would no longer match the rows of this run's CSV."""
    if video_path.exists():
        video_path.unlink()


def process_tracking_data(
    input_dir: str,
    output_dir: str,
    csv_path: Optional[str] = None,
    fps: float = 30.0,
    show_stream: bool = True,
    crop_cm: float = DEFAULT_CROP_CM,
    crop_scale: int = DEFAULT_CROP_SCALE,
    super_resolve: bool = True,
    sr_model_path: Optional[str] = None,
):
    """Step d, part 1: one small picture of the rat per tracked frame.

    Writes into the output folder:
      collected_frames_crop.mp4     the box around the rat, enlarged the plain
                                    way (Lanczos): real camera pixels only
      collected_frames_crop_sr.mp4  the same box enlarged by the super-resolution
                                    network (only when super_resolve is on and
                                    the network can be loaded)
      <csv>_with_frames.csv         the tracking CSV plus region_id,
                                    extracted_frame_idx, crop_x0, crop_y0, crop_scale

    Both videos have the same frames, in the same order and at the same size, so
    frame i of either one belongs to the i-th CSV row that has extracted_frame_idx.
    The whole eye frame is no longer saved (the old collected_frames.mp4)."""
    ip_path = Path(input_dir)
    op_path = Path(output_dir)
    op_path.mkdir(parents=True, exist_ok=True)

    # 1. Locate CSV
    if csv_path:
        csv_file = Path(csv_path)
    else:
        csv_files = _find_files(op_path, "*_Full.csv") + _find_files(ip_path, "*_Full.csv")
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {ip_path} or {op_path}.")
        csv_file = csv_files[0]

    print(f"--- Processing Data ---\nCSV: {csv_file}\nInput: {ip_path}\nOutput: {op_path}\n")

    df = pd.read_csv(csv_file)
    df = add_region_column(df)

    # which video frame each row's picture was taken from (empty = no picture)
    df['extracted_frame_idx'] = pd.NA

    # The box around the rat. Its corner is kept in the CSV so a keypoint found
    # in the cropped video can be put back into the eye frame, and the box is
    # enlarged crop_scale times before it is saved (see crop_pixel_to_eye_pixel).
    crop_size = crop_size_in_pixels(crop_cm)
    if crop_size == 0:
        # the cropped videos are the only videos this step writes now
        print(f"crop_cm={crop_cm:g} would write no video; using {DEFAULT_CROP_CM:g} cm.")
        crop_cm = DEFAULT_CROP_CM
        crop_size = crop_size_in_pixels(crop_cm)
    crop_scale = max(int(crop_scale), 1)
    df['crop_x0'] = pd.NA
    df['crop_y0'] = pd.NA
    df['crop_scale'] = pd.NA

    cropped_vid_path = op_path / CROPPED_VIDEO_NAME
    sr_vid_path = op_path / SR_VIDEO_NAME
    crop_writer = None      # plain enlarged video (real pixels)
    sr_writer = None        # super-resolved video
    saved_size = None       # side of both saved videos, in pixels

    # The super-resolution network, or None when it is switched off or cannot be
    # loaded here (no PyTorch, no model file): the plain video is still written.
    upscaler = None
    if super_resolve:
        upscaler = super_resolution.load_upscaler(sr_model_path)
    if upscaler is None:
        remove_stale_video(sr_vid_path)
    # crops waiting for the network; it is much faster on many pictures at once
    sr_waiting = []

    def write_waiting_crops_super_resolved():
        """Run the waiting crops through the network and append them to the video."""
        if not sr_waiting:
            return
        for picture in upscaler.upscale(sr_waiting):
            # The network always enlarges 4x. With another crop_scale, bring its
            # picture to the size of the plain video so both stay comparable.
            if picture.shape[0] != saved_size:
                picture = cv2.resize(picture, (saved_size, saved_size),
                                     interpolation=cv2.INTER_AREA)
            sr_writer.write(picture)
        sr_waiting.clear()

    # --- OPTIMIZATION 1: Pre-map regions to video files ---
    print("Pre-mapping video paths...")
    unique_regions = df['region_id'].dropna().unique()
    region_to_video = {}
    for region in unique_regions:
        region_to_video[region] = find_video_for_region(region, ip_path)

    # --- OPTIMIZATION 2: Initialize the FastVideoReader ---
    video_reader = FastVideoReader()

    errors = []
    frames_written = 0

    total_rows = len(df)
    has_frame_index = "Frame_Index" in df.columns
    for i, (row_idx, row) in enumerate(df.iterrows(), start=1):
        region_id = row.get("region_id")
        if pd.isna(region_id):
            continue

        # Look up the video instantly from our pre-mapped dictionary
        video_path = region_to_video.get(region_id)
        if not video_path:
            errors.append(f"Row {row_idx}: No video for region {region_id}")
            continue

        # The video frame of this row. The row NUMBER is that frame only when the
        # file has one row per frame from frame 0; a tracker run that started later
        # (Start_Min / Start_Sec) without a sync file does not, and the row number
        # then picked frames from the wrong moment.
        frame_no = int(row_idx)
        if has_frame_index and pd.notna(row["Frame_Index"]):
            frame_no = int(row["Frame_Index"])

        # Fetch frame using our cached reader
        ok, frame, err = video_reader.get_frame(video_path, frame_no)
        if not ok:
            errors.append(f"Row {row_idx}: {err}")
            continue

        if int(region_id) > 5:
            frame = cv2.flip(frame, -1)  # -1 = flip both vertically and horizontally

        frame_h, frame_w = frame.shape[:2]
        if crop_writer is None:
            # First picture: now the frame size is known, open the video file(s).
            # The box can never be larger than the frame it is cut from.
            crop_size = min(crop_size, frame_w - frame_w % 2, frame_h - frame_h % 2)
            saved_size = crop_size * crop_scale
            crop_writer = HighQualityVideoWriter(cropped_vid_path, fps, (saved_size, saved_size))
            if upscaler is not None:
                sr_writer = HighQualityVideoWriter(sr_vid_path, fps, (saved_size, saved_size))

        # Cut the box from the frame as the camera recorded it (already rotated
        # for the bottom row), so the crop keeps the original pixels.
        eye_x, eye_y = rat_position_in_eye_frame(row["Rat_X"], row["Rat_Y"], int(region_id))
        x0, y0 = crop_box_origin(eye_x, eye_y, crop_size, frame_w, frame_h)
        cropped = frame[y0:y0 + crop_size, x0:x0 + crop_size]
        if cropped.shape[:2] != (crop_size, crop_size):
            cropped = cv2.resize(cropped, (crop_size, crop_size))

        # video 1: plain enlargement
        enlarged = enlarge_crop(cropped, crop_scale)
        crop_writer.write(enlarged)

        # video 2: super-resolution, a batch of crops at a time
        if sr_writer is not None:
            sr_waiting.append(cropped.copy())
            if len(sr_waiting) >= SR_BATCH_SIZE:
                write_waiting_crops_super_resolved()

        df.at[row_idx, 'extracted_frame_idx'] = frame_no
        df.at[row_idx, 'crop_x0'] = x0
        df.at[row_idx, 'crop_y0'] = y0
        df.at[row_idx, 'crop_scale'] = crop_scale
        frames_written += 1

        if show_stream:
            cv2.imshow("Processing Stream (Press 'q' to quit)", enlarged)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nInterrupted by user.")
                break

        # Progress bar (updated less frequently to save terminal overhead)
        if i % 10 == 0 or i == total_rows:
            progress = int(i / total_rows * 20)
            sys.stdout.write(f"\rProgress: [{'#' * progress}{'-' * (20 - progress)}] {i}/{total_rows}")
            sys.stdout.flush()

    # Clean up
    if sr_writer is not None:
        write_waiting_crops_super_resolved()    # the last, not yet full, batch
        sr_writer.release()
    if crop_writer is not None:
        crop_writer.release()
    else:
        # not one frame could be written: do not leave last run's videos behind
        remove_stale_video(cropped_vid_path)
        remove_stale_video(sr_vid_path)
    video_reader.release_all()
    cv2.destroyAllWindows()

    output_csv_path = op_path / f"{csv_file.stem}_with_frames.csv"
    df.to_csv(output_csv_path, index=False)

    print(f"\n\nProcessing Complete. {frames_written} frames written.")
    if crop_writer is not None:
        print(f"Rat-centred video ({crop_size}x{crop_size} px box, about {crop_cm:g} cm, "
              f"saved {crop_scale}x larger as {saved_size}x{saved_size} px) "
              f"saved to: {cropped_vid_path}")
    if sr_writer is not None:
        print(f"Super-resolved video (same frames, same size) saved to: {sr_vid_path}")
    print(f"Updated CSV saved to: {output_csv_path}")

    if errors:
        print(f"Encountered {len(errors)} errors. (Showing first 5)")
        for e in errors[:5]: print(e)

def main():
    parser = argparse.ArgumentParser(description="Compile video frames based on tracking regions.")
    parser.add_argument("--input_folder", "-i", required=True, help="Directory containing source videos")
    parser.add_argument("--output_folder", "-o", required=True, help="Directory for output video and CSV search")
    parser.add_argument("--csv", "-c", help="Specific path to tracking CSV (optional)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30)")
    parser.add_argument("--no-vis", action="store_false", dest="show_stream", help="Disable real-time video preview")
    parser.add_argument("--crop_cm", type=float, default=DEFAULT_CROP_CM,
                        help=f"Side of the square cut around the rat for {CROPPED_VIDEO_NAME}, "
                             "in cm (default: %(default)s)")
    parser.add_argument("--no-sr", action="store_false", dest="super_resolve",
                        help=f"Do not write the super-resolved video {SR_VIDEO_NAME}")
    parser.add_argument("--sr_model", default=os.environ.get("DLC_SR_MODEL", "").strip() or None,
                        help="Path of the Real-ESRGAN model file (default: DLC_SR_MODEL, "
                             "else downloaded once into <repo>/models/)")

    parser.add_argument("--crop_scale", type=int, default=DEFAULT_CROP_SCALE,
                        help="How many times the cropped box is enlarged before it is saved "
                             "(default: %(default)s; 1 = keep the camera's own pixels)")

    args = parser.parse_args()

    try:
        process_tracking_data(args.input_folder, args.output_folder, args.csv, args.fps,
                              args.show_stream, args.crop_cm, args.crop_scale,
                              args.super_resolve, args.sr_model)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()