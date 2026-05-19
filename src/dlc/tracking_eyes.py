import os
import re
import sys
import cv2
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict

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

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|\D+', s)]

def find_video_for_region(region_idd: int | str, video_dir: Path) -> Optional[Path]:
    region_id = int(region_idd) + 1
    pattern = f"eye0{region_id}*.mp4" if region_id < 10 else f"eye{region_id}*.mp4"
    matches = sorted(video_dir.rglob(pattern), key=lambda p: _natural_key(p.name))
    return matches[0] if matches else None

class FastVideoReader:
    """Caches VideoCapture objects to prevent reopening files continuously."""
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
        if frame_idx != current_idx:
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

def process_tracking_data(
    input_dir: str, 
    output_dir: str, 
    csv_path: Optional[str] = None, 
    fps: float = 30.0,
    show_stream: bool = True
):
    ip_path = Path(input_dir)
    op_path = Path(output_dir)
    op_path.mkdir(parents=True, exist_ok=True)

    # 1. Locate CSV
    if csv_path:
        csv_file = Path(csv_path)
    else:
        csv_files = list(op_path.rglob("*_Full.csv")) + list(ip_path.rglob("*_Full.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in {ip_path} or {op_path}.")
        csv_file = csv_files[0]

    print(f"--- Processing Data ---\nCSV: {csv_file}\nInput: {ip_path}\nOutput: {op_path}\n")

    df = pd.read_csv(csv_file)
    df = add_region_column(df)
    
    # --- NEW: Initialize the column to store the extracted frame index ---
    df['extracted_frame_idx'] = pd.NA
    
    # --- OPTIMIZATION 1: Pre-map regions to video files ---
    print("Pre-mapping video paths...")
    unique_regions = df['region_id'].dropna().unique()
    region_to_video = {}
    for region in unique_regions:
        region_to_video[region] = find_video_for_region(region, ip_path)
    
    # --- OPTIMIZATION 2: Initialize the FastVideoReader ---
    video_reader = FastVideoReader()

    output_vid_path = op_path / "collected_frames.mp4"
    writer = None
    target_size = None
    errors = []

    total_rows = len(df)
    for i, (row_idx, row) in enumerate(df.iterrows(), start=1):
        region_id = row.get("region_id")
        if pd.isna(region_id):
            continue

        # Look up the video instantly from our pre-mapped dictionary
        video_path = region_to_video.get(region_id)
        if not video_path:
            errors.append(f"Row {row_idx}: No video for region {region_id}")
            continue

        # Fetch frame using our cached reader
        ok, frame, err = video_reader.get_frame(video_path, int(row_idx))
        if not ok:
            errors.append(f"Row {row_idx}: {err}")
            continue
            
        # --- NEW: Record the frame index successfully extracted for this row ---
        df.at[row_idx, 'extracted_frame_idx'] = int(row_idx)

        if int(region_id) > 5:
            frame = cv2.flip(frame, -1)  # -1 = flip both vertically and horizontally

        if writer is None:
            h, w = frame.shape[:2]
            target_size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_vid_path), fourcc, fps, target_size)

        if (frame.shape[1], frame.shape[0]) != target_size:
            frame = cv2.resize(frame, target_size)

        writer.write(frame)

        if show_stream:
            cv2.imshow("Processing Stream (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nInterrupted by user.")
                break

        # Progress bar (updated less frequently to save terminal overhead)
        if i % 10 == 0 or i == total_rows:
            progress = int(i / total_rows * 20)
            sys.stdout.write(f"\rProgress: [{'#' * progress}{'-' * (20 - progress)}] {i}/{total_rows}")
            sys.stdout.flush()

    # Clean up
    if writer:
        writer.release()
    video_reader.release_all()
    cv2.destroyAllWindows()
    
    # --- NEW: Save the updated dataframe to a new CSV file ---
    output_csv_path = op_path / f"{csv_file.stem}_with_frames.csv"
    df.to_csv(output_csv_path, index=False)
    
    print(f"\n\nProcessing Complete. Video saved to: {output_vid_path}")
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

    args = parser.parse_args()

    try:
        process_tracking_data(args.input_folder, args.output_folder, args.csv, args.fps, args.show_stream)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()