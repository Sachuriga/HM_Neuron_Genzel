import sys
import argparse
import deeplabcut
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def find_collected_video(op_path: Path) -> Optional[Path]:
    # find collected_frames.mp4
    matches = list(op_path.rglob("collected_frames.mp4"))
    return matches[0] if matches else None


def find_with_frames_csv(op_path: Path) -> Optional[Path]:
    # find *_with_frames.csv
    matches = list(op_path.rglob("*_with_frames.csv"))
    return matches[0] if matches else None


def run_dlc(config_path: Path, video_path: Path, shuffle: int = 2) -> pd.DataFrame:
    # run DLC, return one row per frame
    deeplabcut.analyze_videos(
        str(config_path),
        [str(video_path)],
        shuffle=shuffle,
        save_as_csv=True,
    )
    # output is <video_stem>DLC_*shuffle<shuffle>*.h5 next to the video
    h5_matches = sorted(video_path.parent.glob(f"{video_path.stem}DLC*shuffle{shuffle}*.h5"))
    if not h5_matches:
        raise FileNotFoundError(f"No DLC output found for {video_path}")
    dlc_df = pd.read_hdf(h5_matches[0])
    # columns -> (bodypart, coord)
    dlc_df.columns = dlc_df.columns.droplevel(0)
    # keep x, y only
    dlc_df = dlc_df.loc[:, dlc_df.columns.get_level_values("coords").isin(["x", "y"])]
    return dlc_df


def merge_coordinates(df: pd.DataFrame, dlc_df: pd.DataFrame) -> pd.DataFrame:
    # written rows = rows put into the video, in DLC frame order
    written_mask = df['extracted_frame_idx'].notna()
    written_rows = df.index[written_mask].to_numpy()
    if len(written_rows) != len(dlc_df):
        print(
            f"Warning: {len(written_rows)} written rows vs {len(dlc_df)} DLC frames. "
            f"Merging up to the shorter length."
        )
    n = min(len(written_rows), len(dlc_df))
    # column names: <bodypart>_<coord>
    flat_cols = [f"{bp}_{coord}" for bp, coord in dlc_df.columns]
    for col in flat_cols:
        df[col] = np.nan
        df[col] = df[col].astype(float)
    dlc_values = dlc_df.to_numpy()
    df.loc[written_rows[:n], flat_cols] = dlc_values[:n]
    return df


def process_dlc_tracking(op_dir: str, config: str, csv_path: Optional[str] = None, shuffle: int = 2):
    op_path = Path(op_dir)
    config_path = Path(config)
    video_path = find_collected_video(op_path)
    if not video_path:
        raise FileNotFoundError(f"No collected_frames.mp4 found in {op_path}.")
    if csv_path:
        csv_file = Path(csv_path)
    else:
        csv_file = find_with_frames_csv(op_path)
        if not csv_file:
            raise FileNotFoundError(f"No *_with_frames.csv found in {op_path}.")
    print(f"--- DLC Tracking ---\nConfig: {config_path}\nVideo: {video_path}\nCSV: {csv_file}\nShuffle: {shuffle}\n")
    df = pd.read_csv(csv_file)
    if 'extracted_frame_idx' not in df.columns:
        raise ValueError("CSV is missing 'extracted_frame_idx'. Run the tracking pipeline first.")
    dlc_df = run_dlc(config_path, video_path, shuffle)
    df = merge_coordinates(df, dlc_df)
    # overwrite the same CSV
    df.to_csv(csv_file, index=False)
    print(f"\nDone. Coordinates written back to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Run DLC on collected frames and merge coordinates back into the CSV.")
    parser.add_argument("--op_folder", "-o", required=True, help="The op folder (input)")
    parser.add_argument("--config", "-g", required=True, help="Path to DLC project config.yaml")
    parser.add_argument("--csv", "-c", help="Specific path to the *_with_frames.csv (optional)")
    parser.add_argument("--shuffle", "-s", type=int, default=2, help="DLC shuffle index")
    args = parser.parse_args()
    try:
        process_dlc_tracking(args.op_folder, args.config, args.csv, args.shuffle)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
