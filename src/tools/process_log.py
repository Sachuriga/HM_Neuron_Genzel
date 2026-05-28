import sys
import re
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

import sys

# OUTPUTS DF FROM LOG FILE

# Helper functions
def parse_video_to_seconds(ts_str):
    """Parses HH:MM:SS.mmm strings into total seconds."""
    if not ts_str:
        return None
    try:
        h, m, s_ms = ts_str.split(":")
        s, ms = s_ms.split(".")
        td = timedelta(hours=int(h), minutes=int(m), seconds=int(s), milliseconds=int(ms))
        return td.total_seconds()
    except Exception:
        return None

# Trim to active trials
# Trim dataframe to filter all rows for which column 'label' contains NaN
def filter_nan_in_col(df, label):
    df = df[df[label].notna()]
    return df

def process_log(paths):
    # --- Corrected Regex Patterns ---

    # 1. Timestamp Regex: accurately handles the optional system time AND the optional colon separator
    ts_line_new = re.compile(
        r'^(?:(?P<level>[A-Z]+)\s*:\s*)?'             # Level (INFO:)
        r'(?:(?P<video>\d{1,2}:\d{1,2}:\d{1,2}\.\d{3})\s*)?' # Video Time
        r'(?:(?P<sys>\d+(?:\.\d+)?)\s*)?'             # Optional Sys Time (No colon here yet)
        r'(?::\s*)?'                                  # Match the colon separator separately so it doesn't end up in msg
        r'(?P<msg>.*)$'                               # The Message (clean)
    )

    # 2. Position Regex: Matches floats/ints and tolerates spaces inside parentheses
    pos_line = re.compile(
        r'The rat position is:\s*\(\s*(?P<x>-?[\d\.]+),\s*(?P<y>-?[\d\.]+)\s*\)\s*@\s*(?P<frame>[\d\.]+)'
    )

    all_dfs = []
    # --- 2. Parse Logs ---
    for log_path in paths.log_paths:
        print(f"Parsing: {log_path}")
        rows_new = []
        with Path(log_path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                m = ts_line_new.match(line)
                if not m: continue
                
                # Extract groups
                level = m.group("level")
                video_time = m.group("video")
                sys_time_str = m.group("sys")
                msg = m.group("msg")

                x = y = frame = None
                node = None
                mpos = pos_line.search(msg)
                
                if mpos:
                    try:
                        # FIX: Convert to float first to avoid ValueError on strings like "100.0"
                        # We then cast to int() for the dataframe to keep pixels as integers
                        x = int(float(mpos.group("x")))
                        y = int(float(mpos.group("y")))
                        frame = int(float(mpos.group("frame")))
                        event = "rat_position"
                    except ValueError:
                        print(f"Skipping malformed number in: {line}")
                        continue
                else:
                    # Clean check on msg (now that the colon is gone)
                    if msg.startswith("Video Imported"): event = "video_imported"
                    elif msg.startswith("Recording Trial"): event = "recording_start"
                    else: event = "message"
                
                rows_new.append({
                    "video_seconds": parse_video_to_seconds(video_time),
                    "sys_time": float(sys_time_str) if sys_time_str else None,
                    "event": event,
                    "x": x, "y": y, 
                    "raw": msg,
                })

        if rows_new:
            all_dfs.append(pd.DataFrame(rows_new))
            print(f" -> Extracted {len(rows_new)} rows.")

    if not all_dfs:
        sys.exit("No valid data parsed from logs.")

    df_log = pd.concat(all_dfs, ignore_index=True)

    trial_starts = np.where(df_log['event'].values == 'recording_start')[0]
    
    df_log.insert(0, 'Trial_Number', None)
    for i, start_idx in enumerate(trial_starts):
        trial_number = int(df_log['raw'][start_idx].split()[-1])
        if i < len(trial_starts) - 1:
            end_idx = trial_starts[i + 1]
        else:
            end_idx = len(df_log)

        df_log.loc[start_idx:end_idx - 1, "Trial_Number"] = trial_number

    df_log = filter_nan_in_col(df_log, 'x')
    df_log = filter_nan_in_col(df_log, 'sys_time')

    # after filter indices change
    trial_starts = np.where(np.diff(df_log['Trial_Number'].values) != 0)[0] + 1
    return df_log, trial_starts
