import sys
import argparse
import re
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import timedelta
import textwrap

# --- Imports for Graph Theory ---
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy import stats 

# --- Helper Functions ---

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

def moving_average(a: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or a.size == 0:
        return a.astype(float, copy=True)
    kernel = np.ones(k) / k
    pad = k // 2
    a_pad = np.pad(a, (pad, pad), mode="edge")
    out = np.convolve(a_pad, kernel, mode="valid")
    if out.size > a.size:
        out = out[:a.size]
    return out

def compute_speed_from_xy(x: np.ndarray, y: np.ndarray, fs: float) -> np.ndarray:
    dt = 1.0 / fs
    vx = np.gradient(x) / dt
    vy = np.gradient(y) / dt
    spd = np.hypot(vx, vy)
    spd = np.nan_to_num(spd, nan=0.0, posinf=0.0, neginf=0.0)
    return spd

def compute_path_length(x: np.ndarray, y: np.ndarray) -> float:
    """Calculates the total Euclidean distance of the trajectory."""
    if len(x) < 2:
        return 0.0
    dx = np.diff(x)
    dy = np.diff(y)
    dists = np.sqrt(dx**2 + dy**2)
    return np.sum(dists)

def parse_node_sequences(txt_path):
    sequences = {}
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        trial_header_re = re.compile(r"Summary Trial\s+(\d+)", re.IGNORECASE)
        node_line_re = re.compile(r"^[\d, ]+$")

        for i, line in enumerate(lines):
            m = trial_header_re.search(line)
            if m:
                trial_id = int(m.group(1))
                if i > 0:
                    prev_line = lines[i-1]
                    if node_line_re.match(prev_line.replace(" ", "").rstrip(',')):
                         sequences[trial_id] = prev_line.strip(', ')
    except Exception as e:
        print(f"Error parsing node sequence file: {e}")
    return sequences

def build_hexmaze_graph(nodes_df):
    G = nx.Graph()
    nodes_df['id_str'] = nodes_df['id'].astype(int).astype(str)
    
    pos_dict = {}
    for idx, row in nodes_df.iterrows():
        node_id = row['id_str']
        G.add_node(node_id, pos=(row['x'], row['y']))
        pos_dict[node_id] = np.array([row['x'], row['y']])

    coords = nodes_df[['x', 'y']].values
    distances = squareform(pdist(coords))
    threshold = 65
    node_ids = nodes_df['id_str'].tolist()
    
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            dist = distances[i, j]
            if dist < threshold:
                G.add_edge(node_ids[i], node_ids[j], weight=dist)

    # Modifications: Remove 501, 502
    for n in ['501', '502']:
        if n in G: G.remove_node(n)

    # Modifications: Add Manual Edges
    manual_edges = [
        ('121', '302'),
        ('324', '401'),
        ('305', '220'),
        ('404', '223'),
        ('201', '124'),
        ('224', '218'),
    ]
    
    for u, v in manual_edges:
        if u in G and v in G:
            p1 = pos_dict[u]
            p2 = pos_dict[v]
            w = np.linalg.norm(p1 - p2)
            G.add_edge(u, v, weight=w)
            
    return G

def get_all_shortest_paths_plot_data(G, start_node, end_node, weight_mode='weight'):
    all_paths_segments = []
    label = "No Path"
    metric_val = 0.0
    
    try:
        if start_node not in G or end_node not in G:
            return [], "Node not found", 0.0
            
        if not nx.has_path(G, start_node, end_node):
            return [], "No Path", 0.0

        paths_iter = nx.all_shortest_paths(G, source=start_node, target=end_node, weight=weight_mode)
        metric_val = nx.shortest_path_length(G, source=start_node, target=end_node, weight=weight_mode)
        
        path_count = 0
        pos = nx.get_node_attributes(G, 'pos')
        
        for path in paths_iter:
            path_count += 1
            current_segments = []
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                current_segments.append((pos[u], pos[v]))
            all_paths_segments.append(current_segments)
        
        if weight_mode == 'weight':
            label = f"Dist: {metric_val:.1f} (N={path_count})"
        else:
            label = f"Hops: {metric_val} (N={path_count})"
            
        return all_paths_segments, label, metric_val
    
    except nx.NetworkXNoPath:
        return [], "No Path", 0.0
    except Exception as e:
        return [], str(e), 0.0


# --- Main Script ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Log Files from Output Folder')
    
    parser.add_argument('-o', "--output", dest='output_folder', required=True, 
                        help='Folder path containing .log files (PDF will be saved here too)')
    
    args = parser.parse_args()

    work_dir = Path(args.output_folder)

    if not work_dir.exists():
        sys.exit(f"Error: The directory {work_dir} does not exist.")

    # 1. Find logs
    LOG_GLOB = str(work_dir / "*.log")
    log_paths = sorted(glob.glob(LOG_GLOB, recursive=True))
    
    if not log_paths:
        sys.exit(f"No .log files found in {work_dir}")

    print(f"Found {len(log_paths)} log files in {work_dir}")
    log_file_stem = Path(log_paths[0]).stem

    # --- 1b. Find and Parse Node Sequence Text File ---
    TXT_GLOB = str(work_dir / "*.txt")
    txt_paths = sorted(glob.glob(TXT_GLOB))
    
    trial_node_sequences = {}
    if txt_paths:
        print(f"Found text file(s): {txt_paths}. Parsing for node sequences...")
        trial_node_sequences = parse_node_sequences(txt_paths[0])
        print(f"Extracted sequences for {len(trial_node_sequences)} trials.")
    else:
        print("No .txt file found for node sequences.")

    # --- 1c. Find and Parse Metadata Excel File ---
    # Look specifically for RecordingMeta.xlsx or fallback to any *RecordingMeta.xlsx
    EXCEL_GLOB = str(work_dir / "RecordingMeta.xlsx")
    excel_paths = sorted(glob.glob(EXCEL_GLOB))
    if not excel_paths:
        excel_paths = sorted(glob.glob(str(work_dir / "*RecordingMeta.xlsx")))
    
    session_meta = None
    trial_metadata = {} # Dictionary to store per-trial metadata
    
    if excel_paths:
        print(f"Found Metadata Excel: {excel_paths[0]}")
        try:
            # Use read_excel instead of read_csv
            meta_df = pd.read_excel(excel_paths[0])
            if not meta_df.empty:
                session_meta = meta_df.iloc[0].to_dict() # For the cover page
                
                # Attempt to find a trial ID column. Typical names: Trial_ID, Trial, trial_id
                trial_col = None
                for col in ['Trial_ID', 'Trial', 'trial_id', 'trial']:
                    if col in meta_df.columns:
                        trial_col = col
                        break
                
                # Iterate over all rows to extract trial-specific goals, starts, and types
                for idx, row in meta_df.iterrows():
                    # If there's an explicit trial column use it, else assume 1-based index (1, 2, 3...)
                    t_id = int(row[trial_col]) if trial_col else idx + 1
                    
                    trial_data = {}
                    if 'Goal_Node' in row and pd.notna(row['Goal_Node']):
                        trial_data['Goal_Node'] = str(int(row['Goal_Node']))
                    if 'Start_Node' in row and pd.notna(row['Start_Node']):
                        trial_data['Start_Node'] = str(int(row['Start_Node']))
                    if 'Trial_Type' in row and pd.notna(row['Trial_Type']):
                        trial_data['Trial_Type'] = str(row['Trial_Type'])
                        
                    trial_metadata[t_id] = trial_data
                    
                print(f"Target Nodes and Trial Types loaded for {len(trial_metadata)} trials.")
        except Exception as e:
            print(f"Error parsing metadata Excel: {e}")
    else:
        print("No RecordingMeta.xlsx found. Proceeding without metadata.")

    # --- Corrected Regex Patterns ---
    
    # 1. Timestamp Regex: accurately handles the optional system time AND the optional colon separator
    ts_line_new = re.compile(
        r'^(?:(?P<level>[A-Z]+)\s*:\s*)?'            # Level (INFO:)
        r'(?:(?P<video>\d{1,2}:\d{1,2}:\d{1,2}\.\d{3})\s*)?' # Video Time
        r'(?:(?P<sys>\d+(?:\.\d+)?)\s*)?'            # Optional Sys Time (No colon here yet)
        r'(?::\s*)?'                                  # Match the colon separator separately so it doesn't end up in msg
        r'(?P<msg>.*)$'                               # The Message (clean)
    )

    # 2. Position Regex: Matches floats/ints and tolerates spaces inside parentheses
    pos_line = re.compile(
        r'The rat position is:\s*\(\s*(?P<x>-?[\d\.]+),\s*(?P<y>-?[\d\.]+)\s*\)\s*@\s*(?P<frame>[\d\.]+)'
    )

    all_dfs = []

    # --- 2. Parse Logs ---
    for log_path in log_paths:
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

    df = pd.concat(all_dfs, ignore_index=True)

    # --- 3. Cleanup ---
    for col in ["video_seconds", "x", "y", "sys_time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 4. Assign Trial IDs ---
    if "trial_id" not in df.columns:
        trial_id_list = []
        current = 1
        trial_re = re.compile(r"Recording\s*Trial\s*(\d+)\b", flags=re.I)

        for _, row in df.iterrows():
            if str(row.get("event", "")).lower() == "recording_start":
                m = trial_re.search(str(row.get("raw", "")))
                if m: current = int(m.group(1))
            trial_id_list.append(current)

        df["trial_id"] = trial_id_list

    # --- 5. Filter and Sort ---
    pos_df = df[df["event"] == "rat_position"].copy()
    sort_cols = []
    if "trial_id" in pos_df.columns: sort_cols.append("trial_id")
    if "sys_time" in pos_df.columns: sort_cols.append("sys_time")
    elif "video_seconds" in pos_df.columns: sort_cols.append("video_seconds")
    
    if sort_cols:
        pos_df = pos_df.sort_values(sort_cols, na_position="last")

    # --- 6. Per-Trial Aggregation ---
    records = []
    grouped = pos_df.groupby("trial_id", sort=False)
    
    for tid, g in grouped:
        if g.empty: continue
        g_valid = g.dropna(subset=["x", "y"])
        if g_valid.empty: continue

        if len(g_valid) > 5:
            g_valid = g_valid.iloc[:-5]

        xy_seq = list(zip(g_valid["x"], g_valid["y"]))
        records.append({
            "trial_id": tid,
            "xy": xy_seq
        })

    per_trial_df = pd.DataFrame.from_records(records)
    if "xy" in per_trial_df.columns:
        per_trial_df["xy"] = per_trial_df["xy"].apply(
            lambda pairs: np.asarray(list(pairs)) if pairs is not None else np.empty((0, 2))
        )

    # --- 7. Plotting Preparation ---
    FS = 30.0
    DT = 1.0 / FS
    
    X_SCALE_DEN = (2352 / 2 / 9)
    Y_SCALE_DEN = (1424 / 2 / 5)

    SMOOTH_SAMPLES_RAW = max(1, int(round((400.0 / 1000.0) * FS))) 
    SMOOTH_SAMPLES_05 = max(1, int(round(0.5 * FS)))
    SMOOTH_SAMPLES_10 = max(1, int(round(1.0 * FS)))
    SMOOTH_SAMPLES_20 = max(1, int(round(2.0 * FS)))
    SMOOTH_SAMPLES_50 = max(1, int(round(5.0 * FS))) 

    # --- Load Node List and Build Graph ---
    node_file_path = Path(".src/tools/node_list_new.csv")
    nodes_data = None
    maze_graph = None

    if node_file_path.exists():
        try:
            nodes_df = pd.read_csv(node_file_path, header=None, names=["id", "x", "y"])
            nodes_df["x_scaled"] = nodes_df["x"] / X_SCALE_DEN
            nodes_df["y_scaled"] = nodes_df["y"] / Y_SCALE_DEN
            nodes_data = nodes_df
            print(f"Loaded {len(nodes_df)} nodes from {node_file_path}")
            
            print("Building Hexmaze Graph for pathfinding...")
            maze_graph = build_hexmaze_graph(nodes_df)
            print("Graph built successfully.")
            
        except Exception as e:
            print(f"Warning: Found {node_file_path} but could not parse/build graph: {e}")
    else:
        print(f"Warning: {node_file_path} not found. Nodes and Paths will not be plotted.")

    pdf_path = work_dir / f"{log_file_stem}_analysis_final.pdf"
    print(f"Generating PDF: {pdf_path}")

    agg_data = {'0.5s': [], '1.0s': [], '2.0s': [], '5.0s': []}
    all_trials_speed_raw_list = []
    global_x_scaled = []
    global_y_scaled = []
    global_speed_vals = []
    
    summary_metrics = []
    print(f"Total Trials Processed: {len(per_trial_df)}")
    print(f"Graph Loaded: {maze_graph is not None}")

    with PdfPages(pdf_path) as pdf:
        
        # --- Part 0: Metadata Summary Page ---
        if session_meta:
            fig_meta = plt.figure(figsize=(10, 8))
            ax_meta = fig_meta.add_subplot(111)
            ax_meta.axis('off')
            
            # Format text
            meta_text = "SESSION METADATA SUMMARY\n\n"
            for k, v in session_meta.items():
                val_str = str(v)
                if len(val_str) > 80:
                    val_str = val_str[:80] + "..."
                meta_text += f"{k:<20}: {val_str}\n"
            
            ax_meta.text(0.1, 0.8, meta_text, fontsize=12, family='monospace', va='top')
            pdf.savefig(fig_meta)
            plt.close(fig_meta)
        
        # --- Part A: Per-Trial Pages ---
        for i, row in per_trial_df.iterrows():
            trial_id = row.get("trial_id", "Unknown")
            xy_arr = row["xy"]
            
            if xy_arr.size == 0: continue
            
            # --- Look up trial-specific metadata ---
            current_meta = trial_metadata.get(trial_id, {})
            current_goal_node = current_meta.get('Goal_Node', None)
            current_trial_type = current_meta.get('Trial_Type', 'Unknown')
            meta_start_node = current_meta.get('Start_Node', None)

            x_raw = xy_arr[:, 0]
            y_raw = xy_arr[:, 1]
            
            # 1. CALCULATE SPEED ON RAW DATA (Before modifying path)
            x_calc = x_raw / X_SCALE_DEN
            y_calc = y_raw / Y_SCALE_DEN
            speed = compute_speed_from_xy(x_calc, y_calc, FS)

            # 2. PLOTTING PATH (Always use FULL path)
            x_plot = x_calc.copy()
            y_plot = y_calc.copy()
            
            # Variables for Scoring Logic
            goal_reached_naturally = False
            first_goal_visit_idx = -1
            
            # Find Goal Location if available
            gx_scaled, gy_scaled = None, None
            gx_raw, gy_raw = None, None
            
            if current_goal_node and nodes_data is not None:
                goal_row = nodes_data[nodes_data['id_str'] == current_goal_node]
                if not goal_row.empty:
                    gx_scaled = goal_row.iloc[0]['x_scaled']
                    gy_scaled = goal_row.iloc[0]['y_scaled']
                    gx_raw = goal_row.iloc[0]['x']
                    gy_raw = goal_row.iloc[0]['y']

                    # --- CHECK IF GOAL IS REACHED IN TRAJECTORY ---
                    # Check distance to goal for every point in trajectory
                    # Graph building used threshold 65. Let's use 50 (raw units) as "arrival".
                    dist_to_goal_sq = (x_raw - gx_raw)**2 + (y_raw - gy_raw)**2
                    arrival_indices = np.where(dist_to_goal_sq < (50**2))[0]
                    
                    if len(arrival_indices) > 0:
                        goal_reached_naturally = True
                        first_goal_visit_idx = arrival_indices[0]
            
            # --- Handling the "Force End" Logic ---
            appended_goal = False
            if current_goal_node and not goal_reached_naturally and gx_scaled is not None:
                # Check distance to last point
                last_x, last_y = x_plot[-1], y_plot[-1]
                dist_to_goal = np.sqrt((last_x - gx_scaled)**2 + (last_y - gy_scaled)**2)
                
                # If far from goal, append goal point for visual connection
                if dist_to_goal > 0.5: # Threshold in scaled units
                    x_plot = np.append(x_plot, gx_scaled)
                    y_plot = np.append(y_plot, gy_scaled)
                    appended_goal = True

            # -- Stats for Correlation Plots --
            avg_speed_trial = np.mean(speed) if len(speed) > 0 else 0
            median_speed_trial = np.median(speed) if len(speed) > 0 else 0

            global_x_scaled.append(x_calc) # Store original for aggregates
            global_y_scaled.append(y_calc)
            global_speed_vals.append(speed)

            speed_raw_smooth = moving_average(speed, SMOOTH_SAMPLES_RAW) 
            speed_05 = moving_average(speed, SMOOTH_SAMPLES_05)
            speed_10 = moving_average(speed, SMOOTH_SAMPLES_10)
            speed_20 = moving_average(speed, SMOOTH_SAMPLES_20)
            speed_50 = moving_average(speed, SMOOTH_SAMPLES_50) 

            all_trials_speed_raw_list.append((trial_id, speed_raw_smooth))
            
            if len(speed) > 1:
                common_len = 100
                norm_time_common = np.linspace(0, 1, common_len)
                curr_norm_time = np.linspace(0, 1, len(speed))
                agg_data['0.5s'].append(np.interp(norm_time_common, curr_norm_time, speed_05))
                agg_data['1.0s'].append(np.interp(norm_time_common, curr_norm_time, speed_10))
                agg_data['2.0s'].append(np.interp(norm_time_common, curr_norm_time, speed_20))
                agg_data['5.0s'].append(np.interp(norm_time_common, curr_norm_time, speed_50))

            time_vec = np.arange(len(speed)) * DT
            norm_time_vec = np.linspace(0, 1, len(speed)) if len(speed) > 1 else np.array([0.0])

            speed_vis = speed_raw_smooth.copy()
            speed_vis[speed_vis > 1] = 1 
            
            if appended_goal:
                speed_vis = np.append(speed_vis, 0.0)

            bins_x, bins_y = 50, 30
            range_map = [[0, 9], [0, 5]]
            H, _, _ = np.histogram2d(x_calc, y_calc, bins=[bins_x, bins_y], range=range_map)
            H = H.T 
            H_rel = H / (H.sum() if H.sum() > 0 else 1)
            H_sec = H * DT 
            H_rel_masked = np.ma.masked_where(H == 0, H_rel)
            H_sec_masked = np.ma.masked_where(H == 0, H_sec)

            # --- Calculate Scores (Physical & Hops) ---
            
            # Logic: If goal reached naturally, score is calculated based on path TO goal only.
            # If not reached, score is based on total path + distance to goal (if appended).
            
            # 1. Physical Dist Calculation for SCORING
            if goal_reached_naturally and first_goal_visit_idx > 0:
                # Truncate path for scoring only
                actual_dist_score_basis = compute_path_length(x_raw[:first_goal_visit_idx+1], y_raw[:first_goal_visit_idx+1])
                score_note = "(Start->FirstGoal)"
            else:
                actual_dist_score_basis = compute_path_length(x_raw, y_raw)
                if appended_goal:
                     added_dist = np.sqrt((x_raw[-1] - gx_raw)**2 + (y_raw[-1] - gy_raw)**2)
                     actual_dist_score_basis += added_dist
                score_note = "(Full Path)"

            # 2. Hops Calculation for SCORING
            actual_hops_score_basis = 0
            seq_str = trial_node_sequences.get(trial_id, "")
            start_node = None
            passed_nodes_list = []
            
            if seq_str:
                try:
                    passed_nodes_list = [t.strip() for t in seq_str.split(',') if t.strip()]
                    if passed_nodes_list:
                        # Use metadata start_node if available, else first item in sequence
                        start_node = meta_start_node if meta_start_node else passed_nodes_list[0]
                        
                        # Logic: Find index of Goal Node in sequence
                        if current_goal_node in passed_nodes_list:
                            # If goal is in list, count hops to FIRST occurrence
                            idx_in_seq = passed_nodes_list.index(current_goal_node)
                            actual_hops_score_basis = idx_in_seq # hops = index (0->0, 0->1 is 1 hop)
                            # Update note if both agree
                            if "Start->FirstGoal" not in score_note: score_note = "(Start->GoalNode)"
                        else:
                            # Goal not in list, use full length
                            actual_hops_score_basis = len(passed_nodes_list) - 1
                            if actual_hops_score_basis < 0: actual_hops_score_basis = 0
                except:
                    pass
            
            optimal_dist_raw = 0.0
            dist_score_msg = "N/A"
            dist_score_val = np.nan 
            
            optimal_hops = 0
            hops_score_msg = "N/A"
            hops_score_val = np.nan 

            # SET TARGET NODE TO CURRENT TRIAL GOAL NODE
            end_node = current_goal_node

            if maze_graph and start_node and end_node:
                try:
                    optimal_dist_raw = nx.shortest_path_length(maze_graph, source=start_node, target=end_node, weight='weight')
                    if actual_dist_score_basis > 0:
                        dist_score_val = np.log(optimal_dist_raw / actual_dist_score_basis)
                        dist_score_msg = f"{dist_score_val:.3f}"
                    else:
                        dist_score_msg = "Err"
                    
                    optimal_hops = nx.shortest_path_length(maze_graph, source=start_node, target=end_node, weight=None)
                    if actual_hops_score_basis > 0:
                        hops_score_val = np.log(optimal_hops / actual_hops_score_basis)
                        hops_score_msg = f"{hops_score_val:.3f}"
                    elif actual_hops_score_basis == 0 and optimal_hops == 0:
                        hops_score_val = 0.000
                        hops_score_msg = "0.000"
                    else:
                        hops_score_msg = "Err"
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    dist_score_msg = "No Path"
                    hops_score_msg = "No Path"
            
            summary_metrics.append({
                'trial_id': trial_id,
                'avg_speed': avg_speed_trial,
                'median_speed': median_speed_trial,
                'dist_log_score': dist_score_val,
                'hops_log_score': hops_score_val
            })

            # --- Setup Figure ---
            fig = plt.figure(figsize=(12, 23)) 
            gs = fig.add_gridspec(6, 2, height_ratios=[0.3, 1, 1, 1, 0.6, 0.6]) 

            ax_text = fig.add_subplot(gs[0, :])
            ax_text.axis('off')
            ax0 = fig.add_subplot(gs[1, 0])
            ax1 = fig.add_subplot(gs[1, 1])
            ax_path_dist = fig.add_subplot(gs[2, 0])
            ax_path_hops = fig.add_subplot(gs[2, 1])
            ax2 = fig.add_subplot(gs[3, 0])
            ax3 = fig.add_subplot(gs[3, 1])
            ax4 = fig.add_subplot(gs[4, :])
            ax5 = fig.add_subplot(gs[5, :])

            wrapped_seq = textwrap.fill(seq_str, width=110)
            summary_txt = (
                f"Trial {trial_id} Summary: {score_note}\n"
                f"Trial Type: {current_trial_type} | Target Goal: {end_node if end_node else 'Unknown'}\n"
                f"Passed Nodes: {wrapped_seq}\n"
                f"Avg Speed: {avg_speed_trial:.3f} m/s | Median Speed: {median_speed_trial:.3f} m/s\n"
                f"--------------------------------------------------\n"
                f"Metric            | Act(Score)| Optimal   | Score [ln(Opt/Act)]\n"
                f"--------------------------------------------------\n"
                f"Physical Distance | {actual_dist_score_basis:8.1f}  | {optimal_dist_raw:8.1f}  | {dist_score_msg}\n"
                f"Topological Hops  | {actual_hops_score_basis:8d}  | {optimal_hops:8d}  | {hops_score_msg}"
            )
            ax_text.text(0.5, 0.5, summary_txt, 
                         fontsize=11, verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
                         bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="black", alpha=1.0))

            # OPTIMIZATION: rasterized=True keeps vector text but rasterizes heavy dots
            sc = ax0.scatter(x_plot, y_plot, c=speed_vis, s=10, vmax=1, cmap='hot', rasterized=True)
            fig.colorbar(sc, ax=ax0, fraction=0.025, pad=0.02, label="Speed (m/s)")
            ax0.set_title(f"Trial {trial_id}: Speed Track")

            if len(x_plot) >= 2:
                pts = np.column_stack([x_plot, y_plot])
                segments = np.stack([pts[:-1], pts[1:]], axis=1)
                t_arr = np.linspace(0.0, 1.0, len(pts) - 1)
                # OPTIMIZATION: rasterized=True
                lc = LineCollection(segments, cmap="cool", norm=mpl.colors.Normalize(0, 1), linewidths=1.5, rasterized=True)
                lc.set_array(t_arr)
                ax1.add_collection(lc)
                fig.colorbar(lc, ax=ax1, fraction=0.025, pad=0.02, label="Time (Norm)")
            ax1.set_title(f"Trial {trial_id}: Actual Path")

            if maze_graph and start_node and end_node:
                all_segs_dist, label_dist, _ = get_all_shortest_paths_plot_data(maze_graph, start_node, end_node, 'weight')
                if all_segs_dist:
                    for path_segs in all_segs_dist:
                        for p1, p2 in path_segs:
                            sx1, sy1 = p1[0] / X_SCALE_DEN, p1[1] / Y_SCALE_DEN
                            sx2, sy2 = p2[0] / X_SCALE_DEN, p2[1] / Y_SCALE_DEN
                            ax_path_dist.plot([sx1, sx2], [sy1, sy2], color='blue', linewidth=3, alpha=0.4)
                    sp = nx.get_node_attributes(maze_graph, 'pos')[start_node]
                    ep = nx.get_node_attributes(maze_graph, 'pos')[end_node]
                    ax_path_dist.scatter(sp[0]/X_SCALE_DEN, sp[1]/Y_SCALE_DEN, c='green', s=150, zorder=10)
                    ax_path_dist.scatter(ep[0]/X_SCALE_DEN, ep[1]/Y_SCALE_DEN, c='red', s=150, zorder=10)
                    ax_path_dist.set_title(f"All Shortest Paths (Physical)\n{label_dist}")
                else:
                    ax_path_dist.text(4.5, 2.5, f"Path Not Found: {label_dist}", ha='center')
                    ax_path_dist.set_title("All Shortest Paths (Physical)")

                all_segs_hops, label_hops, _ = get_all_shortest_paths_plot_data(maze_graph, start_node, end_node, None)
                if all_segs_hops:
                    for path_segs in all_segs_hops:
                        for p1, p2 in path_segs:
                            sx1, sy1 = p1[0] / X_SCALE_DEN, p1[1] / Y_SCALE_DEN
                            sx2, sy2 = p2[0] / X_SCALE_DEN, p2[1] / Y_SCALE_DEN
                            ax_path_hops.plot([sx1, sx2], [sy1, sy2], color='purple', linewidth=3, alpha=0.4)
                    sp = nx.get_node_attributes(maze_graph, 'pos')[start_node]
                    ep = nx.get_node_attributes(maze_graph, 'pos')[end_node]
                    ax_path_hops.scatter(sp[0]/X_SCALE_DEN, sp[1]/Y_SCALE_DEN, c='green', s=150, zorder=10)
                    ax_path_hops.scatter(ep[0]/X_SCALE_DEN, ep[1]/Y_SCALE_DEN, c='red', s=150, zorder=10)
                    ax_path_hops.set_title(f"All Shortest Paths (Topological)\n{label_hops}")
                else:
                    ax_path_hops.text(4.5, 2.5, f"Path Not Found: {label_hops}", ha='center')
                    ax_path_hops.set_title("All Shortest Paths (Topological)")
            else:
                msg = "Graph missing" if not maze_graph else "Start/End not found"
                ax_path_dist.text(4.5, 2.5, msg, ha='center')
                ax_path_hops.text(4.5, 2.5, msg, ha='center')
                ax_path_dist.set_title("Shortest Path (Physical)")
                ax_path_hops.set_title("Shortest Path (Topological)")

            im3 = ax2.imshow(H_rel_masked, interpolation='nearest', origin='upper', 
                             extent=[0, 9, 5, 0], cmap='jet', aspect='auto')
            fig.colorbar(im3, ax=ax2, fraction=0.025, pad=0.02, label="Fraction")
            ax2.set_title(f"Trial {trial_id}: Relative Occupancy")

            im4 = ax3.imshow(H_sec_masked, interpolation='nearest', origin='upper', 
                             extent=[0, 9, 5, 0], cmap='jet', aspect='auto', vmax=5.0)
            fig.colorbar(im4, ax=ax3, fraction=0.025, pad=0.02, label="Seconds")
            ax3.set_title(f"Trial {trial_id}: Absolute Occupancy")

            if nodes_data is not None:
                spatial_axes = [ax0, ax1, ax_path_dist, ax_path_hops, ax2, ax3]
                for sax in spatial_axes:
                    sax.scatter(nodes_data["x_scaled"], nodes_data["y_scaled"], 
                                s=100, facecolors='none', edgecolors='grey', 
                                linewidths=2, alpha=0.3, zorder=5)
                    for _, nrow in nodes_data.iterrows():
                        sax.text(nrow["x_scaled"] + 0.15, nrow["y_scaled"], 
                                 str(int(nrow["id"])), 
                                 color='grey', fontsize=4, 
                                 va='center', zorder=6)

            ax4.plot(time_vec, speed_raw_smooth, color='gray', alpha=0.3, label='Raw (0.4s)', linewidth=1)
            ax4.plot(time_vec, speed_05, color='#1f77b4', linewidth=1.5, label='0.5s')
            ax4.plot(time_vec, speed_10, color='#ff7f0e', linewidth=1.5, label='1.0s')
            ax4.plot(time_vec, speed_20, color='#2ca02c', linewidth=1.5, label='2.0s')
            ax4.plot(time_vec, speed_50, color='#d62728', linewidth=2.0, label='5.0s')
            ax4.set_title(f"Trial {trial_id}: Speed vs Time (Seconds)")
            ax4.set_xlabel("Time (s)")
            ax4.set_ylabel("Speed (m/s)")
            ax4.grid(True, alpha=0.4)
            ax4.legend(loc='upper right', fontsize='small', ncol=5)
            ax4.set_xlim(left=0, right=max(time_vec) if len(time_vec) > 0 else 1)

            ax5.plot(norm_time_vec, speed_raw_smooth, color='gray', alpha=0.3, linewidth=1)
            ax5.plot(norm_time_vec, speed_05, color='#1f77b4', linewidth=1.5, label='0.5s')
            ax5.plot(norm_time_vec, speed_10, color='#ff7f0e', linewidth=1.5, label='1.0s')
            ax5.plot(norm_time_vec, speed_20, color='#2ca02c', linewidth=1.5, label='2.0s')
            ax5.plot(norm_time_vec, speed_50, color='#d62728', linewidth=2.0, label='5.0s')
            ax5.set_title(f"Trial {trial_id}: Speed vs Time (Normalized)")
            ax5.set_xlabel("Normalized Time (0 to 1)")
            ax5.set_ylabel("Speed (m/s)")
            ax5.grid(True, alpha=0.4)
            ax5.set_xlim(0, 1)

            spatial_axes = [ax0, ax1, ax_path_dist, ax_path_hops, ax2, ax3]
            for ax in spatial_axes:
                ax.set_aspect("equal", adjustable="box")
                ax.grid(True, alpha=0.3)
                ax.set_xlim(0, 9)
                ax.set_ylim(5, 0)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Part B: Summary Histograms ---
        if all_trials_speed_raw_list:
            n_trials = len(all_trials_speed_raw_list)
            cols = 4
            rows = math.ceil(n_trials / cols)
            fig_hist, axes_hist = plt.subplots(rows, cols, figsize=(15, 3 * rows), constrained_layout=True)
            axes_flat = axes_hist.flatten() if n_trials > 1 else [axes_hist]
            common_bins = np.linspace(0, 1.5, 31) 
            for idx, (tid, spd) in enumerate(all_trials_speed_raw_list):
                ax = axes_flat[idx]
                ax.hist(spd, bins=common_bins, color='skyblue', edgecolor='black', alpha=0.7)
                ax.set_title(f"Trial {tid}")
                ax.grid(True, alpha=0.3)
            for i in range(n_trials, len(axes_flat)): axes_flat[i].axis('off')
            fig_hist.suptitle("Individual Trial Speed Distributions (Raw/0.4s)", fontsize=16)
            pdf.savefig(fig_hist)
            plt.close(fig_hist)

        # --- Summary Part: Aggregate Figures ---
        if all_trials_speed_raw_list:
            n_trials = len(all_trials_speed_raw_list)
            
            # 1. Combined Distribution
            fig_comb, ax_comb = plt.subplots(figsize=(12, 8))
            cmap = mpl.colormaps['tab20']
            colors = [cmap(i % 20) for i in range(n_trials)]
            common_bins = np.linspace(0, 1.5, 51)
            
            for idx, (tid, spd) in enumerate(all_trials_speed_raw_list):
                ax_comb.hist(spd, bins=common_bins, density=True, histtype='step', 
                             linewidth=1.5, color=colors[idx], label=f'Trial {tid}', alpha=0.7)
            
            all_speeds = np.concatenate([s for _, s in all_trials_speed_raw_list])
            ax_comb.hist(all_speeds, bins=common_bins, density=True, histtype='step',
                         linewidth=3, color='black', label='Aggregate', linestyle='-')
            
            ax_comb.set_title("Combined Speed Probability Distributions")
            ax_comb.set_xlabel("Speed (m/s)")
            ax_comb.set_ylabel("Probability Density")
            if n_trials <= 20:
                ax_comb.legend(loc='upper right', ncol=2, fontsize='small')
            ax_comb.grid(True, alpha=0.3)
            
            pdf.savefig(fig_comb)
            plt.close(fig_comb)

        # --- Part C.2: Thresholded Speed Distribution ---
        if all_trials_speed_raw_list:
            fig_comb_th, ax_comb_th = plt.subplots(figsize=(12, 8))
            common_bins = np.linspace(0, 1.5, 51)
            all_speeds_th = []

            for idx, (tid, spd) in enumerate(all_trials_speed_raw_list):
                spd_th = spd[spd > 0.05]
                if len(spd_th) > 0:
                    ax_comb_th.hist(spd_th, bins=common_bins, density=True, histtype='step', 
                                   linewidth=1.5, color=colors[idx], label=f'Trial {tid}', alpha=0.7)
                    all_speeds_th.append(spd_th)
            
            if all_speeds_th:
                all_speeds_concat = np.concatenate(all_speeds_th)
                ax_comb_th.hist(all_speeds_concat, bins=common_bins, density=True, histtype='step',
                             linewidth=3, color='black', label='Aggregate', linestyle='-')

            ax_comb_th.set_title("Combined Speed Probability Distributions (Speed > 0.05 m/s)")
            ax_comb_th.set_xlabel("Speed (m/s)")
            ax_comb_th.set_ylabel("Probability Density")
            if n_trials <= 20:
                ax_comb_th.legend(loc='upper right', ncol=2, fontsize='small')
            ax_comb_th.grid(True, alpha=0.3)
            
            pdf.savefig(fig_comb_th)
            plt.close(fig_comb_th)

        # --- Part D: Aggregate Mean Speed vs Normalized Time ---
        if len(agg_data['0.5s']) > 0:
            fig_agg, ax_agg = plt.subplots(figsize=(12, 8))
            common_time_axis = np.linspace(0, 1, 100)
            agg_colors = {'0.5s': '#1f77b4', '1.0s': '#ff7f0e', '2.0s': '#2ca02c', '5.0s': '#d62728'}
            
            for label, data_list in agg_data.items():
                if not data_list: continue
                stack = np.vstack(data_list)
                mean_curve = np.mean(stack, axis=0)
                sem_curve = np.std(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])
                c = agg_colors.get(label, 'black')
                ax_agg.plot(common_time_axis, mean_curve, color=c, linewidth=2.5, label=f'Mean ({label})')
                ax_agg.fill_between(common_time_axis, mean_curve - sem_curve, mean_curve + sem_curve, color=c, alpha=0.15)

            ax_agg.set_title(f"Aggregate Speed vs Normalized Time (N={len(agg_data['0.5s'])} Trials)", fontsize=16)
            ax_agg.set_xlabel("Normalized Time (Start -> End)", fontsize=12)
            ax_agg.set_ylabel("Speed (m/s)", fontsize=12)
            ax_agg.set_xlim(0, 1)
            ax_agg.grid(True, which='both', linestyle='--', alpha=0.5)
            ax_agg.legend(fontsize=12)
            pdf.savefig(fig_agg)
            plt.close(fig_agg)

        # --- Part E: AGGREGATE LINE PLOT (Trajectories over Time) ---
        if len(global_x_scaled) > 0:
            fig_traj, ax_traj = plt.subplots(figsize=(12, 10))
            all_segments = []
            all_times = []
            
            for x_i, y_i in zip(global_x_scaled, global_y_scaled):
                if len(x_i) < 2: continue
                pts = np.column_stack([x_i, y_i])
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                t_i = np.linspace(0.0, 1.0, len(pts) - 1)
                all_segments.append(segs)
                all_times.append(t_i)
                
            if all_segments:
                combined_segments = np.concatenate(all_segments, axis=0)
                combined_times = np.concatenate(all_times)
                # RASTERIZED FOR SMALLER PDF
                lc_agg = LineCollection(combined_segments, cmap='cool', 
                                        norm=mpl.colors.Normalize(0, 1), 
                                        linewidths=1.5, alpha=0.3, rasterized=True) 
                lc_agg.set_array(combined_times)
                ax_traj.add_collection(lc_agg)
                cbar_traj = fig_traj.colorbar(lc_agg, ax=ax_traj, fraction=0.046, pad=0.04)
                cbar_traj.set_label("Normalized Time (Start->End)", rotation=270, labelpad=20)
                
                ax_traj.set_xlim(0, 9)
                ax_traj.set_ylim(5, 0)
                ax_traj.set_aspect('equal')
                ax_traj.set_title(f"Aggregate Trajectories (All Trials)", fontsize=16)
                
                if nodes_data is not None:
                    ax_traj.scatter(nodes_data["x_scaled"], nodes_data["y_scaled"], 
                                      s=100, facecolors='none', edgecolors='black', 
                                      linewidths=2, alpha=0.7, zorder=20)
                    for _, nrow in nodes_data.iterrows():
                        ax_traj.text(nrow["x_scaled"] + 0.15, nrow["y_scaled"], 
                                      str(int(nrow["id"])), 
                                      color='black', fontsize=8, fontweight='bold',
                                      va='center', zorder=21)

            pdf.savefig(fig_traj)
            plt.close(fig_traj)

        # --- Part F: OVERLAYS ---
        if len(global_x_scaled) > 0:
            all_x = np.concatenate(global_x_scaled)
            all_y = np.concatenate(global_y_scaled)
            all_spd = np.concatenate(global_speed_vals)

            bins_x, bins_y = 90, 50
            x_edges = np.linspace(0, 9, bins_x + 1)
            y_edges = np.linspace(0, 5, bins_y + 1)
            
            H_count, _, _ = np.histogram2d(all_x, all_y, bins=[x_edges, y_edges])
            n_trials = len(global_x_scaled)
            H_occupancy_avg = (H_count * DT) / n_trials 
            
            H_speed_sum, _, _ = np.histogram2d(all_x, all_y, bins=[x_edges, y_edges], weights=all_spd)
            with np.errstate(divide='ignore', invalid='ignore'):
                H_speed_avg = H_speed_sum / H_count
                H_speed_avg = np.nan_to_num(H_speed_avg, nan=0.0)

            seg_list_occ = []
            val_list_occ = []
            seg_list_spd = []
            val_list_spd = []

            for x_i, y_i in zip(global_x_scaled, global_y_scaled):
                if len(x_i) < 2: continue
                pts = np.column_stack([x_i, y_i])
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                ix = np.searchsorted(x_edges, x_i[:-1]) - 1
                iy = np.searchsorted(y_edges, y_i[:-1]) - 1
                ix = np.clip(ix, 0, bins_x - 1)
                iy = np.clip(iy, 0, bins_y - 1)
                vals_occ = H_occupancy_avg[ix, iy]
                vals_spd = H_speed_avg[ix, iy]
                seg_list_occ.append(segs)
                val_list_occ.append(vals_occ)
                seg_list_spd.append(segs)
                val_list_spd.append(vals_spd)

            if seg_list_occ:
                all_segs_occ = np.concatenate(seg_list_occ, axis=0)
                all_vals_occ = np.concatenate(val_list_occ)
                all_segs_spd = np.concatenate(seg_list_spd, axis=0)
                all_vals_spd = np.concatenate(val_list_spd)

                # Overlay Occupancy (Rasterized)
                fig_ov_occ, ax_ov_occ = plt.subplots(figsize=(12, 10))
                lc_occ = LineCollection(all_segs_occ, cmap='jet', 
                                        norm=mpl.colors.Normalize(vmin=0, vmax=np.max(all_vals_occ)), 
                                        linewidths=1.5, alpha=0.6, rasterized=True)
                lc_occ.set_array(all_vals_occ)
                ax_ov_occ.add_collection(lc_occ)
                cbar_occ = fig_ov_occ.colorbar(lc_occ, ax=ax_ov_occ, fraction=0.046, pad=0.04)
                cbar_occ.set_label("Average Seconds per Trial", rotation=270, labelpad=20)
                ax_ov_occ.set_title("Overlay: Paths colored by Average Spatial Occupancy", fontsize=16)
                ax_ov_occ.set_xlim(0, 9)
                ax_ov_occ.set_ylim(5, 0)
                ax_ov_occ.set_aspect('equal')
                if nodes_data is not None:
                    ax_ov_occ.scatter(nodes_data["x_scaled"], nodes_data["y_scaled"], 
                                    s=100, facecolors='none', edgecolors='black', 
                                    linewidths=2, alpha=0.7, zorder=20)
                pdf.savefig(fig_ov_occ)
                plt.close(fig_ov_occ)

                # Overlay Speed (Rasterized)
                fig_ov_spd, ax_ov_spd = plt.subplots(figsize=(12, 10))
                lc_spd = LineCollection(all_segs_spd, cmap='hot', 
                                        norm=mpl.colors.Normalize(vmin=0, vmax=0.8), 
                                        linewidths=1.5, alpha=0.6, rasterized=True)
                lc_spd.set_array(all_vals_spd)
                ax_ov_spd.add_collection(lc_spd)
                cbar_spd = fig_ov_spd.colorbar(lc_spd, ax=ax_ov_spd, fraction=0.046, pad=0.04)
                cbar_spd.set_label("Average Speed (m/s)", rotation=270, labelpad=20)
                ax_ov_spd.set_title("Overlay: Paths colored by Spatial Average Speed", fontsize=16)
                ax_ov_spd.set_xlim(0, 9)
                ax_ov_spd.set_ylim(5, 0)
                ax_ov_spd.set_aspect('equal')
                if nodes_data is not None:
                    ax_ov_spd.scatter(nodes_data["x_scaled"], nodes_data["y_scaled"], 
                                    s=100, facecolors='none', edgecolors='black', 
                                    linewidths=2, alpha=0.7, zorder=20)
                pdf.savefig(fig_ov_spd)
                plt.close(fig_ov_spd)

        # --- Part G: Cumulative Occupancy Heatmap ---
        if len(global_x_scaled) > 0:
            fig_cum, ax_cum = plt.subplots(figsize=(12, 10))
            all_x = np.concatenate(global_x_scaled)
            all_y = np.concatenate(global_y_scaled)
            bins_x, bins_y = 50, 30
            range_map = [[0, 9], [0, 5]]
            H, _, _ = np.histogram2d(all_x, all_y, bins=[bins_x, bins_y], range=range_map)
            H = H.T 
            H_sec = H * DT 
            H_masked = np.ma.masked_where(H == 0, H_sec)
            im_cum = ax_cum.imshow(H_masked, interpolation='nearest', origin='upper', 
                                   extent=[0, 9, 5, 0], cmap='jet', aspect='equal')
            cbar_cum = fig_cum.colorbar(im_cum, ax=ax_cum, fraction=0.046, pad=0.04)
            cbar_cum.set_label("Cumulative Time (Total Seconds)", rotation=270, labelpad=20)
            ax_cum.set_title("Cumulative Occupancy Map (All Trials)", fontsize=16)
            if nodes_data is not None:
                ax_cum.scatter(nodes_data["x_scaled"], nodes_data["y_scaled"], 
                                  s=100, facecolors='none', edgecolors='black', 
                                  linewidths=2, alpha=0.7, zorder=20)
                for _, nrow in nodes_data.iterrows():
                    ax_cum.text(nrow["x_scaled"] + 0.15, nrow["y_scaled"], 
                                  str(int(nrow["id"])), 
                                  color='black', fontsize=8, fontweight='bold',
                                  va='center', zorder=21)
            pdf.savefig(fig_cum)
            plt.close(fig_cum)

        # --- Part H: Aggregate Trajectories Colored by Trial ID (Rasterized) ---
        if len(global_x_scaled) > 0:
            seg_list_trials = []
            val_list_trials = []
            for i, row in per_trial_df.iterrows():
                tid = row.get("trial_id")
                xy_arr = row["xy"]
                if xy_arr.size < 2: continue
                x = xy_arr[:, 0] / X_SCALE_DEN
                y = (xy_arr[:, 1] / Y_SCALE_DEN) + i*0.005 
                pts = np.column_stack([x, y])
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                t_vals = np.full(len(segs), tid)
                seg_list_trials.append(segs)
                val_list_trials.append(t_vals)
                
            if seg_list_trials:
                combined_segs_trial = np.concatenate(seg_list_trials, axis=0)
                combined_vals_trial = np.concatenate(val_list_trials)
                fig_idx, ax_idx = plt.subplots(figsize=(12, 10))
                lc_idx = LineCollection(combined_segs_trial, cmap='cool', 
                                        norm=plt.Normalize(vmin=combined_vals_trial.min(), vmax=combined_vals_trial.max()),
                                        linewidths=1.5, alpha=0.6, rasterized=True)
                lc_idx.set_array(combined_vals_trial)
                ax_idx.add_collection(lc_idx)
                cbar_idx = fig_idx.colorbar(lc_idx, ax=ax_idx, fraction=0.046, pad=0.04)
                cbar_idx.set_label("Trial Number", rotation=270, labelpad=20)
                if len(per_trial_df) <= 20:
                    cbar_idx.set_ticks(sorted(per_trial_df['trial_id'].unique()))
                ax_idx.set_title("Aggregate Trajectories Colored by Trial Number", fontsize=16)
                ax_idx.set_xlim(0, 9)
                ax_idx.set_ylim(5, 0)
                ax_idx.set_aspect('equal')
                if nodes_data is not None:
                     ax_idx.scatter(nodes_data["x_scaled"], nodes_data["y_scaled"], 
                                    s=100, facecolors='none', edgecolors='black', 
                                    linewidths=2, alpha=0.7, zorder=20)
                pdf.savefig(fig_idx)
                plt.close(fig_idx)

        # --- Part I: Speed vs Quality Metric Correlations (FIXED) ---
        if summary_metrics:
            sum_df = pd.DataFrame(summary_metrics)
            
            # Debug: See what we actually have
            print(f"Plotting Correlations for {len(sum_df)} trials...")
            
            fig_corr, axes_corr = plt.subplots(2, 2, figsize=(15, 12))
            fig_corr.suptitle(f"Speed vs Log Quality Metric Correlations (N={len(sum_df)})", fontsize=16)

            plots_config = [
                ('avg_speed', 'dist_log_score', 'Mean Speed vs Physical Score', axes_corr[0, 0]),
                ('avg_speed', 'hops_log_score', 'Mean Speed vs Topological Hops Score', axes_corr[0, 1]),
                ('median_speed', 'dist_log_score', 'Median Speed vs Physical Score', axes_corr[1, 0]),
                ('median_speed', 'hops_log_score', 'Median Speed vs Topological Hops Score', axes_corr[1, 1])
            ]

            for spd_col, score_col, title_str, ax in plots_config:
                # Get valid data for this specific pair
                plot_data = sum_df[[spd_col, score_col]].dropna()
                
                if not plot_data.empty:
                    x = plot_data[score_col]
                    y = plot_data[spd_col]
                    
                    # 1. ALWAYS Plot the points (Scatter)
                    ax.scatter(x, y, alpha=0.7, edgecolors='b', s=80, c='skyblue', label='Trials')
                    
                    # 2. ONLY calculate regression if we have 2+ points
                    if len(plot_data) > 1:
                        try:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                            # Create line points
                            line_x = np.linspace(x.min(), x.max(), 100)
                            line_y = slope * line_x + intercept
                            ax.plot(line_x, line_y, 'r--', linewidth=2, label=f'r={r_value:.3f}')
                        except Exception as e:
                            print(f"Could not fit line for {title_str}: {e}")

                    ax.set_title(title_str)
                    ax.set_xlabel(f"Log Score: {score_col} [ln(Opt/Act)]")
                    ax.set_ylabel(f"Speed: {spd_col} (m/s)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, "No Valid Data", ha='center', va='center', color='red')
                    ax.set_title(title_str)

            pdf.savefig(fig_corr)
            plt.close(fig_corr)

    # --- Part J: Full Connected Hexmaze Graph (Reference) ---
        if maze_graph is not None:
            fig_graph = plt.figure(figsize=(12, 10))
            ax_graph = fig_graph.add_subplot(111)

            # 1. Extract Scaled Positions
            # The graph was built with RAW coords in 'pos'. We need to scale them for the plot.
            node_pos_scaled = {}
            raw_pos = nx.get_node_attributes(maze_graph, 'pos')

            for node_id, (rx, ry) in raw_pos.items():
                node_pos_scaled[node_id] = (rx / X_SCALE_DEN, ry / Y_SCALE_DEN)

            # 2. Draw Edges
            for u, v in maze_graph.edges():
                if u in node_pos_scaled and v in node_pos_scaled:
                    p1 = node_pos_scaled[u]
                    p2 = node_pos_scaled[v]
                    ax_graph.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                                  color='#333333', linewidth=1.5, alpha=0.5, zorder=1)

            # 3. Draw Nodes
            x_vals = [p[0] for p in node_pos_scaled.values()]
            y_vals = [p[1] for p in node_pos_scaled.values()]

            ax_graph.scatter(x_vals, y_vals, s=450, c='white', edgecolors='black', linewidth=1.5, zorder=2)

            # 4. Draw Labels
            for node_id, (nx_val, ny_val) in node_pos_scaled.items():
                ax_graph.text(nx_val, ny_val, node_id, 
                              ha='center', va='center', fontsize=6, fontweight='bold', zorder=3)

            # 5. Styling
            ax_graph.set_xlim(0, 9)
            ax_graph.set_ylim(5, 0) # Maintain inverted Y-axis to match other plots
            ax_graph.set_aspect('equal')
            ax_graph.set_title("Hexmaze Connectivity Graph (Nodes & Edges)", fontsize=18)
            ax_graph.axis('off') # Clean look for the map
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig_graph)
            plt.close(fig_graph)

        
    print(f"Done. PDF saved to {pdf_path}")