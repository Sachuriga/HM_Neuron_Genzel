# HM Tracker 2025

A batch-processing pipeline for neuroscience experiments — integrates video-based animal tracking (YOLOv11), behavioral node analysis, neural spike sorting (Kilosort4), LFP extraction, and LED-based synchronization into a single orchestrated workflow.

> **Tracker attribution** — `src/TrackerYolov11.py` is based on and substantially modified from [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT). See [Tracker — Modifications from Original](#tracker--modifications-from-original) for a summary of changes.

---

## Overview

The pipeline takes raw Trodes recordings (`.rec`) and multi-camera video, then runs a configurable sequence of steps to produce:

- Annotated tracking video + per-frame position CSV (via YOLOv11)
- Per-trial metrics written back into `RecordingMeta.xlsx`
- LED-synchronized behavioral timestamps (via ICA)
- Behavioral metrics computed from trial node sequences (via hex maze analysis)
- Spike-sorted neural data (via Kilosort4 / SpikeInterface)
- LFP traces (1 kHz, 500 Hz low-pass)
- DeepLabCut-ready video export
- GPU-compressed output video

---

## Directory Layout

```
HM_Tracker_2025/
├── src/
│   ├── TrackerYolov11.py            # Main YOLOv11 tracker
│   ├── Video_LED_Sync_using_ICA.py  # LED sync via ICA
│   ├── join_views.py                # Multi-camera stitching
│   ├── plot_trials.py               # Trial-level plotting
│   ├── sorter/
│   │   ├── sorting.py               # Kilosort4 spike sorting
│   │   └── export_lfp.py            # LFP extraction & export
│   ├── node_analysis/
│   │   ├── hex_maze_analysis.py     # Hex maze behavioral metrics
│   │   └── README.md                # Column-by-column calculation reference
│   └── dlc/
│       └── tracking_eyes.py         # DeepLabCut eye-tracking export
├── runner_windows.bat               # Main Windows orchestrator
├── hm_tracker_paths.example.txt    # Config template
├── RecordingMeta.xlsx               # Per-session metadata
├── reproduce.yml                    # Conda environment spec
└── requirements.txt                 # pip dependencies
```

---

## Setup

### 1. Environment

```bash
conda env create -f reproduce.yml
conda activate si_trodes
pip install -r requirements.txt
```

> Requires CUDA-capable GPU. PyTorch is installed with CUDA 12.8 support (`torch==2.10.0+cu128`).

### 2. External tools

| Tool | Purpose |
|---|---|
| [Trodes](https://spikegadgets.com/trodes/) | `.rec` file export (DIO, raw, LFP) |
| [FFmpeg](https://ffmpeg.org/) | GPU-accelerated video compression |
| YOLOv11 weights | Object detection model (`.pt` file) |

### 3. Path config

Copy the template and edit it:

```bash
cp hm_tracker_paths.example.txt ~/Desktop/hm_tracker_paths.txt
```

Then edit `hm_tracker_paths.txt`:

```
FFMPEG_CMD=C:\path\to\ffmpeg.exe
ONNX_WEIGHTS_PATH=C:\path\to\weights.pt
TRODES_EXPORT_CMD=C:\path\to\trodesexport.exe
TRODES_EXPORT_LFP=C:\path\to\exportLFP.exe
LFP_CHANNELS=1 2 3 4 5 6 7 8
```

The runner reads this file at startup — it must exist on the Desktop before running.

---

## Data Layout

The runner expects input/output folder pairs named `ipN` / `opN` inside the target directory:

```
data_root/
├── ip1/    ← raw input (contains .rec + camera videos)
├── op1/    ← processed output
├── ip2/
└── op2/
```

---

## Running the Pipeline

```bat
runner_windows.bat "C:\path\to\data_root"
```

You will be prompted to select which steps to run:

| Key | Step |
|---|---|
| `1` | Trodes DIO/Raw export |
| `e` | Trodes LFP export (1 kHz, 500 Hz LP) |
| `2` | Video–neural sync (LED ICA) |
| `3` | Multi-camera stitching |
| `4` | YOLOv11 tracker |
| `5` | Trial plotting |
| `6` | GPU video compression (h264_nvenc) |
| `7` | Spike sorting (Kilosort4) |
| `8` | LFP extraction |
| `d` | DeepLabCut video export |
| `9` | Cleanup (`.DIO`, `.raw`, timestamp folders) |

Enter any combination, e.g. `1e234` to run steps 1, e, 2, 3, and 4 in order.

The master process launches one worker per `ipN/opN` pair in parallel, throttling based on CPU/GPU/memory thresholds (configurable at the top of `runner_windows.bat`).

---

## Tracker — How It Works

`src/TrackerYolov11.py` is built around a per-frame detect → classify → update-state loop.

### 1. Detection (`cnn()`)

Every frame is passed through YOLOv11 at confidence threshold 0.7 and input size 1280 px. The model outputs three classes:

| Class | Meaning |
|---|---|
| `head` | Rat's head (counted only, not used for position) |
| `rat` | Rat's body (used for position) |
| `researcher` | Human experimenter |

**Rat position selection** — Only `rat` (body) detections are used for position. All body candidates are sorted by confidence and the highest-confidence box is selected. Head detections increment a separate `Rat-head Count` overlay counter but do not influence the centroid. If no body detection fires on a frame, the last known position (`last_rat_pos`) is carried forward so the state machine never stalls.

**Motion-based skip** — Before running YOLO, a frame-difference check is performed. If fewer than 500 pixels changed since the previous frame, YOLO is skipped and the last known bounding boxes are redrawn from cache. This avoids unnecessary inference on static frames without causing the display to flash.

**Researcher selection** — All `researcher` boxes are stored. The one geometrically closest to the rat's active position is used for trial-trigger and force-end logic, handling the common case of multiple people visible in the arena.

---

### 2. Trial State Machine

```
WAITING (start_trial=True)
    │  rat centroid within 60 px of start node
    ▼
ACTIVE (record_detections=True)
    │  goal reached / timeout / force-end condition
    ▼
INTER-TRIAL (record_detections=False)
    │  researcher within 300 px of rat  →  back to WAITING
```

#### Starting a trial
In WAITING state, `find_start()` checks Euclidean distance from the rat centroid to the current trial's start-node pixel coordinate. If distance < 60 px, recording begins: per-frame buffers are reset, `record_detections` is set to `True`, and the trial timer starts.

#### Trial types and end conditions

Each trial has a type read from `RecordingMeta.xlsx`:

| Type | Label | End condition |
|---|---|---|
| 1 | Normal | Rat centroid within 25 px of goal node |
| 2 | NGL (New Goal Location) | Rat visited goal (within 20 px) AND 10 minutes elapsed |
| 3 | Probe | ≥ 2 minutes elapsed AND researcher within 600 px of goal AND rat within 25 px of goal |
| 4–6 | Special NGL variants | Same as NGL; followed by a 10-minute inter-trial lockout |

**Did-Not-Reach override** — If `Did_Not_Reach = 1` for a trial in the metadata, the normal goal-proximity check is replaced: the trial ends when a researcher stays within 60 px of the rat for ≥ 1 second (rat-pickup detection).

**Researcher proximity end** — For trial types 1 and 2, if the closest researcher comes within 240 px of the rat after at least 5 seconds have elapsed, the trial ends immediately.

**Force-end fallbacks** — Two additional guards prevent trials from running indefinitely:
- Closest researcher to **goal** within 50 px for 10 continuous seconds → trial ends.
- Closest researcher to **goal** within 160 px for 30 continuous seconds → trial ends (probe immunity and unnormal-interval rules apply).

**Unnormal intervals** — Specific time windows can be marked immune in the metadata (`Unnormal_Intervals` column, format `trial_num:start_min-end_min`). During these windows, goal-reach and force-end checks are suppressed.

**Inter-trial lockout** — After a type-4/5/6 trial, a 10-minute lockout is enforced. The overlay shows a countdown; the researcher-proximity trigger that advances to WAITING is blocked until the lockout expires.

---

### 3. Node Logging (`annotate_frame()`)

The maze layout is encoded as a node dictionary (`src/tools/mask.py`) mapping node IDs to pixel coordinates. On every active frame, the tracker checks whether the rat centroid falls within 20 px of any node. When it does:

- The node ID is appended to `saved_nodes`.
- A synchronized timestamp from the framewise CSV is attached.
- Consecutive duplicate node visits are de-duplicated before saving.

---

### 4. Velocity Calculation (`calculate_velocity()`)

After each trial ends, segment velocities are computed from the ordered list of (timestamp, node) pairs:

```
speed = segment_length / time_difference   [m/s]
```

Segment lengths are either looked up from a hardcoded bridge table (known physical distances, e.g. 1.72 m for the 121→302 bridge) or default to 0.30 m for standard maze segments.

---

### 5. Outputs

| File | Contents |
|---|---|
| `<date>_Rat<id>.txt` | Per-trial node sequence, segment timing, and velocity summary |
| `<date>_Rat<id>_Coordinates_Full.csv` | Per-frame: `Frame_Index`, `Timestamp` (sync seconds), `Trial_Num`, `Rat_X/Y`, `Researcher_X/Y` |
| `<date>_Rat<id>.mp4` | Annotated video: bounding boxes, centroid trail (pink), crosshair, node markers, start/goal overlays, FPS counter |
| `<RecordingMeta>.xlsx` (copy) | Source metadata with per-trial columns appended (see table below) |

#### RecordingMeta output columns

| Column | Description |
|---|---|
| `paths` | Comma-separated sequence of visited node IDs (e.g. `101,202,303`) |
| `delay` | Trial duration in seconds (start node entry → end condition) |
| `active_time` | Same as `delay` — trial duration in seconds |
| `avg_speed` | Overall trial speed: total path distance ÷ total path time (m/s) |
| `avg_between_node_speed` | Mean of per-segment speeds across all node transitions (m/s) |
| `trial_start_time` | Sync timestamp (seconds) at the moment the rat enters the start node |
| `trial_end_time` | Sync timestamp (seconds) at the moment the trial end condition fires |

`trial_start_time` and `trial_end_time` are populated only when a framewise timestamp CSV is present in the output folder. The following files are checked in order:

1. `<date>_Rat<id>_framewise_ts.csv`
2. `stitched_framewise_seconds.csv`
3. `stitched_framewise_ts.csv`

If none is found, those columns are left empty.

---

### Tracker — Modifications from Original

`src/TrackerYolov11.py` is based on [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT). Key changes from the original:

- Detection backend replaced with YOLOv11 (Ultralytics)
- Rat position uses `rat` (body) class only; `head` class is counted separately and does not influence the tracked centroid
- Extended trial state machine: NGL variants (types 4–6), Did-Not-Reach logic, researcher-proximity trigger and force-end timers, inter-trial lockout after special trials
- Per-trial metrics (`avg_speed`, `avg_between_node_speed`, `active_time`, `trial_start_time`, `trial_end_time`) written back into a copy of `RecordingMeta.xlsx`
- Motion-based YOLO skip with cached bounding-box redraw to prevent display flicker
- Threaded video writer for non-blocking frame output

---

## Node Analysis

`src/node_analysis/hex_maze_analysis.py` reads trial data from `.xlsx` files and computes behavioral metrics, writing results back into a `*_results.xlsx` copy.

### Maze structure

The maze has **96 nodes** across 4 islands (nodes 101–124, 201–224, 301–324, 401–424) plus 2 homeboxes (501, 502). Two graphs are built at startup:

- **Node graph** — full 96-node maze; used for node-level distances and choice analysis.
- **Island graph** — 4-node graph where each node represents an island; used for island-level metrics.

All-pairs shortest paths are pre-computed once at startup for both graphs.

### Required input columns

| Column | Description |
|---|---|
| `path_to_reach` | Comma-separated node IDs of the rat's full path |
| `start_node_n` | Start node ID |
| `goal_node_n` | Goal node ID |
| `start_island_n` | Island number (1–4) of the start node |
| `goal_island_n` | Island number (1–4) of the goal node |
| `seq_islands` | Comma-separated island sequence visited |
| `exclude_trial` | `0` = include; any other value = skip Step 2 |

### Computed metrics

#### Distance metrics

| Column | Description |
|---|---|
| `distance_start_goal_island` | Shortest island-to-island distance + 1 |
| `distance_start_goal_nodes` | Shortest node-to-node distance + 1 |

#### Path length metrics

| Column | Description |
|---|---|
| `path_length_start_goal_nodes_node_hit` | Total nodes visited (`len(path)`) |
| `path_length_start_goal_island_node_hit` | Total island entries (`len(seq_islands)`) |
| `path_length_start_goal_island_island_hit` | Unique islands visited (`len(set(seq_islands))`) |
| `norm_path_length_start_goal_nodes_node_hit` | Node path length ÷ optimal node distance |
| `norm_path_length_start_goal_island_node_hit` | Island entries ÷ optimal island distance |
| `norm_path_length_start_goal_island_island_hit` | Unique islands ÷ optimal island distance |

#### Core behavioral metrics (Step 1)

| Column | Description |
|---|---|
| `shortest_path` | Minimum hops between start and goal node |
| `n_nodes_visited` | Total nodes in path including revisits |
| `food_reached` | `1` if goal node appears among the last two path nodes |
| `eat_on_1_encounter` | `1` if the last path node is exactly the goal node |
| `dist_tra` | Edges traveled (`n_nodes_visited - 1`); `99` if food not reached |
| `dt_rel_sp` | `dist_tra / shortest_path` — relative path length (1.0 = optimal) |
| `dt_min_sp` | `dist_tra - shortest_path` — extra steps beyond optimal |
| `dir_run_mat_perf` | `1` if food reached AND path length equals `shortest_path` |
| `node_choices_binary` | Comma-separated `0`/`1` per step: `1` = choice minimized distance to goal |
| `perc_correct_choices` | Percentage of steps that were optimal choices |

#### Goal-island entry metrics (Step 2)

Computed from the **last** bridge crossing into the goal island. Skipped for rows where `exclude_trial != 0`.

| Column | Description |
|---|---|
| `isl_node_in` | Node at which the rat last entered the goal island |
| `isl_short_path` | Optimal distance from island-entry node to goal + 1 |
| `isl_dt_trav` | Nodes traveled from island entry to end of path |
| `perf_in_island` | `isl_dt_trav / isl_short_path` — performance ratio within the island |

Rows with empty or unparseable `path_to_reach` are flagged in a `flag` column and highlighted red in the output Excel; all computed columns for that row are left blank.

### Running node analysis

```bash
python src/node_analysis/hex_maze_analysis.py \
    --input_folder  /path/to/ip1 \
    --output_folder /path/to/op1
```

All `.xlsx` files in `--input_folder` are processed. Each produces a `*_results.xlsx` file in `--output_folder`.

For full calculation details see [`src/node_analysis/README.md`](src/node_analysis/README.md).

---

## Key Scripts

### `src/TrackerYolov11.py`
Reads session metadata from `RecordingMeta.xlsx`, finds `stitched.mp4` in the input folder, runs YOLOv11 detection + tracking, and writes an annotated output video, position CSV, and per-trial metrics back into a copy of the metadata spreadsheet. Based on [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT).

### `src/Video_LED_Sync_using_ICA.py`
Extracts LED blink signals from video frames using FastICA, aligns them with Trodes DIO timestamps, and produces a synchronized framewise timestamp CSV for downstream analysis.

### `src/join_views.py`
Stitches frames from multiple camera angles into a single wide video (`stitched.mp4`).

### `src/node_analysis/hex_maze_analysis.py`
Reads raw trial `.xlsx` files, computes behavioral metrics (shortest path, performance ratios, node choice analysis, island-entry analysis), and writes results back into the spreadsheet. See [`src/node_analysis/README.md`](src/node_analysis/README.md) for a full explanation of every computed column.

### `src/plot_trials.py`
Generates trial-level plots from tracked position data.

### `src/sorter/sorting.py`
Runs Kilosort4 via SpikeInterface on the raw Trodes export, outputs spike-sorted units.

### `src/sorter/export_lfp.py`
Reads Trodes-exported LFP channels and exports them to NWB or CSV format.

### `src/dlc/tracking_eyes.py`
Exports video clips formatted for DeepLabCut eye-tracking analysis.

---

## Metadata

`RecordingMeta.xlsx` contains per-session and per-trial information read by the tracker and plotting scripts. The tracker appends computed columns to a copy of this file after each run — the original is never modified.

Required columns:

| Column | Description |
|---|---|
| `Rat_ID` | Subject identifier |
| `Date` | Recording date |
| `Repeat` | Repeat number |
| `Day` | Training day |
| `Session` | Session number |
| `Num_Trials` | Total number of trials in the session |
| `Start_Min` / `Start_Sec` | Optional: resume video from this time offset |
| `Start_At_Trial_Num` | Optional: resume from this trial number |
| `Start_Nodes` | Per-row start node IDs |
| `Goal_Node` | Per-row goal node IDs |
| `Trial_Type` | Per-row trial type (1–6) |
| `Special_Trials` | Per-row special trial flags |
| `Did_Not_Reach` | Per-row: `1` if rat did not reach goal |
| `Unnormal_Intervals` | Per-row immunity windows (`trial:start_min-end_min`) |

---

## License

See [LICENSE](LICENSE).
