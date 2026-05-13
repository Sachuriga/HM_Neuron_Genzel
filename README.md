# HM Tracker 2025

A batch-processing pipeline for neuroscience experiments — integrates video-based animal tracking (YOLOv11), neural spike sorting (Kilosort4), LFP extraction, and behavioral synchronization into a single orchestrated Windows workflow.

---

## Overview

The pipeline takes raw Trodes recordings (`.rec`) and multi-camera video, then runs a configurable sequence of steps to produce:

- Annotated tracking video + position CSV (via YOLOv11)
- LED-synchronized behavioral timestamps (via ICA)
- Spike-sorted neural data (via Kilosort4 / SpikeInterface)
- LFP traces (1 kHz, 500 Hz low-pass)
- DeepLabCut-ready video export
- GPU-compressed output video

---

## Directory Layout

```
HM_Tracker_2025/
├── src/
│   ├── TrackerYolov11.py         # Main YOLOv11 tracker
│   ├── Video_LED_Sync_using_ICA.py  # LED sync via ICA
│   ├── join_views.py             # Multi-camera stitching
│   ├── plot_trials.py            # Trial-level plotting
│   ├── sorter/
│   │   ├── sorting.py            # Kilosort4 spike sorting
│   │   └── export_lfp.py         # LFP extraction & export
│   ├── node_analysis/
│   │   ├── hex_maze_analysis.py  # Hex maze behavioral metrics
│   │   └── README.md             # Column-by-column calculation reference
│   └── dlc/
│       └── tracking_eyes.py      # DeepLabCut eye-tracking export
├── runner_windows.bat            # Main Windows orchestrator
├── hm_tracker_paths.example.txt # Config template
├── RecordingMeta.xlsx            # Per-session metadata
├── reproduce.yml                 # Conda environment spec
└── requirements.txt              # pip dependencies
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

Download and install:

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

The tracker (`src/TrackerYolov11.py`) is built around a per-frame detect → classify → update-state loop. Here is a detailed walkthrough of each layer.

### 1. Detection (`cnn()`)

Every frame is passed through YOLOv11 at confidence threshold 0.7 and input size 1280 px. The model outputs three classes:

| Class | Meaning |
|---|---|
| `head` | Rat's head (preferred for position) |
| `rat` | Rat's body (fallback) |
| `researcher` | Human experimenter |

**Rat position selection** — All detections are collected and sorted by confidence. If a `head` box is found, the tracker locks onto head detections (`locked_to_head = True`) and ignores body boxes for the rest of the trial. If no detection fires on a frame, the last known position (`last_rat_pos`) is carried forward so the state machine never stalls.

**Researcher selection** — All `researcher` boxes are stored. The one geometrically closest to the rat's active position is used for trial-trigger and force-end logic. This handles the common case of multiple people visible in the arena.

---

### 2. Trial State Machine

The tracker runs a simple three-state machine per trial:

```
WAITING (start_trial=True)
    │  rat centroid within 60 px of start node
    ▼
ACTIVE (record_detections=True)
    │  goal reached / timeout / force-end condition
    ▼
INTER-TRIAL (record_detections=False)
    │  researcher within 80 px of rat  →  back to WAITING
```

#### Starting a trial
When in WAITING state, `find_start()` checks Euclidean distance from the rat centroid to the current trial's start-node pixel coordinate. If distance < 60 px, recording begins: per-frame buffers are reset, `record_detections` is set to `True`, and the trial timer starts.

#### Trial types and end conditions

Each trial has a type read from `RecordingMeta.xlsx`. The end condition differs by type:

| Type | Label | End condition |
|---|---|---|
| 1 | Normal | Rat centroid within 25 px of goal node |
| 2 | NGL (New Goal Location) | Rat visited goal (within 20 px) AND 10 minutes elapsed |
| 3 | Probe | More than 2 minutes elapsed AND researcher within 80 px of goal AND rat within 25 px of goal |
| 4–6 | Special NGL variants | Same as NGL; followed by a 10-minute inter-trial lockout |

**Did-Not-Reach override** — If `Did_Not_Reach = 1` for a trial in the metadata, the normal goal-proximity check is skipped. Instead, the trial ends when a researcher stays within 60 px of the rat for ≥ 1 second (rat-pickup detection).

**Force-end fallbacks** — Two additional guards prevent a trial from running indefinitely:
- If the closest researcher to the **goal** stays within 50 px for 10 continuous seconds → trial ends.
- If the closest researcher to the **goal** stays within 80 px for 30 continuous seconds → trial ends (probe immunity rules apply).

**Unnormal intervals** — Specific time windows can be marked immune in the metadata (`Unnormal_Intervals` column, format `trial_num:start_min-end_min`). During these windows, goal-reach and force-end checks are suppressed.

**Inter-trial lockout** — After a type-4/5/6 trial, a 10-minute lockout is enforced. The overlay shows a countdown; the researcher-proximity trigger that moves back to WAITING is blocked until the lockout expires.

---

### 3. Node Logging (`annotate_frame()`)

The maze layout is encoded as a node dictionary (`src/tools/mask.py`) mapping node IDs to pixel coordinates. On every active frame, the tracker checks whether the rat centroid falls within 20 px of any node. When it does:

- The node ID is appended to `saved_nodes`.
- A synchronized timestamp (from the LED-ICA CSV) is attached.
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

The CSV timestamps are merged from the LED-ICA sync file produced in Step 2, so each frame carries an absolute time reference aligned to the neural recording.

---

## Key Scripts

### `src/TrackerYolov11.py`
Reads session metadata from `RecordingMeta.xlsx`, finds `stitched.mp4` in the input folder, runs YOLOv11x detection + tracking, and writes an annotated output video and position CSV.

### `src/Video_LED_Sync_using_ICA.py`
Extracts LED blink signals from video frames using FastICA, aligns them with Trodes DIO timestamps, and produces a synchronized timestamp file for downstream analysis.

### `src/join_views.py`
Stitches frames from multiple camera angles into a single wide video (`stitched.mp4`).

### `src/node_analysis/hex_maze_analysis.py`
Reads raw trial `.xlsx` files, computes behavioral metrics (shortest path, performance ratios, island-entry analysis), and writes results back into the spreadsheet. See [`src/node_analysis/README.md`](src/node_analysis/README.md) for a full explanation of every computed column.

### `src/sorter/sorting.py`
Runs Kilosort4 via SpikeInterface on the raw Trodes export, outputs spike-sorted units.

### `src/sorter/export_lfp.py`
Reads Trodes-exported LFP channels and exports them to NWB or CSV format.

---

## Metadata

`RecordingMeta.xlsx` contains per-session information (subject ID, date, trial structure, etc.) read by the tracker and plotting scripts. Update this file before running a new batch.

---

## License

See [LICENSE](LICENSE).
