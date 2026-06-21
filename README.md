# HM Neuron Genzel

**Author:** Sachuriga S.

A batch-processing pipeline for neuroscience experiments — integrates video-based animal tracking (YOLOv11), behavioral node analysis, neural spike sorting (Mountainsort4), LFP extraction, and LED-based synchronization into a single orchestrated workflow.

> **Attribution** — `src/Video_LED_Sync_using_ICA.py` (Step 2), `src/join_views.py` (Step 3), and `src/TrackerYolov11.py` (Step 4) are based on and modified from [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT). See [Tracker — Modifications from Original](#tracker--modifications-from-original) for a summary of changes to the tracker.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Layout](#directory-layout)
3. [Setup](#setup)
4. [Data Layout](#data-layout)
5. [Running the Pipeline](#running-the-pipeline)
6. [Resource Monitoring](#resource-monitoring)
7. [Pipeline Steps — Detailed Reference](#pipeline-steps--detailed-reference)
   - [Step 1 — Trodes DIO/Raw Export](#step-1--trodes-dioraw-export)
   - [Step e — Trodes LFP Export](#step-e--trodes-lfp-export)
   - [Step 2 — LED Sync (ICA)](#step-2--led-sync-ica)
   - [Step 3 — Multi-Camera Stitching](#step-3--multi-camera-stitching)
   - [Step 4 — YOLOv11 Tracker](#step-4--yolov11-tracker)
   - [Step 5 — Trial Plotting](#step-5--trial-plotting)
   - [Step 6 — GPU Video Compression](#step-6--gpu-video-compression)
   - [Step 7 — Spike Sorting](#step-7--spike-sorting)
   - [Step 8 — LFP Extraction](#step-8--lfp-extraction)
   - [Step d — DeepLabCut Export](#step-d--deeplabcut-export)
   - [Step 9 — Cleanup](#step-9--cleanup)
   - [Step n — Node Analysis](#step-n--node-analysis)
8. [Tracker — How It Works](#tracker--how-it-works)
   - [Model](#model)
9. [Node Analysis — Computed Metrics](#node-analysis--computed-metrics)
10. [Metadata](#metadata)

---

## Overview

The pipeline takes raw Trodes recordings (`.rec`) and multi-camera video, then runs a configurable sequence of steps to produce:

- Annotated tracking video + per-frame position CSV (via YOLOv11)
- Per-trial metrics written back into `RecordingMeta.xlsx`
- LED-synchronized behavioral timestamps (via ICA)
- Behavioral metrics computed from trial node sequences (via hex maze analysis)
- Spike-sorted neural data (via Mountainsort4 / SpikeInterface)
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
│   │   ├── sorting.py               # Mountainsort4 spike sorting
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

**Option A — automated (recommended):** double-click `install.bat` or run it from an Anaconda Prompt.

**Option B — manual:**

```bat
conda env create -f reproduce.yml
conda activate HM_neuron
pip install -r requirements_core.txt
```

> Requires a CUDA-capable GPU and **CUDA 12.8** drivers. PyTorch (`torch==2.10.0+cu128`) is fetched automatically from `download.pytorch.org` — no manual download needed.
>
> If `conda env create` freezes, make sure no older `HM_neuron` environment already exists (`conda env remove -n HM_neuron`).

### 2. External tools

| Tool | Purpose |
|---|---|
| [Trodes](https://spikegadgets.com/trodes/) | `.rec` file export (DIO, raw, LFP) |
| [FFmpeg](https://ffmpeg.org/) | Multi-camera stitching and GPU-accelerated compression |
| YOLOv11 weights | Object detection model (`.pt` file) — see [Model](#model) |

### 3. Path config

Copy the template and fill in your local paths:

```bash
cp hm_tracker_paths.example.txt %USERPROFILE%\Desktop\hm_tracker_paths.txt
```

Edit `hm_tracker_paths.txt`:

```
FFMPEG_CMD=C:\path\to\ffmpeg.exe
ONNX_WEIGHTS_PATH=C:\path\to\weights.pt
TRODES_EXPORT_CMD=C:\path\to\trodesexport.exe
TRODES_EXPORT_LFP=C:\path\to\exportLFP.exe
LFP_CHANNELS=1 2 3 4 5 6 7 8
```

This file must exist on the Desktop before running. Lines starting with `#` are treated as comments.

---

## Data Layout

The runner expects input/output folder pairs named `ipN` / `opN` inside the target directory:

```
data_root/
├── ip1/    ← raw input (contains .rec + eye camera videos)
├── op1/    ← processed output
├── ip2/
└── op2/
```

Each `ipN` folder is processed by one worker. The matching `opN` folder is created if it does not exist.

---

## Running the Pipeline

```bat
runner_windows.bat "C:\path\to\data_root"
```

On launch you are prompted to select which steps to run:

```
Select steps to run (e.g., 123 for steps 1, 2, and 3):
[1] Trodes Export (DIO/Raw)
[e] Trodes Export LFP (per channel)
[2] Sync Script
[3] Stitching
[4] Tracker
[5] Plotting
[6] Compression
[7] Sorting
[8] LFP
[d] DeepLabCut
[9] Cleaning
[n] Node Analysis
```

Type any combination of keys and press Enter, e.g. `1e234` runs steps 1, e, 2, 3, and 4 in that order.

The master process scans all `ipN/opN` pairs in the target directory. For each pair it launches a separate worker in a new `cmd` window. A **20-second gap** is inserted between launches for stability, and each launch is preceded by a resource check (see [Resource Monitoring](#resource-monitoring)).

---

## Resource Monitoring

Before launching each worker, the master checks system load using PowerShell and `nvidia-smi`. If any threshold is exceeded, it pauses for 30 seconds and rechecks:

| Resource | Default threshold | How measured |
|---|---|---|
| CPU | 90% | `Win32_Processor.LoadPercentage` average |
| GPU | 90% | `nvidia-smi --query-gpu=utilization.gpu` |
| RAM | 65% | `(TotalVisibleMemorySize - FreePhysicalMemory) / TotalVisibleMemorySize` |

Thresholds and the wait duration are configurable at the top of `runner_windows.bat`:

```bat
set MAX_CPU=90
set MAX_GPU=90
set MAX_MEM=65
set WAIT_SECONDS=30
```

---

## Pipeline Steps — Detailed Reference

### Step 1 — Trodes DIO/Raw Export

**Key:** `1`  
**Script:** `trodesexport` (external binary)  
**Command:**
```
trodesexport -dio -raw -rec <file.rec>
```

Iterates over every `.rec` file found in the input folder and exports:
- **DIO** (digital I/O) — TTL pulse events, typically used to record LED sync signals and task events.
- **Raw** — raw neural voltage traces at the recording sample rate (default 30 000 Hz), written as `.dat` files per channel group.

Output lands alongside the `.rec` file in the input folder. Required before Step e (LFP), Step 2 (sync), and Step 7 (sorting).

---

### Step e — Trodes LFP Export

**Key:** `e`  
**Script:** `exportLFP` (external binary)  
**Command:**
```
exportLFP -rec <file.rec> -outputrate 1000 -lfplowpass 500
```

Exports Local Field Potential data from each `.rec` file:
- **Sample rate:** 1 000 Hz (downsampled from the raw recording rate)
- **Low-pass filter:** 500 Hz (applied during export by Trodes)
- **Output format:** per-channel `.dat` files inside a `<recording>.LFP/` subfolder

These files are read by Step 8 (LFP extraction).

---

### Step 2 — LED Sync (ICA)

**Key:** `2`  
**Script:** `src/Video_LED_Sync_using_ICA.py` *(based on [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT))*  
**Command:**
```
python Video_LED_Sync_using_ICA.py -i <ip> -o <op> -f 30000
```

Aligns the video timeline to the neural recording timeline using an LED synchronization light visible in the camera frame:

1. **LED detection** — Scans video frames for bright moving regions by subtracting a reference frame, thresholding, and finding contours. The brightest 16×16 region is tracked as the LED.
2. **Signal extraction** — Extracts a per-frame luminance signal from the LED region across the full video.
3. **ICA decomposition** — Runs FastICA on the luminance signal to isolate the LED blink component from background noise and other light sources.
4. **Blink detection** — KMeans clustering separates ON/OFF states; blink timestamps are extracted in video frames.
5. **Alignment** — Blink times in the video are matched against DIO TTL timestamps from the Trodes export (`-f 30000` sets the neural recording sample rate). A linear regression maps video frame indices to neural recording time.
6. **Output** — Writes a framewise timestamp CSV (`stitched_framewise_seconds.csv` or `stitched_framewise_ts.csv`) to the output folder. Each row maps a video frame index to a synchronized timestamp in seconds. This file is consumed by the tracker (Step 4) and the plotting script (Step 5).

---

### Step 3 — Multi-Camera Stitching

**Key:** `3`  
**Script:** `src/join_views.py` *(based on [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT))*  
**Command:**
```
python join_views.py <ip>
```

Stitches multiple individual camera video files (named `eye??_*.mp4`) into a single combined output file:

- Expects **12 videos** by default; videos are sorted naturally and arranged in a **2-row grid** (`ceil(n/2)` columns per row).
- Raw frame size per camera: **600 × 800 px**. A crop is applied before stitching — default **104 px** from each horizontal side and **91 px** from the top (bottom-row cameras start 91 px lower). Effective frame size after crop: **392 × 709 px** per tile.
- Bottom-row cameras are horizontally and vertically flipped (`hflip,vflip`) to correct for inverted mounting.
- The FFmpeg command uses a **nullsrc canvas + overlay** filter chain (not hstack/vstack). Output is encoded with **h264_nvenc** (NVIDIA GPU), preset `fast`, bitrate **4000k**, pixel format `yuv420p`, at **30 fps**.
- Output is written as `stitched.mp4` in the input folder. This file is required by Step 4 (tracker).

---

### Step 4 — YOLOv11 Tracker

**Key:** `4`  
**Script:** `src/TrackerYolov11.py` *(based on [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT))*  
**Requires:** `stitched.mp4` in the input folder  
**Command:**
```
python TrackerYolov11.py --input_folder <ip> --output_folder <op> --onnx_weight <weights.pt>
```

Runs the full tracking pipeline on the stitched video. See [Tracker — How It Works](#tracker--how-it-works) for a complete description.

**Outputs:**

| File | Contents |
|---|---|
| `<date>_Rat<id>.txt` | Per-trial node sequence, segment timing, and velocity summary |
| `<date>_Rat<id>_Coordinates_Full.csv` | Per-frame: `Frame_Index`, `Timestamp`, `Trial_Num`, `Rat_X/Y`, `Researcher_X/Y` |
| `<date>_Rat<id>.mp4` | Annotated video with bounding boxes, centroid trail, crosshair, node markers, overlays |
| `<RecordingMeta>.xlsx` (copy) | Source metadata with per-trial columns appended (see [RecordingMeta output columns](#recordingmeta-output-columns)) |

---

### Step 5 — Trial Plotting

**Key:** `5`  
**Script:** `src/plot_trials.py`  
**Command:**
```
python plot_trials.py --input_folder <ip> --output_folder <op>
```

Generates a PDF report with trial-level visualizations from the tracker output:

- Reads **`.log` files** from the output folder (produced by the tracker). Extracts rat position events (`The rat position is: (x, y) @ frame`) and trial markers (`Recording Trial N`) via regex.
- Optionally reads the `*.txt` node-sequence file from the output folder and `RecordingMeta.xlsx` from the input folder for goal/start node overlays.
- Reconstructs the hex maze graph from the hardcoded node coordinate file (`src/tools/node_list_new.csv`) using NetworkX, connecting nodes within **65 px** of each other. Manually adds bridge edges (121↔302, 324↔401, 305↔220, 404↔223, 201↔124, 224↔218).
- Per-trial speed is computed from pixel coordinates at **30 fps** and smoothed at multiple windows (0.4 s, 0.5 s, 1.0 s, 2.0 s, 5.0 s).
- Outputs a multi-page PDF with per-trial trajectory plots (coloured by speed), aggregate speed distributions, occupancy heatmaps, and path-scoring statistics.

---

### Step 6 — GPU Video Compression

**Key:** `6`  
**Script:** inline in `runner_windows.bat`  
**Command:**
```
ffmpeg -i <video.mp4> -c:v h264_nvenc -preset p6 -cq 28 -c:a copy __temp_compressed.mp4
```

Compresses the annotated tracker output video using NVIDIA hardware encoding:

- **Codec:** `h264_nvenc` — NVIDIA H.264 hardware encoder (requires an NVENC-capable GPU).
- **Preset:** `p6` — high quality, slower encoding.
- **Quality:** CQ 28 — constant quality mode; lower values = higher quality / larger file.
- Audio (if present) is copied without re-encoding.
- Compression writes to a temp file (`__temp_compressed.mp4`) first; on success the temp file replaces the original. On failure the temp file is deleted and the original is preserved.
- Any leftover temp files from a previous crashed run are cleaned up at the start.

> If no NVENC sessions are available (GPU limit reached or no NVIDIA GPU), FFmpeg will fail with an error and the original file is left untouched.

---

### Step 7 — Spike Sorting

**Key:** `7`  
**Script:** `src/sorter/sorting.py`  
**Command:**
```
python sorting.py --input_folder <ip> --output_folder <op> [--config <hm_tracker_paths.txt>]
```

Runs the full spike-sorting pipeline on raw Trodes-exported `.dat` files via SpikeInterface + **Mountainsort4**:

1. **Data loading** — Searches for `**/*.raw/*_group0.dat` files under the input folder. Reads raw voltage traces using the official Trodes reader (`readTrodesExtractedDataFile3.py`). Applies gain (`0.195 mV/unit`) and offset.
2. **Probe geometry** — Attaches an **8 × 4 tetrode grid** (32 tetrodes, 128 channels total). Tetrodes are spaced **250 µm** apart; contacts within each tetrode are arranged in a ±10 µm diamond pattern. Contact shape: circle, radius 5 µm.
3. **Preprocessing:**
   - Bandpass filter: **300–6 000 Hz**
   - Bad-channel interpolation (**50 µm** radius): a per-rat list of bad channels (see config below) is interpolated from their neighbours
   - Common reference: optional reference channel(s) per rat, followed by a global median common-average reference
   - Whitening (float32)
4. **Sorting** — **Mountainsort4** is run via SpikeInterface (`adjacency_radius=50 µm`, pre-filtering and whitening disabled since already done).
5. **Analysis** — Waveforms extracted (1 ms before, 2 ms after spike peak), 3-component PCA per channel, quality metrics (SNR, ISI violation, firing rate).
6. **Export** — Results exported to `phy_export/` for manual curation in Phy. All intermediate files are deleted; only `phy_export/` is retained.

#### Per-rat channel configuration

Bad channels and reference channels are read from `hm_tracker_paths.txt` (the same Desktop config file used by the runner; pass it with `--config`, or it falls back to `~/Desktop/hm_tracker_paths.txt`). The rat is matched against the start of each recording's file name (e.g. `rat1_..._group0.dat` → the `RAT1` keys).

```
# space- or comma-separated; plain ids and NT notation can be mixed
BAD_CHANNELS_RAT1=0 1 2 3 NT8ch1 NT8ch2
REF_CHANNEL_RAT1=NT17ch1
```

- **`BAD_CHANNELS_<RAT>`** — channels to interpolate. Omit/leave empty for none.
- **`REF_CHANNEL_<RAT>`** — channel(s) referenced against *before* the global median CAR (several ids are averaged together). If omitted, only the global median CAR is applied.
- If a rat has no config entry, `RAT1` falls back to a built-in default bad-channel list; other rats default to none.

#### Tetrode (NT) channel mapping

The 128 channels are wired as 32 tetrodes of 4. Channels may be written as plain 0-based ids (`0`–`127`) or as `NT<t>ch<c>` (`t` = 1–32, `c` = 1–4), which map by:

```
channel = (NT − 1) × 4 + (ch − 1)
```

| Tetrode | ch1 | ch2 | ch3 | ch4 |
|---|---|---|---|---|
| NT1  | 0   | 1   | 2   | 3   |
| NT2  | 4   | 5   | 6   | 7   |
| NT3  | 8   | 9   | 10  | 11  |
| NT4  | 12  | 13  | 14  | 15  |
| NT5  | 16  | 17  | 18  | 19  |
| NT6  | 20  | 21  | 22  | 23  |
| NT7  | 24  | 25  | 26  | 27  |
| NT8  | 28  | 29  | 30  | 31  |
| NT9  | 32  | 33  | 34  | 35  |
| NT10 | 36  | 37  | 38  | 39  |
| NT11 | 40  | 41  | 42  | 43  |
| NT12 | 44  | 45  | 46  | 47  |
| NT13 | 48  | 49  | 50  | 51  |
| NT14 | 52  | 53  | 54  | 55  |
| NT15 | 56  | 57  | 58  | 59  |
| NT16 | 60  | 61  | 62  | 63  |
| NT17 | 64  | 65  | 66  | 67  |
| NT18 | 68  | 69  | 70  | 71  |
| NT19 | 72  | 73  | 74  | 75  |
| NT20 | 76  | 77  | 78  | 79  |
| NT21 | 80  | 81  | 82  | 83  |
| NT22 | 84  | 85  | 86  | 87  |
| NT23 | 88  | 89  | 90  | 91  |
| NT24 | 92  | 93  | 94  | 95  |
| NT25 | 96  | 97  | 98  | 99  |
| NT26 | 100 | 101 | 102 | 103 |
| NT27 | 104 | 105 | 106 | 107 |
| NT28 | 108 | 109 | 110 | 111 |
| NT29 | 112 | 113 | 114 | 115 |
| NT30 | 116 | 117 | 118 | 119 |
| NT31 | 120 | 121 | 122 | 123 |
| NT32 | 124 | 125 | 126 | 127 |

---

### Step 8 — LFP Extraction

**Key:** `8`  
**Script:** `src/sorter/export_lfp.py`  
**Command:**
```
python export_lfp.py --input_folder <ip> --output_folder <op>
```

Reads the Trodes-exported LFP `.dat` files (produced by Step e) and compiles them into analysis-ready output under `LFP_Output/`:

- **File discovery** — Scans for `*.LFP/*.dat` files, excluding the timestamps file. Falls back to a flat `*LFP*.dat` glob if the subfolder structure is absent.
- **Channel parsing** — Extracts ntrode and channel numbers from filenames (pattern `_nt<N>ch<C>.dat`). Voltage scaled by `voltagescaling` from file header (default **0.195 mV/unit**).
- **Timestamps** — Reads the companion `*.timestamps.dat` file; converts raw counts to seconds (zero-referenced).
- **Channel selection:**
  - **EMG** — Welch power spectrum over the first 60 s; channel with maximum power in the **20–200 Hz** band.
  - **EEG (cleanest)** — SNR = sleep-band power (0.5–30 Hz) ÷ noise-band power (>50 Hz); top **3 channels** selected.
- **Awakeness index** — Computed per 1-second epoch: z-scored EMG RMS × **0.6** + z-scored theta/delta ratio × **0.4**, upsampled back to sample rate. Theta band: **5–9 Hz**, delta band: **0.5–4 Hz**.
- **Output files** — `lfp_data.npy` **(n_samples × n_channels)**, per-channel `.npy` files, `lfp_timestamps.npy`, `channel_map.npy`, `emg_data.npy`, `awakeness.npy`, `theta_delta_ratio.npy`, `channel_snr_scores.npy`.

---

### Step d — DeepLabCut Export

**Key:** `d`  
**Script:** `src/dlc/tracking_eyes.py`  
**Command:**
```
python tracking_eyes.py --input_folder <ip> --output_folder <op>
```

Extracts eye-camera frames aligned to the rat's tracked position for DeepLabCut training or analysis:

1. **Position data** — Reads `*_Coordinates_Full.csv` from the output folder (or input folder as fallback).
2. **Region mapping** — Divides the stitched frame (1176 × 712 px) into a grid of 196 × 356 px regions. Each frame's rat XY position is mapped to a region ID indicating which individual eye camera captured it.
3. **Video lookup** — Pre-maps each unique region ID to the corresponding raw eye camera video file (`eye01_*.mp4`, `eye02_*.mp4`, etc.).
4. **Frame extraction** — For each row in the CSV, seeks to the corresponding frame in the matched eye camera video using a cached `VideoCapture` (sequential reads where possible to avoid costly `cap.set()` calls).
5. **Output** — Writes all extracted frames as a single `collected_frames.mp4` to the output folder. The source frame index is stored back into the CSV as `extracted_frame_idx`.

---

### Step 9 — Cleanup

**Key:** `9`  
**Script:** inline in `runner_windows.bat`

Deletes intermediate folders from the input directory to recover disk space after processing is complete:

| Deleted | Why it exists |
|---|---|
| `*.DIO/` | Trodes DIO export (consumed by Step 2) |
| `*.raw/` | Trodes raw export (consumed by Step 7) |
| `*timestampoffset*/` | Trodes timestamp offset folders (no longer needed after sync) |

> **Irreversible.** Run only after confirming Steps 2 and 7 have completed successfully. The original `.rec` file is never touched.

---

### Step n — Node Analysis

**Key:** `n`  
**Script:** `src/node_analysis/hex_maze_analysis.py`  
**Command:**
```
python hex_maze_analysis.py --input_folder <ip> --output_folder <op>
```

Reads all `.xlsx` files in the input folder and computes behavioral metrics from trial node sequences. Each input file produces a `*_results.xlsx` in the output folder. See [Node Analysis — Computed Metrics](#node-analysis--computed-metrics) for a full reference of every computed column.

### Step w — NWB / LFP Package

**Key:** `w` (labelled `nwblfp` in the menu)
**Script:** `src/nwb/create_nwb.py`
**Command** (run once at the master level, after every parallel/sequential step has completed):
```
python create_nwb.py --rat_nr %NWB_RAT_NR% --noroot --ip <ROOT_DIR> --op <ROOT_DIR>
```

Packages the rat's processed data (Coordinates CSVs, paths TXT, tracker log, optional LFP `.npy` files) into [NWB (Neurodata Without Borders)](https://www.nwb.org/) files — one `.nwb` per session, written to `ROOT_DIR`.

This step is not parallelized: it runs once at the master level after sorting and cleaning, because `create_nwb.py` scans the whole rat directory in one pass. The rat number is read from `NWB_RAT_NR` in `hm_tracker_paths.txt` (default `1`). The auxiliary helpers live in `src/tools/` (`pathnames.py`, `process_log.py`, `process_txt.py`, `process_dataframe.py`).

---

## Tracker — How It Works

`src/TrackerYolov11.py` is based on [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT) and is built around a per-frame detect → classify → update-state loop.

### Model

The YOLOv11 model was trained on a custom dataset labeled by the authors specifically for this setup — rat body, rat head, and researcher detections in a hex maze environment.

**Dataset:** [box_hm_rat on Roboflow Universe](https://universe.roboflow.com/rathm-wjck3/box_hm_rat) *(self-labeled)*

[![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/rathm-wjck3/box_hm_rat)

---

### Detection (`cnn()`)

Every frame is passed through YOLOv11 at confidence threshold 0.7 and input size 1280 px. The model outputs three classes:

| Class | Used for |
|---|---|
| `head` | Counted only (`Rat-head Count` overlay); does not affect position |
| `rat` | Position tracking — the body centroid drives all trial logic |
| `researcher` | Trial-trigger and force-end logic |

**Motion-based skip** — Before running YOLO, a frame-difference check compares the current frame against the previous one (Gaussian blur, absolute diff, dilation, pixel count). If fewer than 500 changed pixels are detected, YOLO is skipped. The last known bounding boxes are redrawn from a cache so the display never flickers. When YOLO does run and finds detections, the cache is updated; if YOLO runs but finds nothing, the cache is left unchanged (stale boxes remain visible rather than flashing away).

**Rat position** — All `rat` (body) candidates are sorted by confidence; the highest-confidence box centroid becomes `self.Rat`. If no body is detected this frame, `last_rat_pos` is used as a fallback so the state machine never stalls.

**Researcher selection** — All `researcher` boxes are stored in `all_researchers`. The one closest (Euclidean) to the rat's active position is used for proximity checks.

---

### Trial State Machine

```
WAITING  (start_trial=True, record_detections=False)
    │  rat centroid within 60 px of start node
    ▼
ACTIVE   (record_detections=True)
    │  end condition met (see trial types below)
    ▼
INTER-TRIAL  (start_trial=False, record_detections=False)
    │  researcher within 300 px of rat  →  back to WAITING
```

#### Trial types and end conditions

| Type | Label | End condition |
|---|---|---|
| 1 | Normal | Rat centroid ≤ 25 px from goal node |
| 2 | NGL | Rat visited goal (≤ 20 px) AND 10 minutes elapsed |
| 3 | Probe | ≥ 2 min elapsed AND researcher ≤ 600 px from goal AND rat ≤ 25 px from goal |
| 4–6 | Special NGL | Same as NGL; 10-minute inter-trial lockout after the trial ends |

**Researcher proximity end** — For types 1 and 2, if the closest researcher comes within 150 px of the rat after at least 5 seconds have elapsed, the trial ends immediately.

**Force-end fallbacks:**
- Closest researcher to the **goal** within 50 px for 10 continuous seconds → trial ends.
- Closest researcher to the **goal** within 160 px for 30 continuous seconds → trial ends (probe immunity and unnormal-interval rules apply).

**Unnormal intervals** — Time windows specified in the `Unnormal_Intervals` metadata column (`trial_num:start_min-end_min`) suppress goal-reach and force-end checks during that window.

**Inter-trial lockout** — After a type-4/5/6 trial, the next trial cannot start until 10 minutes have elapsed **from the start of the special trial** (not from when it ended). A countdown overlay shows the remaining time; the researcher-proximity trigger is blocked until the lockout expires.

#### Time-locked special trials

A row's `Special_Trials` cell can specify when a particular trial unlocks, using the format `trial_num@MM:SS` (e.g. `3@5:30`). The schedule check runs every frame **before** any other trial logic, so it takes precedence over researcher-proximity, DNR, and goal-reach checks.

| Phase | Behavior |
|---|---|
| Before unlock time | Trial N's start node is gated — even if the rat sits on it, `find_start` returns without triggering. The previously-active trial keeps running with its normal end conditions. |
| Unlock time arrives, earlier trial still active | The earlier trial is force-ended with reason `"forced by special trial schedule"` regardless of why it was running. |
| After force-end | `start_trial` is armed directly — TrigA (researcher near rat) and TrigB (researcher covering start node) are bypassed. The inter-trial lockout is still enforced inside `find_start`: if the previous trial was type 4/5/6, the new trial cannot actually begin until 10 minutes have elapsed since the previous trial's start, even though the start node is "appeared". |

**Schedule-only end mode.** When a trial's *next* trial number is in the schedule list, the current trial enters "schedule-only end" mode: every other end path (NGL 10-min timeout, researcher-near-rat 150 px, researcher-at-goal 10 s / 30 s, unnormal-interval timeout) is suppressed for it. Only the schedule force-end can terminate that trial. Trials whose successor is NOT scheduled keep their original end conditions — in particular, a type 4/5/6 NGL trial without a scheduled successor still ends at its 10-minute timer as before.

Multiple time-locked trials can be defined in the same session. A plain trial number with no `@TIME` is accepted as a marker but has no runtime effect.

Console output at session start lists the loaded schedule:

```
Special trial schedule (trial_num → session seconds):
   Trial 3 → 05:30.00
   Trial 7 → 12:00.00
```

Each force-end logs:

```
[SCHEDULE] Trial 3 unlock time 330.00s reached at session 330.13s — force-ending active trial 2.
```

---

### Node Logging

On every active frame, the tracker checks if the rat centroid falls within 20 px of any maze node. When it does, the node ID is appended to `saved_nodes` with the synchronized timestamp from the framewise CSV. Consecutive duplicate visits to the same node are de-duplicated before saving.

---

### Velocity Calculation

After each trial ends, segment velocities are computed from the (timestamp, node) sequence:

```
speed = segment_length / time_difference   [m/s]
```

Segment lengths use a hardcoded bridge table for known cross-island distances (e.g. 1.72 m for 121→302) and default to 0.30 m for standard intra-island segments.

---

### RecordingMeta output columns

After processing, a copy of `RecordingMeta.xlsx` is written to the output folder with these columns appended:

| Column | Description |
|---|---|
| `paths` | Comma-separated visited node IDs (e.g. `101,202,303`) |
| `delay` | Trial duration in seconds (start node entry → end condition) |
| `active_time` | Same as `delay` |
| `avg_speed` | Total path distance ÷ total path time (m/s) |
| `avg_between_node_speed` | Mean of per-segment speeds across node transitions (m/s) |
| `trial_start_time` | Sync timestamp (s) when rat enters the start node |
| `trial_end_time` | Sync timestamp (s) when end condition fires |

`trial_start_time` and `trial_end_time` are populated only when a framewise timestamp CSV is present. Files are checked in this order:

1. `<date>_Rat<id>_framewise_ts.csv`
2. `stitched_framewise_seconds.csv`
3. `stitched_framewise_ts.csv`

---

### Tracker — Modifications from Original

Based on [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT). Key changes:

- Detection backend replaced with YOLOv11 (Ultralytics)
- Rat position uses `rat` (body) class only; `head` is counted separately and does not affect the tracked centroid
- Extended trial state machine: NGL variants (types 4–6), researcher-proximity trigger and force-end timers, inter-trial lockout
- Per-trial metrics (`avg_speed`, `avg_between_node_speed`, `active_time`, `trial_start_time`, `trial_end_time`) written back into a copy of `RecordingMeta.xlsx`
- Motion-based YOLO skip with cached bounding-box redraw to prevent display flicker
- Threaded video writer for non-blocking frame output

---

## Node Analysis — Computed Metrics

`src/node_analysis/hex_maze_analysis.py` processes `.xlsx` trial files and appends computed columns.

### Maze structure

96 nodes across 4 islands (101–124, 201–224, 301–324, 401–424) plus 2 homeboxes (501, 502). Two graphs are pre-computed at startup:

- **Node graph** — full 96-node maze for node-level shortest paths and choice analysis.
- **Island graph** — 4-node graph (one node per island) for island-level distance metrics.

### Required input columns

| Column | Description |
|---|---|
| `path_to_reach` | Comma-separated node IDs of the full path |
| `start_node_n` | Start node ID |
| `goal_node_n` | Goal node ID |
| `start_island_n` | Island number (1–4) of the start node |
| `goal_island_n` | Island number (1–4) of the goal node |
| `seq_islands` | Comma-separated island sequence visited |
| `exclude_trial` | `0` = include in Step 2; anything else = skip |
| `comment` | Used as the `flag` message when `path_to_reach` is empty |

### Distance metrics

| Column | Formula |
|---|---|
| `distance_start_goal_island` | `island_graph_distance(start, goal) + 1` |
| `distance_start_goal_nodes` | `node_graph_distance(start, goal) + 1` |

### Path length metrics

| Column | Description |
|---|---|
| `path_length_start_goal_nodes_node_hit` | `len(path)` — total nodes visited |
| `path_length_start_goal_island_node_hit` | `len(seq_islands)` — total island entries |
| `path_length_start_goal_island_island_hit` | `len(set(seq_islands))` — unique islands visited |
| `norm_path_length_start_goal_nodes_node_hit` | Node path length ÷ optimal node distance |
| `norm_path_length_start_goal_island_node_hit` | Island entries ÷ optimal island distance |
| `norm_path_length_start_goal_island_island_hit` | Unique islands ÷ optimal island distance |

### Core behavioral metrics (Step 1)

| Column | Description |
|---|---|
| `shortest_path` | Minimum hops between start and goal node |
| `n_nodes_visited` | Total nodes visited including revisits |
| `food_reached` | `1` if goal appears among the last two path nodes |
| `eat_on_1_encounter` | `1` if the last path node is exactly the goal |
| `dist_tra` | Edges traveled; sentinel `99` if food not reached |
| `dt_rel_sp` | `dist_tra / shortest_path` — relative path length (1.0 = optimal) |
| `dt_min_sp` | `dist_tra − shortest_path` — extra steps beyond optimal |
| `dir_run_mat_perf` | `1` if food reached AND path length equals `shortest_path` |
| `node_choices_binary` | Per-step `0`/`1`: `1` = step minimised remaining distance to goal |
| `perc_correct_choices` | `(sum of 1s / total steps) × 100` |

### Goal-island entry metrics (Step 2)

Computed from the last bridge crossing into the goal island (detected as a step where consecutive node IDs differ by ≥ 50). Skipped when `exclude_trial != 0`.

| Column | Description |
|---|---|
| `isl_node_in` | Node at last entry into the goal island |
| `isl_short_path` | `node_graph_distance(isl_node_in, goal) + 1` |
| `isl_dt_trav` | Nodes from island entry to end of path |
| `perf_in_island` | `isl_dt_trav / isl_short_path` — performance within the island |

Rows with missing or unparseable `path_to_reach` are flagged in a `flag` column and highlighted red in the output; all computed columns for that row are left blank.

---

## Metadata

`RecordingMeta.xlsx` contains per-session and per-trial information. The original file is never modified — the tracker writes a copy to the output folder.

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
| `Special_Trials` | Per-row special trial flags. Accepts a plain trial number (`3`) or the time-locked form `trial_num@MM:SS` (e.g. `3@5:30`) — the trial's start node won't trigger until the given session time, and any earlier active trial is force-ended when that time arrives. See [Time-locked special trials](#time-locked-special-trials). |
| `Did_Not_Reach` | `1` if rat did not reach the goal for this trial (read by the tracker but currently has no effect on trial end logic) |
| `Unnormal_Intervals` | Immunity windows per trial (`trial:start_min-end_min`) |

---

## License

See [LICENSE](LICENSE).
