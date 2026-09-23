# HM Neuron Genzel

**Authors:**

- **Sachuriga S.** — Set up the whole processing pipeline initially.
- **Jacob van Rosmalen** — Set up the pipeline for integrating the behavioral and ephys data into `.nwb` (Neurodata Without Borders).
- **Phan Minh** — Developed the DeepLabCut processing line for tracking the rat's keypoints.
- **Jill Gerritsen** — Tested and validated the whole processing pipeline.

A batch-processing pipeline for rat hexmaze neuroscience experiments — from raw multi-camera video and Trodes ephys recordings all the way to spike-sorted, NWB-packaged sessions with place-field, Bayesian-decoding, theta/gamma, phase-precession and population-UMAP analyses, plus drive-integrity QC tooling.

> **Attribution** — `src/tracker/Video_LED_Sync_using_ICA.py` (Step 2), `src/tracker/join_views.py` (Step 3), and `src/tracker/TrackerYolov11.py` (Step 4) are based on and modified from [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT). See [Tracker — Modifications from Original](#tracker--modifications-from-original).

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Layout](#directory-layout)
3. [Setup](#setup)
4. [Data Layout](#data-layout)
5. [Running the Pipeline](#running-the-pipeline)
6. [Resource Monitoring](#resource-monitoring)
7. [Environment Variables — Full Reference](#environment-variables--full-reference)
8. [Pipeline Steps — Detailed Reference](#pipeline-steps--detailed-reference)
   - [Step 1 — Trodes DIO/Raw/Analog Export](#step-1--trodes-diorawanalog-export)
   - [Step e — Trodes LFP + Analog Export](#step-e--trodes-lfp--analog-export)
   - [Step 2 — LED Sync (ICA)](#step-2--led-sync-ica)
   - [Step 3 — Multi-Camera Stitching](#step-3--multi-camera-stitching)
   - [Step 4 — YOLOv11 Tracker](#step-4--yolov11-tracker)
   - [Step 5 — Trial Plotting](#step-5--trial-plotting)
   - [Step 6 — Video Compression](#step-6--video-compression)
   - [Step 7 — Spike Sorting](#step-7--spike-sorting)
   - [Step c — Continue After Sorting](#step-c--continue-after-sorting)
   - [Step r — Recompute Metrics](#step-r--recompute-metrics-after-manual-phy-curation)
   - [Step 8 — LFP + Motion + EMG-from-LFP](#step-8--lfp--motion--emg-from-lfp)
   - [Step d — DeepLabCut Export + Inference](#step-d--deeplabcut-export--inference)
   - [Step 9 — Cleanup](#step-9--cleanup)
   - [Step f — Fix .txt Unix Timestamps](#step-f--fix-txt-unix-timestamps)
   - [Step n — Node Analysis](#step-n--node-analysis)
   - [Step w — NWB Packaging](#step-w--nwb-packaging)
   - [Step u — Add Curated Units to NWB](#step-u--add-curated-units-to-nwb)
   - [Step v — Visualize NWB Units](#step-v--visualize-nwb-units)
   - [Step b — Bayesian Decoder + Video Overlays + Predictive-Coding Test](#step-b--bayesian-decoder--video-overlays--predictive-coding-test)
   - [Step m — Neural Population UMAP](#step-m--neural-population-umap)
   - [Step t — Drive Scan (QC)](#step-t--drive-scan-qc)
9. [Drive Coverage GUI](#drive-coverage-gui)
10. [Analysis Conventions (clocks, coordinates, gating)](#analysis-conventions-clocks-coordinates-gating)
11. [Standalone Utilities](#standalone-utilities)
12. [Tracker — How It Works](#tracker--how-it-works)
13. [Node Analysis — Computed Metrics](#node-analysis--computed-metrics)
14. [Metadata (RecordingMeta.xlsx)](#metadata-recordingmetaxlsx)
15. [Troubleshooting & Docs](#troubleshooting--docs)
16. [License](#license)

---

## Overview

The pipeline takes raw Trodes recordings (`.rec`) and 12-camera hexmaze video, and runs a configurable sequence of steps to produce:

**Behaviour / video tier**

- LED-synchronized behavioral timestamps via ICA (video clock ↔ ephys clock)
- A single stitched maze video and an annotated YOLOv11 tracking video + per-frame position CSV
- Per-trial metrics written back into a copy of `RecordingMeta.xlsx`
- Trial-level trajectory/speed/occupancy PDF reports (with decoded-position overlays once step b has run)
- DeepLabCut keypoint tracking of eye-camera close-ups
- Behavioral metrics from trial node sequences (hex maze graph analysis)
- Hardware-accelerated video compression (auto-probed NVENC/QSV/AMF/VAAPI/VideoToolbox with CPU fallback)

**Ephys tier**

- Spike sorting (MountainSort5 default / MountainSort4) via SpikeInterface, per tetrode, with automated good/mua/noise labels and Phy export for manual curation
- LFP extraction (1500 Hz, one channel per tetrode), headstage IMU motion, sleep-scoring aids (EMG channel, awakeness index, Buzsáki EMG-from-LFP)

**Integration / analysis tier (NWB)**

- One `.nwb` file per session: positions (rat / researcher / DLC keypoints), trials, LFP, curated units with waveforms + CellExplorer-style cell types
- Per-unit visualization PDFs: rate maps per trial, place fields, event-locked PETHs (trial start / goal arrival / bridge entry), spike–theta & spike–gamma phase coupling, theta–gamma coupling (PAC), classic phase precession on the linearised maze
- Population Bayesian position decoding (with prediction leads), decoded/spike overlays rendered onto the real behaviour video, and a predictive-coding control analysis
- Population-activity UMAP embeddings (Gardner et al. 2022 method) with rich behavioural colourings
- Raw-drive integrity scanning + a Qt "Drive Coverage" GUI for reconciling experiment spreadsheets against acquisition drives

---

## Directory Layout

```text
HM_Tracker_2025/
├── runner.py                  # cross-platform pipeline runner — the single source of truth
├── tracker_gui.py             # "genzeltracker" Qt GUI launcher (imports the menu from runner.py)
├── list_drive.py              # standalone drive-tree lister (debugging aid)
├── pyproject.toml             # package "genzeltracker" (extras: [mac], [gpu])
├── scripts/
│   ├── runner_unix.sh         # thin wrapper: exec python runner.py "$@"
│   └── runner_windows.bat     # thin wrapper: python runner.py %*
├── examples/
│   ├── hm_tracker_paths.example.txt   # config template
│   └── RecordingMeta.xlsx             # per-session metadata template
├── requirements/
│   ├── requirements.txt       # exact-version freeze (cu128), optional
│   ├── constraints.txt        # pip constraints
│   └── reproduce.yml          # minimal conda env (HM_neuron, python 3.13)
├── docs/                      # ERROR_REFERENCE, RecordingMeta_README, TROUBLESHOOTING*,
│                              # experiment protocol, media/
└── src/
    ├── tracker/               # steps 2–5: sync, stitching, YOLOv11 tracker, trial plotting
    ├── sorter/                # steps 7/c/r/8: sorting, curation tools, LFP/motion/EMG export,
    │                          # make_channel_map.py, Trodes reader
    ├── dlc/                   # step d: eye-frame export + DeepLabCut inference
    ├── node_analysis/         # step n: hex_maze_analysis.py (+ its own README)
    ├── nwb/                   # steps w/u/v/b/m: create_nwb, add_units, visualize_nwb,
    │                          # theta_events, decode_position, make_videos,
    │                          # predictive_coding, neural_umap, spike_metrics
    └── tools/                 # vcodec.py, gpuslot.py, scan_drive.py, scan_drive_gui.py
                               # (+ organize/reset/fix/find/summarize/prepare_meta/preprocess_check),
                               # node_list_new.csv, maze_roi.txt, define_maze_roi.py
```

---

## Setup

### 1. Environment

```bat
conda env create -f requirements/reproduce.yml
conda activate HM_neuron

:: Linux / Windows (full GPU pipeline, CUDA 12.8):
pip install -e ".[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128

:: macOS (CPU subset: sync / plotting / node analysis / NWB analyses / GUI):
pip install -e ".[mac]"
```

Dependencies live in `pyproject.toml` (extras `[gpu]` / `[mac]`); installing also creates the **`genzeltracker`** GUI command. Install **editable** (`-e`) — the repo is a script collection, and the GUI finds `runner.py` and `src/` in place.

> - `[gpu]` needs a CUDA-capable GPU and **CUDA 12.8** drivers (`torch==2.10.0+cu128` from the extra index).
> - `mountainsort4` builds isosplit5 from C++ — on Windows install *Microsoft C++ Build Tools* first.
> - `deeplabcut` ships its own torch pin that can conflict with this repo's; verify DLC in your env matches the version your model was trained with before running Step d inference.
> - The three numpy statements are intentionally distinct: `pyproject.toml` wants `numpy<2` (install-time), `requirements/requirements.txt` is a freeze of a working env, `constraints.txt` pins for legacy zarr. When in doubt, follow `pyproject.toml`.
> - If `conda env create` freezes, remove an existing env first (`conda env remove -n HM_neuron`).

### 2. External tools

| Tool | Purpose |
|---|---|
| [Trodes](https://spikegadgets.com/trodes/) (`trodesexport`, `exportLFP`) | `.rec` export: DIO, raw, analog, LFP |
| [FFmpeg](https://ffmpeg.org/) (+ffprobe) | Stitching, compression, drive-scan video checks |
| YOLOv11 weights (`.pt`) | Tracker detection model — see [Model](#model) |
| [Phy](https://github.com/cortex-lab/phy) | Manual spike-sorting curation (between steps 7/c and r) |

### 3. Path config (`hm_tracker_paths.txt`)

Copy the template to your Desktop (or point `HM_CONFIG_FILE` anywhere):

```bash
cp examples/hm_tracker_paths.example.txt ~/Desktop/hm_tracker_paths.txt
```

Format: `KEY=VALUE` lines, `#` comments; whitespace around `=` tolerated. **Every key is exported into the environment** for child steps; the same file is also passed via `--config` to the sorter (which parses its own keys from it). Recognised keys:

```text
# tool paths
FFMPEG_CMD=C:\path\to\ffmpeg.exe          # auto-repaired to system ffmpeg if the path is stale
ONNX_WEIGHTS_PATH=C:\path\to\weights.pt   # YOLOv11 model (step 4)
TRODES_EXPORT_CMD=C:\path\to\trodesexport.exe
TRODES_EXPORT_LFP=C:\path\to\exportLFP.exe

# LED sync (step 2)
SYNC_START_SEC=45        # skip this many s before locating the LED (0 = off)
SYNC_LED=auto            # auto | red | blue

# DeepLabCut (step d)
DLC_CONFIG_PATH=         # path to the trained project's config.yaml; blank = export-only
DLC_SHUFFLE=2

# NWB (step w)
NWB_RAT_NR=1             # fallback only — RecordingMeta.xlsx Rat_ID wins

# spike sorting (step 7) — global
SORTER=mountainsort5     # or mountainsort4
FREQ_MIN=600
FREQ_MAX=8000
DETECT_THRESHOLD=5
DETECT_SIGN=0            # -1 neg, 0 both, 1 pos

# spike sorting — per rat (RAT token matched as a prefix of the recording name)
BAD_CHANNELS_RAT1=0 1 2 3 NT8ch1 NT8ch2   # interpolated (50 µm radius)
REF_CHANNEL_RAT1=NT17ch1                  # referenced before the global median CAR
EEG_TETRODES_RAT1=NT1 NT32                # whole tetrodes dropped before sorting
SLEEP_CHANNELS_RAT1=cortex:NT28 sr:NT10 pyr:NT5   # sleep-scoring tetrode roles (step 8)
```

Channel ids accept plain 0-based numbers (`0`–`127`) or tetrode notation `NT<t>ch<c>` — `channel = (t−1)×4 + (c−1)`; 32 tetrodes × 4 channels (see the [NT mapping table](#tetrode-nt-channel-mapping)).

---

## Data Layout

The runner expects input/output folder pairs named `ipN` / `opN` inside the target directory:

```text
data_root/
├── ip1/    ← raw input (.rec, eye??_*.mp4 + .meta, *RecordingMeta.xlsx)
├── op1/    ← processed output (created if missing)
├── ip2/
└── op2/
```

- **Parallel steps** (`1 e 2 3 4 5 8 d n`) process each `ipN → opN` pair in its own worker.
- **Sequential steps** (`7 c r 9 f w u v b m t`) run at master level after all workers finish — steps `7 c r 9 f u v b m` loop one folder at a time; steps `w` and `t` run **once over the whole root**. Steps c/r/u/v/b/m also accept **session-date folders** (`YYYYMMDD/` whose name appears in a file inside them) as targets — the layout used on the processed network share.
- **Step 6** (compression) is always pulled out and runs **last**, over every top-level `.mp4` of every `op*` folder.

---

## Running the Pipeline

### GUI

```bash
genzeltracker
```

The Qt window imports the step menu straight from `runner.py` (GUI and CLI can never disagree), offers per-step checkboxes and one-click presets, then launches `runner.py` with `HM_STEPS` set:

| Preset | Steps |
|---|---|
| Tracker implanted | `1e2345678d` |
| Tracker non-implanted | `23456d` |
| After manual curation | `wuvbm` |
| Retrack | `346d` |
| Full pipeline | `1e234567c89wuvnbmt` |
| Trodes export (DIO/raw/analog + LFP) | `1e` |
| Sync + stitch + track | `234` |
| Spike sorting (+ continue) | `7c` |
| LFP + motion + EMG (sleep) | `e8` |
| NWB packaging (nwb + units + visualise) | `wuv` |
| Analysis (decode / UMAP) | `nbm` |
| Drive scan (QC) | `t` |

### CLI

```bash
python runner.py /path/to/data_root          # interactive step prompt
HM_STEPS=wuvbm python runner.py /path/to/data_root   # non-interactive
scripts/runner_unix.sh /path/to/data_root    # thin wrappers (PYTHON env picks the interpreter)
scripts\runner_windows.bat "C:\path\to\data_root"
```

The interactive menu:

```text
[1] Trodes Export (DIO/Raw/Analog)
[e] Trodes Export LFP + Analog (per channel)
[2] Sync Script
[3] Stitching
[4] Tracker
[5] Plotting
[6] Compression (always runs LAST, over all op folders)
[7] Sorting
[c] Continue After Sorting (metrics + BombCell + Phy, no re-sort)
[r] Recompute Metrics (after manual Phy curation)
[8] LFP + Motion (IMU Accel) + EMG-from-LFP
[d] deeplabcut (extract eye frames + run DLC inference -> keypoints in CSV)
[9] Cleaning
[f] Fix .txt unix timestamps (repair re-tracked sessions; framewise ts<->seconds mapping)
[n] Node Analysis
[w] nwblfp (NWB / LFP package)
[u] Add curated Units (metrics + waveforms) to NWB (runs after w)
[v] Visualize NWB units (summary + per-unit rate-map PDFs; runs after u)
[b] Bayesian position decoder + spikes/decoded-on-video overlays per session (good and good+mua)
[m] Neural population UMAP per session (good & good+mua, all + pyramidal-only; Gardner et al. 2022)
[t] Drive scan (videos playable + ephys has pre/task/post + non-zero .rec)
```

Type any combination (order irrelevant — execution order is fixed: parallel workers `1→e→2→3→4→5→8→d→n`, then sequential `7→c→r→9→f→w→u→v→b→m→t`, then `6` last). A stray `s` is ignored with a notice — the cross-session summary moved to the HM_Rat_Analysis repo.

Each parallel worker opens its **own terminal window** by default (`runner.py --worker <ip> <op> <steps> <marker>`); the master polls a `<tmp>/hm_worker_<name>.done` marker. Set `WORKER_WINDOWS=0` to run workers as background threads logging to `<tmp>/hm_worker_<name>.log`. Workers are launched `LAUNCH_GAP` (20 s) apart.

> Closing a worker's terminal before it finishes means its `.done` marker is never written — the master waits forever. Kill and re-run instead.

---

## Resource Monitoring

Before each worker launch, the master checks system load (psutil for CPU/RAM, `nvidia-smi` for GPU — each reads as 0% when unavailable, so the gate never blocks on missing tools). If any threshold is exceeded it waits `WAIT_SECONDS` and rechecks.

| Env var | Default | Meaning |
|---|---|---|
| `MAX_CPU` | 90 | max CPU % |
| `MAX_GPU` | 90 | max GPU % (`nvidia-smi`) |
| `MAX_MEM` | 65 | max RAM % |
| `WAIT_SECONDS` | 30 | pause between rechecks |
| `LAUNCH_GAP` | 20 | seconds between worker launches |
| `DISABLE_RESOURCE_CHECK` | 0 | `1` skips the gate entirely |

---

## Environment Variables — Full Reference

Everything in `hm_tracker_paths.txt` is exported as env; these can also be set directly in the shell.

| Variable | Default | Used by |
|---|---|---|
| `HM_STEPS` | — (interactive) | runner: non-interactive step selection (set by the GUI) |
| `HM_CONFIG_FILE` | `~/Desktop/hm_tracker_paths.txt` | runner + export_lfp: config location |
| `WORKER_WINDOWS` | 1 | runner: 0 = background-thread workers |
| `MAX_CPU` / `MAX_GPU` / `MAX_MEM` | 90 / 90 / 65 | resource gate |
| `WAIT_SECONDS` / `LAUNCH_GAP` | 30 / 20 | resource gate / launch pacing |
| `DISABLE_RESOURCE_CHECK` | 0 | 1 disables the gate |
| `FREQ` | 30000 | step 2: Trodes DIO sample rate (`-f`) |
| `SYNC_START_SEC` | 45 | step 2: LED detection skip window (0 = off) |
| `SYNC_LED` | auto | step 2: auto \| red \| blue |
| `SYNC_DEBUG` | — | step 2: 1 = per-video debug output (`sync_debug/`) |
| `FFMPEG_CMD` | PATH `ffmpeg` | steps 3/6, vcodec, scan_drive |
| `FFPROBE_CMD` | sibling of `FFMPEG_CMD` | step 6 probe, scan_drive |
| `FFMPEG_VCODEC` | auto-probe | vcodec: force an encoder (e.g. `libx264`) |
| `FFMPEG_HWACCEL` | off (software) | vcodec: decode acceleration for stitching (`auto` probes) |
| `NVENC_SLOTS` | measured | gpuslot: override NVENC session limit |
| `NVENC_SLOT_TIMEOUT` | 1800 | gpuslot: wait for a GPU slot before CPU fallback |
| `NVENC_SLOT_DIR` | `<tmp>/hm_tracker_gpu_slots` | gpuslot: slot/cache dir |
| `ONNX_WEIGHTS_PATH` | — | step 4: YOLO weights |
| `TRODES_EXPORT_CMD` / `TRODES_EXPORT_LFP` | — | steps 1/e binaries |
| `DLC_CONFIG_PATH` | — (inference skipped) | step d part 2 |
| `DLC_SHUFFLE` | 2 | step d part 2 |
| `NWB_RAT_NR` | 1 | step w fallback rat number |
| `DECODE_FOLDS` | 1 | step b: visualisation-track folds |
| `DECODE_CV_FOLDS` | 5 | step b: held-out accuracy folds |
| `DECODE_LEADS` | `0 1 3` | step b: prediction leads (`""` disables multi-lead mode) |
| `DECODE_VIDEO` | 1 | step b: 0 disables both video overlays |
| `DECODE_VIDEO_LEADS` | `0 1 2 3` | step b: leads on the decoded video |
| `PREDICTIVE_CODING` | 1 | step b: 0 disables the control analysis |
| `PC_CV_FOLDS` / `PC_SHUFFLE` | 5 / 8 | predictive-coding folds / shuffle nulls |
| `PYTHON` | python3/python | scripts/runner_unix.sh interpreter override |

Sorter keys (`SORTER`, `FREQ_MIN/MAX`, `DETECT_*`, `BAD_CHANNELS_<RAT>`, `REF_CHANNEL_<RAT>`, `EEG_TETRODES_<RAT>`, `SLEEP_CHANNELS_<RAT>`) are read from the `--config` **file** by the sorter/LFP steps, not from env.

---

## Pipeline Steps — Detailed Reference

### Step 1 — Trodes DIO/Raw/Analog Export

**Script:** `trodesexport` (external). Per `.rec` in `ip`:

```bash
trodesexport -dio -raw -analogio -rec <file.rec>
```

Exports DIO TTL events (LED sync pulses), raw 30 kHz voltage `.dat` per channel group, and analog (headstage IMU) channels — into subfolders next to the `.rec` (`<stem>.DIO/`, `<stem>.raw/`, …). Required before steps 2 (sync), 7 (sorting) and the step-8 raw fallback. Skipped with a warning if the binary path doesn't resolve.

### Step e — Trodes LFP + Analog Export

**Script:** `exportLFP` + `trodesexport` (external). Per `.rec`:

```bash
exportLFP -rec <file.rec> -outputrate 1500 -lfplowpass 700
trodesexport -analogio -rec <file.rec>
```

LFP at **1500 Hz** (low-pass **700 Hz**) into `<recording>.LFP/` — one `.dat` per channel + `*.timestamps.dat`; analog IMU into `<recording>.analog/`. Consumed by step 8. The 1500 Hz rate is load-bearing: the EMG-from-LFP method needs Nyquist > 600 Hz.

### Step 2 — LED Sync (ICA)

**Script:** `src/tracker/Video_LED_Sync_using_ICA.py`

```bash
python src/tracker/Video_LED_Sync_using_ICA.py -i <ip> -o <op> -f 30000 \
       --start-sec 45 --sync-led auto [--debug]
```

Aligns the video timeline to the Trodes clock via the blinking sync LED in each eye camera:

1. **LED localization** per `*eye*.mp4` — max frame-difference region (top 100 px ignored), 16×16 crop.
2. **FastICA** (3 components, seeded) on the crop luminance + KMeans ON/OFF → blink train.
3. **Component classification** — red blinks at 0.5 Hz, blue at 2.5 Hz (frequency within 10 %, duty cycle near 0.5; assumes 30 fps).
4. **Regression** of blink times against the DIO TTL edge trains (`Din1` = blue, `Din2` = red; recursive multi-pattern `.dat` discovery) maps every frame timestamp onto the ephys clock.
5. **Outputs (op):** `stitched_framewise_ts.csv` (*Corrected Time Stamp*) and `stitched_framewise_seconds.csv` (*Seconds From Creation* — **the master session clock**, see [Analysis Conventions](#analysis-conventions-clocks-coordinates-gating)).

Options & rescue knobs: `--start-sec`/`SYNC_START_SEC` skips the first N s for detection only (LED often repositioned early; per-frame output still covers everything); `--sync-led`/`SYNC_LED` forces a colour (auto = prefer blue, fall back red); manual overrides in **ip**: `led_crop_override.txt` (`filename, x, y`) and `led_ica_override.txt` (`filename, color, component`); `--debug`/`SYNC_DEBUG=1` writes `sync_debug/` diagnostics (localization/crop/ICA PNGs + CSVs) *before* the sync stage so they survive failures. Videos with no detectable LED are parked in `<ip>/temp_no_led/` during the run and restored after (an aborted run can leave them there).

### Step 3 — Multi-Camera Stitching

**Script:** `src/tracker/join_views.py`

```bash
python src/tracker/join_views.py <ip> [-n 12] [-g 'eye??_*.mp4'] [-c libx264] [-d 60] [-D]
```

Stitches the 12 `eye??_*.mp4` views into one canvas: 2 rows × 6 columns, per-view crop 104 px horizontal / 91 px vertical (600×800 source frames), bottom row hflip+vflip, ~2352×1424 16-px-aligned canvas at 30 fps → **`stitched.mp4` written into the ip folder** (step 4 looks for it there). `-d N` stitches only the first N seconds to `stitched_<N>s.mp4` (benchmarking; never clobbers a finished stitch). `-D` prints the ffmpeg command without running.

Encoding is delegated to **`src/tools/vcodec.py`**: it probes *real encodes at the real canvas size* down a ladder of hardware encoders (NVENC → QSV → AMF → Media Foundation → VAAPI → VideoToolbox) and falls back to `libx264`; `FFMPEG_VCODEC` forces a codec, `FFMPEG_HWACCEL=auto` probes decode acceleration (software decode is the measured-faster default for the small eye frames). **`src/tools/gpuslot.py`** rations the limited NVENC sessions across parallel workers: the session limit is measured once per machine and cached; workers without a slot wait up to `NVENC_SLOT_TIMEOUT` then encode on CPU. `python src/tools/gpuslot.py` shows slot state; `--measure` re-measures after a driver update.

> The `-cx/-cy/-rows` flags exist but are currently ignored — crop 104/91 and 2 rows are hardcoded in the command builder.

### Step 4 — YOLOv11 Tracker

**Script:** `src/tracker/TrackerYolov11.py` — **requires `<ip>/stitched.mp4`** (step silently skipped otherwise) and `*RecordingMeta.xlsx` in ip.

```bash
python src/tracker/TrackerYolov11.py --input_folder <ip> --output_folder <op> --onnx_weight <weights.pt>
```

Runs the full detect → classify → trial-state-machine loop on the stitched video (details in [Tracker — How It Works](#tracker--how-it-works)). Every frame is processed at `DISPLAY_SIZE` **1176×712** — all pixel coordinates in CSVs, logs and the maze ROI live in this frame. Fixed inputs resolved from the repo root (the runner chdirs there): node map `src/tools/node_list_new.csv` and maze polygon `src/tools/maze_roi.txt` (restricts *rat* detections only; the run refuses to start on a resolution mismatch — redraw with `src/tools/define_maze_roi.py`). Sync CSVs are read from **op**, seconds files first (priority: `stitched_framewise_seconds.csv` → `<date>_Rat<N>_framewise_seconds.csv` → `stitched_framewise_ts.csv` → `<date>_Rat<N>_framewise_ts.csv`) — the unix-clock `_ts` files are a last resort only, so re-runs keep the `.txt` on session seconds.

**Outputs (op):**

| File | Contents |
|---|---|
| `<date>_Rat<N>.mp4` | Annotated video (boxes, trail, node markers, overlays) |
| `<date>_Rat<N>_Coordinates_Full.csv` | Per frame: `Frame_Index, Timestamp, Trial_Num, Rat_X/Y, Researcher_X/Y, JP_S_X/Y, JP_L_X/Y` |
| `<date>_Rat<N>.txt` | Per-trial node sequences, segment timing, velocities |
| `log_<date>_Rat<N>.log` | Run log (overwritten each run by design) |
| `RecordingMeta.xlsx` (copy) | Original + appended per-trial columns (see [RecordingMeta output columns](#recordingmeta-output-columns)) |
| `<date>_Rat<N>_framewise_seconds.csv` / `..._framewise_ts.csv` | Both sync CSVs, session-prefixed at the end of the run |

> Not headless: a live cv2 preview window opens; pressing `q` aborts the run.

### Step 5 — Trial Plotting

**Script:** `src/tracker/plot_trials.py`

```bash
python src/tracker/plot_trials.py --input_folder <ip> --output_folder <op> \
       [--no-seam-repair] [--max-speed 0.6] [--jump-ratio 3.0] [--max-spread 1.0]
```

- Prefers `*Coordinates_Full_with_frames.csv`, else `*Coordinates_Full.csv` + `stitched_framewise_seconds.csv` joined by `Frame_Index` — putting trial windows on the same stitched-seconds clock the decoder and NWB use. Falls back to parsing the `.log` (with a warning; no frame-index clock).
- Repairs one-frame "teleports" across camera-stitch seams (crossing a tile seam AND exceeding `--max-speed` AND `--jump-ratio`× local speed → interpolated).
- **Decoded-track overlay:** every `<op>/decoding/decoded_*.npz` from step b is loaded; per-trial decoded panels and actual-vs-decoded pages are added automatically.
- Output: `<op>/<YYYYMMDD>_Rat<N>_analysis_final.pdf` (session-anchored name — reruns overwrite in place). Pages: metadata summary, per-trial trajectory/speed, histograms, occupancy/speed maps, learning-index, graph-theory metrics, decoded-track pages.

### Step 6 — Video Compression

Master-level, **always last**, over every top-level `.mp4` in every `op*` folder. The encoder comes from `vcodec.select(mode='quality')`, probed at the **largest actual frame size found** (so an old GPU that can't encode the ~2352×1424 stitch fails at probe time, not mid-run); CPU fallback `libx264 -preset veryfast -crf 28`. Each video is re-encoded to `__temp_compressed.mp4` then atomically moved over the original; on failure the temp file is deleted and the original preserved. Force a codec with `FFMPEG_VCODEC=libx264`.

### Step 7 — Spike Sorting

**Script:** `src/sorter/sorting.py`

```bash
python src/sorter/sorting.py --input_folder <ip> --output_folder <op> --config <hm_tracker_paths.txt>
```

1. **Discovery** — `**/*.raw/*_group0.dat` under ip. **Split-recording grouping:** files sharing (rat, date, phase) — phase from the `.raw` folder name (`pre` / `maze|mazs|awake|hab` / `post`) — are one job: parts ordered by acquisition time and lazily concatenated, so a crash-restarted phase sorts as one continuous recording.
2. **Loading** — file-backed (memmap layout via `trodesBinaryLayout()`), never into RAM; fs 30 kHz, gain **0.195 µV/bit**.
3. **Probe** — 8×4 tetrode grid (32 tetrodes × 4 = 128 channels), 250 µm spacing, ±10 µm diamond contacts, `group` per tetrode.
4. **Preprocessing** — drop `EEG_TETRODES_<RAT>` channels → bandpass `FREQ_MIN`–`FREQ_MAX` (600–8000) → interpolate `BAD_CHANNELS_<RAT>` (50 µm) → optional `REF_CHANNEL_<RAT>` single-reference → global-median CAR → saved to `processed_binary/`. No manual whitening (the sorter whitens itself).
5. **Sorting** — MountainSort5 (default, scheme 2) or MountainSort4 per tetrode (`run_sorter_by_property('group')`), selectable via `SORTER`.
6. **Post-sorting** (shared with step c): duplicate-spike removal (0.3 ms censor) + redundant-unit removal → SortingAnalyzer (waveforms 1/2 ms, 3-PC, quality metrics) → automated **good/mua/noise** labels (noise gate: SNR > 2.5 & >100 spikes; mua gate: isolation distance > 8 & ISI violations < 0.2) → **Phy export** with `copy_binary=False` (`params.py` points at `processed_binary/` — the >100 GB recording is not duplicated) → writes `quality_check_labels.csv` and pre-fills `cluster_group.tsv` so units arrive pre-labelled in Phy.
7. **Cleanup** — everything deleted except `phy_export/` and `processed_binary/`.

Output: `<op>/<stem>_<sorter>_sorting_output/`. Curate with `phy template-gui phy_export/params.py`.

> - **Re-running step 7 wipes the whole sorting-output folder** including manual curation inside it.
> - **Never delete `processed_binary/`** — Phy's raw-trace view and step r depend on it.
> - The sorter scripts exit 0 even on fatal errors so batches continue — check logs, not exit codes.

#### Tetrode (NT) channel mapping

`channel = (NT − 1) × 4 + (ch − 1)`, NT 1–32, ch 1–4 → channels 0–127 (NT1 = 0–3, NT2 = 4–7, … NT32 = 124–127).

### Step c — Continue After Sorting

**Script:** `src/sorter/continue_sorting.py --output_folder <op> [--n_jobs 4] [--keep-intermediates]`

Re-runs the shared post-sorting stage (metrics, labels, Phy export, cleanup) on an existing `final_sorting_result/` + `processed_binary/` — for sorts that crashed *after* the sorting itself. Skips folders already finalized (only `phy_export` left). The "BombCell" in the menu label is `spikeinterface.curation.bombcell_label_units` used purely as the thresholding engine.

### Step r — Recompute Metrics (after manual Phy curation)

**Script:** `src/sorter/recompute_metrics.py --output_folder <op|sorting_output|phy_export> [--n_jobs 4]`

Recomputes all quality + template metrics on the **curated** sorting (loaded directly from `spike_times.npy` + `spike_clusters.npy` so manual splits/merges are honoured) and writes into `phy_export/`: `curated_quality_metrics.csv`, `curated_template_metrics.csv`, per-metric `cluster_<metric>.tsv` (visible as Phy columns), **`curated_templates.npy`** (+ unit/channel id files — Phy's own `templates.npy` is stale after merges), refreshed `quality_check_labels.csv`, and overwrites `cluster_group.tsv` with recomputed labels (merges/splits survive; label-only manual edits are recomputed).

### Step 8 — LFP + Motion + EMG-from-LFP

Three scripts run in order per worker, all writing into `<op>/LFP_Output/` with a `<Rat>_<YYYYMMDD_HHMMSS>_` session prefix:

```bash
python src/sorter/export_lfp.py          --input_folder <ip> --output_folder <op> --output_rate 1500
python src/sorter/export_emg_from_lfp.py --input_folder <ip> --output_folder <op>
python src/sorter/export_motion.py       --input_folder <ip> --output_folder <op>
```

**`export_lfp.py`** — reads `<recording>.LFP/*.dat` (voltage × header `voltagescaling`, default **0.195 µV/bit**; `--output_rate 1500` overrides the sometimes-lying header fs). Multiple `.LFP` folders = sessions concatenated chronologically; only channels common to all sessions kept, sorted ntrode-then-channel. Outputs:

| File (`{pfx}` = session prefix) | Contents |
|---|---|
| `{pfx}lfp_data.npy` | (n_samples × n_channels) float32 µV |
| `{pfx}lfp_timestamps.npy` | synthetic gapless seconds = `arange(n)/fs` |
| `{pfx}session_boundaries.npy` | `{name, start, n}` per concatenated session |
| `{pfx}channel_map.npy` | `{index, ntrode, channel, source_file}` per column |
| `channels_npy/{pfx}lfp_nt<NN>_ch<CC>.npy` | one file per channel (fast single-channel reads) |
| `{pfx}sleep_channels.npy` | `{cortex, sr, pyr}` tetrodes from `SLEEP_CHANNELS_<RAT>` |
| `{pfx}emg_data.npy` / `{pfx}emg_channel_index.npy` | channel with max 20–200 Hz power (raw-LFP "EMG") |
| `{pfx}cleanest_channel_indices.npy` / `{pfx}channel_snr_scores.npy` | top-3 sleep-band-SNR channels |
| `{pfx}awakeness.npy`, `{pfx}emg_rms.npy`, `{pfx}theta_delta_ratio.npy` | per-sample awakeness index (0.6·zEMG + 0.4·zθ/δ) |

**`export_emg_from_lfp.py`** — Buzsáki EMG-from-LFP (300–600 Hz band-pass, sliding ±0.2 s mean pairwise cross-channel correlation): `{pfx}emg_from_lfp_5hz.npy` (+timestamps), `{pfx}emg_from_lfp.npy` upsampled onto the LFP time axis. Falls back to raw 30 kHz `.dat` / `.rec` if the LFP export is missing or too low-rate.

**`export_motion.py`** — headstage IMU `AccelX/Y/Z` from `.analog` folders, resampled to 1500 Hz: `{pfx}motion.npy`, `{pfx}motion_timestamps.npy`, `{pfx}motion_accel.npy` (movement envelope, TheStateEditor port), `{pfx}motion_session_boundaries.npy`.

> - **Timestamps are synthetic** (sample/fs, gapless): real inter-recording gaps are not represented — use `session_boundaries.npy`.
> - Two different "EMG" families exist: `emg_data`/`emg_rms` (one raw LFP channel) vs `emg_from_lfp*` (cross-channel correlation muscle-tone estimate).
> - Running `export_lfp.py` by hand **must** include `--output_rate 1500` or downstream timestamps can be ~20× off.

### Step d — DeepLabCut Export + Inference

**Scripts:** `src/dlc/tracking_eyes.py` → `src/dlc/dlc_coordinates.py`

Part 1 maps every tracked frame to the eye camera that captured it (1176×712 gridded into 6×2 regions of 196×356 px; bottom-row frames flipped 180°) and compiles the per-frame close-ups into `collected_frames.mp4` + `<csv>_with_frames.csv` (adds `region_id`, `extracted_frame_idx`). Opens a live preview by default (`--no-vis` disables; `q` aborts).

Part 2 runs **only if `DLC_CONFIG_PATH` points to an existing DLC `config.yaml`** (shuffle from `DLC_SHUFFLE`, default 2): `deeplabcut.analyze_videos` on `collected_frames.mp4`, merges per-bodypart `_x/_y` back into the CSV rows, **re-centres all keypoints on `mid_brain`** (that bodypart must exist), overwrites the CSV in place, and writes a DLC-labelled QC video.

### Step 9 — Cleanup

Inline in `runner.py` (`clean_folder`): deletes top-level directories in each **ip** folder matching `*.DIO`, `*.raw`, `*timestampoffset*`. The `.rec` files are never touched. Irreversible; errors silently ignored — run only after steps 2/7 are verified.

### Step f — Fix .txt Unix Timestamps

**Script:** `src/tracker/fix_txt_timestamps.py --output_folder <op> [--dry_run]`

Remediation for sessions re-tracked before the sync-CSV rename fix: their `<date>_Rat<N>.txt` carries unix timestamps (~1.7e9 s) instead of session seconds. The two sync CSVs are frame-aligned, so unix → seconds is an exact per-frame mapping: the step builds it from `*framewise_ts.csv` ↔ `*framewise_seconds.csv` and rewrites the `.txt` in place (original kept once as `<name>.txt.unixbak`). Only numeric values > 1e6 in the two timestamp positions (`Trial End (Sync Seconds):` and each transition's `(t_start, t_end)`) are converted — healthy files are left byte-identical, so the step is idempotent; `N/A` and durations are untouched. Re-run `w → u` afterwards if the session's NWB `Trials_Data` should pick up the corrected values.

### Step n — Node Analysis

**Script:** `src/node_analysis/hex_maze_analysis.py --input_folder <ip> --output_folder <op>`

Processes every `.xlsx` in ip (sheet `raw`, else the first sheet) and writes a formatting-preserving copy `<name>_results.xlsx` with behavioural metrics computed from trial node sequences. See [Node Analysis — Computed Metrics](#node-analysis--computed-metrics). Rows with unusable paths are flagged and highlighted red.

### Step w — NWB Packaging

**Script:** `src/nwb/create_nwb.py` — runs **once over the whole root** at master level:

```bash
python create_nwb.py --rat_nr $NWB_RAT_NR --noroot --ip <ROOT> --op <ROOT>
```

Discovers session folders (`op*` or `YYYYMMDD`-named) and packages each into `<op>/Rat<N>_<YYYYMMDD>.nwb` (atomic write via `.tmp.nwb`). **Requires `*Coordinates_Full_with_frames.csv`** per session (skipped otherwise). Contents:

- `session_start_time` — tz-aware (Europe/Amsterdam), anchored so that clock-zero round-trips with the unix `Timestamp` column; scalars (`Rat_ID, Date, Repeat, Day, Session, Goal_Node`) from `RecordingMeta.xlsx` row 0 (authoritative — `--rat_nr` is only a fallback).
- `acquisition/lfp` — `TimeSeries` (µV) from `LFP_Output`'s `lfp_data.npy`/`lfp_timestamps.npy` if present.
- `processing/Behavior`:
  - `Position` — one `SpatialSeries` per tracked entity (`Rat`, `Researcher`, `JP_S`, `JP_L`), pixel units, timestamps on the stitched-seconds clock;
  - `DLC_Position` — per-bodypart series (mid-brain-centred), if step d ran;
  - `Metrics` — `region_id`, `extracted_frame_idx`;
  - `Trials_Data` (`DynamicTable`) — per-node-transition rows from the tracker `.txt`, plus **`Trial_start_s` / `Trial_end_s`** (trial windows on the position/spike clock — the *only* trial times comparable with positions; `Trial_start_time`/`Trial_end_time` are on the incompatible behavioural-sync clock).

### Step u — Add Curated Units to NWB

**Script:** `src/nwb/add_units.py --output_folder <op> [--n_jobs 4] [--skip-waveforms]`

Appends the curated Phy units into the session NWB (in place). Templates are **always recomputed** from the recording referenced by `phy_export/params.py` (`dat_path` → `processed_binary/`, since the export uses `copy_binary=False`); cached/stale template files are never trusted. Units columns: `spike_times` (seconds, position clock), `waveform_mean`, `phy_cluster_id`, `sorting_group`, **`quality_label`** (manual Phy `cluster_group.tsv` — the human truth), **`auto_quality_label`** (automated), all curated metric CSV columns, plus recomputed `firing_rate_hz`, `trough_to_peak_s`, `peak_half_width_s`, `trough_half_width_s`, `acg_tau_rise_ms`, and **`cell_type`**.

**Cell-type rule** (`src/nwb/spike_metrics.py`, CellExplorer + FR gate): *interneuron* if FR > 10 Hz, OR trough-to-peak ≤ 0.425 ms (narrow), OR (trough-to-peak > 0.425 ms AND ACG τ_rise > 6 ms, wide); else *pyramidal*.

> Step u **skips** an NWB that already has Units. Regeneration recipe after manual curation: **r → w → u** (`HM_STEPS=rwu python runner.py <root>`).

### Step v — Visualize NWB Units

**Script:** `src/nwb/visualize_nwb.py`

```bash
python src/nwb/visualize_nwb.py --output_folder <op> [--bin_cm 5.0] [--smooth 2.0] [--speed 0.05] \
       [--units CID ...] [--no_theta_events]
```

Writes `<op>/visualization/<pfx>summary.pdf` + one `<pfx>Unit_<cid>.pdf` per **good** unit. `--units` restricts which unit PDFs are written (fast iteration); `--speed` is the run-epoch gate applied to all rate maps.

**`summary.pdf`:** CellExplorer classification plane + firing-rate gate → quality-metric histograms → place-field outline overlay of all pyramidal cells → small-multiple rate maps sorted by spatial information.

**`Unit_<cid>.pdf` page order:**

1. Unit summary (amplitude vs time, waveform template, stats + classification rule, wide/narrow autocorrelograms)
2. Full-duration spikes-on-path + rate map (all of a unit's maps share one colour ceiling = 0.75× its full-duration peak)
3. One spatial page per trial (goal = gold star, start = green ring)
4. Free-roaming trials again as their own pages; before/after the 2nd free-roaming trial (remapping check, when present)
5. Between-trials maps on the **Researcher** trajectory (the rat is carried between trials, its own tracker freezes)
6. Speed–firing-rate correlation panels
7. **theta_events pages** (`src/nwb/theta_events.py`; skipped by `--no_theta_events`), each split three ways — **goal running** (type-1 trials) / **free roaming** (type-4/5) / **all**:
   - **PETH + classic vertical-tick rasters** (±3 s, 100 ms bins) around **trial start**, **goal arrival** (0.20 m radius, first arrival per goal run with trial-end fallback; every debounced entry in free-roaming) and **bridge entry** (stepping onto one of the long ~0.7 m **inter-island** bridges only — short honeycomb edges don't count). All PETH panels of a unit share one rate scale; rates are coverage-corrected at record edges.
   - **Spike–theta phase coupling** (6–10 Hz, double-plotted histograms, R / Rayleigh p / preferred phase; 0° = theta peak; shared y scale).
   - **Spike–gamma phase coupling** — mid gamma 50–90 Hz and high gamma 90–150 Hz rows (shared y scale).
   - **Session theta–gamma page** — phase–amplitude coupling (18 theta-phase bins, Tort MI) per band + gamma event rates (envelope > mean+2 SD, peak > 3 SD, ≥3 cycles).
   - **Classic phase precession** on the linearised hexmaze: 1-D axis = graph distance to the goal (Dijkstra over the maze graph incl. inter-island corridors); place fields detected in **2-D at ≥35 % of peak** (1-D-only detection is degenerate on a maze where paths converge); every speed-gated *approach* pass through a field is normalised 0 = entry → 1 = exit, all passes and fields pooled into one phase-vs-position plot per split with a circular-linear fit (negative slope = precession).

   Theta/gamma come from the `LFP_Output` export: the channel with the best theta/delta ratio in a movement-rich block is band-passed + Hilbert-transformed; the LFP maze recording and the spike clock share t = 0 (guarded via `session_boundaries.npy`; an unalignable multi-session export skips the phase pages loudly rather than computing phases on the wrong epoch). Without an LFP export the PETH pages still render.

### Step b — Bayesian Decoder + Video Overlays + Predictive-Coding Test

Runs per session for **both** quality sets `good` and `good+mua`:

1. **`src/nwb/decode_position.py`** — population Bayesian decoding (Poisson likelihood, flat prior; 10 cm bins, 0.5 s time bins, tuning curves and evaluated bins **speed-gated to run epochs** at 0.05 m/s). With `DECODE_LEADS` (default `0 1 3`) it also *predicts the future position* at each lead and compares them cross-validated. Outputs to `<op>/decoding/`:
   - `decoded_<tag>.npz` — full-data lead-0 track (`t` on the stitched-seconds clock, metres) that plot_trials overlays; unprefixed by design.
   - `<pfx>decode_<tag>.pdf` — decoding report + per-trial panels; `decode_leads_<tag>.pdf` + `leads_summary_<tag>.npz` — cross-validated error-vs-lead comparison.
   - `<pfx>trial_unit_metrics_<tag>.csv/.pdf` + NWB scratch table — per-(trial, unit) spatial info / field size / selectivity / FR / decoding error / task performance (`log10(shortest/actual hops)`) / between-node speed, with correlation figures.
2. **`src/nwb/make_videos.py --which decoded`** — one mp4 per goal trial in `<op>/videos/`: the decoded position per lead (jet colours, accumulating dots, "prediction-reach" arrow from the animal) overlaid on the **real annotated behaviour video** (video frame k ↔ `Frame_Index` k, verified). Free-roaming trials are excluded by default.
3. **`src/nwb/predictive_coding.py`** — is a long-lead "prediction" genuine or just behavioural autocorrelation/goal occupancy? Four pages into `<op>/predictive_coding/`: cross-validated neural error vs *persistence*, *constant-velocity* and *behaviour-only* baselines + a spike-shuffle null band (over moving bins); overshoot vs distance-to-goal; decoded-density maps neural vs shuffle; goal-switch density (or a note when the session has a single fixed goal).
4. Once per op: **`make_videos.py --which spikes --quality good`** — per goal trial, the top-20 good pyramidal cells by spatial information, spikes (speed-gated, jet-coloured per unit, FR-ranked perpendicular jitter) accumulated on the real video.

make_videos options: `--which spikes|decoded|both`, `--n_units 20`, `--leads 0 1 2 3`, `--quality good [mua]`, `--exclude_types 4 5`, `--trials N..`, `--stride N` (keep every Nth frame; output fps = source fps / N — fps itself comes from the source video), `--hold_s 0.6`, `--bin_cm 10`, `--time_bin 0.5`, `--no_jitter`, `--fr_offset_px 15`, `--speed_thresh 0.05`.

> The visualisation track (`decoded_*.npz`) is deliberately in-sample (smooth); quote accuracy only from the cross-validated `decode_leads_*.pdf` / `leads_summary_*.npz`.

### Step m — Neural Population UMAP

**Script:** `src/nwb/neural_umap.py` — Gardner/Hermansen (Nature 2022) population embedding: 0.1 s bins, 0.3 s Gaussian smoothing, √-transform, per-neuron z-score, 3-D UMAP (cosine metric), moving bins only (>0.05 m/s), ≤25 000 bins. The runner produces **four embeddings per session**: `good`, `good+mua`, each also pyramidal-only (`--cell_type pyramidal`). Outputs `<op>/umap/<pfx>umap_<tag>{,.pdf,.npz}` with tags `good`, `good_mua`, `good_pyr`, `good_mua_pyr`.

Each PDF has one 3-D page + one 2-D-projections page per colouring (maze-mapped colourings get a maze-key panel): X / Y position, **speed (jet, capped at 0.6 m/s)**, session time, trial number, distance to goal, trial type, **before/after the 2nd free-roaming trial**, **hexmaze node-vs-bridge** (rainbow nodes, grey bridges incl. inter-island corridors), **island** (the maze's 4 hexagon clusters), **1st & 2nd goal run**, **the 3 free-roaming periods**, and **goal + its 3 adjacent nodes** (rest grey).

### Step t — Drive Scan (QC)

**Script:** `src/tools/scan_drive.py`

```bash
python src/tools/scan_drive.py --root <drive-or-folder> [--rat Rat5] [--no-videos] [--deep] [--workers 8]
```

Integrity scan of a raw acquisition drive (`Rat<N>_*/<YYYYMMDD>/...` layout; a bare drive root works too — the scanner auto-locates the folder holding the Rat directories). Checks: (1) every raw camera video (`eye01–eye12`, deduplicated; derived mp4s ignored) decodes (ffprobe, or a pure-python moov/ftyp fallback; `--deep` fully decodes); (2) every session with ephys has **pre + task (maze/mazs/awake/hab) + post**; (3) no zero-byte `.rec`/`logger_raw.dat`. Extra flags: cross-rat-named recordings, copy leftovers, empty recording folders, **split recordings** (a phase recorded in parts after an acquisition-PC crash, including parts nested inside the phase folder). Outputs to the root: `drive_scan.xlsx` (inventory / issues / files + per-animal sheets), `drive_scan_report.md`, `drive_scan_inventory.md`.

> In the pipeline the runner passes the *processing* root; for raw-drive QC run it standalone against the acquisition drive.

---

## Drive Coverage GUI

```bash
python src/tools/scan_drive_gui.py
```

A PyQt6 app ("HexMaze — Drive Coverage Checker") that cross-checks the experiment spreadsheet against what is actually on the drives:

1. **Load roster** — pick the experiment Excel (`Raw` sheet; columns `subject, day, session, Date, Implant`). `Implant=0` → video-only expectation, `Implant=1` → video + pre/task/post ephys.
2. **Add drive folders** (any number; each directly contains the `Rat<N>_*` folders, e.g. `F:\HM_neurons`).
3. **Scan drives** — per expected session: status **OK / PARTIAL / MISSING**, video count, per-phase GB (`✗` absent; red `⚠` = present but *short*, < 0.6× that rat's own per-phase median), split flags, found-in paths. Double-click opens the folder; on MISSING rows it offers the deep search.
4. **Find scattered data…** — deep-search *within the selected folders* for sessions filed in odd places; classifies rows as *elsewhere* / *not-found* / *orphan* (on disk but not in the sheet) / unfiled *video* dumps (auto-assigned to rat+date by rig conventions).

Menu bar: **Fix / Prepare** — *Organize drive…* (pick one of your listed drive folders; roster sessions already on that drive are filed into its `HM_neurons` archive by rename only — data on other drives is listed but never copied or touched), *Reset → raw…* (un-file all camera folders back to per-drive `raw/`), *Fix video names…* (canonicalise `eye<NN>_<date>_<time>.mp4`), *Prepare RecordingMeta…* (write missing tracker metadata files from the sheet); **Reports** — *Summary figure…* (status matrix per rat/repeat), *Preprocess progress…* (per-session pipeline-step completion from the `HM_neuron_preprocess` tree); **Options** — auto-RecordingMeta-after-scan (default ON; only ever creates missing files), search depth, include system drive.

**Safety model:** every writing action first builds a read-only, colour-coded *plan* and writes nothing until explicitly confirmed; nothing is ever overwritten or deleted; conflicts are skipped for a human. All "Export CSV…" buttons actually write `.xlsx` workbooks.

---

## Analysis Conventions (clocks, coordinates, gating)

These conventions hold across steps 5/w/u/v/b/m — worth internalising before touching the analysis code:

- **The master session clock** is `stitched_framewise_seconds.csv` → *Seconds From Creation* (renamed to `<date>_Rat<N>_framewise_seconds.csv` when a tracker run finishes; consumers match both). NWB positions, spike times and trial windows (`Trial_start_s/Trial_end_s`, `build_trials`) all live on it; the LFP maze recording starts at its zero. The tracker's `trial_start_time`/`trial_end_time` (RecordingMeta copy) are on a *different*, drifting behavioural-sync clock — never mix them with positions.
- **Pixel frame:** everything the tracker writes is in the 1176×712 display frame. **Metre frame:** analyses divide by `SCALE_X = 2352/2/9 ≈ 130.7 px/m`, `SCALE_Y = 1424/2/5 = 142.4 px/m` into a fixed maze box `MAZE_EXTENT = (0–9, 0–5) m`; the y axis is inverted (camera y grows downward). The canonical node map is `src/tools/node_list_new.csv` (98 nodes, 4 islands + homeboxes).
- **Speed gating:** all place-cell analyses — rate maps, spatial information, decoder tuning curves *and* evaluated bins, spike videos, precession passes — use the same run-epoch gate, **0.05 m/s**.
- **Splits:** event/theta/gamma/precession analyses report three columns: *goal running* (type-1 trials), *free roaming* (type-4/5), *all*.
- **Quality:** `quality_label` (manual Phy) is the human truth; `auto_quality_label` is the automated label. Analyses run on `good` (and `good+mua` where noted).
- **Session prefix:** analysis outputs carry a `<Rat>_<YYYYMMDD_HHMMSS>_` prefix so files from different sessions never collide; a few files are deliberately unprefixed because other tools glob for them (`decoded_*.npz`, `leads_summary_*.npz`, `decode_leads_*.pdf`).

---

## Standalone Utilities

| Tool | Purpose |
|---|---|
| `python src/sorter/make_channel_map.py --input_folder <trodes_export>` | Rebuild `channel_map.npy` from `.LFP` filenames alone (column order identical to `lfp_data.npy`; sanity-checks against it when present) |
| `python list_drive.py <root> [--files] [--depth 6]` | Dump a drive/folder tree (+sizes) to a text file for debugging |
| `python src/tools/define_maze_roi.py --video stitched.mp4 [--frame N]` | Redraw the maze polygon (`src/tools/maze_roi.txt`) after a camera change |
| `python src/tools/gpuslot.py [--measure]` | Show / re-measure the machine's NVENC session limit |
| `python src/tools/vcodec.py [w h]` | Show which encoder the pipeline would pick at a given frame size |
| `python src/sorter/readTrodesExtractedDataFile3.py --info <file.dat>` | Inspect a Trodes `.dat` without loading it |

---

## Tracker — How It Works

`src/tracker/TrackerYolov11.py` is based on [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT) and is built around a per-frame detect → classify → update-state loop.

### Model

The YOLOv11 model was trained on a custom dataset labeled by the authors specifically for this setup — rat body, rat head, and researcher detections in a hex maze environment.

**Dataset:** [box_hm_rat on Roboflow Universe](https://universe.roboflow.com/rathm-wjck3/box_hm_rat) *(self-labeled)*

[![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/rathm-wjck3/box_hm_rat)

### Detection (`cnn()`)

Every frame is passed through YOLOv11 at confidence threshold 0.7 and input size 1280 px. The model outputs three classes:

| Class | Used for |
|---|---|
| `head` | Boxes drawn (cached) only; not counted, does not affect position |
| `rat` | Position tracking — the body centroid drives all trial logic |
| `researcher` | Trial-trigger and force-end logic |

**Maze ROI gating** — rat detections are restricted to the maze polygon (`src/tools/maze_roi.txt`); researcher detections are deliberately unrestricted (they work outside the maze). The ROI records the resolution it was drawn at and the tracker refuses to start on a mismatch with `DISPLAY_SIZE` (1176×712) — redraw with `define_maze_roi.py` rather than editing.

**Motion-based skip** — Before running YOLO, a frame-difference check (Gaussian blur, absolute diff, dilation, changed-pixel count) skips YOLO below 500 changed pixels; cached bounding boxes are redrawn so the display never flickers. When YOLO runs and finds detections the cache updates; when it finds nothing the cache is left in place.

**Rat position** — highest-confidence `rat` box centroid; `last_rat_pos` fallback so the state machine never stalls.

**Researcher selection** — of all `researcher` boxes, the one closest to the rat's active position is used for proximity checks.

### Trial State Machine

```text
WAITING  (start_trial=True, record_detections=False)
    │  rat centroid within 60 px of start node
    ▼
ACTIVE   (record_detections=True)
    │  end condition met (see trial types below)
    ▼
INTER-TRIAL  (start_trial=False, record_detections=False)
    │  TrigA: researcher within 300 px of rat, or
    │  TrigB: researcher covering the start node (≤ 40 px)   →  back to WAITING
```

#### Trial types and end conditions

| Type | Label | End condition |
|---|---|---|
| 1 | Normal | Rat centroid ≤ 25 px from goal node |
| 2 | NGL | Rat visited goal (≤ 20 px) AND 10 minutes elapsed |
| 3 | Probe | ≥ 2 min elapsed AND researcher ≤ 600 px from goal AND rat ≤ 25 px from goal |
| 4–6 | Special NGL | Same as NGL; 10-minute inter-trial lockout measured **from the special trial's start** |

**Researcher-proximity end** — applies to **all** trial types: the trial ends when the closest researcher comes within 150 px of the rat, but only once *armed* (the researcher must first have been >150 px away during the trial) and only after a per-type minimum time — 5 s for types 1/2, 10 min for types 3–6. Suppressed in schedule-only end mode.

**`Did_Not_Reach` trials** — the column is read, but it currently has **no effect on trial ending**: an intended "rat picked up" end (researcher ≤ 60 px for a cumulative 1 s, goal-reach disabled) exists only in a shadowed duplicate of `object_detection` and never runs — the goal-reach end still fires for DNR trials.

**Force-end fallbacks:**

- Closest researcher to the **goal** within 50 px for 10 continuous seconds → trial ends.
- Closest researcher to the **goal** within 160 px for 30 continuous seconds → trial ends (probe immunity and unnormal-interval rules apply).

**Unnormal intervals** — Time windows in the `Unnormal_Intervals` column (`trial_num:start_min-end_min`) suppress the goal-reach ends and the 160 px/30 s researcher-at-goal force-end during the window. The 50 px/10 s force-end and the researcher-near-rat end still fire.

**Inter-trial lockout** — After a type-4/5/6 trial, the next trial cannot start until 10 minutes have elapsed **from the start of the special trial**. A countdown overlay shows the remaining time; the researcher-proximity trigger is blocked until the lockout expires.

#### Time-locked special trials

A row's `Special_Trials` cell can specify when a trial unlocks: `trial_num@MM:SS` (e.g. `3@5:30`; **format the cell as Text** or Excel eats the trial number). The schedule check runs once per frame after the detection/trial-state logic, and only while a trial is active.

| Phase | Behavior |
|---|---|
| Before unlock time | Trial N's start node is gated — even if the rat sits on it, nothing triggers. The active trial keeps running normally. |
| Unlock time arrives, earlier trial still active | The earlier trial is force-ended (`"forced by special trial schedule"`). |
| After force-end | `start_trial` is armed directly (TrigA/TrigB bypassed); the type-4/5/6 lockout is still enforced. |

**Schedule-only end mode.** When a trial's *next* trial number is in the schedule, four end paths for the current trial are suppressed (NGL timeout, researcher-near-rat, researcher-at-goal timers, unnormal-interval timeout) so the schedule force-end can take over. The plain goal-reach end (type 1) and the probe-complete end are **not** suppressed and can still finish the trial early.

**Scheduled-trial minimum duration** — a trial started by schedule blocks all end triggers for its first 5 seconds.

**`Special_Trials_End`** — a separate column of `trial_num@MM:SS` **termination** locks: `end_trial()` refuses every end path for that trial before the scheduled time (logged `[END-LOCK]`), then force-ends it at that time (`"forced by end schedule"`).

Multiple time-locked trials can coexist; a plain trial number with no `@TIME` is accepted as a marker with no runtime effect. The loaded schedule is printed at session start, and each force-end is logged with session seconds.

### Node Logging

On every active frame, the rat centroid is checked against the maze nodes — radius 20 px for ordinary nodes, **26 px for the current goal node**; on a goal-reach end the goal node is appended to the path if not already last. Node visits carry the synchronized timestamp from the framewise CSV; consecutive duplicates are de-duplicated.

### Velocity Calculation

After each trial, segment velocities are computed from the (timestamp, node) sequence: `speed = segment_length / time_difference` [m/s]. Segment lengths use a hardcoded bridge table for cross-island distances (e.g. 1.72 m for 121→302) and default to 0.30 m for intra-island segments.

### RecordingMeta output columns

A copy of `RecordingMeta.xlsx` is written to the output folder with these per-trial columns appended:

| Column | Description |
|---|---|
| `paths` | Comma-separated visited node IDs |
| `delay` / `active_time` | Trial duration in seconds (start-node entry → end condition) |
| `avg_speed` | Total path distance ÷ total path time (m/s) |
| `avg_between_node_speed` | Mean of per-segment speeds (m/s) |
| `trial_start_time` / `trial_end_time` | Sync timestamps (s) — behavioural-sync clock, **not** comparable with NWB position timestamps |

Framewise timestamp CSV priority: `stitched_framewise_seconds.csv` → `<date>_Rat<id>_framewise_seconds.csv` → `stitched_framewise_ts.csv` → `<date>_Rat<id>_framewise_ts.csv` (seconds clock preferred; both files are session-prefixed at the end of each run).

### Tracker — Modifications from Original

Based on [genzellab/HM_RAT](https://github.com/genzellab/HM_RAT). Key changes:

- Detection backend replaced with YOLOv11 (Ultralytics); rat position uses the `rat` body class only
- Maze-ROI spatial gating of rat detections with fail-fast resolution checking
- Extended trial state machine: NGL variants (types 4–6), researcher-proximity trigger/arming and force-end timers, inter-trial lockout, time-locked start (`Special_Trials`) and end (`Special_Trials_End`) schedules, `Did_Not_Reach` pickup-end handling
- Per-trial metrics written back into a copy of `RecordingMeta.xlsx`
- Motion-based YOLO skip with cached bounding-box redraw; threaded video writer

---

## Node Analysis — Computed Metrics

`src/node_analysis/hex_maze_analysis.py` processes `.xlsx` trial files and appends computed columns (values written into existing headers when present — legacy header aliases accepted — else appended; flagged rows highlighted red).

### Maze structure

96 nodes across 4 islands (101–124, 201–224, 301–324, 401–424) plus 2 homeboxes (501, 502). Two graphs are pre-computed: the full **node graph** and a 4-node **island graph**.

### Required input columns

| Column | Description |
|---|---|
| `path_to_reach` | Comma-separated node IDs of the full path |
| `start_node_n` / `goal_node_n` | Start / goal node IDs |
| `start_island_n` / `goal_island_n` | Island numbers (1–4) |
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
| `norm_path_length_…` (×3) | Each of the above ÷ its optimal distance |

### Core behavioral metrics (Step 1)

| Column | Description |
|---|---|
| `shortest_path` | Minimum hops between start and goal node |
| `n_nodes_visited` | Total nodes visited including revisits |
| `food_reached` | `1` if goal appears among the last two path nodes |
| `eat_on_1_encounter` | `1` if the last path node is exactly the goal |
| `dist_tra` | Edges traveled; sentinel `99` if food not reached |
| `dt_rel_sp` | `dist_tra / shortest_path` (1.0 = optimal) |
| `dt_min_sp` | `dist_tra − shortest_path` |
| `dir_run_mat_perf` | `1` if food reached AND path length equals `shortest_path` |
| `node_choices_binary` | Per-step `0`/`1`: `1` = step minimised remaining distance to goal |
| `perc_correct_choices` | `(sum of 1s / total steps) × 100` |

### Goal-island entry metrics (Step 2)

Computed from the last bridge crossing into the goal island (a step where consecutive node IDs differ by ≥ 50). Skipped when `exclude_trial != 0`.

| Column | Description |
|---|---|
| `isl_node_in` | Node on the DEPARTURE side of the last island crossing (the node before the ≥50-id jump); entry metrics are measured from it |
| `isl_short_path` | `node_graph_distance(isl_node_in, goal) + 1` |
| `isl_dt_trav` | Nodes from island entry to end of path |
| `perf_in_island` | `isl_dt_trav / isl_short_path` |

---

## Metadata (RecordingMeta.xlsx)

Per-session and per-trial input for the tracker (template: `examples/RecordingMeta.xlsx`; filling guide: `docs/RecordingMeta_README.md`). The original file is never modified — the tracker writes a copy to the output folder.

| Column | Description |
|---|---|
| `Rat_ID` / `Date` / `Repeat` / `Day` / `Session` | Session identity (also authoritative for the NWB) |
| `Num_Trials` | Total trials in the session |
| `Start_Min` / `Start_Sec` | Optional: start the video at this offset |
| `Stop_Min` / `Stop_Sec` | Optional: stop processing at this offset |
| `Start_At_Trial_Num` | Optional; only takes effect together with `Start_Min`/`Start_Sec`, and only relabels the trial number — start/goal/type rows are NOT skipped ahead |
| `Start_Nodes` / `Goal_Node` | Per-row start / goal node IDs |
| `Trial_Type` | Per-row trial type (1–6) |
| `Special_Trials` | `3` or time-locked `trial_num@MM:SS` start schedule (**cell must be Text**) |
| `Special_Trials_End` | `trial_num@MM:SS` termination locks: no end path fires before the time, force-end at it (**cell must be Text**) |
| `Did_Not_Reach` | Read by the tracker but currently has **no effect** on trial ending (the intended pickup-end is dead code — see Tracker section) |
| `Unnormal_Intervals` | Immunity windows per trial (`trial:start_min-end_min`) |

---

## Troubleshooting & Docs

`docs/` contains: `ERROR_REFERENCE.md` (exact error message → cause/fix), `TROUBLESHOOTING.md` (LED sync), `TROUBLESHOOTING_EPHYS.md` (sorting/LFP), `TROUBLESHOOTING_PIPELINE.md` (per-step), `TROUBLESHOOTING_SETUP.md` (env/config), `RecordingMeta_README.md`, and the experiment protocol.

Quick pointers:

- **Sync fails / wrong LED** — `SYNC_DEBUG=1` and inspect `sync_debug/`; pin the LED with `led_crop_override.txt` / `led_ica_override.txt`; adjust `SYNC_START_SEC`.
- **Compression/stitching falls back to CPU** — check `python src/tools/gpuslot.py` and `python src/tools/vcodec.py 2352 1424`; force `FFMPEG_VCODEC=libx264` to rule the GPU out.
- **Phy can't show raw traces** — `processed_binary/` was deleted; re-run step 7 (this wipes curation) or restore it.
- **Step u "already has Units"** — by design; regenerate with `r → w → u`.
- **Theta/phase pages missing in step v** — no `LFP_Output` for that session, or an unalignable multi-session LFP export (the log says which).
- **macOS `._*` AppleDouble files on SMB shares** — harmless; every consumer in the pipeline skips them.

---

## License

See [LICENSE](LICENSE).
