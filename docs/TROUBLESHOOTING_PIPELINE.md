# Troubleshooting: Pipeline Steps

## Step 1 / Step e — Trodes Export (DIO / Raw / LFP)

**No `.DIO/`, `.raw/`, or `.LFP/` folder generated after running**

- Confirm Trodes is installed and the paths in `hm_tracker_paths.txt` are correct.
- Check that the `.rec` file is not corrupted and is accessible.
- Ensure there is enough free disk space in the input folder (recordings can be several hundred GB).
- If DIO channels are missing (LED sync data absent), check that the DIO cables were connected during recording.

---

## Step 2 — LED Sync (ICA)

**Error: `No contours found in <file>`**

The script could not detect the LED in the video.

1. Open the video in VLC and verify the blue and red LEDs are visibly blinking.
2. If they are not visible, the wrong video file may be in the folder.
3. If the LED is present but the detection still fails, a manual LED crop coordinate file (`.led_crop`) may need to be provided.

**Warning: `INVALID XY COORDINATES FOUND FOR LED`**

The detected LED position is outside expected bounds (8–592 x, 8–792 y). The `.led_crop` file may contain wrong values, or the LED is at the edge of the frame.

**Warning: `Frame counts do not match`**

The video frame count does not match the metadata `.meta` file.

- Verify that the `.meta` file belongs to this specific video (same recording session).
- If the video was re-encoded or trimmed, the frame count may have changed. Use the original exported video.

**Output CSV is empty or not generated (`stitched_framewise_seconds.csv`)**

- Check that the Blue and Red light columns are not empty in the Step 2 log.
- If they are empty, the DIO files were not copied or the ephys export (Step 1) failed. See Step 1 above.
- Verify the individual camera videos are blinking correctly before re-running.

---

## Step 3 — Multi-Camera Stitching

**Error: `No videos found at location [...]`**

The script expects video files named `eye??_*.mp4` (e.g., `eye01_recording.mp4`).

1. Check that all 12 camera video files are in the input folder.
2. Confirm the filenames match the pattern. If they are named differently (e.g., `cam01_*.mp4`), rename them.
3. If fewer than 12 cameras were used, pass the `-n` flag with the correct count when calling the script.

**Error: `Found unexpected number of videos`**

The number of `.mp4` files found does not match what the script expects.

- Count the actual video files in the folder.
- If some are missing, check the recording setup.
- If extra files are present (e.g., a previously stitched output), move them out of the folder before re-running.

**FFmpeg fails or produces a black/corrupt output**

- Confirm FFmpeg is installed and the path in `hm_tracker_paths.txt` is correct: run `ffmpeg -version` in a terminal.
- NVIDIA GPU is required for `h264_nvenc`. If the GPU is unavailable, FFmpeg will return an error. Check GPU availability with `nvidia-smi`.
- If crop coordinates are wrong, the output may have black borders. Do not modify `join_views.py` crop values unless the camera hardware has changed.

---

## Step 4 — YOLOv11 Tracker

**Error: `Error loading model` then exit**

The YOLO weights file could not be loaded.

1. Verify `ONNX_WEIGHTS_PATH` in `hm_tracker_paths.txt` points to the correct `.onnx` file.
2. If the file is missing, re-download the model weights from the project archive.
3. Check GPU availability — if the model cannot be moved to the GPU, the process exits.

**Warning: `No timestamp CSV found`**

Step 2 output is missing. The tracker will continue but log entries will have no neural timestamps.

- Run Step 2 first to generate `stitched_framewise_ts.csv`.
- Confirm the output folder path is the same for both steps.

**Not all trials appear in the output video or `.txt` log**

- After Step 4, open the labeled video and count the trials.
- If trials are missing, check that the start nodes and goal nodes in `RecordingMeta.xlsx` are correctly filled in for every trial.
- Confirm trial numbers in the worksheet are sequential and match what was run.
- If start node assignment looks wrong, see the Tracker Troubleshooting section in `TROUBLESHOOTING.md`.

**Tracker output appears for correct number of trials but node paths are wrong**

- The tracker logs a node whenever the rat centroid is within 20 px of a node center.
- If the stitched video has calibration drift or the camera was shifted, node positions in `node_list_new.csv` may no longer match the physical maze layout. Contact your supervisor.

---

## Step 5 — Trial Plotting

**Error: `No log files found in output folder`**

The `.log` file from Step 4 is missing. Run Step 4 first, then re-run Step 5.

**PDF is generated but trials are missing or blank**

- Check the `.txt` node sequence file from Step 4. Each trial should have a `Summary Trial N` header followed by a comma-separated node list.
- If start/goal node overlays are missing from the plots, `RecordingMeta.xlsx` was not found in the input folder. Confirm the file is in the correct location.

---

## Step 6 — Video Compression

**Error: `FFmpeg compression failed for ...`**

NVIDIA hardware encoding (NVENC) failed.

- The GPU may have too many concurrent NVENC sessions. This happens when multiple jobs are running in parallel. Wait for other jobs to finish.
- Check GPU memory with `nvidia-smi`. If memory is full, close other GPU-heavy processes.
- If FFmpeg is not found, verify `FFMPEG_CMD` in `hm_tracker_paths.txt`.
- The original video is never deleted if compression fails — it remains untouched.

**Warning: `No valid .mp4 file found to compress`**

Step 4 did not produce a `.mp4` output file. Run Step 4 first.

---

## Step 9 — Cleanup

**Cleanup runs but important files are deleted**

Step 9 permanently deletes the `.DIO/`, `.raw/`, and `*timestampoffset*/` folders inside the input folder. These are the intermediate Trodes exports.

- Do not run Step 9 until Steps 2 and 7 (spike sorting) have successfully completed and their outputs have been verified.
- The original `.rec` file is never touched by cleanup.
- If cleanup ran prematurely, you will need to re-run Steps 1/e and 2/7 from the original `.rec` file.

---

## Step n — Node Analysis

**Error rows highlighted red in output `.xlsx`**

The script flags rows it cannot process and highlights them red. The `flag` column contains the reason.

Common causes:
- **`unknown node(s) in path`** — A node ID in the path does not exist in the maze topology. Valid IDs are 101–124, 201–224, 301–324, 401–424, 501, 502. Fix the source data (tracker `.txt` output) or check `node_list_new.csv`.
- **`start_node_n != first path node`** — The start node in the metadata does not match the first node in the recorded path. Correct the worksheet.
- **Empty `path_to_reach`** — The tracker produced no node sequence for this trial. Check the `.txt` output from Step 4 for this trial.

**No `.xlsx` found in input folder**

Confirm the `RecordingMeta.xlsx` file (filled in with start/goal nodes and trial metadata) is in the input folder before running Step n.
