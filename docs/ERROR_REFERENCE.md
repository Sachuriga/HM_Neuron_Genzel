# Error Message Quick Reference

Look up the exact error or warning message you see to find the likely cause and fix.

| Message | Script | Likely Cause | Fix |
|---|---|---|---|
| `Config file not found` | `runner_windows.bat` | `hm_tracker_paths.txt` not on Desktop | Copy example from repo, rename, fill in paths |
| `trodesexport not found at: ...` | `runner_windows.bat` | `TRODES_EXPORT_CMD` path is wrong | Fix path in `hm_tracker_paths.txt` |
| `exportLFP not found at: ...` | `runner_windows.bat` | `TRODES_EXPORT_LFP` path is wrong | Fix path in `hm_tracker_paths.txt` |
| `No videos found at location [...]` | `join_views.py` | Camera videos missing or named incorrectly | Rename to `eye??_*.mp4`, verify 12 files present |
| `Found unexpected number of videos` | `join_views.py` | Video count doesn't match expected | Use `-n` flag or remove extra `.mp4` files |
| `Can't find requested path` | `join_views.py` | Input folder doesn't exist | Verify input folder path |
| `No contours found in <file>` | `Video_LED_Sync_using_ICA.py` | LED not visible in video | Check LED is lit; verify correct video file |
| `INVALID XY COORDINATES FOUND FOR LED` | `Video_LED_Sync_using_ICA.py` | LED at frame edge or `.led_crop` file wrong | Check/fix `.led_crop` file |
| `Frame counts do not match` | `Video_LED_Sync_using_ICA.py` | Video and `.meta` file are from different recordings | Verify `.meta` file matches video |
| `Error loading model` → exit | `TrackerYolov11.py` | YOLO weights not found or corrupt | Check `ONNX_WEIGHTS_PATH`, re-download weights |
| `WARNING: No GPU detected. Running on CPU.` | `TrackerYolov11.py` | CUDA unavailable | Install driver, run `nvidia-smi` to verify |
| `Warning: No timestamp CSV found` | `TrackerYolov11.py` | Step 2 not run or output missing | Run Step 2 first |
| `No log files found in output folder` | `plot_trials.py` | Step 4 not run | Run Step 4 first |
| `FFmpeg compression failed` | `runner_windows.bat` | NVENC session limit or GPU out of memory | Reduce parallel jobs; check `nvidia-smi` |
| `No valid .mp4 file found to compress` | `runner_windows.bat` | Step 4 did not produce output | Run Step 4 first |
| `No LFP .dat files found` | `sorting.py`, `export_lfp.py` | Step e (LFP export) not run | Run `exportLFP` with `-rec` flag |
| `Warning: 'readTrodesExtractedDataFile3.py' not found` | `sorting.py` | Helper script missing from `src/sorter/` | Obtain file from supervisor, place in `src/sorter/` |
| `Settings format not supported` | `readTrodesExtractedDataFile3.py` | `.dat` file header malformed or wrong Trodes version | Re-export from Trodes; check file header |
| `No .xlsx files found` | `hex_maze_analysis.py` | `RecordingMeta.xlsx` not in input folder | Move metadata file to input folder |
| `unknown node(s) in path: [...]` | `hex_maze_analysis.py` | Node ID not in maze topology | Fix node ID; valid range: 101–124, 201–224, 301–324, 401–424, 501–502 |
| `start_node_n != first path node` | `hex_maze_analysis.py` | Metadata start node doesn't match tracked path | Correct start node in worksheet |
| `No CSV file found` | `tracking_eyes.py` | Step 4 tracking output missing | Run Step 4 first |

---

## Output Files Checklist

Use this to verify each step completed successfully before moving to the next.

| Step | Expected Output | Location |
|---|---|---|
| Step 1 | `<recording>.DIO/` folder | Input folder |
| Step 1 | `<recording>.raw/` folder | Input folder |
| Step e | `<recording>.LFP/` folder containing `_nt*ch*.dat` files | Input folder |
| Step 2 | `stitched_framewise_ts.csv` or `stitched_framewise_seconds.csv` | Output folder |
| Step 3 | `stitched.mp4` | Input folder |
| Step 4 | `<date>_Rat<id>.txt` (node log) | Output folder |
| Step 4 | `<date>_Rat<id>_Coordinates_Full.csv` | Output folder |
| Step 4 | `<date>_Rat<id>.mp4` (labeled video) | Output folder |
| Step 5 | Multi-page PDF with trial plots | Output folder |
| Step 6 | Compressed `.mp4` (replaces Step 4 output) | Output folder |
| Step 7 | `phy_export/` folder | Output folder |
| Step 8 | `LFP_Output/` folder with `.npy` files | Output folder |
| Step n | `*_results.xlsx` | Input or output folder |
