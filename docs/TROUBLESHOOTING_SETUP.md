# Troubleshooting: Setup & Environment

## Config File

**Error: `Config file not found: %USERPROFILE%\Desktop\hm_tracker_paths.txt`**

The batch script requires a config file on your Desktop. It will not run without it.

1. Copy `hm_tracker_paths.example.txt` from the repo root to your Desktop.
2. Rename it to `hm_tracker_paths.txt`.
3. Open it and fill in all paths:
   - `FFMPEG_CMD` — full path to `ffmpeg.exe`
   - `ONNX_WEIGHTS_PATH` — full path to the `.onnx` model weights file
   - `TRODES_EXPORT_CMD` — full path to `trodesexport.exe`
   - `TRODES_EXPORT_LFP` — full path to `exportLFP.exe`
   - `LFP_CHANNELS` — channel list for LFP export

---

## External Tools Not Found

**Warning: `trodesexport not found at: ...`**
**Warning: `exportLFP not found at: ...`**

The paths in `hm_tracker_paths.txt` are incorrect or Trodes is not installed.

1. Confirm Trodes is installed on this machine.
2. Find the actual path to `trodesexport.exe` and `exportLFP.exe`.
3. Update both paths in `hm_tracker_paths.txt` on your Desktop.

---

## Python Environment Issues

**`conda env create` freezes or hangs**

The most common causes are a Python version that conflicts with neuro packages, or a contradictory numpy constraint that the solver cannot resolve.

1. Remove any existing environment and re-create from the current `reproduce.yml` (which uses Python 3.10):
   ```
   conda env remove -n HM_neuron
   conda env create -f reproduce.yml
   conda activate HM_neuron
   pip install -r requirements.txt
   ```

2. If it still hangs, use the faster `libmamba` solver:
   ```
   conda install -n base conda-libmamba-solver
   conda config --set solver libmamba
   conda env create -f reproduce.yml
   ```

**Error: Package not found, version conflict, or import error after pip install**

1. Verify the PyTorch index URL is present at the top of `requirements.txt`:
   ```
   --extra-index-url https://download.pytorch.org/whl/cu128
   ```
   Without this line, `torch==2.10.0+cu128` cannot be found and pip will fail.

2. If Qt-related errors appear, only one Qt binding is allowed. `PySide6` has been removed from `requirements.txt` — if it was previously installed, uninstall it:
   ```
   pip uninstall PySide6 shiboken6
   ```

---

## GPU / CUDA Issues

**Warning: `No GPU detected. Running on CPU.`**

YOLO tracking will be extremely slow on CPU. Fix GPU detection before running.

1. Check that the GPU is recognized by the driver:
   ```
   nvidia-smi
   ```
   If this fails, reinstall the NVIDIA driver.

2. Verify CUDA 12.8 is accessible:
   ```
   python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
   ```
   Expected: `True 12.8`

3. If the CUDA version is wrong, reinstall PyTorch:
   ```
   pip install torch==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128
   ```

4. Minimum GPU requirement: NVIDIA GPU with CUDA compute capability 7.0 or higher (RTX or recent GTX series). Older cards will not work.

---

## Resource Threshold Warnings

The batch runner checks CPU, GPU, and RAM before launching each step. If thresholds are exceeded it will wait.

Default thresholds (top of `runner_windows.bat`):
- `MAX_CPU` = 90%
- `MAX_GPU` = 90%
- `MAX_MEM` = 65%
- `WAIT_SECONDS` = 30

If the pipeline is stuck waiting, check task manager for runaway processes or reduce the number of parallel jobs.
