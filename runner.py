#!/usr/bin/env python3
# ============================================================
#   HM Tracker — cross-platform runner (single source of truth)
#
#   Usage:  python runner.py /path/to/data_root
#
#   This ONE file drives the whole pipeline on macOS, Linux and
#   Windows. The old runner_unix.sh / runner_windows.bat are now
#   thin launchers that just call this. Edit the menu / steps here
#   ONCE and both platforms stay in sync.
#
#   Behaviour (ported from both old runners):
#     - Parallel steps (1 e 2 3 4 5 6 8 d n) run one background
#       worker per ip*/op* pair, gated by a CPU/GPU/MEM monitor.
#     - Sequential master steps (7 c r 9 w u) run afterwards, one
#       folder at a time, in that order.
#     - Each parallel worker logs to <tmp>/hm_worker_<name>.log.
#     - A missing tool (nvidia-smi, trodes, ...) never blocks the run.
#
#   Env overrides: MAX_CPU MAX_GPU MAX_MEM WAIT_SECONDS LAUNCH_GAP
#   FFMPEG_VCODEC SYNC_START_SEC NWB_RAT_NR DISABLE_RESOURCE_CHECK
#   HM_CONFIG_FILE
# ============================================================

import os
import sys
import time
import json
import shlex
import shutil
import tempfile
import threading
import subprocess
from pathlib import Path

try:
    import psutil  # optional; used for the CPU/MEM resource gate
except Exception:
    psutil = None

# Always operate from the repo root (this file's directory), because the step
# scripts are referenced as ./src/... relative paths.
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

PYTHON = sys.executable  # run child python steps with the same interpreter

# ------------------------------------------------------------
#                 THE MENU — single source of truth
# ------------------------------------------------------------
# (key, label). Order here is the order shown. Whether a step runs in a
# parallel per-folder worker or sequentially at master level is decided by
# SEQUENTIAL_STEPS below — everything else is a parallel worker step.
MENU = [
    ("1", "Trodes Export (DIO/Raw/Analog)"),
    ("e", "Trodes Export LFP + Analog (per channel)"),
    ("2", "Sync Script"),
    ("3", "Stitching"),
    ("4", "Tracker"),
    ("5", "Plotting"),
    ("6", "Compression"),
    ("7", "Sorting"),
    ("c", "Continue After Sorting (metrics + BombCell + Phy, no re-sort)"),
    ("r", "Recompute Metrics (after manual Phy curation)"),
    ("8", "LFP + Motion (IMU Accel)"),
    ("d", "deeplabcut"),
    ("9", "Cleaning"),
    ("n", "Node Analysis"),
    ("w", "nwblfp (NWB / LFP package)"),
    ("u", "Add curated Units (metrics + waveforms) to NWB (runs after w)"),
    ("v", "Visualize NWB units (summary + per-unit rate-map PDFs; runs after u)"),
    ("b", "Bayesian position decoder + decoded-on-video overlay per session (good and good+mua)"),
    ("m", "Neural population UMAP per session (good and good+mua; Gardner et al. 2022)"),
    ("s", "Session summary (cross-session per-animal plots by date/repeat/session)"),
    ("t", "Drive scan (videos playable + ephys has pre/task/post + non-zero .rec)"),
]

# Sequential master-level steps, in execution order. Everything NOT in here is
# a parallel worker step.
SEQUENTIAL_STEPS = ["7", "c", "r", "9", "w", "u", "v", "b", "m", "s", "t"]


# ------------------------------------------------------------
#                 CONFIGURATION (SHARED)
# ------------------------------------------------------------
def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MAX_CPU = _int_env("MAX_CPU", 90)
MAX_GPU = _int_env("MAX_GPU", 90)
MAX_MEM = _int_env("MAX_MEM", 65)
WAIT_SECONDS = _int_env("WAIT_SECONDS", 30)
LAUNCH_GAP = _int_env("LAUNCH_GAP", 20)   # seconds between worker launches
FREQ = 30000


def load_config():
    """Locate and parse hm_tracker_paths.txt (KEY=VALUE lines). Exports every
    key into os.environ so child steps inherit them, and returns the path."""
    cfg = os.environ.get("HM_CONFIG_FILE") or str(Path.home() / "Desktop" / "hm_tracker_paths.txt")
    if not Path(cfg).is_file():
        print(f"[ERROR] Config file not found: {cfg}\n")
        print("Please create hm_tracker_paths.txt on your Desktop with lines like:")
        print("  FFMPEG_CMD=/usr/local/bin/ffmpeg")
        print("  ONNX_WEIGHTS_PATH=/path/to/weights.pt")
        print("  TRODES_EXPORT_CMD=/path/to/trodesexport")
        print("  TRODES_EXPORT_LFP=/path/to/exportLFP")
        print("  LFP_CHANNELS=1 2 3 4 5 6 7 8\n")
        print("See hm_tracker_paths.example.txt in the repo for a template.")
        sys.exit(1)

    for raw in Path(cfg).read_text(errors="replace").splitlines():
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = "".join(key.split())
        if key:
            os.environ[key] = val
    return cfg


# ------------------------------------------------------------
#                 RESOURCE MONITOR
# ------------------------------------------------------------
# Each returns an int percent (0..100); 0 on any failure so a missing tool
# never blocks the pipeline.
def get_cpu_load():
    if psutil is not None:
        try:
            return int(psutil.cpu_percent(interval=0.3))
        except Exception:
            return 0
    return 0


def get_mem_usage():
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().percent)
        except Exception:
            return 0
    return 0


def get_gpu_load():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        first = out.stdout.strip().splitlines()
        return int(float(first[0].strip())) if first else 0
    except Exception:
        return 0


def wait_for_resources():
    if os.environ.get("DISABLE_RESOURCE_CHECK", "0") == "1":
        return
    while True:
        cpu, gpu, mem = get_cpu_load(), get_gpu_load(), get_mem_usage()
        if cpu > MAX_CPU:
            print(f"    [WAIT] High CPU: {cpu}%. Pausing {WAIT_SECONDS}s...")
            time.sleep(WAIT_SECONDS); continue
        if gpu > MAX_GPU:
            print(f"    [WAIT] High GPU: {gpu}%. Pausing {WAIT_SECONDS}s...")
            time.sleep(WAIT_SECONDS); continue
        if mem > MAX_MEM:
            print(f"    [WAIT] High MEM: {mem}%. Pausing {WAIT_SECONDS}s...")
            time.sleep(WAIT_SECONDS); continue
        print(f"    [CHECK] CPU: {cpu}% | GPU: {gpu}% | MEM: {mem}% - OK.")
        return


# ------------------------------------------------------------
#                 SMALL HELPERS
# ------------------------------------------------------------
def run(cmd, cwd=None, out=None):
    """Run a command, streaming to `out` (a file for workers, or None=console for
    master steps). Never raises; returns the exit code (127 if not found)."""
    cmd = [str(c) for c in cmd]
    banner = "    $ " + " ".join(cmd)
    print(banner, file=out, flush=True) if out else print(banner)
    # Force child processes to emit UTF-8 so Unicode in their prints (e.g. the '->'
    # arrow) does not crash on Windows' default cp1252 console/file encoding.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        return subprocess.run(cmd, cwd=cwd, stdout=out, stderr=out, env=env).returncode
    except FileNotFoundError as e:
        msg = f"    [ERROR] command not found: {e}"
        print(msg, file=out, flush=True) if out else print(msg)
        return 127


def launch_worker_window(ip, op, steps, marker, title):
    """Open a NEW terminal window running one parallel worker, so its progress is
    visible live (the pre-unification 'Job-<ip>' console behaviour). The worker
    writes `marker` when it finishes so the master can wait for it. Returns True if
    a window was opened, False to fall back to an in-process background thread."""
    repo = os.path.dirname(os.path.abspath(__file__))
    args = [sys.executable or PYTHON, "-u", os.path.join(repo, "runner.py"),
            "--worker", str(ip), str(op), steps, str(marker)]
    try:
        if sys.platform.startswith("win"):
            # start "title" cmd /k <cmd>  — new console that stays open when done.
            inner = subprocess.list2cmdline(args)
            subprocess.Popen(f'start "HM Job-{title}" cmd /k {inner}',
                             shell=True, cwd=repo)
            return True
        if sys.platform == "darwin":
            cmd = "cd " + shlex.quote(repo) + " && " + " ".join(shlex.quote(a) for a in args)
            subprocess.Popen(["osascript", "-e",
                              f'tell application "Terminal" to do script {json.dumps(cmd)}'])
            return True
        cmd = "cd " + shlex.quote(repo) + " && " + " ".join(shlex.quote(a) for a in args) + "; exec bash"
        for term in (["gnome-terminal", "--"], ["konsole", "-e"],
                     ["x-terminal-emulator", "-e"], ["xterm", "-e"]):
            try:
                subprocess.Popen(term + ["bash", "-lc", cmd])
                return True
            except FileNotFoundError:
                continue
        return False
    except Exception as e:
        print(f"[WARN] could not open a worker window for {title}: {e}")
        return False


def _tool(var):
    """Configured tool path from the environment, if it exists on disk."""
    p = os.environ.get(var, "")
    return p if p and Path(p).exists() else ""


def log(out, msg):
    print(msg, file=out, flush=True)


# ------------------------------------------------------------
#                 THE WORKER (parallel, per ip/op pair)
# ------------------------------------------------------------
def run_worker(ip, op, steps, out):
    """Run the per-folder parallel steps for one ip/op pair, logging to `out`.
    Sequential steps (7 c r 9 w u) are handled at master level, not here."""
    ip, op = Path(ip), Path(op)
    op.mkdir(parents=True, exist_ok=True)
    log(out, f"[INFO] Running steps [{steps}] for {ip}")
    if not steps.strip():
        log(out, "[WORKER] No steps to run, exiting.")
        return

    freq = os.environ.get("FREQ", FREQ)
    sync_start = os.environ.get("SYNC_START_SEC", "45")
    sync_led = os.environ.get("SYNC_LED", "auto")        # auto|red|blue (which LED drives sync)
    vcodec = os.environ.get("FFMPEG_VCODEC", "h264_nvenc")
    recs = sorted(ip.glob("*.rec"))

    # --- STEP 1: Trodes DIO/Raw/Analog export (per .rec) ---
    if "1" in steps:
        log(out, "[STEP 1] Running Trodes DIO/Raw/Analog Export (per .rec)...")
        trodes = _tool("TRODES_EXPORT_CMD")
        if trodes:
            if not recs:
                log(out, f"[WARNING] No .rec files found in '{ip}'")
            for f in recs:
                log(out, f"    Exporting {f.name}")
                run([trodes, "-dio", "-raw", "-analogio", "-rec", f], out=out)
        else:
            log(out, f"[WARNING] trodesexport not found at: {os.environ.get('TRODES_EXPORT_CMD','')}")

    # --- STEP e: Trodes LFP + Analog export (per .rec) ---
    if "e" in steps:
        log(out, "[STEP e] Running Trodes LFP + Analog Export (per .rec)...")
        lfp, trodes = _tool("TRODES_EXPORT_LFP"), _tool("TRODES_EXPORT_CMD")
        if not recs:
            log(out, f"[WARNING] No .rec files found in '{ip}'")
        for f in recs:
            log(out, f"    --- {f.name} ---")
            if lfp:
                log(out, "    Exporting LFP (1000Hz, LP 500Hz)...")
                run([lfp, "-rec", f, "-outputrate", "1000", "-lfplowpass", "500"], out=out)
            else:
                log(out, f"[WARNING] exportLFP not found at: {os.environ.get('TRODES_EXPORT_LFP','')}")
            if trodes:
                log(out, "    Exporting analog/AUX (headstage IMU)...")
                run([trodes, "-analogio", "-rec", f], out=out)
            else:
                log(out, f"[WARNING] trodesexport not found at: {os.environ.get('TRODES_EXPORT_CMD','')}")

    # --- STEP 2: LED sync ---
    if "2" in steps and Path("./src/tracker/Video_LED_Sync_using_ICA.py").exists():
        log(out, f"[STEP 2] Running Sync Script (LED detection starts after {sync_start}s, "
                 f"sync LED: {sync_led})...")
        run([PYTHON, "-u", "./src/tracker/Video_LED_Sync_using_ICA.py",
             "-i", ip, "-o", op, "-f", freq, "--start-sec", sync_start,
             "--sync-led", sync_led], out=out)

    # --- STEP 3: stitching ---
    if "3" in steps and Path("./src/tracker/join_views.py").exists():
        log(out, "[STEP 3] Running Stitching...")
        run([PYTHON, "-u", "./src/tracker/join_views.py", ip], out=out)

    # --- STEP 4: tracker ---
    if "4" in steps and (ip / "stitched.mp4").exists():
        log(out, "[STEP 4] Running Tracker...")
        run([PYTHON, "-u", "./src/tracker/TrackerYolov11.py", "--input_folder", ip,
             "--output_folder", op, "--onnx_weight", os.environ.get("ONNX_WEIGHTS_PATH", "")], out=out)

    # --- STEP 5: plotting ---
    if "5" in steps and Path("./src/tracker/plot_trials.py").exists():
        log(out, "[STEP 5] Running Plotting...")
        run([PYTHON, "-u", "./src/tracker/plot_trials.py",
             "--input_folder", ip, "--output_folder", op], out=out)

    # --- STEP 6: GPU compression ---
    if "6" in steps:
        log(out, f"[STEP 6] Running Compression (codec: {vcodec})...")
        ffmpeg = os.environ.get("FFMPEG_CMD", "ffmpeg")
        temp_file = op / "__temp_compressed.mp4"
        if temp_file.exists():
            temp_file.unlink()
        video = next((f for f in sorted(op.glob("*.mp4"))
                      if f.name != "__temp_compressed.mp4"), None)
        if video is not None:
            rc = run([ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
                      "-stats", "-i", video, "-c:v", vcodec, "-preset", "p6", "-cq", "28",
                      "-c:a", "copy", temp_file], out=out)
            if rc == 0:
                shutil.move(str(temp_file), str(video))
                log(out, f"[SUCCESS] Video compressed: {video}")
            else:
                log(out, f"[ERROR] FFmpeg compression failed for {video}")
                log(out, f"        Check codec '{vcodec}' (Mac: try h264_videotoolbox).")
                if temp_file.exists():
                    temp_file.unlink()
        else:
            log(out, f"[WARNING] No valid .mp4 file found in '{op}' to compress.")

    # --- STEP 8: LFP + Motion/IMU extraction ---
    if "8" in steps:
        if Path("./src/sorter/export_lfp.py").exists():
            log(out, "[STEP 8] Running LFP Extraction...")
            run([PYTHON, "-u", "./src/sorter/export_lfp.py",
                 "--input_folder", ip, "--output_folder", op], out=out)
        if Path("./src/sorter/export_motion.py").exists():
            log(out, "[STEP 8] Running Motion (IMU Accel) Extraction...")
            run([PYTHON, "-u", "./src/sorter/export_motion.py",
                 "--input_folder", ip, "--output_folder", op], out=out)

    # --- STEP d: DeepLabCut export ---
    if "d" in steps and Path("./src/dlc/tracking_eyes.py").exists():
        log(out, "[STEP d] Exporting video for DeepLabCut...")
        run([PYTHON, "-u", "./src/dlc/tracking_eyes.py",
             "--input_folder", ip, "--output_folder", op], out=out)

    # --- STEP n: node analysis ---
    if "n" in steps and Path("./src/node_analysis/hex_maze_analysis.py").exists():
        log(out, "[STEP n] Running Node Analysis...")
        run([PYTHON, "-u", "./src/node_analysis/hex_maze_analysis.py",
             "--input_folder", ip, "--output_folder", op], out=out)

    log(out, f"[COMPLETE] Worker finished for {ip}")


def clean_folder(target):
    """Delete .DIO / .raw / *timestampoffset* folders under `target`."""
    target = Path(target)
    for pattern in ("*.DIO", "*.raw", "*timestampoffset*"):
        for d in sorted(target.glob(pattern)):
            if d.is_dir():
                print(f"    Deleting: {d.name}")
                shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------
#                 SEQUENTIAL (master) STEP RUNNERS
# ------------------------------------------------------------
def _run_per_op(label, tag, script, ops, config, extra_ip=None):
    """Run a python step sequentially over every op folder (used by 7/c/r/u)."""
    print("\n" + "=" * 56)
    print(f"[MASTER] Running {label} sequentially...")
    print("=" * 56)
    total = len(ops)
    for i, (ip, op) in enumerate(ops, 1):
        print(f"\n[{tag} {i}/{total}] Processing: {op}")
        if not Path(script).exists():
            print(f"[{tag}] {script} NOT found.")
            continue
        cmd = [PYTHON, "-u", script]
        if extra_ip:
            cmd += ["--input_folder", ip]
        cmd += ["--output_folder", op, "--config", config]
        rc = run(cmd)
        print(f"[{tag} {i}/{total}] {'Done.' if rc == 0 else 'Python exited with error. Continuing...'}")
    print(f"\n[MASTER] {label} complete for all {total} folder(s).")


def _is_date_name(name):
    return len(name) == 8 and name.isdigit()


def sequential_targets(root, ops):
    """Folder targets for the steps that don't need an ip folder (c/r/u/v). Besides
    the op* folders derived from ip*, include every op* folder in the root AND every
    session-date-named folder (YYYYMMDD) whose name ALSO appears in a file/subfolder
    inside it — i.e. the folder really belongs to that session (e.g. '20260629'
    holding 'Rat6_20260629.nwb' or 'Rat6_HM_..._20260629_..._sorting_output').
    Returns a list of (folder, folder) pairs (ip unused by these steps)."""
    seen = {}
    for _ip, op in ops:
        seen[op.resolve()] = op
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        key = d.resolve()
        if key in seen:
            continue
        if d.name.startswith("op"):
            seen[key] = d
        elif _is_date_name(d.name):
            try:
                if any(d.name in c.name for c in d.iterdir()):
                    seen[key] = d
            except OSError:
                pass
    return [(op, op) for op in seen.values()]


# ------------------------------------------------------------
#                 MASTER
# ------------------------------------------------------------
def main():
    print("=" * 56)
    print("          SMART PARALLEL MODE (Multi-Step)")
    print("=" * 56)
    print(f"[CONFIG] Max CPU: {MAX_CPU}% | Max GPU: {MAX_GPU}% | Max MEM: {MAX_MEM}%\n")

    if len(sys.argv) < 2 or not sys.argv[1]:
        print(f"Usage: {Path(sys.argv[0]).name} /path/to/data_root")
        sys.exit(1)
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"[ERROR] Data folder not found: {sys.argv[1]}")
        sys.exit(1)
    root = root.resolve()

    config = load_config()
    print(f"[DEBUG] Target Root Directory: [{root}]\n")

    print("Select steps to run (e.g., 123 for steps 1, 2, and 3):")
    for key, label in MENU:
        print(f"[{key}] {label}")
    print()
    selection = input("Enter steps: ")

    # Split the selection into sequential master steps and parallel worker steps.
    has = {k: (k in selection) for k in SEQUENTIAL_STEPS}
    parallel_steps = "".join(c for c in selection if c not in SEQUENTIAL_STEPS)
    parallel_trim = "".join(parallel_steps.split())

    # Scan ip* folders. Each ipN -> opN.
    ip_dirs = sorted(d for d in root.glob("ip*") if d.is_dir())
    if not ip_dirs:
        print(f"[WARNING] No ip* folders found under {root}")
    ops = []  # list of (ip_path, op_path) — for ip-dependent steps (workers/7/9)
    for ip_path in ip_dirs:
        num = ip_path.name[len("ip"):]
        ops.append((ip_path, root / f"op{num}"))

    # Targets for the ip-INDEPENDENT steps (c/r/u/v): op* folders + session-date
    # folders whose name matches a file inside them. This lets those steps run on
    # session-named folders, not just op*.
    seq_ops = sequential_targets(root, ops)
    if seq_ops:
        print(f"[DEBUG] ip-independent step targets: "
              f"{', '.join(op.name for _i, op in seq_ops)}")

    # --- Launch parallel workers. By default each opens its OWN terminal window so
    # its progress is visible live (set WORKER_WINDOWS=0 to run them as quiet
    # background threads logging to files instead). ---
    tmp = Path(tempfile.gettempdir())
    use_windows = os.environ.get("WORKER_WINDOWS", "1") != "0"
    threads, markers, count = [], [], 0
    if parallel_trim:
        for ip_path, op_path in ops:
            print(f"\n[QUEUE] Preparing: {ip_path.name}")
            wait_for_resources()
            count += 1
            name = ip_path.name
            marker = tmp / f"hm_worker_{name}.done"
            try:
                marker.unlink()
            except FileNotFoundError:
                pass
            opened = launch_worker_window(ip_path, op_path, parallel_steps, marker, name) \
                if use_windows else False
            if opened:
                markers.append((marker, name))
                print(f"[MASTER] Job launched in its own window: Job-{name}")
            else:                                   # fallback: background thread + log file
                logpath = tmp / f"hm_worker_{name}.log"
                print(f"[MASTER] Launching worker for {name} (log: {logpath})")
                logf = open(logpath, "w")

                def _job(ip=ip_path, op=op_path, lf=logf):
                    try:
                        run_worker(ip, op, parallel_steps, lf)
                    finally:
                        lf.close()

                t = threading.Thread(target=_job, daemon=True)
                t.start()
                threads.append((t, name))
            print(f"[MASTER] Job launched. Waiting {LAUNCH_GAP}s for stability...")
            time.sleep(LAUNCH_GAP)

    if count > 0:
        print("\n" + "=" * 56)
        print(f"[MASTER] Launched {count} parallel job(s). Waiting for all to finish...")
        print("=" * 56)
        for marker, name in markers:            # windowed workers signal via a marker file
            while not marker.exists():
                time.sleep(2)
            print(f"[MASTER] Worker finished: Job-{name}")
        for t, name in threads:                 # background-thread workers
            t.join()
            print(f"[MASTER] Worker finished: {name}")
        print("[MASTER] All parallel workers have completed.")

    # --- Sequential master steps, in fixed order ---
    if has["7"]:
        _run_per_op("SORTING", "SORT", "./src/sorter/sorting.py", ops, config, extra_ip=True)
    if has["c"]:
        _run_per_op("CONTINUE-AFTER-SORTING", "CONT", "./src/sorter/continue_sorting.py", seq_ops, config)
    if has["r"]:
        _run_per_op("RECOMPUTE-METRICS (curated Phy)", "RECOMP", "./src/sorter/recompute_metrics.py", seq_ops, config)

    if has["9"]:
        print("\n" + "=" * 56)
        print("[MASTER] Running CLEANING sequentially (after sorting)...")
        print("=" * 56)
        for i, (ip, _op) in enumerate(ops, 1):
            print(f"\n[CLEAN {i}/{len(ops)}] Cleaning: {ip}")
            clean_folder(ip)
        print(f"\n[MASTER] Cleaning complete for all {len(ops)} folder(s).")

    if has["w"]:
        rat_nr = os.environ.get("NWB_RAT_NR", "1")
        print("\n" + "=" * 56)
        print(f"[MASTER] Running NWB / LFP packaging (rat_nr={rat_nr})...")
        print("=" * 56)
        if Path("./src/nwb/create_nwb.py").exists():
            rc = run([PYTHON, "-u", "create_nwb.py", "--rat_nr", rat_nr,
                      "--noroot", "--ip", root, "--op", root], cwd="./src/nwb")
            print("[NWB] Done." if rc == 0 else "[NWB] Python exited with error.")
        else:
            print(f"[NWB] create_nwb.py NOT found at: {SCRIPT_DIR / 'src/nwb/create_nwb.py'}")

    if has["u"]:
        _run_per_op("ADD-UNITS (curated Phy -> NWB)", "UNITS", "./src/nwb/add_units.py", seq_ops, config)

    if has["v"]:
        _run_per_op("VISUALIZE-NWB (summary + per-unit PDFs)", "VIZ", "./src/nwb/visualize_nwb.py", seq_ops, config)

    if has["b"]:
        # viz_folds=1 -> the continuous decoded_*.npz track plot_trials overlays is
        # full-data (smooth). cv_folds -> the reported accuracy (comparison PDF +
        # step-s summary) is cross-validated, i.e. tested on held-out data.
        folds = os.environ.get("DECODE_FOLDS", "1")          # visualisation track
        cv_folds = os.environ.get("DECODE_CV_FOLDS", "5")    # accuracy (held-out)
        # prediction leads (seconds ahead) compared in one PDF; lead 0 also saves the
        # decoded_*.npz that plot_trials overlays. Set DECODE_LEADS="" to disable.
        leads = os.environ.get("DECODE_LEADS", "0 1 3").split()
        # after each decode, overlay the decoded position onto the real behaviour
        # video (make_videos.py, one mp4 per goal trial). DECODE_VIDEO=0 disables it;
        # DECODE_VIDEO_LEADS picks the leads shown in the overlay (default 0 1 2 3).
        make_vid = os.environ.get("DECODE_VIDEO", "1") != "0"
        vid_leads = os.environ.get("DECODE_VIDEO_LEADS", "0 1 2 3").split()
        # always produce BOTH a good-only and a good+mua decode per session.
        qual_sets = [["good"], ["good", "mua"]]
        print("\n" + "=" * 56)
        print(f"[MASTER] Running POSITION-DECODER sequentially (good and good+mua, "
              f"viz folds: {folds}, CV folds: {cv_folds}, leads: {' '.join(leads) or 'none'})...")
        print("=" * 56)
        total = len(seq_ops)
        for i, (_ip, op) in enumerate(seq_ops, 1):
            print(f"\n[DECODE {i}/{total}] Decoding: {op}")
            if Path("./src/nwb/decode_position.py").exists():
                for quals in qual_sets:
                    cmd = [PYTHON, "-u", "./src/nwb/decode_position.py", "--output_folder", op,
                           "--config", config, "--folds", folds, "--quality", *quals]
                    if leads:                   # CV accuracy PDF + full-data lead-0 npz
                        cmd += ["--leads", *leads, "--cv_folds", cv_folds]
                    rc = run(cmd)
                    print(f"[DECODE {i}/{total}] units {'+'.join(quals)}: "
                          f"{'Done.' if rc == 0 else 'Python exited with error. Continuing...'}")
                    # overlay the decoded position on the real behaviour video
                    # (one mp4 per goal trial) for this quality set.
                    if make_vid and vid_leads and Path("./src/nwb/make_videos.py").exists():
                        vcmd = [PYTHON, "-u", "./src/nwb/make_videos.py", "--output_folder", op,
                                "--config", config, "--which", "decoded", "--quality", *quals,
                                "--leads", *vid_leads]
                        vrc = run(vcmd)
                        print(f"[DECODE {i}/{total}] units {'+'.join(quals)} decoded-video: "
                              f"{'Done.' if vrc == 0 else 'Python exited with error. Continuing...'}")
            else:
                print("[DECODE] decode_position.py NOT found.")
        print(f"\n[MASTER] Position-decoder complete for all {total} folder(s).")

    if has["m"]:
        # UMAP population embedding per session, both GOOD-only and GOOD+MUA.
        print("\n" + "=" * 56)
        print("[MASTER] Running NEURAL-UMAP sequentially (good and good+mua)...")
        print("=" * 56)
        total = len(seq_ops)
        for i, (_ip, op) in enumerate(seq_ops, 1):
            print(f"\n[UMAP {i}/{total}] Embedding: {op}")
            if Path("./src/nwb/neural_umap.py").exists():
                for quals in (["good"], ["good", "mua"]):
                    rc = run([PYTHON, "-u", "./src/nwb/neural_umap.py", "--output_folder", op,
                              "--config", config, "--quality", *quals])
                    print(f"[UMAP {i}/{total}] units {'+'.join(quals)}: "
                          f"{'Done.' if rc == 0 else 'Python exited with error. Continuing...'}")
            else:
                print("[UMAP] neural_umap.py NOT found.")
        print(f"\n[MASTER] Neural-UMAP complete for all {total} folder(s).")

    if has["s"]:
        # cross-session aggregation over ALL NWBs under the root (one call, not per-op)
        print("\n" + "=" * 56)
        print("[MASTER] Running SESSION-SUMMARY (cross-session per-animal plots)...")
        print("=" * 56)
        if Path("./src/nwb/session_summary.py").exists():
            rc = run([PYTHON, "-u", "./src/nwb/session_summary.py", "--root", root, "--config", config])
            print("[SUMMARY] Done." if rc == 0 else "[SUMMARY] Python exited with error.")
        else:
            print("[SUMMARY] session_summary.py NOT found.")

    if has["t"]:
        # raw-drive integrity scan over the whole root (one call, not per-op)
        print("\n" + "=" * 56)
        print("[MASTER] Running DRIVE SCAN (videos + ephys phases + non-zero .rec)...")
        print("=" * 56)
        if Path("./src/tools/scan_drive.py").exists():
            rc = run([PYTHON, "-u", "./src/tools/scan_drive.py", "--root", str(root), "--config", config])
            print("[DRIVE-SCAN] Done." if rc == 0 else "[DRIVE-SCAN] Python exited with error.")
        else:
            print("[DRIVE-SCAN] scan_drive.py NOT found.")

    flags = " | ".join(f"{k}:{int(has[k])}" for k in SEQUENTIAL_STEPS)
    print("\n" + "=" * 56)
    print(f"[MASTER] Done. Parallel jobs: {count} | {flags}")
    print("=" * 56)


def _worker_main(argv):
    """Entry point for a single parallel worker launched in its own terminal
    window: `runner.py --worker <ip> <op> <steps> [<marker>]`. Runs the worker with
    output to this console and touches <marker> on completion so the master waits."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # scripts use ./src/... paths
    ip, op, steps = argv[0], argv[1], argv[2]
    marker = argv[3] if len(argv) > 3 else None
    load_config()                                          # inherit paths/FREQ/etc. into env
    try:
        run_worker(Path(ip), Path(op), steps, sys.stdout)
    finally:
        if marker:
            try:
                Path(marker).write_text("done")
            except Exception:
                pass
        print("\n[WORKER] Finished. You can close this window.")


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):        # UTF-8 console so '->' etc. never crash
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2:])
    else:
        main()
