#!/usr/bin/env python3
# ============================================================
#   HM Tracker — cross-platform runner (single source of truth)
#
#   Usage:  python runner.py /path/to/data_root
#
#   This ONE file drives the whole pipeline on macOS, Linux and
#   Windows. The old scripts/runner_unix.sh / scripts/runner_windows.bat are now
#   thin launchers that just call this. Edit the menu / steps here
#   ONCE and both platforms stay in sync.
#
#   Behaviour (ported from both old runners):
#     - Parallel steps (1 e 2 3 4 5 8 d n) run one background
#       worker per ip*/op* pair, gated by a CPU/GPU/MEM monitor.
#     - Sequential master steps (7 c r 9 w u) run afterwards, one
#       folder at a time, in that order.
#     - Each parallel worker logs to <tmp>/hm_worker_<name>.log.
#     - A missing tool (nvidia-smi, trodes, ...) never blocks the run.
#     - TREE MODE — a root without ip* folders, e.g. the share's task folder
#       where each session sits at <rat>/<YYYYMMDD>/: the session folders are
#       found below the root, and the per-session steps (l u v b 5 m) run as
#       parallel jobs, one per session, with a live resource monitor (see
#       run_post_jobs). A root with ip* folders runs exactly as before.
#
#   Env overrides: MAX_CPU MAX_GPU MAX_MEM WAIT_SECONDS LAUNCH_GAP
#   FFMPEG_VCODEC SYNC_START_SEC NWB_RAT_NR DISABLE_RESOURCE_CHECK
#   HM_CONFIG_FILE POST_JOBS TREE_DEPTH MONITOR_SECONDS POST_LOG_DIR
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
    ("1", "Trodes Export (DIO + Analog/IMU)"),
    ("2", "Sync Script"),
    ("3", "Stitching"),
    ("4", "Tracker (+ fix .txt timestamps + write/refresh the session NWB: behaviour/trials)"),
    ("5", "Plotting"),
    ("6", "Compression (always runs LAST, over all op folders)"),
    ("7", "Sorting + clock QC/repair"),
    ("c", "Continue After Sorting (metrics + BombCell + Phy, no re-sort)"),
    ("r", "Recompute Metrics (after manual Phy curation)"),
    ("8", "LFP + Motion (IMU Accel) + EMG-from-LFP + sleep NWB"),
    ("l", "Rewrite the NWB LFP from the old LFP_Output export (5 Hz stubs / 30 kHz stamps; at LFP_OUTPUT_RATE)"),
    ("d", "deeplabcut (extract eye frames + run DLC inference -> keypoints in CSV)"),
    ("9", "Cleaning"),
    ("n", "Node Analysis"),
    ("u", "Add curated Units (metrics + waveforms) to NWB (runs after 4)"),
    ("v", "Visualize NWB units (summary + per-unit rate-map PDFs; runs after u)"),
    ("b", "Bayesian position decoder + spikes/decoded-on-video overlays per session (good and good+mua)"),
    ("m", "Neural population UMAP per session (good & good+mua, all + pyramidal-only; Gardner et al. 2022)"),
    ("t", "Drive scan (videos playable + ephys has pre/task/post + non-zero .rec)"),
]

# Sequential master-level steps, in execution order. Everything NOT in here is
# a parallel worker step.
SEQUENTIAL_STEPS = ["7", "c", "r", "9", "l", "u", "v", "b", "m", "t"]

# Tree mode (a root without ip* folders): these steps run per session, as parallel
# jobs, in this order — l before v (the theta pages read the LFP), u before v/b/m
# (they read the units), b before 5 (plot_trials overlays the decoded track).
POST_JOB_STEPS = ["l", "u", "v", "b", "5", "m"]

# Retired keys, still accepted so saved presets / HM_STEPS strings keep working:
# both now run inside step 4. Typed alone, they refresh the NWB of every op target.
RETIRED_STEPS = {"f": "fix .txt timestamps", "w": "NWB behaviour/trials"}


# ------------------------------------------------------------
#                 CONFIGURATION (SHARED)
# ------------------------------------------------------------
def _int_env(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _str_env(*names, default=""):
    """First non-empty value among ``names``, whitespace-stripped.

    load_config() already strips the values it reads from hm_tracker_paths.txt;
    stripping again here also covers variables set directly in the shell.
    """
    for name in names:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    return default


def _flag_env(*names):
    """A yes/no setting from the config. Off unless it reads as true.

    Written out because "KEEP_NPY=0" is a non-empty string and would otherwise
    be truthy — i.e. switching the setting off would switch it on.
    """
    return _str_env(*names).lower() in ("1", "true", "yes", "on")


MAX_CPU = _int_env("MAX_CPU", 90)
MAX_GPU = _int_env("MAX_GPU", 90)
MAX_MEM = _int_env("MAX_MEM", 65)
WAIT_SECONDS = _int_env("WAIT_SECONDS", 30)
LAUNCH_GAP = _int_env("LAUNCH_GAP", 20)   # seconds between worker launches
TREE_DEPTH = _int_env("TREE_DEPTH", 4)    # tree mode: how deep to look for session folders
MONITOR_SECONDS = _int_env("MONITOR_SECONDS", 30)   # tree mode: status line interval
POLL_SECONDS = 2.0                        # tree mode: how often jobs are checked
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
        print("  TRODES_EXPORT_CMD=/path/to/trodesexport\n")
        print("See examples/hm_tracker_paths.example.txt in the repo for a template.")
        sys.exit(1)

    for raw in Path(cfg).read_text(errors="replace").splitlines():
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = "".join(key.split())
        if key:
            # Strip the spaces around the value: "TRODES_EXPORT_CMD = /path"
            # must give "/path", not " /path" — a path with a leading space
            # does not exist, so the tool was reported as "not found".
            os.environ[key] = val.strip()

    # Auto-use the system ffmpeg (on PATH) when FFMPEG_CMD is unset or points to a
    # path that doesn't exist here — e.g. a Windows path in the config on Linux/Mac,
    # where ffmpeg is normally installed system-wide. Exported so all steps inherit it.
    ff = os.environ.get("FFMPEG_CMD", "").strip()
    if not (ff and (Path(ff).is_file() or shutil.which(ff))):
        system_ff = shutil.which("ffmpeg")
        if system_ff:
            if ff:
                print(f"[INFO] FFMPEG_CMD '{ff}' not found here — using system "
                      f"ffmpeg on PATH: {system_ff}")
            os.environ["FFMPEG_CMD"] = system_ff
        elif ff:
            print(f"[WARN] FFMPEG_CMD '{ff}' not found and no ffmpeg on PATH.")
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
            # start "title" cmd /k "<cmd>"  — new console that stays open when done.
            # The extra outer quotes are required: with >2 quotes on the line (python
            # under "C:\Users\Genzel Lab\..." + a %TEMP% marker), cmd /k strips the first
            # and last quote and runs 'C:\Users\Genzel'. Wrapping makes it strip ours.
            inner = subprocess.list2cmdline(args)
            subprocess.Popen(f'start "HM Job-{title}" cmd /k "{inner}"',
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


def pick_vcodec(size=(1920, 1080)):
    """Encoder for compression: GPU when any GPU path works, else CPU."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from tools import vcodec
    return vcodec.select(mode="quality", size=size)


def log(out, msg):
    print(msg, file=out, flush=True)


_SESSION_RE = __import__("re").compile(r"(?P<rat>[A-Za-z]+\d+).*?(?P<dt>\d{8}_\d{6})")


def session_pfx(folder):
    """rat_sessiondate_ prefix from a .rec (or the folder name) in `folder`, else ''."""
    cands = sorted(Path(folder).glob("*.rec")) or [Path(folder)]
    for c in cands:
        m = _SESSION_RE.search(c.stem)
        if m:
            return f"{m.group('rat')}_{m.group('dt')}_"
    return ""


# ------------------------------------------------------------
#                 THE WORKER (parallel, per ip/op pair)
# ------------------------------------------------------------
def run_worker(ip, op, steps, out):
    """Run the per-folder parallel steps for one ip/op pair, logging to `out`.
    Sequential steps (7 c r 9 u ...) are handled at master level, not here."""
    ip, op = Path(ip), Path(op)
    op.mkdir(parents=True, exist_ok=True)
    log(out, f"[INFO] Running steps [{steps}] for {ip}")
    if not steps.strip():
        log(out, "[WORKER] No steps to run, exiting.")
        return

    freq = os.environ.get("FREQ", FREQ)
    sync_start = os.environ.get("SYNC_START_SEC", "45")
    sync_led = os.environ.get("SYNC_LED", "auto")        # auto|red|blue (which LED drives sync)
    recs = sorted(ip.glob("*.rec"))

    # --- STEP 1: Trodes DIO/Analog export (per .rec) ---
    #  No -raw: steps 7 and 8 read the wideband straight from the .rec
    #  (src/sorter/trodes_rec.py), so the ~27 GiB/h raw copy is not written.
    if "1" in steps:
        log(out, "[STEP 1] Running Trodes DIO/Analog Export (per .rec)...")
        trodes = _tool("TRODES_EXPORT_CMD")
        if trodes:
            if not recs:
                log(out, f"[WARNING] No .rec files found in '{ip}'")
            for f in recs:
                log(out, f"    Exporting {f.name}")
                run([trodes, "-dio", "-analogio", "-rec", f], out=out)
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
        if run([PYTHON, "-u", "./src/tracker/join_views.py", ip], out=out):
            log(out, "[ERROR] Stitching failed — stitched.mp4 not produced, so the tracker "
                     "(step 4) and everything downstream of it will be skipped.")

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
    #  Compression is NOT run here anymore. It always runs LAST, at master level,
    #  over ALL op folders (so it can't be interrupted by other steps and never
    #  compresses a video that a later step still needs). See compress_all_op_videos().

    # --- STEP 8: LFP + Motion/IMU extraction ---
    if "8" in steps:
        # Per-sample data goes into the session NWB. The LFP is built for ALL
        # channels straight from the .rec (or an old *.raw/*_group0.dat); without
        # either, it falls back to an old exportLFP export (<recording>.LFP/, one
        # channel per nTrode, from the retired step e). Motion reads the analog
        # (IMU) export of step 1. KEEP_NPY=1 in hm_tracker_paths.txt also writes
        # the old LFP_Output/*.npy alongside it.
        keep_npy = ["--keep-npy"] if _flag_env("KEEP_NPY", "HM_KEEP_NPY") else []
        lfp_rate = _str_env("LFP_OUTPUT_RATE", default="1500")
        if Path("./src/sorter/export_lfp.py").exists():
            log(out, "[STEP 8] Running LFP Extraction...")
            run([PYTHON, "-u", "./src/sorter/export_lfp.py",
                 "--input_folder", ip, "--output_folder", op,
                 "--output_rate", lfp_rate] + keep_npy, out=out)
        # EMG-from-LFP needs the raw wideband (300-600 Hz); runs after the LFP
        # export so it can upsample onto lfp_timestamps.npy. Reads the .rec.
        if Path("./src/sorter/export_emg_from_lfp.py").exists():
            log(out, "[STEP 8] Running EMG-from-LFP (raw wideband, Buzsáki)...")
            run([PYTHON, "-u", "./src/sorter/export_emg_from_lfp.py",
                 "--input_folder", ip, "--output_folder", op] + keep_npy, out=out)
        if Path("./src/sorter/export_motion.py").exists():
            log(out, "[STEP 8] Running Motion (IMU Accel) Extraction...")
            run([PYTHON, "-u", "./src/sorter/export_motion.py",
                 "--input_folder", ip, "--output_folder", op] + keep_npy, out=out)
        # Package LFP + EMG + motion into the session NWB last, once all three
        # exports above have produced their .npy files. This CREATES the session
        # NWB that steps w and u later append to, and is what the sleep-scoring
        # GUI reads (and writes each scorer's result back into).
        if Path("./src/nwb/export_sleep_nwb.py").exists():
            log(out, "[STEP 8] Packaging sleep-scoring inputs into the session NWB...")
            run([PYTHON, "-u", "./src/nwb/export_sleep_nwb.py",
                 "--input_folder", ip, "--output_folder", op], out=out)

    # --- STEP d: DeepLabCut export + inference (extract frames -> run DLC -> write) ---
    if "d" in steps and Path("./src/dlc/tracking_eyes.py").exists():
        log(out, "[STEP d] Exporting eye frames for DeepLabCut...")
        run([PYTHON, "-u", "./src/dlc/tracking_eyes.py",
             "--input_folder", ip, "--output_folder", op], out=out)
        # Run DeepLabCut on the collected_frames.mp4 just exported and merge the
        # predicted keypoints back into the CSV. Uses the GPU like step 4; the
        # per-worker resource gate (MAX_GPU) throttles concurrent launches. Skipped
        # unless DLC_CONFIG_PATH (the DLC project's config.yaml) points to a real file.
        dlc_cfg = os.environ.get("DLC_CONFIG_PATH", "").strip()
        if Path("./src/dlc/dlc_coordinates.py").exists():
            if Path(dlc_cfg).is_file():
                log(out, "[STEP d] Running DeepLabCut inference...")
                run([PYTHON, "-u", "./src/dlc/dlc_coordinates.py",
                     "--output_folder", op, "--config", dlc_cfg,
                     "--shuffle", os.environ.get("DLC_SHUFFLE", "2")], out=out)
            else:
                log(out, f"[STEP d] Skipping DLC inference: DLC_CONFIG_PATH not set or not found: '{dlc_cfg}'")

    # --- STEP n: node analysis ---
    if "n" in steps and Path("./src/node_analysis/hex_maze_analysis.py").exists():
        log(out, "[STEP n] Running Node Analysis...")
        run([PYTHON, "-u", "./src/node_analysis/hex_maze_analysis.py",
             "--input_folder", ip, "--output_folder", op], out=out)

    # --- STEP 4 (end): fix .txt timestamps + write/refresh the session NWB ---
    #  Last in the worker so the NWB picks up DLC keypoints (step d) and joins
    #  the LFP/EMG/motion step 8 just wrote, whichever of the two ran first.
    if "4" in steps:
        refresh_session_nwb(op, out)

    log(out, f"[COMPLETE] Worker finished for {ip}")


def refresh_session_nwb(op, out=None):
    """Former steps f + w for one op folder: convert any unix timestamps in the
    tracker .txt to session seconds, then write the behaviour/trials into the
    session NWB (created if absent; a re-track REPLACES the old behaviour, while
    step 8's LFP, sleep scorings and step u's units are kept)."""
    op = Path(op).resolve()
    log_ = (lambda m: log(out, m)) if out else print
    if Path("./src/tracker/fix_txt_timestamps.py").exists():
        log_("[STEP 4] Fixing .txt timestamps (unix -> session seconds)...")
        run([PYTHON, "-u", "./src/tracker/fix_txt_timestamps.py",
             "--output_folder", op], out=out)
    if Path("./src/nwb/create_nwb.py").exists():
        rat_nr = os.environ.get("NWB_RAT_NR", "1")
        log_(f"[STEP 4] Writing behaviour/trials into the session NWB (rat_nr={rat_nr})...")
        run([PYTHON, "-u", "create_nwb.py", "--rat_nr", rat_nr, "--noroot",
             "--ip", op.parent, "--op", op.parent, "--session_folder", op.name],
            cwd=str(SCRIPT_DIR / "src" / "nwb"), out=out)
    if Path("./src/nwb/clock_qc.py").exists():
        # Behaviour vs ephys clock, from the data (theta amplitude vs running speed).
        # Reports only; a failing session is flagged loudly with the repair command.
        log_("[STEP 4] Checking the behaviour clock against the ephys clock...")
        run([PYTHON, "-u", "./src/nwb/clock_qc.py", op], out=out)


def clock_qc_after_sorting(ops):
    """Step 7, after sorting: check the behaviour clock against the ephys clock and
    repair it automatically when needed.

    clock_qc.py --auto re-times the op folder's video-clock files (framewise CSVs,
    coordinates Timestamp, tracker .txt/.log) only when the check FAILS and the
    repaired clock passes the same check; originals go to <op>/clock_repair_backup/. A repaired
    session's NWB is then refreshed (step 4 end) so positions and trials carry the
    corrected clock. A failure no repair can verify is left untouched and reported.
    """
    if not Path("./src/nwb/clock_qc.py").exists():
        return
    print("\n" + "=" * 56)
    print("[MASTER] Clock QC (behaviour vs ephys) with automatic repair...")
    print("=" * 56)
    review, repaired = [], []
    for i, (_ip, op) in enumerate(ops, 1):
        print(f"\n[CLOCK {i}/{len(ops)}] {op}")
        rc = run([PYTHON, "-u", "./src/nwb/clock_qc.py", op, "--auto"])
        if rc == 10:
            print(f"[CLOCK {i}/{len(ops)}] repaired — refreshing the session NWB")
            refresh_session_nwb(op)
            repaired.append(op)
        elif rc == 3:
            review.append(op)
        elif rc != 0:
            print(f"[CLOCK {i}/{len(ops)}] clock_qc.py exited with {rc}")
    if repaired:
        # figures and decodes made before the repair are on the old clock
        print("\n[CLOCK] Clock repaired (originals in <op>/clock_repair_backup/) for:")
        for op in repaired:
            print(f"   {op}")
        print("Steps 5 / v / b / m outputs made before this run are on the old clock: "
              "re-run them for these sessions (later steps of this run already use the fix).")
    if review:
        print("\n" + "*" * 70)
        print("[CLOCK] Behaviour/ephys clocks disagree and no repair verified for:")
        for op in review:
            print(f"   {op}")
        print("Re-run step 2 for these (needs the raw .meta + DIO), see <op>/clock_qc/.")
        print("*" * 70)


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


def _ffprobe_cmd():
    """ffprobe path: FFPROBE_CMD, else next to FFMPEG_CMD, else bare 'ffprobe'.

    Setting FFMPEG_CMD to a full path usually means that install is NOT on PATH,
    so bare 'ffprobe' would not resolve either — and a silent miss here downgrades
    the encoder probe to a guessed frame size."""
    explicit = os.environ.get("FFPROBE_CMD", "")
    if explicit:
        return explicit
    ffmpeg = os.environ.get("FFMPEG_CMD", "")
    if ffmpeg:
        p = Path(ffmpeg)
        sibling = p.with_name("ffprobe" + p.suffix)
        if sibling.exists():
            return str(sibling)
    return "ffprobe"


def _video_size(video):
    """(w, h) of `video` via ffprobe, or None if it cannot be read."""
    ffprobe = _ffprobe_cmd()
    try:
        p = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video)],
                           capture_output=True, text=True, timeout=30)
        w, h = p.stdout.strip().split("x")[:2]
        return int(w), int(h)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def compress_all_op_videos(op_dirs):
    """STEP 6 (final phase): GPU-compress EVERY top-level .mp4 in each op folder,
    in-place. Runs last so it never re-encodes a video another step still reads."""
    ffmpeg = os.environ.get("FFMPEG_CMD", "ffmpeg")
    todo = [(op, [f for f in sorted(op.glob("*.mp4")) if f.name != "__temp_compressed.mp4"])
            for op in op_dirs]

    # Probe at the largest frame we actually have to encode: an old GPU can manage
    # 1080p and still refuse the ~2352x1424 stitch, and we would rather find that
    # out now than after re-encoding half the folder.
    sizes = [s for _, vids in todo for s in map(_video_size, vids) if s]
    probe_size = max(sizes, key=lambda s: s[0] * s[1]) if sizes else (1920, 1080)
    enc = pick_vcodec(size=probe_size)
    # VAAPI encodes from a hardware surface: device before the input, frames
    # uploaded by a filter. Everything else takes ordinary software frames.
    vf_args = ["-vf", enc.filter_chain] if enc.filter_chain else []

    print("\n" + "=" * 56)
    print(f"[MASTER] STEP 6 — Compression (codec: {enc.codec} {' '.join(enc.args)}, probed at "
          f"{probe_size[0]}x{probe_size[1]}) over {len(op_dirs)} op folder(s), all videos, "
          f"running LAST...")
    print("=" * 56)
    n_ok = n_fail = 0
    for op, videos in todo:
        if not videos:
            print(f"[COMPRESS] {op.name}: no .mp4 to compress.")
            continue
        for video in videos:
            temp_file = op / "__temp_compressed.mp4"
            if temp_file.exists():
                temp_file.unlink()
            print(f"\n[COMPRESS] {op.name}/{video.name} ...")
            rc = run([ffmpeg, "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
                      "-stats", *enc.global_args, "-i", video, *vf_args,
                      "-c:v", enc.codec, *enc.args, "-c:a", "copy", temp_file])
            if rc == 0:
                shutil.move(str(temp_file), str(video))
                print(f"[SUCCESS] Compressed: {video}")
                n_ok += 1
            else:
                print(f"[ERROR] FFmpeg failed for {video} (codec '{enc.codec}'). "
                      f"Set FFMPEG_VCODEC to force another encoder, e.g. libx264.")
                if temp_file.exists():
                    temp_file.unlink()
                n_fail += 1
    print(f"\n[MASTER] Compression done: {n_ok} ok, {n_fail} failed.")


# ------------------------------------------------------------
#        PER-SESSION STEP BODIES (master loops and tree jobs)
# ------------------------------------------------------------
def decode_settings():
    """Step b's settings, from the config / environment."""
    return {
        # viz_folds=1 -> the continuous decoded_*.npz track plot_trials overlays is
        # full-data (smooth). cv_folds -> the reported accuracy (comparison PDF +
        # step-s summary) is cross-validated, i.e. tested on held-out data.
        "folds": os.environ.get("DECODE_FOLDS", "1"),          # visualisation track
        "cv_folds": os.environ.get("DECODE_CV_FOLDS", "5"),    # accuracy (held-out)
        # prediction leads (seconds ahead) compared in one PDF; lead 0 also saves the
        # decoded_*.npz that plot_trials overlays. Set DECODE_LEADS="" to disable.
        "leads": os.environ.get("DECODE_LEADS", "0 1 3").split(),
        # after decoding, overlay analysis onto the real behaviour video via
        # make_videos.py (one mp4 per goal trial): the decoded position per quality
        # set, plus a spikes-on-path video (top-20 good pyramidal by spatial info,
        # once per op). DECODE_VIDEO=0 disables both; DECODE_VIDEO_LEADS picks the
        # decoded-overlay leads (default 0 1 2 3).
        "make_vid": os.environ.get("DECODE_VIDEO", "1") != "0",
        "vid_leads": os.environ.get("DECODE_VIDEO_LEADS", "0 1 2 3").split(),
        # predictive-coding test: is the long-lead 'prediction' genuine, or behaviour/
        # occupancy? figures -> <op>/predictive_coding/. PREDICTIVE_CODING=0 disables.
        "run_pc": os.environ.get("PREDICTIVE_CODING", "1") != "0",
        "pc_cv": os.environ.get("PC_CV_FOLDS", "5"),
        "pc_shuf": os.environ.get("PC_SHUFFLE", "8"),
        # always produce BOTH a good-only and a good+mua decode per session.
        "qual_sets": [["good"], ["good", "mua"]],
    }


def decode_session(op, config, tag, out=None):
    """Step b for one op folder: decode (good, good+mua), the decoded-on-video and
    spikes-on-video overlays, and the predictive-coding test. Returns the exit codes."""
    say = (lambda m: log(out, m)) if out else print
    s = decode_settings()
    folds, cv_folds, leads = s["folds"], s["cv_folds"], s["leads"]
    rcs = []
    if not Path("./src/nwb/decode_position.py").exists():
        say("[DECODE] decode_position.py NOT found.")
        return [127]
    for quals in s["qual_sets"]:
        cmd = [PYTHON, "-u", "./src/nwb/decode_position.py", "--output_folder", op,
               "--config", config, "--folds", folds, "--quality", *quals]
        if leads:                   # CV accuracy PDF + full-data lead-0 npz
            cmd += ["--leads", *leads, "--cv_folds", cv_folds]
        rc = run(cmd, out=out)
        rcs.append(rc)
        say(f"{tag} units {'+'.join(quals)}: "
            f"{'Done.' if rc == 0 else 'Python exited with error. Continuing...'}")
        # overlay the decoded position on the real behaviour video
        # (one mp4 per goal trial) for this quality set.
        if s["make_vid"] and s["vid_leads"] and Path("./src/nwb/make_videos.py").exists():
            vcmd = [PYTHON, "-u", "./src/nwb/make_videos.py", "--output_folder", op,
                    "--config", config, "--which", "decoded", "--quality", *quals,
                    "--leads", *s["vid_leads"]]
            vrc = run(vcmd, out=out)
            rcs.append(vrc)
            say(f"{tag} units {'+'.join(quals)} decoded-video: "
                f"{'Done.' if vrc == 0 else 'Python exited with error. Continuing...'}")
        # predictive-coding test (neural vs behaviour/kinematics/shuffle
        # baselines + overshoot + decode-density figures) for this quality.
        if s["run_pc"] and Path("./src/nwb/predictive_coding.py").exists():
            prc = run([PYTHON, "-u", "./src/nwb/predictive_coding.py",
                       "--output_folder", op, "--config", config,
                       "--quality", *quals, "--cv_folds", s["pc_cv"],
                       "--n_shuffle", s["pc_shuf"]], out=out)
            rcs.append(prc)
            say(f"{tag} units {'+'.join(quals)} predictive-coding: "
                f"{'Done.' if prc == 0 else 'Python exited with error. Continuing...'}")
    # spikes-on-video (top-N good pyramidal cells by spatial info),
    # once per op — the goal-trial spike-path overlay on the real video.
    if s["make_vid"] and Path("./src/nwb/make_videos.py").exists():
        src = run([PYTHON, "-u", "./src/nwb/make_videos.py", "--output_folder", op,
                   "--config", config, "--which", "spikes", "--quality", "good"], out=out)
        rcs.append(src)
        say(f"{tag} spikes-on-video: "
            f"{'Done.' if src == 0 else 'Python exited with error. Continuing...'}")
    return rcs


def umap_session(op, config, tag, out=None):
    """Step m for one op folder: good and good+mua, each all cell types and
    pyramidal-only (4 embeddings). Returns the exit codes."""
    say = (lambda m: log(out, m)) if out else print
    if not Path("./src/nwb/neural_umap.py").exists():
        say("[UMAP] neural_umap.py NOT found.")
        return [127]
    rcs = []
    for quals in (["good"], ["good", "mua"]):
        for ct in (None, "pyramidal"):
            cmd = [PYTHON, "-u", "./src/nwb/neural_umap.py", "--output_folder", op,
                   "--config", config, "--quality", *quals]
            if ct:
                cmd += ["--cell_type", ct]
            rc = run(cmd, out=out)
            rcs.append(rc)
            say(f"{tag} units {'+'.join(quals)}"
                f"{' ' + ct if ct else ''}: "
                f"{'Done.' if rc == 0 else 'Python exited with error. Continuing...'}")
    return rcs


def run_post_job(op, steps, config, out):
    """Tree mode, one session: run the selected POST_JOB_STEPS for `op`, in that
    order. A step that fails does not stop the next one. Returns the failed keys."""
    op = Path(op)
    labels = dict(MENU)
    failed = []
    for key in POST_JOB_STEPS:
        if key not in steps:
            continue
        log(out, f"\n[POST-STEP] {key} — {labels[key]}")     # the monitor shows this
        t0 = time.time()
        if key == "b":
            rcs = decode_session(op, config, "[DECODE]", out)
        elif key == "m":
            rcs = umap_session(op, config, "[UMAP]", out)
        else:
            script, extra = {
                "l": ("./src/nwb/rewrite_nwb_lfp.py", []),
                "u": ("./src/nwb/add_units.py", []),
                "v": ("./src/nwb/visualize_nwb.py", []),
                # the op folder holds what plot_trials reads; there is no ip here
                "5": ("./src/tracker/plot_trials.py", ["--input_folder", op]),
            }[key]
            if not Path(script).exists():
                log(out, f"[POST-STEP] {script} NOT found.")
                rcs = [127]
            elif key == "5":
                rcs = [run([PYTHON, "-u", script, *extra, "--output_folder", op], out=out)]
            else:
                rcs = [run([PYTHON, "-u", script, "--output_folder", op, "--config", config], out=out)]
        ok = all(rc == 0 for rc in rcs)
        log(out, f"[POST-STEP] {key} {'done' if ok else 'FAILED'} "
                 f"({(time.time() - t0) / 60:.1f} min)")
        if not ok:
            failed.append(key)
    log(out, f"\n[POST] {op}: " + ("all steps done." if not failed
                                   else f"failed step(s): {' '.join(failed)}"))
    return failed


# ------------------------------------------------------------
#        TREE MODE: session folders below the root, parallel jobs
# ------------------------------------------------------------
# Folders that are never a session and never hold one: outputs written into a
# session folder, and Trodes/sorter export folders.
_NOT_SESSION_DIRS = {"lfp_output", "visualization", "decoding", "videos", "umap",
                     "predictive_coding", "clock_qc", "clock_repair_backup",
                     "lfp_fix_backup", "channels_npy", "figures", "__pycache__"}
_NOT_SESSION_SUFFIXES = ("_sorting_output", ".raw", ".dio", ".lfp", ".rec",
                         ".analog", ".time", ".spikes", ".mountainsort")


def _is_session_folder(d, names, top_level):
    """An op folder of this pipeline: it holds a session NWB, or it is named after
    a session date that also appears in a file inside it (sequential_targets' rule:
    '20260629' holding 'Rat6_20260629.nwb'), or it is a top-level op* folder."""
    if any(n.lower().endswith(".nwb") and not n.startswith("._") and ".tmp." not in n
           for n in names):
        return True
    if _is_date_name(d.name) and any(d.name in n for n in names):
        return True
    return top_level and d.name.startswith("op")


def find_session_folders(root, max_depth=None):
    """Every session op folder at most `max_depth` levels below `root`, sorted.

    For trees where op folders have been renamed and grouped, e.g. the share's
    task/rat5_491390/20260630/. A session folder is not searched further (its
    subfolders are its outputs), and output/export folders are skipped, so only a
    few directory listings are made — this matters on a network drive."""
    root = Path(root)
    max_depth = _int_env("TREE_DEPTH", TREE_DEPTH) if max_depth is None else max_depth
    found, todo = [], [(root, 0)]
    while todo:
        folder, depth = todo.pop(0)
        try:
            subdirs = sorted(d for d in folder.iterdir() if d.is_dir())
        except OSError as e:
            print(f"[TREE] cannot list {folder}: {e}")
            continue
        for d in subdirs:
            low = d.name.lower()
            if d.name.startswith((".", "~")) or low in _NOT_SESSION_DIRS \
                    or low.endswith(_NOT_SESSION_SUFFIXES):
                continue
            try:
                names = [c.name for c in d.iterdir()]
            except OSError:
                continue
            if _is_session_folder(d, names, top_level=(depth == 0)):
                found.append(d)
            elif depth + 1 < max_depth:
                todo.append((d, depth + 1))
    return sorted(found)


def read_session_list(list_file):
    """The session folders listed in a text file, one folder per line.

    Used instead of find_session_folders when HM_SESSIONS_FILE is set, e.g. by the
    GUI's "Only failed of a run…" (the sessions a previous run did not finish).
    Blank lines and '#' comments are skipped; a folder that does not exist is
    reported and left out."""
    folders = []
    for line in Path(list_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        folder = Path(line)
        if folder.is_dir():
            folders.append(folder)
        else:
            print(f"[TREE] listed session folder not found, skipped: {line}")
    return folders


def session_label(folder, root):
    """'Rat5_20260630' from the folder's session NWB, else its path below root."""
    nwbs = [p for p in Path(folder).glob("*.nwb")
            if not p.name.startswith("._") and ".tmp." not in p.name]
    if len(nwbs) == 1:
        return nwbs[0].stem
    try:
        parts = Path(folder).relative_to(root).parts
    except ValueError:
        parts = (Path(folder).name,)
    return "_".join(parts) or Path(folder).name


def post_jobs_limit(n_sessions):
    """How many sessions run at once: POST_JOBS from the config (HM_POST_JOBS, set
    by the GUI, wins over it), or 'auto' — half
    the CPU cores, one job per 12 GB of RAM, at most 6 (a decode or UMAP job can
    take several GB; the launch gate below keeps a margin on top of this)."""
    raw = _str_env("HM_POST_JOBS", "POST_JOBS", default="auto").lower()   # GUI's first
    if raw != "auto":
        try:
            return max(1, min(int(raw), n_sessions))
        except ValueError:
            print(f"[WARN] POST_JOBS={raw!r} is not a number — using auto.")
    cpus = os.cpu_count() or 2
    mem_gb = psutil.virtual_memory().total / 2 ** 30 if psutil is not None else 16.0
    return max(1, min(cpus // 2, int(mem_gb // 12), 6, n_sessions))


def resources_free():
    """(True, '') when a new job may start, else (False, why). Same limits as the
    parallel workers' wait_for_resources(), without blocking; read now, so the
    config file's values count too (load_config runs after import)."""
    if os.environ.get("DISABLE_RESOURCE_CHECK", "0") == "1":
        return True, ""
    cpu, gpu, mem = get_cpu_load(), get_gpu_load(), get_mem_usage()
    for name, val, lim in (("CPU", cpu, _int_env("MAX_CPU", MAX_CPU)),
                           ("GPU", gpu, _int_env("MAX_GPU", MAX_GPU)),
                           ("MEM", mem, _int_env("MAX_MEM", MAX_MEM))):
        if val > lim:
            return False, f"{name} {val}% > {lim}%"
    return True, ""


class ResourceMonitor:
    """CPU / memory / GPU load and network traffic (a share is read over the
    network) for the tree-mode status lines."""

    def __init__(self):
        self.t, self.net = time.time(), self._net()

    @staticmethod
    def _net():
        if psutil is None:
            return None
        try:
            c = psutil.net_io_counters()
            return c.bytes_recv, c.bytes_sent
        except Exception:
            return None

    def line(self):
        cpu, mem, gpu = get_cpu_load(), get_mem_usage(), get_gpu_load()
        now, net = time.time(), self._net()
        traffic = ""
        if net and self.net:
            secs = max(now - self.t, 1e-3)
            traffic = (f" | net in {(net[0] - self.net[0]) / secs / 1e6:.1f} "
                       f"out {(net[1] - self.net[1]) / secs / 1e6:.1f} MB/s")
        self.t, self.net = now, net
        mem_txt = f"{mem}%"
        if psutil is not None:
            vm = psutil.virtual_memory()
            mem_txt += f" ({(vm.total - vm.available) / 2 ** 30:.1f}/{vm.total / 2 ** 30:.0f} GB)"
        return f"CPU {cpu}% | MEM {mem_txt} | GPU {gpu}%{traffic}"


def _job_memory_gb(pid):
    """Resident memory of a job and its children (GB), or None without psutil."""
    if psutil is None:
        return None
    try:
        p = psutil.Process(pid)
        return sum(q.memory_info().rss for q in [p] + p.children(recursive=True)) / 2 ** 30
    except Exception:
        return None


def _job_step(logpath):
    """The step a job is on: the last '[POST-STEP] <key>' line of its log."""
    try:
        with open(logpath, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 32768))
            tail = f.read().decode("utf-8", "replace")
        marks = [ln for ln in tail.splitlines() if ln.startswith("[POST-STEP] ")]
        return marks[-1].split()[1] if marks else "-"
    except OSError:
        return "?"


def _stop_job(proc):
    """Stop a job and the step script it is running."""
    try:
        if psutil is not None:
            for c in psutil.Process(proc.pid).children(recursive=True):
                c.kill()
        proc.kill()
    except Exception:
        pass


def run_post_jobs(targets, steps, config, root, job_cmd=None):
    """Tree mode: run the per-session steps for every session folder as parallel
    jobs — one child process per session (runner.py --post-job), each logging to its
    own file. A job starts only while fewer than POST_JOBS run, LAUNCH_GAP seconds
    after the previous one, and while CPU / GPU / MEM are under MAX_CPU / MAX_GPU /
    MAX_MEM; every MONITOR_SECONDS a status block shows the machine and each job.
    `job_cmd(op, steps, config)` builds a job's command (tests swap in a stand-in).
    Returns one result dict per session."""
    order = "".join(k for k in POST_JOB_STEPS if k in steps)
    n_jobs = post_jobs_limit(len(targets))
    gap = _int_env("LAUNCH_GAP", LAUNCH_GAP)                 # the config file's, if set
    every = _int_env("MONITOR_SECONDS", MONITOR_SECONDS)
    limits = tuple(_int_env(k, d) for k, d in (("MAX_CPU", MAX_CPU), ("MAX_GPU", MAX_GPU),
                                                ("MAX_MEM", MAX_MEM)))
    logdir = Path(_str_env("POST_LOG_DIR", default=str(Path(tempfile.gettempdir()) / "hm_post")))
    logdir = logdir / time.strftime("%Y%m%d_%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)
    queue, seen = [], {}
    for _ip, op in targets:
        label = session_label(op, root)
        seen[label] = seen.get(label, 0) + 1
        queue.append((label if seen[label] == 1 else f"{label}_{seen[label]}", op))

    print("\n" + "=" * 56)
    print(f"[TREE] {len(queue)} session(s) x steps [{order}], up to {n_jobs} at a time "
          f"(POST_JOBS), {gap} s apart, started while CPU<={limits[0]}% GPU<={limits[1]}% "
          f"MEM<={limits[2]}%")
    print(f"[TREE] One log per session in {logdir}")
    print("=" * 56)
    if psutil is None:
        print("[TREE] psutil is not installed: CPU/MEM are not monitored or gated "
              "(pip install psutil).")

    running, results = {}, []
    monitor = ResourceMonitor()
    last_launch = last_status = 0.0
    waiting = ""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        while queue or running:
            for label, job in list(running.items()):            # finished jobs
                rc = job["proc"].poll()
                if rc is None:
                    continue
                job["logf"].close()
                minutes = (time.time() - job["start"]) / 60
                results.append({"session": label, "folder": str(job["op"]), "status":
                                "ok" if rc == 0 else "failed", "minutes": round(minutes, 1),
                                "log": str(job["log"])})
                print(f"[TREE] {'done  ' if rc == 0 else 'FAILED'} {label} ({minutes:.1f} min)"
                      + ("" if rc == 0 else f" — see {job['log']}"))
                del running[label]

            now = time.time()
            if queue and len(running) < n_jobs and now - last_launch >= gap:
                ok, waiting = resources_free()
                if ok:
                    label, op = queue.pop(0)
                    logpath = logdir / f"{label}.log"
                    logf = open(logpath, "w", encoding="utf-8")
                    cmd = (job_cmd(op, order, config) if job_cmd else
                           [PYTHON, "-u", str(SCRIPT_DIR / "runner.py"), "--post-job",
                            str(op), order, str(config)])
                    proc = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR), stdout=logf,
                                            stderr=subprocess.STDOUT, env=env)
                    running[label] = {"proc": proc, "op": op, "start": now,
                                      "log": logpath, "logf": logf}
                    last_launch = now
                    print(f"[TREE] start  {label}  ({len(running)} running, {len(queue)} queued)")

            if now - last_status >= every:
                last_status = now
                n_failed = sum(r["status"] != "ok" for r in results)
                print(f"\n[MON {time.strftime('%H:%M:%S')}] {monitor.line()} | {len(running)} "
                      f"running, {len(queue)} queued, {len(results)} done ({n_failed} failed)")
                for label, job in running.items():
                    gb = _job_memory_gb(job["proc"].pid)
                    print(f"    {label:<28} step {_job_step(job['log']):<2} "
                          f"{(now - job['start']) / 60:6.1f} min"
                          + (f"  {gb:5.1f} GB" if gb is not None else ""))
                if queue and waiting and len(running) < n_jobs:
                    print(f"    next start waits: {waiting}")
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n[TREE] Interrupted — stopping the running jobs...")
        for label, job in running.items():
            _stop_job(job["proc"])
            job["logf"].close()
            results.append({"session": label, "folder": str(job["op"]), "status": "interrupted",
                            "minutes": round((time.time() - job["start"]) / 60, 1),
                            "log": str(job["log"])})
        results += [{"session": label, "folder": str(op), "status": "not started",
                     "minutes": 0.0, "log": ""} for label, op in queue]
        _post_summary(results, logdir)
        raise
    _post_summary(results, logdir)
    return results


def _post_summary(results, logdir):
    import csv
    with open(logdir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["session", "status", "minutes", "folder", "log"])
        w.writeheader()
        w.writerows(results)
    bad = [r for r in results if r["status"] != "ok"]
    print("\n" + "=" * 56)
    print(f"[TREE] {len(results) - len(bad)} of {len(results)} session(s) done"
          + (f"; {len(bad)} not:" if bad else "."))
    for r in bad:
        print(f"    {r['status']:<11} {r['session']}  {r['log']}")
    print(f"[TREE] Summary: {logdir / 'summary.csv'}")
    print("=" * 56)


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

    # Steps may come non-interactively via HM_STEPS (used by the genzeltracker GUI);
    # otherwise prompt for them interactively.
    selection = os.environ.get("HM_STEPS")
    if selection is None:
        print("Select steps to run (e.g., 123 for steps 1, 2, and 3):")
        for key, label in MENU:
            print(f"[{key}] {label}")
        print()
        selection = input("Enter steps: ")
    else:
        print(f"[STEPS] (from HM_STEPS) {selection}")

    # Step [e] (Trodes exportLFP + analog export) is retired: step 8 builds the
    # LFP for every channel straight from the .rec, and the analog (headstage
    # IMU) export that step 8's motion needs is part of step 1. So an "e" in an
    # old preset or HM_STEPS string (e.g. "e8") runs step 1 instead.
    if "e" in selection:
        print("[STEPS] Step [e] is retired: step 8 builds the LFP from the .rec "
              "and step 1 exports the analog (IMU) data. Running step 1 for 'e'.")
        selection = selection.replace("e", "" if "1" in selection else "1")

    # Split the selection into sequential master steps and parallel worker steps.
    # Step 6 (compression) is special: it always runs LAST at master level over all
    # op folders, so pull it out of the parallel worker steps here.
    has = {k: (k in selection) for k in SEQUENTIAL_STEPS}
    do_compress = "6" in selection
    parallel_steps = "".join(c for c in selection
                             if c not in SEQUENTIAL_STEPS and c != "6"
                             and c not in RETIRED_STEPS)
    parallel_trim = "".join(parallel_steps.split())

    # Step [s] (cross-session summary) moved to the HM_Rat_Analysis repo. Without
    # this notice a stale "s" would match no worker step and silently do nothing.
    if "s" in selection:
        print("[STEPS] Step [s] (session summary) now lives in the HM_Rat_Analysis "
              "repo: python session_summary.py --root <folder>. Ignoring 's'.")
        parallel_steps = parallel_steps.replace("s", "")
        parallel_trim = parallel_trim.replace("s", "")

    # Scan ip* folders. Each ipN -> opN.
    ip_dirs = sorted(d for d in root.glob("ip*") if d.is_dir())
    ops = []  # list of (ip_path, op_path) — for ip-dependent steps (workers/7/9)
    for ip_path in ip_dirs:
        num = ip_path.name[len("ip"):]
        ops.append((ip_path, root / f"op{num}"))

    # TREE MODE: no ip* folders — the root holds session folders that already went
    # through tracking/sorting, possibly renamed and grouped (the share's
    # task/<rat>/<YYYYMMDD>/). They are found below the root, and the per-session
    # steps (POST_JOB_STEPS) run as parallel jobs instead of one folder at a time.
    tree = not ip_dirs
    sessions_file = os.environ.get("HM_SESSIONS_FILE", "").strip()
    if tree:
        if sessions_file:
            # only the listed sessions (e.g. the failed ones of an earlier run);
            # no tree walk, which also saves the directory listings on the share
            print(f"[TREE] Only the session folders listed in {sessions_file}:")
            seq_ops = [(d, d) for d in read_session_list(sessions_file)]
        else:
            print(f"[TREE] No ip* folders under {root}: looking for session folders "
                  f"(up to {_int_env('TREE_DEPTH', TREE_DEPTH)} levels deep)...")
            seq_ops = [(d, d) for d in find_session_folders(root)]
        print(f"[TREE] {len(seq_ops)} session folder(s):")
        for _ip, op in seq_ops:
            try:
                shown = op.relative_to(root)
            except ValueError:                  # a listed folder outside the root
                shown = op
            print(f"    {shown}")
        need_ip = [k for k in "1234789dn" if k in selection]     # these read ip*/<raw>
        if need_ip:
            print(f"[TREE] Step(s) {' '.join(need_ip)} read the raw recordings (ip* folders) "
                  f"and are skipped here.")
    else:
        # Targets for the ip-INDEPENDENT steps (c/r/u/v): op* folders + session-date
        # folders whose name matches a file inside them. This lets those steps run on
        # session-named folders, not just op*.
        seq_ops = sequential_targets(root, ops)
        if sessions_file:
            print(f"[STEPS] HM_SESSIONS_FILE applies to a session tree only; {root} has "
                  f"ip* folders, so it is ignored.")
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
            name = f"{session_pfx(ip_path)}{ip_path.name}"   # rat_sessiondate_ log/marker
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
        clock_qc_after_sorting(ops)
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

    # Retired f / w: step 4 does both now. Without step 4 in the selection, run
    # them here over every op target so old presets (e.g. "rwu") still refresh.
    retired = [k for k in RETIRED_STEPS if k in selection]
    if retired:
        print(f"\n[STEPS] Step(s) {', '.join(retired)} are now part of step 4 "
              f"({'; '.join(RETIRED_STEPS[k] for k in retired)}).")
        if "4" not in selection:
            for i, (_ip, op) in enumerate(seq_ops, 1):
                print(f"\n[NWB {i}/{len(seq_ops)}] Refreshing: {op}")
                refresh_session_nwb(op)

    # Tree mode: the per-session steps run as parallel jobs, one per session.
    post = "".join(k for k in POST_JOB_STEPS if k in selection) if tree else ""
    post_results = []
    if post:
        if seq_ops:
            post_results = run_post_jobs(seq_ops, post, config, root)
        else:
            print(f"[TREE] No session folders found below {root} — steps {post} not run.")

    if has["l"] and not tree:
        _run_per_op("REWRITE-NWB-LFP (old LFP_Output export -> NWB)", "LFP",
                    "./src/nwb/rewrite_nwb_lfp.py", seq_ops, config)

    if has["u"] and not tree:
        _run_per_op("ADD-UNITS (curated Phy -> NWB)", "UNITS", "./src/nwb/add_units.py", seq_ops, config)

    if has["v"] and not tree:
        _run_per_op("VISUALIZE-NWB (summary + per-unit PDFs)", "VIZ", "./src/nwb/visualize_nwb.py", seq_ops, config)

    if has["b"] and not tree:
        dec = decode_settings()
        print("\n" + "=" * 56)
        print(f"[MASTER] Running POSITION-DECODER sequentially (good and good+mua, "
              f"viz folds: {dec['folds']}, CV folds: {dec['cv_folds']}, "
              f"leads: {' '.join(dec['leads']) or 'none'})...")
        print("=" * 56)
        total = len(seq_ops)
        for i, (_ip, op) in enumerate(seq_ops, 1):
            print(f"\n[DECODE {i}/{total}] Decoding: {op}")
            decode_session(op, config, f"[DECODE {i}/{total}]")
        print(f"\n[MASTER] Position-decoder complete for all {total} folder(s).")

    if has["m"] and not tree:
        # UMAP population embedding per session: good and good+mua, each ALL cell
        # types and a PYRAMIDAL-only version (4 embeddings/session).
        print("\n" + "=" * 56)
        print("[MASTER] Running NEURAL-UMAP sequentially (good & good+mua; all + pyramidal)...")
        print("=" * 56)
        total = len(seq_ops)
        for i, (_ip, op) in enumerate(seq_ops, 1):
            print(f"\n[UMAP {i}/{total}] Embedding: {op}")
            umap_session(op, config, f"[UMAP {i}/{total}]")
        print(f"\n[MASTER] Neural-UMAP complete for all {total} folder(s).")

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

    # --- STEP 6: compression, always LAST, over every op folder ---
    if do_compress:
        op_dirs = sorted(d for d in root.glob("op*") if d.is_dir())
        compress_all_op_videos(op_dirs)

    flags = " | ".join(f"{k}:{int(has[k])}" for k in SEQUENTIAL_STEPS)
    tree_txt = (f" | tree: {sum(r['status'] == 'ok' for r in post_results)}/{len(post_results)} "
                f"sessions [{post}]") if post else ""
    print("\n" + "=" * 56)
    print(f"[MASTER] Done. Parallel jobs: {count} | 6:{int(do_compress)} | {flags}{tree_txt}")
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


def _post_job_main(argv):
    """Entry point for one tree-mode job (see run_post_jobs):
    `runner.py --post-job <op> <steps> <config>`. Runs the per-session steps with
    output to stdout (the master points it at the session's log file). Exit code 1
    when any step failed."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # scripts use ./src/... paths
    op, steps, config = argv[0], argv[1], argv[2]
    os.environ["HM_CONFIG_FILE"] = config                  # the master's config file
    load_config()
    print(f"[POST] {op}: steps [{steps}]", flush=True)
    return 1 if run_post_job(Path(op), steps, config, sys.stdout) else 0


if __name__ == "__main__":
    for _s in (sys.stdout, sys.stderr):        # UTF-8 console so '->' etc. never crash
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker_main(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "--post-job":
        sys.exit(_post_job_main(sys.argv[2:]))
    else:
        main()
