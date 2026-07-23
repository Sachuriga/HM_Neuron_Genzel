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
    ("6", "Compression (always runs LAST, over all op folders)"),
    ("7", "Sorting"),
    ("c", "Continue After Sorting (metrics + BombCell + Phy, no re-sort)"),
    ("r", "Recompute Metrics (after manual Phy curation)"),
    ("8", "LFP + Motion (IMU Accel) + EMG-from-LFP"),
    ("d", "deeplabcut (extract eye frames + run DLC inference -> keypoints in CSV)"),
    ("9", "Cleaning"),
    ("n", "Node Analysis"),
    ("w", "nwblfp (NWB / LFP package)"),
    ("u", "Add curated Units (metrics + waveforms) to NWB (runs after w)"),
    ("v", "Visualize NWB units (summary + per-unit rate-map PDFs; runs after u)"),
    ("b", "Bayesian position decoder + spikes/decoded-on-video overlays per session (good and good+mua)"),
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
        print("See examples/hm_tracker_paths.example.txt in the repo for a template.")
        sys.exit(1)

    for raw in Path(cfg).read_text(errors="replace").splitlines():
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = "".join(key.split())
        if key:
            os.environ[key] = val

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


def pick_vcodec(size=(1920, 1080)):
    """(codec, args) for compression: GPU when any GPU path works, else CPU."""
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
                log(out, "    Exporting LFP (1500Hz, LP 700Hz)...")
                run([lfp, "-rec", f, "-outputrate", "1500", "-lfplowpass", "700"], out=out)
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
        if Path("./src/sorter/export_lfp.py").exists():
            log(out, "[STEP 8] Running LFP Extraction...")
            run([PYTHON, "-u", "./src/sorter/export_lfp.py",
                 "--input_folder", ip, "--output_folder", op,
                 "--output_rate", "1500"], out=out)
        # EMG-from-LFP needs the raw wideband (300-600 Hz); runs after the LFP
        # export so it can upsample onto lfp_timestamps.npy. Needs step 1 (-raw).
        if Path("./src/sorter/export_emg_from_lfp.py").exists():
            log(out, "[STEP 8] Running EMG-from-LFP (raw wideband, Buzsáki)...")
            run([PYTHON, "-u", "./src/sorter/export_emg_from_lfp.py",
                 "--input_folder", ip, "--output_folder", op], out=out)
        if Path("./src/sorter/export_motion.py").exists():
            log(out, "[STEP 8] Running Motion (IMU Accel) Extraction...")
            run([PYTHON, "-u", "./src/sorter/export_motion.py",
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


def _video_size(video):
    """(w, h) of `video` via ffprobe, or None if it cannot be read."""
    ffprobe = os.environ.get("FFPROBE_CMD", "ffprobe")
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
    vcodec, vargs = pick_vcodec(size=probe_size)

    print("\n" + "=" * 56)
    print(f"[MASTER] STEP 6 — Compression (codec: {vcodec} {' '.join(vargs)}, probed at "
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
                      "-stats", "-i", video, "-c:v", vcodec, *vargs,
                      "-c:a", "copy", temp_file])
            if rc == 0:
                shutil.move(str(temp_file), str(video))
                print(f"[SUCCESS] Compressed: {video}")
                n_ok += 1
            else:
                print(f"[ERROR] FFmpeg failed for {video} (codec '{vcodec}'). "
                      f"Set FFMPEG_VCODEC to force another encoder, e.g. libx264.")
                if temp_file.exists():
                    temp_file.unlink()
                n_fail += 1
    print(f"\n[MASTER] Compression done: {n_ok} ok, {n_fail} failed.")


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

    # Split the selection into sequential master steps and parallel worker steps.
    # Step 6 (compression) is special: it always runs LAST at master level over all
    # op folders, so pull it out of the parallel worker steps here.
    has = {k: (k in selection) for k in SEQUENTIAL_STEPS}
    do_compress = "6" in selection
    parallel_steps = "".join(c for c in selection
                             if c not in SEQUENTIAL_STEPS and c != "6")
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
        # after decoding, overlay analysis onto the real behaviour video via
        # make_videos.py (one mp4 per goal trial): the decoded position per quality
        # set, plus a spikes-on-path video (top-20 good pyramidal by spatial info,
        # once per op). DECODE_VIDEO=0 disables both; DECODE_VIDEO_LEADS picks the
        # decoded-overlay leads (default 0 1 2 3).
        make_vid = os.environ.get("DECODE_VIDEO", "1") != "0"
        vid_leads = os.environ.get("DECODE_VIDEO_LEADS", "0 1 2 3").split()
        # predictive-coding test: is the long-lead 'prediction' genuine, or behaviour/
        # occupancy? figures -> <op>/predictive_coding/. PREDICTIVE_CODING=0 disables.
        run_pc = os.environ.get("PREDICTIVE_CODING", "1") != "0"
        pc_cv = os.environ.get("PC_CV_FOLDS", "5")
        pc_shuf = os.environ.get("PC_SHUFFLE", "8")
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
                    # predictive-coding test (neural vs behaviour/kinematics/shuffle
                    # baselines + overshoot + decode-density figures) for this quality.
                    if run_pc and Path("./src/nwb/predictive_coding.py").exists():
                        prc = run([PYTHON, "-u", "./src/nwb/predictive_coding.py",
                                   "--output_folder", op, "--config", config,
                                   "--quality", *quals, "--cv_folds", pc_cv,
                                   "--n_shuffle", pc_shuf])
                        print(f"[DECODE {i}/{total}] units {'+'.join(quals)} predictive-coding: "
                              f"{'Done.' if prc == 0 else 'Python exited with error. Continuing...'}")
                # spikes-on-video (top-N good pyramidal cells by spatial info),
                # once per op — the goal-trial spike-path overlay on the real video.
                if make_vid and Path("./src/nwb/make_videos.py").exists():
                    src = run([PYTHON, "-u", "./src/nwb/make_videos.py", "--output_folder", op,
                               "--config", config, "--which", "spikes", "--quality", "good"])
                    print(f"[DECODE {i}/{total}] spikes-on-video: "
                          f"{'Done.' if src == 0 else 'Python exited with error. Continuing...'}")
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

    # --- STEP 6: compression, always LAST, over every op folder ---
    if do_compress:
        op_dirs = sorted(d for d in root.glob("op*") if d.is_dir())
        compress_all_op_videos(op_dirs)

    flags = " | ".join(f"{k}:{int(has[k])}" for k in SEQUENTIAL_STEPS)
    print("\n" + "=" * 56)
    print(f"[MASTER] Done. Parallel jobs: {count} | 6:{int(do_compress)} | {flags}")
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
