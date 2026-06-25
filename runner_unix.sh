#!/usr/bin/env bash
# ============================================================
#   HM Tracker — macOS / Linux runner (port of runner_windows.bat)
#
#   Usage:  ./runner_unix.sh /path/to/data_root
#
#   Mirrors the Windows orchestrator:
#     - parallel steps run one background worker per ip*/op* pair
#     - steps 7 (sort), 9 (clean) and w (nwb) run sequentially afterwards
#     - a CPU/GPU/MEM resource gate throttles new worker launches
#
#   Differences from the .bat (platform adaptations):
#     - Workers run as background jobs (no new terminal window). Each
#       worker's output is written to /tmp/hm_worker_<name>.log AND streamed.
#     - GPU check is skipped automatically when nvidia-smi is absent (Macs).
#     - Video compression codec is configurable via FFMPEG_VCODEC
#       (default h264_nvenc to match Windows; set h264_videotoolbox on Mac).
#     - Set DISABLE_RESOURCE_CHECK=1 to skip the CPU/GPU/MEM gate.
# ============================================================

# Do NOT use `set -e`: like the batch script we want to continue past
# individual step failures.
set -uo pipefail

# Always operate from the repo root (the directory of this script).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OS="$(uname -s)"

# ------------------------------------------------------------
#                 CONFIGURATION (SHARED)
# ------------------------------------------------------------
MAX_CPU="${MAX_CPU:-90}"
MAX_GPU="${MAX_GPU:-90}"
MAX_MEM="${MAX_MEM:-65}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
LAUNCH_GAP="${LAUNCH_GAP:-20}"   # seconds to wait between worker launches
FREQ=30000

# Pick a python interpreter.
if command -v python >/dev/null 2>&1; then
    PYTHON="${PYTHON:-python}"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="${PYTHON:-python3}"
else
    echo "[ERROR] No python interpreter found on PATH."
    exit 1
fi

# Config file: ~/Desktop/hm_tracker_paths.txt (override with HM_CONFIG_FILE).
CONFIG_FILE="${HM_CONFIG_FILE:-$HOME/Desktop/hm_tracker_paths.txt}"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    echo
    echo "Please create hm_tracker_paths.txt on your Desktop with lines like:"
    echo "  FFMPEG_CMD=/usr/local/bin/ffmpeg"
    echo "  ONNX_WEIGHTS_PATH=/path/to/weights.pt"
    echo "  TRODES_EXPORT_CMD=/path/to/trodesexport"
    echo "  TRODES_EXPORT_LFP=/path/to/exportLFP"
    echo "  LFP_CHANNELS=1 2 3 4 5 6 7 8"
    echo
    echo "See hm_tracker_paths.example.txt in the repo for a template."
    exit 1
fi

# Parse KEY=VALUE lines (skip blanks and # comments) into exported vars.
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"                       # strip CR from CRLF files
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "${line#"${line%%[![:space:]]*}"}" == \#* ]] && continue
    [[ "$line" != *"="* ]] && continue
    key="${line%%=*}"
    val="${line#*=}"
    key="${key//[[:space:]]/}"
    [[ -z "$key" ]] && continue
    export "$key=$val"
done < "$CONFIG_FILE"

# Video codec for step 6 (Windows uses NVENC; Mac users: h264_videotoolbox).
FFMPEG_VCODEC="${FFMPEG_VCODEC:-h264_nvenc}"

# Provide harmless defaults for vars the steps reference, so `set -u`
# doesn't abort when the config omits an optional tool path.
: "${FFMPEG_CMD:=ffmpeg}"
: "${ONNX_WEIGHTS_PATH:=}"
: "${TRODES_EXPORT_CMD:=}"
: "${TRODES_EXPORT_LFP:=}"
: "${NWB_RAT_NR:=1}"

# ------------------------------------------------------------
#                 RESOURCE MONITOR HELPERS
# ------------------------------------------------------------
# Each returns an integer percent (0..100); 0 on any failure so a
# missing tool never blocks the pipeline.

get_cpu_load() {
    local idle used
    if [[ "$OS" == "Darwin" ]]; then
        idle="$(top -l 2 -n 0 2>/dev/null | grep -E '^CPU usage' | tail -1 \
                | sed -E 's/.* ([0-9.]+)% idle.*/\1/')"
        [[ -z "$idle" ]] && { echo 0; return; }
        used="$(awk -v i="$idle" 'BEGIN{printf "%d", 100 - i}')"
        echo "${used:-0}"
    elif [[ -r /proc/stat ]]; then
        local a1 b1 c1 d1 a2 b2 c2 d2 rest
        read -r _ a1 b1 c1 d1 rest < /proc/stat
        sleep 0.3
        read -r _ a2 b2 c2 d2 rest < /proc/stat
        local idle_d=$(( d2 - d1 ))
        local total_d=$(( (a2+b2+c2+d2) - (a1+b1+c1+d1) ))
        (( total_d <= 0 )) && { echo 0; return; }
        echo $(( 100 - (100 * idle_d / total_d) ))
    else
        echo 0
    fi
}

get_gpu_load() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        local g
        g="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)"
        g="${g//[[:space:]]/}"
        echo "${g:-0}"
    else
        echo 0
    fi
}

get_mem_usage() {
    if [[ "$OS" == "Darwin" ]]; then
        local free
        free="$(memory_pressure 2>/dev/null | grep -Ei 'free percentage' \
                | sed -E 's/.*: *([0-9]+)%.*/\1/')"
        [[ -z "$free" ]] && { echo 0; return; }
        echo $(( 100 - free ))
    elif command -v free >/dev/null 2>&1; then
        free | awk '/^Mem:/ {printf "%d", ($3/$2)*100}'
    else
        echo 0
    fi
}

wait_for_resources() {
    [[ "${DISABLE_RESOURCE_CHECK:-0}" == "1" ]] && return 0
    while true; do
        local cpu gpu mem
        cpu="$(get_cpu_load)"; gpu="$(get_gpu_load)"; mem="$(get_mem_usage)"
        cpu="${cpu%%.*}"; gpu="${gpu%%.*}"; mem="${mem%%.*}"
        cpu="${cpu:-0}"; gpu="${gpu:-0}"; mem="${mem:-0}"
        if (( cpu > MAX_CPU )); then
            echo "    [WAIT] High CPU: ${cpu}%. Pausing ${WAIT_SECONDS}s..."
            sleep "$WAIT_SECONDS"; continue
        fi
        if (( gpu > MAX_GPU )); then
            echo "    [WAIT] High GPU: ${gpu}%. Pausing ${WAIT_SECONDS}s..."
            sleep "$WAIT_SECONDS"; continue
        fi
        if (( mem > MAX_MEM )); then
            echo "    [WAIT] High MEM: ${mem}%. Pausing ${WAIT_SECONDS}s..."
            sleep "$WAIT_SECONDS"; continue
        fi
        echo "    [CHECK] CPU: ${cpu}% | GPU: ${gpu}% | MEM: ${mem}% - OK."
        return 0
    done
}

# ------------------------------------------------------------
#                      THE WORKER
# ------------------------------------------------------------
# Runs the per-folder parallel steps. Sorting(7)/clean(9)/nwb(w) are
# handled at master level, not here.
run_worker() {
    local IP="$1" OP="$2" STEPS="$3"
    mkdir -p "$OP"

    echo "[INFO] Running steps [$STEPS] for $IP"
    [[ -z "$STEPS" ]] && { echo "[WORKER] No steps to run, exiting."; return; }

    # --- STEP 1 (DIO/Raw export) ---
    if [[ "$STEPS" == *"1"* ]]; then
        echo "[STEP 1] Running Trodes DIO/Raw Export..."
        if [[ -n "$TRODES_EXPORT_CMD" && -x "$TRODES_EXPORT_CMD" ]]; then
            for f in "$IP"/*.rec; do
                [[ -e "$f" ]] || continue
                "$TRODES_EXPORT_CMD" -dio -raw -rec "$f"
            done
        else
            echo "[WARNING] trodesexport not found at: $TRODES_EXPORT_CMD"
        fi
    fi

    # --- STEP e (LFP export) ---
    if [[ "$STEPS" == *"e"* ]]; then
        echo "[STEP e] Running Trodes LFP Export (1000Hz, LP 500Hz)..."
        if [[ -n "$TRODES_EXPORT_LFP" && -x "$TRODES_EXPORT_LFP" ]]; then
            for f in "$IP"/*.rec; do
                [[ -e "$f" ]] || continue
                echo "    Exporting LFP from $(basename "$f")"
                "$TRODES_EXPORT_LFP" -rec "$f" -outputrate 1000 -lfplowpass 500
            done
        else
            echo "[WARNING] exportLFP not found at: $TRODES_EXPORT_LFP"
        fi
    fi

    # --- STEP 2 (LED sync) ---
    if [[ "$STEPS" == *"2"* ]]; then
        echo "[STEP 2] Running Sync Script..."
        [[ -f ./src/Video_LED_Sync_using_ICA.py ]] && \
            "$PYTHON" -u ./src/Video_LED_Sync_using_ICA.py -i "$IP" -o "$OP" -f "$FREQ"
    fi

    # --- STEP 3 (stitching) ---
    if [[ "$STEPS" == *"3"* ]]; then
        echo "[STEP 3] Running Stitching..."
        [[ -f ./src/join_views.py ]] && "$PYTHON" -u ./src/join_views.py "$IP"
    fi

    # --- STEP 4 (tracker) ---
    if [[ "$STEPS" == *"4"* ]]; then
        echo "[STEP 4] Running Tracker..."
        if [[ -f "$IP/stitched.mp4" ]]; then
            "$PYTHON" -u ./src/TrackerYolov11.py --input_folder "$IP" \
                --output_folder "$OP" --onnx_weight "$ONNX_WEIGHTS_PATH"
        fi
    fi

    # --- STEP 5 (plotting) ---
    if [[ "$STEPS" == *"5"* ]]; then
        echo "[STEP 5] Running Plotting..."
        [[ -f ./src/plot_trials.py ]] && \
            "$PYTHON" -u ./src/plot_trials.py --input_folder "$IP" --output_folder "$OP"
    fi

    # --- STEP 6 (GPU compression) ---
    if [[ "$STEPS" == *"6"* ]]; then
        echo "[STEP 6] Running Compression (codec: $FFMPEG_VCODEC)..."
        local temp_file="$OP/__temp_compressed.mp4"
        [[ -f "$temp_file" ]] && rm -f "$temp_file"
        local video_file=""
        for f in "$OP"/*.mp4; do
            [[ -e "$f" ]] || continue
            [[ "$(basename "$f")" == "__temp_compressed.mp4" ]] && continue
            video_file="$f"; break
        done
        if [[ -n "$video_file" ]]; then
            if "$FFMPEG_CMD" -nostdin -y -hide_banner -loglevel warning -stats \
                    -i "$video_file" -c:v "$FFMPEG_VCODEC" -preset p6 -cq 28 \
                    -c:a copy "$temp_file"; then
                mv -f "$temp_file" "$video_file"
                echo "[SUCCESS] Video compressed: $video_file"
            else
                echo "[ERROR] FFmpeg compression failed for $video_file"
                echo "        Check codec '$FFMPEG_VCODEC' (Mac: try h264_videotoolbox)."
                [[ -f "$temp_file" ]] && rm -f "$temp_file"
            fi
        else
            echo "[WARNING] No valid .mp4 file found in '$OP' to compress."
        fi
    fi

    # --- STEP 8 (LFP extraction) ---
    if [[ "$STEPS" == *"8"* ]]; then
        echo "[STEP 8] Running LFP Extraction..."
        [[ -f ./src/sorter/export_lfp.py ]] && \
            "$PYTHON" -u ./src/sorter/export_lfp.py --input_folder "$IP" --output_folder "$OP"
    fi

    # --- STEP d (DeepLabCut export) ---
    if [[ "$STEPS" == *"d"* ]]; then
        echo "[STEP d] Exporting video for DeepLabCut..."
        [[ -f ./src/dlc/tracking_eyes.py ]] && \
            "$PYTHON" -u ./src/dlc/tracking_eyes.py --input_folder "$IP" --output_folder "$OP"
    fi

    # --- STEP n (node analysis) ---
    if [[ "$STEPS" == *"n"* ]]; then
        echo "[STEP n] Running Node Analysis..."
        [[ -f ./src/node_analysis/hex_maze_analysis.py ]] && \
            "$PYTHON" -u ./src/node_analysis/hex_maze_analysis.py \
                --input_folder "$IP" --output_folder "$OP"
    fi

    echo "[COMPLETE] Worker finished for $IP"
}

clean_folder() {
    local target="$1"
    shopt -s nullglob
    local d
    for d in "$target"/*.DIO "$target"/*.raw "$target"/*timestampoffset*; do
        if [[ -d "$d" ]]; then
            echo "    Deleting: $(basename "$d")"
            rm -rf "$d"
        fi
    done
    shopt -u nullglob
}

# ------------------------------------------------------------
#                         MASTER
# ------------------------------------------------------------
echo "========================================================"
echo "          SMART PARALLEL MODE (Multi-Step)"
echo "========================================================"
echo "[CONFIG] Max CPU: ${MAX_CPU}% | Max GPU: ${MAX_GPU}% | Max MEM: ${MAX_MEM}%"
echo

if [[ $# -lt 1 || -z "${1:-}" ]]; then
    echo "Usage: $0 /path/to/data_root"
    exit 1
fi

if [[ ! -d "$1" ]]; then
    echo "[ERROR] Data folder not found: $1"
    exit 1
fi
ROOT_DIR="$(cd "$1" && pwd)"
echo "[DEBUG] Target Root Directory: [$ROOT_DIR]"
echo

echo "Select steps to run (e.g., 123 for steps 1, 2, and 3):"
echo "[1] Trodes Export (DIO/Raw)"
echo "[e] Trodes Export LFP (per channel)"
echo "[2] Sync Script"
echo "[3] Stitching"
echo "[4] Tracker"
echo "[5] Plotting"
echo "[6] Compression"
echo "[7] Sorting"
echo "[8] LFP"
echo "[d] deeplabcut"
echo "[9] Cleaning"
echo "[n] Node Analysis"
echo "[w] nwblfp (NWB / LFP package)"
echo
read -r -p "Enter steps: " MY_SELECTION

# Steps 7, 9 and w run sequentially after the parallel workers.
HAS_SORT=0; HAS_CLEAN=0; HAS_NWB=0
PARALLEL_STEPS="$MY_SELECTION"
if [[ "$MY_SELECTION" == *"7"* ]]; then HAS_SORT=1;  PARALLEL_STEPS="${PARALLEL_STEPS//7/}"; fi
if [[ "$MY_SELECTION" == *"9"* ]]; then HAS_CLEAN=1; PARALLEL_STEPS="${PARALLEL_STEPS//9/}"; fi
if [[ "$MY_SELECTION" == *"w"* ]]; then HAS_NWB=1;   PARALLEL_STEPS="${PARALLEL_STEPS//w/}"; fi
PARALLEL_STEPS_TRIM="${PARALLEL_STEPS//[[:space:]]/}"

# Scan ip* folders.
shopt -s nullglob
ip_dirs=("$ROOT_DIR"/ip*)
shopt -u nullglob
if [[ ${#ip_dirs[@]} -eq 0 ]]; then
    echo "[WARNING] No ip* folders found under $ROOT_DIR"
fi

sort_ips=(); sort_ops=(); sort_names=()
worker_pids=(); worker_logs=()
count=0

for ip_path in "${ip_dirs[@]}"; do
    [[ -d "$ip_path" ]] || continue
    dir_name="$(basename "$ip_path")"
    num="${dir_name#ip}"
    op_path="$ROOT_DIR/op${num}"

    # Collect for sequential sorting (sorting.py creates op* if missing).
    sort_ips+=("$ip_path"); sort_ops+=("$op_path"); sort_names+=("$dir_name")

    # Launch a parallel worker only if there are non-sequential steps.
    if [[ -n "$PARALLEL_STEPS_TRIM" ]]; then
        echo
        echo "[QUEUE] Preparing: $dir_name"
        wait_for_resources

        count=$((count + 1))
        log="/tmp/hm_worker_${dir_name}.log"
        echo "[MASTER] Launching worker for $dir_name (log: $log)"
        # Run worker in background; stream output to log and console.
        run_worker "$ip_path" "$op_path" "$PARALLEL_STEPS" > "$log" 2>&1 &
        worker_pids+=("$!")
        worker_logs+=("$log")

        echo "[MASTER] Job launched. Waiting ${LAUNCH_GAP}s for stability..."
        sleep "$LAUNCH_GAP"
    fi
done

# Wait for all parallel workers to finish.
if (( count > 0 )); then
    echo
    echo "========================================================"
    echo "[MASTER] Launched $count parallel job(s). Waiting for all to finish..."
    echo "========================================================"
    for i in "${!worker_pids[@]}"; do
        wait "${worker_pids[$i]}"
        echo "[MASTER] Worker finished (log: ${worker_logs[$i]})"
    done
    echo "[MASTER] All parallel workers have completed."
fi

# --- Sorting (sequential, one folder at a time) ---
if (( HAS_SORT == 1 )); then
    echo
    echo "========================================================"
    echo "[MASTER] Running SORTING sequentially (1 folder at a time)..."
    echo "========================================================"
    total=${#sort_ips[@]}
    for i in "${!sort_ips[@]}"; do
        idx=$((i + 1))
        cur_ip="${sort_ips[$i]}"; cur_op="${sort_ops[$i]}"
        echo
        echo "[SORT $idx/$total] Processing: $cur_ip"
        if [[ -f ./src/sorter/sorting.py ]]; then
            if "$PYTHON" -u ./src/sorter/sorting.py --input_folder "$cur_ip" \
                    --output_folder "$cur_op" --config "$CONFIG_FILE"; then
                echo "[SORT $idx/$total] Done."
            else
                echo "[SORT $idx/$total] Python exited with error. Continuing..."
            fi
        fi
    done
    echo
    echo "[MASTER] Sorting complete for all $total folder(s)."
fi

# --- Cleaning (sequential, after sorting) ---
if (( HAS_CLEAN == 1 )); then
    echo
    echo "========================================================"
    echo "[MASTER] Running CLEANING sequentially (after sorting)..."
    echo "========================================================"
    total=${#sort_ips[@]}
    for i in "${!sort_ips[@]}"; do
        idx=$((i + 1))
        cur_ip="${sort_ips[$i]}"
        echo
        echo "[CLEAN $idx/$total] Cleaning: $cur_ip"
        clean_folder "$cur_ip"
    done
    echo
    echo "[MASTER] Cleaning complete for all $total folder(s)."
fi

# --- NWB / LFP packaging (master level) ---
if (( HAS_NWB == 1 )); then
    echo
    echo "========================================================"
    echo "[MASTER] Running NWB / LFP packaging (rat_nr=$NWB_RAT_NR)..."
    echo "========================================================"
    if [[ -f ./src/nwb/create_nwb.py ]]; then
        ( cd ./src/nwb && "$PYTHON" -u create_nwb.py --rat_nr "$NWB_RAT_NR" \
            --noroot --ip "$ROOT_DIR" --op "$ROOT_DIR" ) \
            && echo "[NWB] Done." || echo "[NWB] Python exited with error."
    else
        echo "[NWB] create_nwb.py NOT found at: $SCRIPT_DIR/src/nwb/create_nwb.py"
    fi
fi

echo
echo "========================================================"
echo "[MASTER] Done. Parallel jobs: $count | Sorting: $HAS_SORT | NWB: $HAS_NWB"
echo "========================================================"
