# -*- coding: utf-8 -*-
"""
Created on Sun November 30 14:08:05 2025

@author: Sachuriga
Based on / modified from: https://github.com/genzellab/HM_RAT
"""

import os
import cv2
import matplotlib
# Force matplotlib to not use any Xwindow backend.
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
import pandas as pd
from datetime import datetime , time , timedelta
import re
import functools as ft
from sklearn.linear_model import LinearRegression
import argparse
import sys
import shutil


# ──────────────────────────────────────────────────────────────────────────────
# LED-DETECTION DEBUG DUMP
# ──────────────────────────────────────────────────────────────────────────────
# When enabled (via --debug, or SYNC_DEBUG=1), the pipeline writes per-video
# diagnostics to <output>/sync_debug/ so a "clearly blinking but not detected"
# LED can be traced to the failing stage:
#   • <stem>_localization.png   reference frame + detected crop box + blink map
#   • <stem>_crop_trace.png     the 16x16 crop patch and its intensity over time
#   • <stem>_ica.png            the ICA component traces with ON/OFF states
#   • sync_debug_videos.csv     per-video fps / frame-count / crop location
#   • sync_debug_components.csv per-component duty cycle, frequency, accept flags
class _SyncDebug:
    def __init__(self):
        self.enabled = False
        self.dir = None
        self.videos = []
        self.components = []

    def setup(self, output_path):
        base = Path(output_path) if output_path else Path.cwd()
        self.dir = base / "sync_debug"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.enabled = True
        print(f"[DEBUG] LED-detection debug dump → {self.dir}")

    def path(self, name):
        return str(self.dir / name)

    def add_video(self, **kw):
        self.videos.append(kw)

    def add_component(self, **kw):
        self.components.append(kw)

    def write_summary(self):
        if not self.enabled:
            return
        if self.videos:
            pd.DataFrame(self.videos).to_csv(
                self.dir / "sync_debug_videos.csv", index=False)
        if self.components:
            pd.DataFrame(self.components).to_csv(
                self.dir / "sync_debug_components.csv", index=False)
        print(f"[DEBUG] Wrote sync debug summary CSVs → {self.dir}")


DEBUG = _SyncDebug()


# Skip the first SYNC_START_SEC seconds of every eye video when LOCATING the LED
# and when EXTRACTING its blink for the sync regression. Use this when the LED is
# repositioned early in a session (e.g. moved at ~33s): everything before the
# cutoff has the LED in the wrong place and would corrupt the crop and the
# measured blink frequency. The per-frame timestamp OUTPUT still covers every
# frame — only detection/fitting is windowed. Configurable via --start-sec.
SYNC_START_SEC = 45.0


def _start_frame_for(fps):
    """Frame index corresponding to SYNC_START_SEC, using fps (fallback 30)."""
    if not fps or fps <= 0 or fps != fps:  # 0 / None / NaN
        fps = 30.0
    return int(round(SYNC_START_SEC * fps))


def get_led_coords_from_videoframes(file_path, process_frame_count):
    cap = cv2.VideoCapture(str(file_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_process = process_frame_count if process_frame_count is not None else frame_count

    # Start localization after SYNC_START_SEC so the LED is scanned in its final
    # (post-move) position. Seeking is fine here — localization only needs a
    # position, not frame-accurate alignment with the metadata.
    start_frame = _start_frame_for(cap.get(cv2.CAP_PROP_FPS))
    if start_frame > 0 and start_frame < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, ref_frame = cap.read()
    # Accumulate the per-pixel MAX frame-difference incrementally instead of
    # storing every thresholded frame — this keeps memory flat when scanning
    # many frames (e.g. 1000) for the blink.
    max_diff = None
    frames_seen = 0

    for i in range(frames_to_process):
        ret, frame = cap.read()
        if frame is None:
            break

        # Negative values are ignored on image subtraction, so using unsigned subtraction.
        subtracted = cv2.subtract(frame, ref_frame)
        subtracted += cv2.subtract(ref_frame, frame)

        # Ignoring the top 100 pixels in image due to noise from the timestamp prints
        th = cv2.threshold(subtracted[100:, :], 50, 255, cv2.THRESH_BINARY)[1]
        if max_diff is None:
            max_diff = th
        else:
            np.maximum(max_diff, th, out=max_diff)
        frames_seen += 1

    cap.release()

    if frames_seen == 0:
        raise RuntimeError(f"No frames processed for {file_path}; cannot detect LED.")

    avg_frame = max_diff.astype("uint8")
    avg_frame = cv2.cvtColor(avg_frame, cv2.COLOR_RGB2GRAY)
    avg_th = cv2.threshold(avg_frame, 0, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(avg_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if not contours:
        raise RuntimeError(f"No contours found in {file_path}; check thresholding or LED visibility.")

    # Compute areas for all contours
    areas = [cv2.contourArea(c) for c in contours]

    # Select only “small” contours (likely LEDs)
    small_contours = [(i, a) for i, a in enumerate(areas) if a < 100]

    if small_contours:
        # Pick the small contour with the largest area
        largest_contour_index = max(small_contours, key=lambda t: t[1])[0]
    else:
        # Fallback: use the overall largest contour, or raise if you prefer
        print(
            f"Warning: no contour with area < 100 found in {file_path}. "
            f"Using largest contour (area={max(areas):.1f}) as fallback."
        )
        largest_contour_index = int(np.argmax(areas))

    x, y, w, h = cv2.boundingRect(contours[largest_contour_index])
    cv2.rectangle(ref_frame, (x, y + 100), (x + w, y + 100 + h), (0, 255, 0), 2)

    cx, cy = int(x + (w / 2)), int(y + 100 + (h / 2))

    if DEBUG.enabled:
        stem = Path(file_path).stem
        n_small = len(small_contours)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB))
        ax[0].set_title(f"Detected LED crop @ ({cx}, {cy})\n"
                        f"{n_small} small contour(s) <100px; "
                        f"picked area={areas[largest_contour_index]:.1f}")
        ax[0].plot(cx, cy, 'r+', markersize=14)
        ax[1].imshow(avg_th, cmap='gray')
        ax[1].set_title("Blink map (max frame-diff, top 100px excluded)")
        fig.tight_layout()
        fig.savefig(DEBUG.path(f"{stem}_localization.png"), dpi=110)
        plt.close(fig)

    return cx, cy


def get_dio_files(path: Path):
    dio = {}

    # === INIT: 3-tier fallback ===
    init_patterns = [
        '*[0-9].dio_Controller_Din1.dat',    # Plan A1 (most common old style)
        '*maze.dio_Controller_Din1.dat',      # Plan A2 (your new request)
        '*hab.dio_Controller_Din1.dat',      # Plan A2 (your new request)
        '*aze.dio_MCU_Din1.dat',                # Plan B (new MCU naming)
        '*[0-9].dio_MCU_Din1.dat',
        '*[0-9]_hab.dio_Controller_Din1.dat'
    ]

    init_files = []
    for pattern in init_patterns:
        candidates = sorted(path.rglob(pattern))
        if candidates:
            init_files = candidates
            print(f"Found 'init' files with pattern: {pattern} → {len(candidates)} file(s)")
            break
    else:
        print("Warning: No 'init' files found with any known pattern!")

    dio['init'] = init_files

    # === BLUE & RED: 2-tier fallback (Controller → MCU) ===
    blue_red_patterns = {
        'blue': [
            '*merged.dio_Controller_Din1.dat',   # Plan A
            '*merged.dio_MCU_Din1.dat'           # Plan B
        ],
        'red': [
            '*merged.dio_Controller_Din2.dat',   # Plan A
            '*merged.dio_MCU_Din2.dat'           # Plan B
        ]
    }

    for key, patterns in blue_red_patterns.items():
        files = []
        for pattern in patterns:
            candidates = sorted(path.rglob(pattern))
            if candidates:
                files = candidates
                if pattern.endswith('_MCU_'):
                    print(f"{key.capitalize()} → using MCU naming (Plan B)")
                break
        dio[key] = files
        if not files:
            print(f"Warning: No '{key}' files found!")

    return dio


def get_video_files_with_metadata(basepath, led_xy_manual=True, time_stamp=True, info=True):
    path = Path(basepath).resolve()
    all_videos = list(sorted(path.glob('*eye*.mp4')))

    temp_dir = path / "temp_no_led"
    temp_dir.mkdir(exist_ok=True)

    videos_filepath_list = []
    crop_xy_dict = {}

    if led_xy_manual:
        videos_filepath_list = all_videos
        crop_file_list = list(sorted(path.glob('*.led_crop')))
        if crop_file_list:
            with open(crop_file_list[0]) as f:
                crop_txt = f.readlines()
            for line in crop_txt:
                try:
                    vid_path, x, y = line.split(',')
                    crop_xy_dict[vid_path] = (int(x), int(y))
                except ValueError:
                    print("Faulty line:", line, 'Maybe led coordinates are missing?')
                    break
        else:
            raise Exception("File containing led crop coordinates not found.")
    else:
        n_frame = 1000
        for video_file_path in all_videos:
            try:
                xy = get_led_coords_from_videoframes(video_file_path, n_frame)
            except RuntimeError as e:
                print(e)
                print(f"Moving {video_file_path.name} to {temp_dir} and skipping it.")
                shutil.move(str(video_file_path), str(temp_dir / video_file_path.name))

                meta_candidate = video_file_path.with_suffix('.meta')
                if meta_candidate.exists():
                    print(f"Moving associated meta file {meta_candidate.name} to {temp_dir}.")
                    shutil.move(str(meta_candidate), str(temp_dir / meta_candidate.name))
                else:
                    print(f"Warning: no .meta file found for {video_file_path.name}")
                continue

            crop_xy_dict[str(video_file_path)] = xy
            videos_filepath_list.append(video_file_path)

    if time_stamp:
        meta_filepath_list = []
        for v in videos_filepath_list:
            meta_candidate = v.with_suffix('.meta')
            if meta_candidate.exists():
                meta_filepath_list.append(meta_candidate)
            else:
                print(f"Warning: no .meta file for video {v}")

    dio_file_path_dict = get_dio_files(path)

    if info:
        print(f"Following {len(videos_filepath_list)} videos will be processed:")
        for file in videos_filepath_list:
            print(str(file))
        print(f"Following {len(meta_filepath_list)} meta files will be processed:")
        for file in meta_filepath_list:
            print(str(file))
        print(f"Following {len(crop_xy_dict)} crop co-ordinates will be processed:")
        for file in crop_xy_dict:
            print(str(file), crop_xy_dict[file])
        print(f"Following {len(dio_file_path_dict)} dio files will be processed:")
        for file in dio_file_path_dict:
            try:
                print(str(file), str(dio_file_path_dict[file][0]))
            except Exception:
                print(str(file), str(dio_file_path_dict[file]))

    return videos_filepath_list, crop_xy_dict, meta_filepath_list, dio_file_path_dict


def process_ica_signals(demixed, mix_weights, time_meta, debug_stem=None):
    fps = 30.0
    eD = 0.5       # expected Duty cycle of 0.5
    ef_red = 0.5   # expected frequency of 0.5 Hz
    ef_blue = 2.5  # expected frequency of 2.5 Hz

    dD = np.zeros(demixed.shape[1])
    df_red = np.zeros(demixed.shape[1])
    df_blue = np.zeros(demixed.shape[1])

    colors = {0: 'red', 1: 'blue', None: 'gray'}
    N = -1
    N_ICA = -1  # numbers of samples to use for ICA, -1 for all

    df_red_out = None
    df_blue_out = None
    debug_states = []  # (component, y_km) for the trace plot

    for n in range(demixed.shape[1]):
        flip_ica = mix_weights[n] < 0
        if flip_ica:
            demixed[:, n] = -demixed[:, n]

        km = KMeans(n_clusters=2, random_state=0, n_init=10).fit(demixed[:, n].reshape(-1, 1))
        y_km = km.predict(demixed[:, n].reshape(-1, 1))
        centers = km.cluster_centers_.ravel()

        flip_kmeans = centers[0] > centers[1]
        flip = flip_ica ^ flip_kmeans
        if flip_kmeans:
            y_km = np.abs(y_km-1)

        duty_cycle = y_km.sum()/len(y_km)
        freq = (np.diff(y_km)>0).sum()/len(y_km) * fps
        dD[n] = abs(eD-duty_cycle)
        df_red[n] = abs(ef_red - freq)
        df_blue[n] = abs(ef_blue - freq)

        good_DC = dD[n] < 0.2 * eD
        good_freq = np.array([df_red[n] < ef_red * 0.1, df_blue[n] < ef_blue * 0.1])
        is_signal = good_DC and good_freq.sum()
        signal_color = good_freq.argmax() if is_signal else None
        print(f"ICA signal number: {n}, DutyCycle:{duty_cycle}, Freq:{freq}")
        sig_col = colors[signal_color]

        if DEBUG.enabled:
            debug_states.append((demixed[:, n].copy(), y_km.copy()))
            DEBUG.add_component(
                video=debug_stem, component=n,
                duty_cycle=round(float(duty_cycle), 4),
                freq_hz=round(float(freq), 4),
                good_duty=bool(good_DC),
                near_red_0p5hz=bool(good_freq[0]),
                near_blue_2p5hz=bool(good_freq[1]),
                accepted_as=sig_col if sig_col != 'gray' else 'none',
            )

        if sig_col == 'red':
            df_red_out = pd.DataFrame({'key': [], "LED_Intensity": []})
            df_red_out['key'] = time_meta[0:(len(demixed[:N, n]-1))]
            df_red_out["LED_Intensity"] = demixed[:N, n]
        elif sig_col == 'blue':
            df_blue_out = pd.DataFrame({'key': [], "LED_Intensity": []})
            df_blue_out['key'] = time_meta[0:(len(demixed[:N, n]-1))]
            df_blue_out["LED_Intensity"] = demixed[:N, n]

    if DEBUG.enabled and debug_states:
        nrows = len(debug_states)
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 2.2 * nrows), squeeze=False)
        for n, (sig, y_km) in enumerate(debug_states):
            comp = DEBUG.components[-nrows + n]
            ax = axes[n][0]
            # Normalise the source for overlay against the 0/1 ON-OFF state.
            rng = np.ptp(sig) or 1.0
            ax.plot((sig - sig.min()) / rng, lw=0.5, label='ICA source')
            ax.plot(y_km, lw=0.8, alpha=0.7, label='ON/OFF (KMeans)')
            ax.set_title(f"comp {n}: freq={comp['freq_hz']} Hz, "
                         f"duty={comp['duty_cycle']}, accepted={comp['accepted_as']}")
            ax.set_ylim(-0.1, 1.2)
            if n == 0:
                ax.legend(loc='upper right', fontsize=8)
        axes[-1][0].set_xlabel("frame")
        fig.tight_layout()
        fig.savefig(DEBUG.path(f"{debug_stem}_ica.png"), dpi=110)
        plt.close(fig)

    return df_red_out, df_blue_out


def pred_cpu_ts_from_gpu_ts(gpu, cpu):
    reg = LinearRegression().fit(gpu.reshape(-1, 1), cpu)
    reg_ts = reg.predict(gpu.reshape(-1, 1))
    offset = (reg_ts - cpu)[:1000].mean()
    Corr_ts = reg_ts - offset
    print("Results of GPU to CPU delay, drift correction:")
    print(f"First GPU timestamp:{gpu[0]}, First CPU timestamp: {cpu[0]}, First predicted timestamp: {reg_ts[0]}")
    print(f"Calculated Offset from 1000 samples: {offset}, Final corrected timestamp:{Corr_ts[0]}")
    return Corr_ts


def vis_gpu_cpu_ts(path='/home/genzel/param/sync_inp_files'):
    path = Path(path).resolve()
    meta_filepath_list = list(sorted(path.glob('*.meta')))
    for filepath in meta_filepath_list:
        ts_data = np.genfromtxt(filepath, delimiter=',', names=True)
        corr_cpu_ts = pred_cpu_ts_from_gpu_ts(ts_data['callback_gpu_ts'], ts_data['callback_clock_ts'])
        df = pd.DataFrame()
        df['extracted_seconds_timestamp'] = pd.to_datetime(corr_cpu_ts, unit='s', utc=True)
        df['extracted_seconds_timestamp'] = df['extracted_seconds_timestamp'].dt.tz_convert('CET').dt.tz_localize(None)
        error = ts_data['callback_clock_ts'] - corr_cpu_ts
        plt.figure()
        plt.plot(error)
        plt.title("Error in original and predicted CPU timestamp")
        plt.figure()
        plt.plot(ts_data['callback_gpu_ts'] - ts_data['callback_clock_ts'])

        
def process_video_with_metadata(file_path, xy_coord, meta_filepath, process_frame_count):
    cap = cv2.VideoCapture(str(file_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_process = process_frame_count if process_frame_count is not None else frame_count
    
    ts_data = np.genfromtxt(meta_filepath, delimiter=',', names=True)
    corr_cpu_ts = pred_cpu_ts_from_gpu_ts(ts_data['callback_gpu_ts'], ts_data['callback_clock_ts'])
    df = pd.DataFrame()
    print("----------------------------------------------------------")
    print(corr_cpu_ts[0])
    df['extracted_seconds_timestamp'] = pd.to_datetime(corr_cpu_ts, unit='s', utc=True)
    print(df['extracted_seconds_timestamp'][0], df['extracted_seconds_timestamp'][0].tzinfo)
    df['extracted_seconds_timestamp'] = df['extracted_seconds_timestamp'].dt.tz_convert('CET').dt.tz_localize(None)
    print(df['extracted_seconds_timestamp'][0], df['extracted_seconds_timestamp'][0].tzinfo)
    print("----------------------------------------------------------")
    
    meta_frames = len(df['extracted_seconds_timestamp'])
    frames_match = (frame_count == meta_frames)
    if not frames_match:
        print("Frame counts do not match!!!")
        print(f"Frame count from video({frame_count})")
        print(f"Frame count from metadata({meta_frames})")

    # The detector assumes 30 fps when converting edges → Hz; flag any mismatch
    # since it directly biases the measured blink frequency.
    print(f"[fps] Video reports {video_fps:.4f} fps (detector assumes 30.0)")

    if DEBUG.enabled:
        DEBUG.add_video(
            video=Path(file_path).name,
            crop_x=xy_coord[0], crop_y=xy_coord[1],
            video_fps=round(float(video_fps), 4),
            video_frames=frame_count,
            meta_frames=meta_frames,
            frames_match=frames_match,
        )

    if ((xy_coord[0]-8 < 0) or (xy_coord[1]-8 < 0) or (xy_coord[0]+8 > 600) or (xy_coord[1]+8 > 800)):
        print("INVALID XY COORDINATES FOUND FOR LED!!!", xy_coord[0], xy_coord[1])
        return None, None, None
    
    rgb_frames = np.empty((frames_to_process, 16, 16, 3))
    for i in range(frames_to_process):
        ret, frame = cap.read()
        if frame is None:
            break
            
        start_point = (xy_coord[0]-8, xy_coord[1]-8)
        # PERFORMANCE FIX 1: Crop first, then convert color
        frame = frame[start_point[1]:start_point[1]+16, start_point[0]:start_point[0]+16]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        rgb_frames[i, :, :, :] = frame
        if i % 1000 == 0:
            print("Processed frames:", i, " at ", datetime.now(), end='\r')

    cap.release()

    if DEBUG.enabled:
        stem = Path(file_path).stem
        # Mean intensity of the crop patch over time — a real blink shows up as
        # a clear square wave here regardless of ICA. Also preview the patch so
        # you can confirm the LED is actually inside the 16x16 crop.
        patch_mean = rgb_frames.reshape(rgb_frames.shape[0], -1).mean(axis=1)
        mid = rgb_frames[rgb_frames.shape[0] // 2].astype('uint8')
        fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                               gridspec_kw={'width_ratios': [1, 4]})
        ax[0].imshow(mid)
        ax[0].set_title("16x16 crop\n(mid-frame)")
        ax[1].plot(patch_mean, lw=0.6)
        ax[1].set_title("Crop mean intensity vs frame")
        ax[1].set_xlabel("frame")
        fig.tight_layout()
        fig.savefig(DEBUG.path(f"{stem}_crop_trace.png"), dpi=110)
        plt.close(fig)

    # Window the blink extraction to AFTER SYNC_START_SEC: before the cutoff the
    # LED may sit elsewhere (repositioned), which corrupts the crop and dilutes
    # the measured frequency. The FULL per-frame timestamps are still returned so
    # the final output covers every frame.
    skip_frames = _start_frame_for(video_fps)
    n_read = rgb_frames.shape[0]
    if skip_frames >= n_read - 10:  # too few frames left after the cutoff
        if skip_frames > 0:
            print(f"[WARN] {Path(file_path).name}: only {n_read} frames read; "
                  f"start cutoff ({skip_frames}) leaves too few — using full video.")
        skip_frames = 0
    elif skip_frames > 0:
        print(f"[start] Using frames after {SYNC_START_SEC:g}s "
              f"(skipping first {skip_frames}) for LED detection.")

    full_frame_ts = df['extracted_seconds_timestamp']            # all frames (output)
    win_ts = full_frame_ts.iloc[skip_frames:].reset_index(drop=True)  # detection window

    nc = 3
    ica = FastICA(n_components=nc, random_state=0)
    X = rgb_frames[skip_frames:].reshape(n_read - skip_frames, -1).astype(float)
    demixed = ica.fit_transform(X)
    mix_weights = ica.mixing_.mean(axis=0)

    debug_stem = Path(file_path).stem if DEBUG.enabled else None
    red_ica_df, blue_ica_df = process_ica_signals(
        demixed, mix_weights, win_ts, debug_stem=debug_stem)

    return red_ica_df, blue_ica_df, full_frame_ts


# PERFORMANCE FIX 2: Fully Vectorized COM Extraction
def extract_com_from_merged_ica(agg_ica):
    if agg_ica.empty:
        return pd.DataFrame({'Center_of_mass': []})

    agg_ica_thresh = agg_ica.Total_Intensity > 0
    Time_in_seconds = agg_ica['key'].values
    ICA_vals = agg_ica_thresh.astype(int).values
    
    sig_med = np.diff(ICA_vals)
    sig_med = np.append(0, sig_med) 
    
    rising_edge = np.where(sig_med == 1)[0]
    falling_edge = np.where(sig_med == -1)[0]
    
    if len(rising_edge) == 0 or len(falling_edge) == 0:
        return pd.DataFrame({'Center_of_mass': []})

    min_len = min(len(rising_edge), len(falling_edge))
    
    if Time_in_seconds[rising_edge[0]] < Time_in_seconds[falling_edge[0]]:  
        r_edges = Time_in_seconds[rising_edge[:min_len]]
        f_edges = Time_in_seconds[falling_edge[:min_len]]
    else:
        r_edges = Time_in_seconds[rising_edge[:min_len-1]]
        f_edges = Time_in_seconds[falling_edge[1:min_len]]
        
    coms = r_edges + (f_edges - r_edges) / 2.0
    return pd.DataFrame({'Center_of_mass': coms})


# PERFORMANCE FIX 3: Optimize Dataframe Merge
def merge_ica_and_extract_com(red_ica_list, blue_ica_list):
    if red_ica_list:
        valid_reds = [df.set_index('key') for df in red_ica_list if not df.empty and 'key' in df.columns]
        if valid_reds:
            red_concat = pd.concat(valid_reds, axis=1)
            red_ica_total = red_concat.sort_index().interpolate().reset_index()
            red_ica_total['Total_Intensity'] = red_ica_total.filter(like='LED_Intensity').sum(axis=1)
            red_ica_total = red_ica_total[['key', 'Total_Intensity']]
        else:
            red_ica_total = pd.DataFrame({'key': [], 'Total_Intensity': []})
    else:
        red_ica_total = pd.DataFrame({'key': [], 'Total_Intensity': []})

    if blue_ica_list:
        valid_blues = [df.set_index('key') for df in blue_ica_list if not df.empty and 'key' in df.columns]
        if valid_blues:
            blue_concat = pd.concat(valid_blues, axis=1)
            blue_ica_total = blue_concat.sort_index().interpolate().reset_index()
            blue_ica_total['Total_Intensity'] = blue_ica_total.filter(like='LED_Intensity').sum(axis=1)
            blue_ica_total = blue_ica_total[['key', 'Total_Intensity']]
        else:
            blue_ica_total = pd.DataFrame({'key': [], 'Total_Intensity': []})
    else:
        blue_ica_total = pd.DataFrame({'key': [], 'Total_Intensity': []})
        
    red_ica_com = extract_com_from_merged_ica(red_ica_total)
    blue_ica_com = extract_com_from_merged_ica(blue_ica_total)
    
    return red_ica_com, blue_ica_com, red_ica_total, blue_ica_total


def readTrodesExtractedDataFile(filename):
    with open(filename, 'rb') as f:
        if f.readline().decode('ascii').strip() != '<Start settings>':
            raise Exception("Settings format not supported")
        fields = True
        fieldsText = {}
        for line in f:
            if(fields):
                line = line.decode('ascii').strip()
                if line != '<End settings>':
                    vals = line.split(': ')
                    fieldsText.update({vals[0].lower(): vals[1]})
                else:
                    fields = False
                    dt = parseFields(fieldsText['fields'])
                    fieldsText['data'] = np.zeros([1], dtype = dt)
                    break
        dt = parseFields(fieldsText['fields'])
        data = np.fromfile(f, dt)
        fieldsText.update({'data': data})
        return fieldsText

def parseFields(fieldstr):
    sep = re.split(r'\s', re.sub(r"\>\<|\>|\<", ' ', fieldstr).strip())
    typearr = []
    for i in range(0, sep.__len__(), 2):
        fieldname = sep[i]
        repeats = 1
        ftype = 'uint32'
        if sep[i+1].__contains__('*'):
            temptypes = re.split(r'\*', sep[i+1])
            ftype = temptypes[temptypes[0].isdigit()]
            repeats = int(temptypes[temptypes[1].isdigit()])
        else:
            ftype = sep[i+1]
        try:
            fieldtype = getattr(np, ftype)
        except AttributeError:
            print(ftype + " is not a valid field type.\n")
            exit(1)
        else:
            # Scalar fields use (name, type); (name, type, 1) triggers a numpy
            # FutureWarning and will later be reinterpreted as shape (1,).
            if repeats == 1:
                typearr.append((str(fieldname), fieldtype))
            else:
                typearr.append((str(fieldname), fieldtype, repeats))
    return np.dtype(typearr)


def extract_dio_com(dio_file_path_dict, sampling_freq, fallback_start_time=None):
    sys_time_dict = readTrodesExtractedDataFile(dio_file_path_dict['init'][0])
    try:
        sys_time = int(sys_time_dict['system_time_at_creation']) / 1000
        timestamp_at_creation = int(sys_time_dict['timestamp_at_creation'])
    except (KeyError, ValueError):
        if fallback_start_time is None:
            raise RuntimeError("system_time_at_creation missing and no fallback_start_time provided.")
        print("Warning: using fallback_start_time from video metadata.")
        timestamp_at_creation = int(sys_time_dict['first_timestamp'])
        sys_time = fallback_start_time

    print("----------------------------------------------------------")
    print(sys_time)
    timestamp_at_creation = int(sys_time_dict['timestamp_at_creation'])
    first_timestamp = int(sys_time_dict['first_timestamp'])
    sys_time_dt = pd.to_datetime(sys_time, unit='s', utc=False)
    
    red_dict_dio = readTrodesExtractedDataFile(dio_file_path_dict['red'][0])
    red_DIO = red_dict_dio['data']
    print(sys_time_dt, sys_time_dt.tzinfo, timestamp_at_creation, red_DIO[0])
    red_DIO_ts = [(
        (sys_time_dt + timedelta(seconds=float((i[0] - timestamp_at_creation) / sampling_freq))).timestamp(),
        sys_time_dt + timedelta(seconds=float((i[0] - timestamp_at_creation) / sampling_freq))
    ) for i in red_DIO]

    red_DIO_df  = pd.DataFrame({"Time_Stamp_(DIO)": [datetime.utcfromtimestamp(i[0]) for i in red_DIO_ts], 
                                "Time_in_seconds_(DIO)": [str(i[0]) for i in red_DIO_ts], 
                                "State": [i[1] for i in red_DIO_ts]})
    
    blue_dict_dio = readTrodesExtractedDataFile(dio_file_path_dict['blue'][0])
    blue_DIO = blue_dict_dio['data']
    blue_DIO_ts = [(
        (sys_time_dt + timedelta(seconds=float((i[0] - timestamp_at_creation) / sampling_freq))).timestamp(),
        sys_time_dt + timedelta(seconds=float((i[0] - timestamp_at_creation) / sampling_freq))
    ) for i in blue_DIO]

    blue_DIO_df  = pd.DataFrame({"Time_Stamp_(DIO)": [datetime.utcfromtimestamp(i[0]) for i in blue_DIO_ts], 
                                 "Time_in_seconds_(DIO)": [str(i[0]) for i in blue_DIO_ts], 
                                 "State": [i[1] for i in blue_DIO_ts]})
    
    # PERFORMANCE FIX 4: Vectorize DIO calculation
    time_stamps_red = red_DIO_df["Time_Stamp_(DIO)"].values
    if len(time_stamps_red) > 0:
        start_idx = 0 if red_DIO_df["State"][0] == 1 else 1
        t1_red = time_stamps_red[start_idx::2]
        t2_red = time_stamps_red[start_idx+1::2]
        min_len_red = min(len(t1_red), len(t2_red))
        coms_red = t1_red[:min_len_red] + (t2_red[:min_len_red] - t1_red[:min_len_red]) / 2
        com_dio_red = pd.DataFrame({'Center_of_mass': coms_red})
    else:
        com_dio_red = pd.DataFrame({'Center_of_mass': []})
        
    time_stamps_blue = blue_DIO_df["Time_Stamp_(DIO)"].values
    if len(time_stamps_blue) > 0:
        start_idx_blue = 0 if blue_DIO_df["State"][0] == 1 else 1
        t1_blue = time_stamps_blue[start_idx_blue::2]
        t2_blue = time_stamps_blue[start_idx_blue+1::2]
        min_len_blue = min(len(t1_blue), len(t2_blue))
        coms_blue = t1_blue[:min_len_blue] + (t2_blue[:min_len_blue] - t1_blue[:min_len_blue]) / 2
        com_dio_blue = pd.DataFrame({'Center_of_mass': coms_blue})
    else:
        com_dio_blue = pd.DataFrame({'Center_of_mass': []})

    return com_dio_red, com_dio_blue, sys_time, timestamp_at_creation, first_timestamp


def visualise_ica_dio_coms(dio_com_red, ica_com_red, dio_com_blue, ica_com_blue):
    # Skip any channel with no center-of-mass points (e.g. blue is empty in the
    # red-only fallback); matplotlib's stem() errors on empty input.
    series = [
        (dio_com_red,  0.6, 'red',    'ro', 'Red_DIO'),
        (ica_com_red,  0.6, 'orange', 'yo', 'Red_ICA'),
        (dio_com_blue, 0.5, 'blue',   'bo', 'Blue_DIO'),
        (ica_com_blue, 0.5, 'cyan',   'co', 'Blue_ICA'),
    ]

    fig, ax = plt.subplots()
    proxies, legend_names = [], []
    for df, amp, linefmt, markerfmt, name in series:
        df["Amp"] = amp
        if len(df["Center_of_mass"]) == 0:
            continue
        h = ax.stem(df["Center_of_mass"], df["Amp"], linefmt=linefmt, markerfmt=markerfmt)
        proxies.append(h)
        legend_names.append(name)

    if proxies:
        plt.legend(proxies, legend_names, loc='best', numpoints=1)


def pred_dio_ts_from_ica_ts_and_verify(ica_train, dio_train, test_cpu_primary, test_cpu_other, frame_wise_ts, vis_on=False):
    # The regression is fit on the driving (primary) LED's edge train; the
    # "other" LED is only used for cross-check and may be empty (e.g. the red
    # fallback runs with no blue signal).
    reg = LinearRegression().fit(ica_train.reshape(-1, 1), dio_train)
    pred_frame_wise_ts = reg.predict(frame_wise_ts.reshape(-1, 1))

    def _pred(series):
        if series is None or len(series) == 0:
            return np.array([]), None
        p = reg.predict(series.reshape(-1, 1))
        return p, p[0] - series[0]

    pred_primary, off_primary = _pred(test_cpu_primary)
    pred_other,   off_other   = _pred(test_cpu_other)

    offset = off_primary if off_primary is not None else off_other
    if off_primary is not None and off_other is not None and not np.isclose(off_primary, off_other):
        print(f"[WARN] Offset mismatch: primary({off_primary}) vs other({off_other})")
    print("Offset for final correction(s) is: ", offset)

    if len(pred_primary):
        pred_primary = pred_primary - offset
    if len(pred_other):
        pred_other = pred_other - offset
    pred_frame_wise_ts = pred_frame_wise_ts - offset

    if vis_on:
        plt.figure()
        plt.plot(pred_primary)
        plt.title("Predicted ts vs Frame number")

        if len(pred_primary):
            plt.figure()
            plt.plot(pred_primary - test_cpu_primary)
            plt.title("Predicted ts-cpu vs Frame number")

        val_dio = reg.predict(ica_train.reshape(-1, 1))
        plt.figure()
        plt.plot(val_dio - dio_train)
        plt.title("pred dio on train - dio ground truth vs Frame number")

        plt.figure()
        plt.plot(pred_frame_wise_ts - frame_wise_ts)
        plt.title("pred framewise ts - cpu avg framewise ts vs Frame number")
    return pred_primary, pred_other, pred_frame_wise_ts


def trim_ts_before_first_overlap(ica_ts_red, dio_ts_red, ica_ts_blue, dio_ts_blue):
    start_point_ica = 0
    is_dio_longer = False
    trimmed_dio_red = dio_ts_red.values[1:-2]
    print(f"trimmed red dio len: {trimmed_dio_red.shape}, before trim: {dio_ts_red.shape} ")
    trimmed_ica_red_front = ica_ts_red[(ica_ts_red > dio_ts_red.values[0])].to_numpy()
    print(f"trimmed red ica front len: {trimmed_ica_red_front.shape}, before trim: {ica_ts_red.shape} ")
    trimmed_ica_red = trimmed_ica_red_front[start_point_ica:len(trimmed_dio_red)+start_point_ica]
    print(f"trimmed red ica len: {trimmed_ica_red.shape}, before trim: {ica_ts_red.shape} ")

    try:
        min_len = min(len(trimmed_dio_blue), len(trimmed_ica_blue))
        diff = trimmed_dio_blue[:min_len] - trimmed_ica_blue[:min_len]
    except Exception as e:
        print(e, "/n Shape of ICA signal and DIO signal did not match. Auto-assuming DIO is longer.")
        is_dio_longer = True
        
    if is_dio_longer:
        trimmed_dio_red = trimmed_dio_red[:len(trimmed_ica_red)]
        diff = trimmed_dio_red - trimmed_ica_red
    print("Red: Trimmed dio - Trimmed ICA difference is: ", diff)
    plt.figure()
    plt.plot(diff)
    plt.title("diff between RED : trimmed dio and trimmed ica vs Frame number")
    
    # Blue channel may be absent (red-only fallback): skip its alignment.
    if len(ica_ts_blue) == 0 or len(dio_ts_blue) == 0:
        print("Blue channel empty — skipping blue trim (red-only fallback).")
        trimmed_ica_blue = np.array([])
        trimmed_dio_blue = np.array([])
    else:
        trimmed_dio_blue = dio_ts_blue[(dio_ts_blue > dio_ts_red.values[0]) &
                                       (dio_ts_blue < dio_ts_red.values[-1])].to_numpy()
        print(f"trimmed blue dio len: {trimmed_dio_blue.shape}, before trim: {dio_ts_blue.shape} ")
        trimmed_ica_blue_front = ica_ts_blue[(ica_ts_blue > dio_ts_red.values[0])].to_numpy()
        print(f"trimmed blue ica front len: {trimmed_ica_blue_front.shape}, before trim: {ica_ts_blue.shape} ")
        trimmed_ica_blue = trimmed_ica_blue_front[5*start_point_ica:len(trimmed_dio_blue)+5*start_point_ica]
        print(f"trimmed blue ica len: {trimmed_ica_blue.shape}, before trim: {ica_ts_blue.shape} ")

        if is_dio_longer:
            trimmed_dio_blue = trimmed_dio_blue[:len(trimmed_ica_blue)]

        diff = trimmed_dio_blue - trimmed_ica_blue
        print("Blue: Trimmed dio - Trimmed ICA difference is: ", diff)
        plt.figure()
        plt.plot(diff)
        plt.title("diff between trimmed dio and trimmed ica vs Frame number")

    return trimmed_ica_red, trimmed_dio_red, trimmed_ica_blue, trimmed_dio_blue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    
    help_text = "Input folder contains the following : 1. Eye video files: .mp4 formats (12 files, for each eye) \n \
 2. If LED locations to be extracted manually, X,y co-ordinates of crops for LED positions : .led_crop format (1 file containing 12 xy co-ordinates) \n \
 3. Time stamp files containing framewise CPU-GPU clock timestamps: .meta format (12 files) \n \
 4. Time stamps recorded from LED controller referred to as DIO: .dat format (3 files for red,blue and \
 initial systime) \n \
      a. Rat4_20201109_maze.dio_Controller_Din1.dat for initial time stamp \n \
      b. Rat4_20201109_maze_merged.dio_Controller_Din1.dat for blue DIO \n \
      c. Rat4_20201109_maze_merged.dio_Controller_Din2.dat for red DIO \n \
    Example: python Video_LED_Sync_using_ICA.py -i '/home/genzel/param/sync_inp_files' -o '/home/genzel/param/outpath/' "
    
    parser.add_argument('-i', "--input", dest='input_path', help=help_text)
    parser.add_argument('-o', "--output", dest='output_path', help='full path for generating framewise timestamps synchronised with DIO time')
    parser.add_argument('-f', "--samp_freq", dest='sampling_freq', help='Sampling freq for Spike Gadget DIO recordings')
    parser.add_argument("--debug", action='store_true',
                        help="Write LED-detection diagnostics to <output>/sync_debug/ "
                             "(localization overlay, crop trace, ICA traces, and "
                             "per-video/per-component CSV summaries).")
    parser.add_argument("--start-sec", dest='start_sec', type=float, default=SYNC_START_SEC,
                        help="Skip the first N seconds of each eye video when "
                             "locating/detecting the LED (default 45; the LED was "
                             "repositioned early in the session). The per-frame "
                             "timestamp output still covers every frame. Set 0 to disable.")
    args = parser.parse_args()

    if args.input_path is None:
        sys.exit("Please provide path to input and output video files! See --help")
    print('Input path: ', args.input_path, 'Output log path: ', args.output_path)
    print('Sampling freq: ', int(args.sampling_freq))

    SYNC_START_SEC = args.start_sec
    print(f"LED detection start cutoff: {SYNC_START_SEC:g}s")

    if args.debug or os.environ.get('SYNC_DEBUG') == '1':
        DEBUG.setup(args.output_path or args.input_path)

    vfl, xy_dict, meta_file_list, dio_file_path_dict = get_video_files_with_metadata(args.input_path, led_xy_manual=False)

    red_ica_list = []
    blue_ica_list = []
    full_ts_list = []          # full per-frame timestamps per video (for output)
    process_frame_count = None
    for itr, video_file_path in enumerate(vfl):
        print("\n")
        print("Processing for eye:", itr)
        print("Filepath:", video_file_path)
        print("XY coordinates for crop:", xy_dict[str(video_file_path)])

        red_ica_out, blue_ica_out, full_frame_ts = process_video_with_metadata(
            video_file_path, xy_dict[str(video_file_path)],
            meta_file_list[itr], process_frame_count)

        if red_ica_out is None:
            print("Red ICA signal manquant pour :", str(video_file_path))
            red_ica_out = pd.DataFrame({"key": [], "LED_Intensity": []})

        if blue_ica_out is None:
            print("Blue ICA signal manquant pour :", str(video_file_path))
            blue_ica_out = pd.DataFrame({"key": [], "LED_Intensity": []})

        red_ica_list.append(red_ica_out)
        blue_ica_list.append(blue_ica_out)
        if full_frame_ts is not None and len(full_frame_ts) > 0:
            full_ts_list.append(full_frame_ts)
        print("=================")

    # Write debug CSVs now, before the sync stage (which may sys.exit on a
    # folder with no usable LED), so diagnostics are always available.
    DEBUG.write_summary()

    temp_dir = Path(args.input_path).resolve() / "temp_no_led"
    if temp_dir.exists():
        print(f"Moving files back from {temp_dir} to {args.input_path}...")
        for f in temp_dir.iterdir():
            target = Path(args.input_path).resolve() / f.name
            shutil.move(str(f), str(target))
        
        try:
            temp_dir.rmdir()
            print("Cleaned up and removed 'temp_no_led' folder.")
        except OSError as e:
            print(f"Could not remove 'temp_no_led' folder (it might not be empty): {e}")

    # Prefer the blue LED for sync; fall back to red when blue was not detected
    # in any video (e.g. the blue LED was absent or ICA could not isolate a
    # ~2.5 Hz component). The per-frame timestamps ('key') are identical for
    # either LED, so only the edge-train alignment differs.
    have_blue = any(df.shape[0] > 0 for df in blue_ica_list)
    have_red  = any(df.shape[0] > 0 for df in red_ica_list)
    if have_blue:
        sync_color = 'blue'
    elif have_red:
        print("[WARN] No blue LED detected in any video — falling back to RED LED for sync.")
        sync_color = 'red'
    else:
        sys.exit("[ERROR] Neither blue nor red LED detected in any video; cannot sync this folder.")

    # avg_ts_per_frame drives the FINAL per-frame output, so it must span every
    # frame — build it from the full per-frame timestamps (not the windowed ICA
    # signal, which starts after SYNC_START_SEC).
    if not full_ts_list:
        sys.exit("[ERROR] No per-frame timestamps collected; cannot build output.")

    final_size = min(len(ts) for ts in full_ts_list)
    print(f"Sync LED: {sync_color} | Final size (all frames):", final_size)

    sum_ts = np.zeros((final_size,))
    for eye_ts in full_ts_list:
        ts_df = pd.to_datetime(eye_ts).astype('int64') / 10**9
        sum_ts = sum_ts + ts_df.to_numpy()[:final_size]

    avg_ts_per_frame = sum_ts / len(full_ts_list)

    ica_com_red, ica_com_blue, red_ica_total, blue_ica_total = merge_ica_and_extract_com(red_ica_list, blue_ica_list)

    fallback_start = avg_ts_per_frame[0]  # already computed before this call

    dio_com_red, dio_com_blue, system_start_time, timestamp_at_creation, first_timestamp = extract_dio_com(
        dio_file_path_dict, int(args.sampling_freq), fallback_start_time=fallback_start)

    visualise_ica_dio_coms(dio_com_red, ica_com_red, dio_com_blue, ica_com_blue)

    ts_ica_red  = pd.to_datetime(ica_com_red['Center_of_mass']).astype('int64') // 10**9
    ts_dio_red  = pd.to_datetime(dio_com_red['Center_of_mass']).astype('int64') // 10**9

    ts_dio_blue = pd.to_datetime(dio_com_blue['Center_of_mass']).astype('int64') // 10**9
    ts_ica_blue = pd.to_datetime(ica_com_blue['Center_of_mass']).astype('int64') // 10**9

    ica_train_red, dio_train_red, ica_train_blue, dio_train_blue = trim_ts_before_first_overlap(
        ts_ica_red, ts_dio_red, ts_ica_blue, ts_dio_blue)

    red_ica_corrected_s  = pd.to_datetime(red_ica_total['key']).astype('int64') // 10**9
    blue_ica_corrected_s = pd.to_datetime(blue_ica_total['key']).astype('int64') // 10**9

    # Fit the sync regression on whichever LED drives it; the other channel is
    # only cross-checked and may be empty.
    if sync_color == 'blue':
        ica_train, dio_train = ica_train_blue, dio_train_blue
        primary_corrected = blue_ica_corrected_s.to_numpy()
        other_corrected   = red_ica_corrected_s.to_numpy()
    else:
        ica_train, dio_train = ica_train_red, dio_train_red
        primary_corrected = red_ica_corrected_s.to_numpy()
        other_corrected   = blue_ica_corrected_s.to_numpy()

    if len(ica_train) == 0 or len(dio_train) == 0:
        sys.exit(f"[ERROR] No overlapping {sync_color} ICA/DIO edges to fit the sync regression.")

    pred_dio_primary, pred_dio_other, pred_framewise_ts = pred_dio_ts_from_ica_ts_and_verify(
        ica_train, dio_train, primary_corrected, other_corrected, avg_ts_per_frame, vis_on=False)

    diff = pred_dio_primary - primary_corrected
    print(f"[{sync_color} sync] Min diff in seconds (final corrected vs cpu corrected):", np.min(diff))
    print(f"[{sync_color} sync] Max diff in seconds (final corrected vs cpu corrected):", np.max(diff))

    pred_ts_df = pd.DataFrame(pred_framewise_ts, columns=['Corrected Time Stamp'])
    pred_ts_df.to_csv(args.output_path + "/stitched_framewise_ts.csv", index_label='Frame Number')

    relative_seconds = pred_framewise_ts - system_start_time +  (timestamp_at_creation/int(args.sampling_freq)) -  (first_timestamp/int(args.sampling_freq))
    
    pred_seconds_df = pd.DataFrame(relative_seconds, columns=['Seconds From Creation'])
    pred_seconds_df.to_csv(args.output_path + "/stitched_framewise_seconds.csv", index_label='Frame Number')
    print("Saved framewise seconds-based timestamps to stitched_framewise_seconds.csv")