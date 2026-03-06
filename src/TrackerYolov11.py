# -*- coding: utf-8 -*-
'''
Title: Tracker (Headless / Excel Column Lists / Auto-Video) - UPDATED FOR YOLO11
Description: 1. Reads metadata from *RecordingMeta.xlsx (Handles vertical lists).
             2. Automatically finds 'stitched.mp4' in input_folder.
             3. Optimized for Batch/Massive Analysis.
             4. Updated to use Ultralytics YOLO11x.pt
             5. Added function to copy RecordingMeta.xlsx to output folder.
'''

from itertools import groupby
from datetime import date, timedelta, datetime
from pathlib import Path
from collections import deque
from tools import mask
import cv2
# import onnxruntime  <-- REMOVED
# ... 现有的 imports ...
from ultralytics import YOLO 
# --- ADDED FOR TILING ---
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
# ------------------------
import os
import math
import time
import logging
import threading
import queue 
import numpy as np
import pandas as pd
import sys
import argparse
import glob
import shutil # <-- ADDED FOR FILE COPYING
from tqdm import tqdm



# --- CONFIGURATION ---
FONT = cv2.FONT_HERSHEY_TRIPLEX
font = cv2.FONT_HERSHEY_PLAIN 
colors = np.random.uniform(0, 255, size=(100, 3))

def points_dist(p1, p2):
    dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return dist

def convert_milli(time):
    sec = (time / 1000) % 60
    minute = (time / (1000 * 60)) % 60
    hr = (time / (1000 * 60 * 60)) % 24
    return f'{int(hr):02d}:{int(minute):02d}:{sec:.3f}'

def safe_int_str(val):
    """Converts float/int to string without .0 for integers"""
    try:
        if pd.isna(val): return ""
        return str(int(float(val)))
    except:
        return str(val)

# --- CLASS: Threaded Video Writer ---
class ThreadedVideoWriter:
    def __init__(self, path, fourcc, fps, frame_size):
        self.output_file = cv2.VideoWriter(path, fourcc, fps, frame_size)
        self.queue = queue.Queue()
        self.stopped = False
        self.thread = threading.Thread(target=self.write_frames, daemon=True)
        self.thread.start()

    def write(self, frame):
        if not self.stopped:
            self.queue.put(frame)

    def write_frames(self):
        while True:
            if self.stopped and self.queue.empty():
                break
            try:
                frame = self.queue.get(timeout=1) 
                self.output_file.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue

    def release(self):
        self.stopped = True
        self.thread.join()
        self.output_file.release()

# --- CLASS: Tracker ---
class Tracker:
    def __init__(self, vp, nl, out, metadata, onnx_weight):
        '''Tracker class initialisations'''
        self.metadata = metadata 
        self.out_path = out 
        self.model_path = onnx_weight # Renamed for clarity, though variable passed is same
        threads = list()
        
        # Load Network in main thread context to ensure model loads correctly onto GPU/CPU
        self.load_network(self.model_path)

        session = threading.Thread(target=self.load_session, args=(vp, nl, 1, out))
        threads.append(session)
        session.start()
        session.join()
            
        print('\n -Network loaded- ')

        print("Caching node dictionary...")
        self.nodes_dict = mask.create_node_dict(self.node_list)

        self.start_nodes_locations = self.find_location(self.start_nodes, self.goal)
        print('\n  ________  SUMMARY SESSION  ________  ')
        print('\nPath video file:', self.save_video)
        print('\nPath .log and .txt files:', self.save)
        print('\nTotal trials current session:', self.num_trials, '\n\nGoal location node ', self.goal)
        
        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        logfile_name = '{}/log_{}_{}.log'.format(out, str(self.date), 'Rat' + self.rat)
        
        if not os.path.exists(out):
            os.makedirs(out, exist_ok=True)

        fh = logging.FileHandler(str(logfile_name))
        formatter = logging.Formatter('%(levelname)s : %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info('Video Imported: {}'.format(vp))
        self.logger.info(f'The log format is: Video Timestamp(hh:mm:ss.ms), UTC Synchronised Timestamp in seconds, Rat position')
        
        print('\nCreating log files...')

        self.ts_file_loaded = False
        try:
            specific_ts_path = os.path.join(out, f'{str(self.date)}_Rat{str(self.rat)}_framewise_ts.csv')
            if os.path.exists(specific_ts_path):
                 self.sync_ts_dict = pd.read_csv(specific_ts_path, index_col=0).to_dict()
                 print("Loaded timestamp file: " + os.path.basename(specific_ts_path))
                 self.ts_file_loaded = True
            else:
                 stitched_ts_path = os.path.join(out, 'stitched_framewise_ts.csv')
                 if os.path.exists(stitched_ts_path):
                     print("Specific timestamp file not found. Loading 'stitched_framewise_ts.csv'...")
                     self.sync_ts_dict = pd.read_csv(stitched_ts_path, index_col=0).to_dict()
                     self.ts_file_loaded = True
                 else:
                     raise FileNotFoundError
        except Exception:
             print("Warning: No timestamp CSV found. Logs might lack sync times.")
             self.sync_ts_dict = {"Corrected Time Stamp": {}} 

        self.frame_data_log = []

        self.run_vid()
    
    def change_name_csv(self, output_path):
        csvfile_name = os.path.join(output_path,f'{str(self.date)}_Rat{str(self.rat)}_framewise_ts.csv')
        stitched_name = os.path.join(output_path,'stitched_framewise_ts.csv')
        
        if os.path.exists(stitched_name):
            try:
                if os.path.exists(csvfile_name):
                    os.remove(csvfile_name)
                os.rename(stitched_name, csvfile_name)
                print(f"File renamed to: {os.path.basename(csvfile_name)}")
            except OSError as e:
                print(f"Error renaming file: {e}")
        else:
            pass
        
    def load_network(self, model_path):
        import torch
        print(f"Loading YOLOv11 model from: {model_path}")
        
        try:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            if self.device != 'cpu':
                print(f" >> SUCCESS: GPU Detected: {torch.cuda.get_device_name(0)}")
            else:
                print(" >> WARNING: No GPU detected. Running on CPU.")

            # Load standard Ultralytics YOLO model
            self.model = YOLO(model_path)
            self.model.to(self.device)
            
            # Get class names dictionary
            self.model_names = self.model.names
            print(f"Model loaded successfully. Classes: {self.model_names}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def load_session(self, vp, nl, n, out):
        self.start_point = self.metadata['start_point']
        self.custom_trial = self.metadata['custom_trial']
        self.rat = self.metadata['rat']
        self.date = self.metadata['date']
        self.num_trials = self.metadata['num_trials']
        self.goal = self.metadata['goal']
        self.trial_type = self.metadata['trial_type']
        self.start_nodes = self.metadata['start_nodes_list']
        self.special_trials = self.metadata['special_trials_list']
        self.repeat = self.metadata['repeat']
        self.day_num = self.metadata['day']
        self.session_num = self.metadata['session']

        self.node_list = str(nl)
        self.cap = cv2.VideoCapture(str(vp))
        self.start_trial = True 
        self.end_session = False 
        self.check = False 
        self.record_detections = False 
        self.goal_location = None
        self.reached = False
        self.frame = None
        self.disp_frame = None
        self.pos_centroid = None 
        self.center_researcher = None
        
        self.last_rat_pos = None         
        self.last_researcher_pos = None 

        if self.start_point is None:
           self.trial_num = 1
        else:
           self.trial_num = int(self.custom_trial)
        self.counter = 0 
        self.count_rat = 0
        self.count_head = 0
        self.start_time = 0 
        
        self.normal_trial = False
        self.NGL = False
        self.probe = False
        self.unnormal_intervals = self.metadata.get('unnormal_intervals', {})

        self.goal_residence_timer = 0.0
        self.centroid_list = deque(maxlen=500)
        self.node_pos = []
        self.time_points = []
        self.node_id = [] 
        self.saved_nodes = []
        self.saved_velocities = []
        self.summary_trial = []
        self.store_fps = [] 
        self.locked_to_head = False   
        self.start_node_center = None
        self.covering_start_node = False
        self.cover_required_time = 1
        self.start_node_radius = 20
        self.goal_node_radius = 25
        self.save = '{}/{}_{}'.format(out, str(self.date), 'Rat' + self.rat + '.txt') 
        
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.save_video = '{}/{}_{}.mp4'.format(out, str(self.date), 'Rat' + self.rat) 
        self.vid_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.out = ThreadedVideoWriter('{}'.format(self.save_video), self.codec, self.vid_fps, (1176, 712))
        
        self.researcher_goal_timer = 0.0

    def run_vid(self):
        print('\nStarting video processing (Live Stream Enabled).....\n')
        
        # --- GUI SETUP ---
        window_name = f"Tracker - Rat {self.rat}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(window_name, 1176, 712) # 设置一个合理的初始大小
        # -----------------

        if self.start_point is None:
            with open(self.save, 'a+') as file:
                file.write(f"Rat number: {self.rat} , Date: {self.date} \n")
        self.Start_Time = time.time()
        
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        
        if self.start_point is not None:
            frame_index = int(float(self.start_point) * self.vid_fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        pbar = tqdm(total=total_frames - frame_index, unit='frames', desc='Processing', ncols=100)

        while True:
            success, self.frame = self.cap.read()
            if not success:
                if not self.end_session:
                    self.calculate_velocity(self.time_points)
                    self.save_to_file(self.save)
                break

            self.frame_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.converted_time = convert_milli(int(self.frame_time))
            
            frame_itr = frame_index 
            
            pbar.update(1)

            # Resize matches your original logic
            self.disp_frame = cv2.resize(self.frame, (1176, 712))
            
            self.t1 = time.time()
            self.cnn(self.disp_frame) 
            self.annotate_frame(self.disp_frame)
            
            # Write to video file
            self.out.write(self.disp_frame)
            
            # --- SHOW VIDEO WINDOW (STREAM) ---
            cv2.imshow(window_name, self.disp_frame)
            
            # Wait 1ms for key press. If 'q' is pressed, stop the loop.
            # waitKey is REQUIRED for imshow to redraw the window.
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                print("\nUser interrupted execution via Window (Pressed 'q').")
                break
            # ----------------------------------
            
            rat_x = self.pos_centroid[0] if self.pos_centroid else np.nan
            rat_y = self.pos_centroid[1] if self.pos_centroid else np.nan
            
            res_x = self.Researcher[0] if self.Researcher else np.nan
            res_y = self.Researcher[1] if self.Researcher else np.nan

            jp_s_x, jp_s_y = np.nan, np.nan
            jp_l_x, jp_l_y = np.nan, np.nan
            if self.record_detections:
                trial_num = self.trial_num
            else:
                trial_num = np.nan
            self.frame_data_log.append({
                'Frame_Index': frame_itr,
                'Trial_Num': trial_num,
                'Rat_X': rat_x,
                'Rat_Y': rat_y,
                'Researcher_X': res_x,
                'Researcher_Y': res_y,
                'JP_S_X': jp_s_x,
                'JP_S_Y': jp_s_y,
                'JP_L_X': jp_l_x,
                'JP_L_Y': jp_l_y
            })

            if self.record_detections:
                ts_val = self.sync_ts_dict.get("Corrected Time Stamp", {}).get(frame_itr, "N/A")
                if self.saved_nodes:
                    self.logger.info(
                        f'{self.converted_time} {ts_val} : The rat position is: {self.pos_centroid} @ {self.saved_nodes[-1]}')
                else:
                    self.logger.info(
                        f'{self.converted_time} {ts_val} : The rat position is: {self.pos_centroid}')

            if self.end_session:
                break
            
            frame_index += 1    

        pbar.close()
        
        self.export_tracking_data()

        end = time.time()
        hours, rem = divmod(end - self.Start_Time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("\nTracking process finished in: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
        
        self.cap.release()
        self.out.release() 
        
        # --- CLEANUP GUI ---
        cv2.destroyAllWindows()
        # -------------------

    def export_tracking_data(self):
        print("\n>> Compiling tracking data to CSV...")
        
        df_tracking = pd.DataFrame(self.frame_data_log)
        
        if not df_tracking.empty:
            df_tracking['Frame_Index'] = df_tracking['Frame_Index'].astype(int)

        if self.ts_file_loaded:
            try:
                raw_ts_data = self.sync_ts_dict.get('Corrected Time Stamp', self.sync_ts_dict)
                df_master = pd.DataFrame.from_dict(raw_ts_data, orient='index', columns=['Timestamp'])
                df_master.index.name = 'Frame_Index'
                df_master.index = df_master.index.astype(int)
                df_master.sort_index(inplace=True)

                df_final = pd.merge(df_master, df_tracking, on='Frame_Index', how='left')

                if 'Timestamp_y' in df_final.columns:
                    df_final.rename(columns={'Timestamp_x': 'Timestamp'}, inplace=True)
                    df_final.drop(columns=['Timestamp_y'], inplace=True)
                
                df_tracking = df_final

            except Exception as e:
                print(f"Warning: Merge failed, saving partial data only. Error: {e}")
        
        cols = ['Frame_Index', 'Timestamp', 'Trial_Num', 'Rat_X', 'Rat_Y', 
                'Researcher_X', 'Researcher_Y', 'JP_S_X', 'JP_S_Y', 'JP_L_X', 'JP_L_Y']
        
        cols = [c for c in cols if c in df_tracking.columns]
        df_tracking = df_tracking[cols]
        
        filename = f"{self.date}_Rat{self.rat}_Coordinates_Full.csv"
        save_path = os.path.join(self.out_path, filename)
        
        df_tracking.to_csv(save_path, index=False)
        print(f">> Full coordinate data saved to: {save_path}")

    def find_start(self, center_rat):
        node = self.start_nodes_locations[self.counter]
        self.locked_to_head = False 
        if points_dist(center_rat, node) < 60:
            self.logger.info('Recording Trial {}'.format(self.trial_num))
            
            if self.trial_num == 1 and int(self.trial_type) != 1:
                self.start_time = (self.frame_time / (1000 * 60)) % 60
                if int(self.trial_type) == 3:
                    self.probe = True
                if int(self.trial_type) == 2:
                    self.NGL = True
            if int(self.trial_type) == 4:
                for n in self.special_trials:
                    if int(n) == self.trial_num:
                        self.NGL = True
                        self.start_time = (self.frame_time / (1000 * 60)) % 60
            if not self.probe and not self.NGL:
                    self.normal_trial = True

            self.node_pos = []
            self.centroid_list = []
            self.time_points = []
            self.summary_trial = []
            self.saved_nodes = []
            self.node_id = [] 
            self.saved_velocities = []
            self.record_detections = True  
            
            self.researcher_goal_timer = 0.0
            
            self.pos_centroid = node
            self.centroid_list.append(self.pos_centroid)
            self.start_trial = False  
            
    def check_immunity(self):
        if self.trial_num in self.unnormal_intervals:
            start_block, end_block = self.unnormal_intervals[self.trial_num]
            current_abs_minutes = (self.frame_time / (1000 * 60)) % 60
            if start_block <= current_abs_minutes <= end_block:
                return True
        return False
    
    # --- MODIFIED CNN FUNCTION FOR YOLO11 ---
    def cnn(self, frame):
        # 1. Standard YOLO Inference on the full frame
        # stream=True is more memory efficient for video processing
        results = self.model(frame, conf=0.7, verbose=False, imgsz=1280)

        self.Rat = None
        self.Researcher = None
        
        rat_candidates = []
        researcher_candidates = []

        detected_head_this_frame = False
        detected_rat_body_this_frame = False

        # 2. Parse results (Standard Ultralytics Result object)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get metadata
                confidence = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model_names[cls_id]
                
                # Calculate centroid
                centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # Drawing logic (Matches your original style)
                color = colors[cls_id % len(colors)]
                cv2.rectangle(self.disp_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(self.disp_frame, f"{label} {confidence:.2f}", 
                            (x1, y1 + 20), font, 1, (255, 255, 255), 1)

                # Classification Logic
                if label == 'head':
                    rat_candidates.append((confidence, centroid, 'head'))
                    detected_head_this_frame = True
                elif label == 'rat':
                    rat_candidates.append((confidence, centroid, 'rat'))
                    detected_rat_body_this_frame = True
                elif label == 'researcher':
                    researcher_candidates.append((confidence, centroid))

        # 3. Decision logic (Remains exactly as you had it)
        if rat_candidates:
            rat_candidates.sort(key=lambda x: x[0], reverse=True)
            best_conf, best_centroid, best_label = rat_candidates[0]

            if best_label == 'head':
                self.locked_to_head = True

            if self.locked_to_head and best_label != 'head':
                head_cands = [c for c in rat_candidates if c[2] == 'head']
                if head_cands:
                    _, best_centroid, _ = head_cands[0]

            self.Rat = best_centroid

        if researcher_candidates:
            researcher_candidates.sort(key=lambda x: x[0], reverse=True)
            self.Researcher = researcher_candidates[0][1]

        # 4. Post-Detection State Machine logic (Rest of your existing function...)
        # [Keep the rest of your logic from "if self.Rat is not None:" onwards]
        if self.Rat is not None:
            self.last_rat_pos = self.Rat
        if self.Researcher is not None:
            self.last_researcher_pos = self.Researcher

        active_rat_pos = self.Rat if self.Rat is not None else self.last_rat_pos
        
        # Logic check for Un-normal trials
        if not self.start_trial and not self.end_session and self.trial_num in self.unnormal_intervals:
            _, end_block_abs = self.unnormal_intervals[self.trial_num]
            current_abs_minutes = (self.frame_time / (1000 * 60)) % 60
            
            if current_abs_minutes >= end_block_abs:
                self.normal_trial = False
                self.NGL = False
                self.probe = False
                self.reached = False
                self.end_trial()
                self.start_trial = True  
                self.trial_num += 1      
                self.check = False       
                return

        if self.Researcher and active_rat_pos and not self.record_detections:
            dist = points_dist(active_rat_pos, self.Researcher)

            if (not self.start_trial and not self.end_session and 
                not self.record_detections and dist <= 800): 
                
                self.start_trial = True
                self.trial_num += 1
                self.check = False

        if active_rat_pos:
            if self.start_trial:
                self.find_start(active_rat_pos)
            
            if self.record_detections:
                if detected_head_this_frame:
                    self.count_head += 1
                elif detected_rat_body_this_frame:
                    self.count_rat += 1
                
                self.object_detection(rat=active_rat_pos)
                
                if self.Researcher is not None and self.goal_location is not None:
                    dist_to_goal = points_dist(self.Researcher, self.goal_location)
                    
                    if dist_to_goal <= 50:
                        self.researcher_goal_timer += (1.0 / self.vid_fps)
                        
                        if self.researcher_goal_timer >= 10.0:
                            self.normal_trial = False
                            self.NGL = False
                            self.probe = False
                            self.end_trial()
                            self.researcher_goal_timer = 0.0
                    else:
                        self.researcher_goal_timer = 0.0

        if self.Researcher is not None and self.goal_location is not None:
            dist_to_goal = points_dist(self.Researcher, self.goal_location)
            
            if dist_to_goal <= 80:
                
                allow_end = True

                if self.probe:
                    current_min = (self.frame_time / (1000 * 60)) % 60
                    duration = current_min - self.start_time
                    if duration < 0: duration += 60
                    if duration < 2.0:
                        allow_end = False

                if self.check_immunity():
                    allow_end = False
                    
                if allow_end:
                    self.researcher_goal_timer += (1.0 / self.vid_fps)
                    
                    if self.researcher_goal_timer >= 10.0:
                        self.normal_trial = False
                        self.NGL = False
                        self.probe = False
                        
                        self.end_trial()
                        self.researcher_goal_timer = 0.0
                else:
                    self.researcher_goal_timer = 0.0

        researcher_covers_start = False
        if (not self.start_trial and not self.record_detections and 
            not self.end_session and self.counter < len(self.start_nodes_locations)):
            
            self.start_node_center = self.start_nodes_locations[self.counter]

            if self.Researcher is not None: 
                dist_to_start = points_dist(self.Researcher, self.start_node_center)
                if dist_to_start <= 40:
                    researcher_covers_start = True

            if researcher_covers_start:
                if not self.covering_start_node:
                    self.covering_start_node = True
                    self.cover_start_timer = 0.0

                self.cover_start_timer += self.frame_time

                if self.cover_start_timer >= self.cover_required_time:
                    self.start_trial = True
                    self.trial_num += 1
                    self.check = False
                    self.covering_start_node = False
                    self.cover_start_timer = 0.0
            else:
                if self.covering_start_node:
                    self.covering_start_node = False
                    self.cover_start_timer = 0.0

    def object_detection(self, rat):
        self.pos_centroid = rat
        self.centroid_list.append(self.pos_centroid)
        
        is_immune = self.check_immunity()

        if self.NGL:
            minutes = self.timer(start=self.start_time)
            if not self.reached:
                if points_dist(self.pos_centroid, self.goal_location) <= 20:
                    self.reached = True
            if minutes >= 10:
                print('\n\n >>> Ten minute passed... Goal location reached:', self.reached)
                if self.reached:
                    if not is_immune:
                        print('\n\n >>> End New Goal Location Trial - timeout', self.trial_num, ' out of ',
                            self.num_trials)
                        self.NGL = False
                        self.reached = False
                        self.end_trial()

        if self.probe:
            minutes = self.timer(start=self.start_time)
            if minutes >= 2:
                if points_dist(self.pos_centroid, self.goal_location) <= self.goal_node_radius:
                    if not is_immune:
                        self.probe = False
                        self.end_trial()
                    else:
                        pass 

        if self.normal_trial:
            if points_dist(self.pos_centroid, self.goal_location) <= self.goal_node_radius:
                if not is_immune:
                    self.normal_trial = False
                    self.end_trial()
                else:
                    pass 
    
    def end_trial(self):
        self.pos_centroid = self.goal_location
        self.centroid_list.append(self.pos_centroid)
        self.annotate_frame(self.disp_frame)
        if self.saved_nodes:
            self.logger.info(
                f'{self.converted_time} : The rat position is: {self.pos_centroid} @ {self.saved_nodes[-1]}')
        else:
            self.logger.info(
                f'{self.converted_time} : The rat position is: {self.pos_centroid}') 
        self.calculate_velocity(self.time_points)
        self.save_to_file(self.save)

        self.counter += 1 

        if self.counter == int(self.num_trials):
            self.end_session = True
        self.record_detections = False
        self.count_rat = 0
        self.count_head = 0

    def timer(self, start):
        end = (self.frame_time / (1000 * 60)) % 60
        duration = end - start
        if duration < 0:
            duration = duration + 60
        return int(duration)

    def calculate_velocity(self, time_points):
        bridges = {('124', '201'): 0.60,
                   ('121', '302'): 1.72,
                   ('223', '404'): 1.69,
                   ('324', '401'): 0.60,
                   ('305', '220'): 0.60}
        if len(time_points) > 2:
            lenght = 0
            speed = 0
            format = '%H:%M:%S.%f'
            for i in range(0, len(time_points)):
                start_node = time_points[i][1]
                start_time = datetime.strptime((time_points[i][0]), format).time()
                j = i + 1
                if j == len(time_points):
                    self.last_node = time_points[i][1]
                else:
                    end_node = time_points[j][1]
                    end_time = datetime.strptime((time_points[j][0]), format).time()
                    difference = timedelta(hours=end_time.hour - start_time.hour,
                                           minutes=end_time.minute - start_time.minute,
                                           seconds=end_time.second - start_time.second,
                                           microseconds=end_time.microsecond - start_time.microsecond).total_seconds()
                    if (start_node, end_node) in bridges:
                        lenght = bridges[(start_node, end_node)]

                    elif (end_node, start_node) in bridges:
                        lenght = bridges[(end_node, start_node)]

                    else:
                        lenght = 0.30 
                    try:
                        speed = round(float(lenght) / float(difference), 3)
                    except ZeroDivisionError:
                        speed = 0
                    finally:
                        self.summary_trial.append(
                            [(start_node, end_node), (time_points[i][0], time_points[j][0]), difference, lenght, speed])
                        self.saved_velocities.append(speed)

    @staticmethod
    def annotate_node(frame, point, node, t):
        if t == 1:
            cv2.circle(frame, point, 20, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, str(node), (point[0] - 16, point[1]),
                        fontScale=0.5, fontFace=FONT, color=(0, 255, 0), thickness=1,
                        lineType=cv2.LINE_AA)
            cv2.putText(frame, 'Start', (point[0] - 16, point[1] - 22),
                        fontScale=0.5, fontFace=FONT, color=(0, 255, 0), thickness=1,
                        lineType=cv2.LINE_AA)

        if t == 2:
            cv2.circle(frame, point, 20, color=(20, 110, 245), thickness=1)
            cv2.putText(frame, str(node), (point[0] - 16, point[1]),
                        fontScale=0.5, fontFace=FONT, color=(0, 69, 255), thickness=1,
                        lineType=cv2.LINE_AA)
        if t == 3:
            cv2.circle(frame, point, 20, color=(0, 0, 250), thickness=2)
            cv2.putText(frame, str(node), (point[0] - 16, point[1]),
                        fontScale=0.5, fontFace=FONT, color=(0, 0, 255), thickness=1,
                        lineType=cv2.LINE_AA)
            cv2.putText(frame, 'End', (point[0] - 16, point[1] - 22),
                        fontScale=0.5, fontFace=FONT, color=(0, 0, 255), thickness=1,
                        lineType=cv2.LINE_AA)

    def annotate_frame(self, frame):
        nodes_dict = self.nodes_dict 
        
        cv2.putText(frame, str(self.converted_time), (970, 670),
                    fontFace=FONT, fontScale=0.75, color=(240, 240, 240), thickness=1)
        fps = 1. / (time.time() - self.t1) 
        self.store_fps.append(fps)
        cv2.putText(frame, "FPS: {:.2f}".format(fps), (970, 650), fontFace=FONT, fontScale=0.75, color=(240, 240, 240),
                    thickness=1)
        self.annotate_node(frame, point=self.goal_location, node=self.goal, t=3)
        
        if self.start_trial and self.counter < len(self.start_nodes):
            cv2.putText(frame, f'Next trial: {self.trial_num}', (60, 60),
                        fontFace=FONT, fontScale=0.75, color=(255, 255, 255), thickness=1)
            cv2.putText(frame, 'Waiting start new trial...', (60, 80),
                        fontFace=FONT, fontScale=0.75, color=(255, 255, 255), thickness=1)

            start_pos = self.start_nodes_locations[self.counter]
            start_node_name = self.start_nodes[self.counter]
            self.annotate_node(frame, point=start_pos, node=start_node_name, t=1)

        if self.record_detections:
            for node_name in nodes_dict:
                if points_dist(self.pos_centroid, nodes_dict[node_name]) <= 20:
                    self.saved_nodes.append(node_name)
                    self.node_pos.append(nodes_dict[node_name])

                    if len(self.time_points) == 0:
                        self.time_points.append([self.converted_time, node_name])
                    if node_name != self.saved_nodes[(len(self.saved_nodes)) - 2]:
                        self.time_points.append([self.converted_time, node_name])

            cv2.putText(frame, 'Trial:' + str(self.trial_num), (60, 60),
                        fontFace=FONT, fontScale=0.75, color=(255, 255, 255), thickness=1)
            cv2.putText(frame, 'Currently writing to file...', (60, 80),
                        fontFace=FONT, fontScale=0.75, color=(255, 255, 255), thickness=1)
            cv2.putText(frame, "Rat Count: " + str(self.count_rat), (40, 130),
                        fontFace=FONT, fontScale=0.65, color=(255, 255, 255), thickness=1)
            cv2.putText(frame, "Rat-head Count: " + str(self.count_head), (40, 160),
                        fontFace=FONT, fontScale=0.65, color=(255, 255, 255), thickness=1)

            if len(self.centroid_list) >= 2:
                for i in range(1, len(self.centroid_list)):
                    cv2.line(frame, self.centroid_list[i], self.centroid_list[i - 1],
                             color=(255, 0, 60), thickness=1)
            cv2.line(frame, (self.pos_centroid[0] - 5, self.pos_centroid[1]),
                     (self.pos_centroid[0] + 5, self.pos_centroid[1]),
                     color=(0, 255, 0), thickness=2)
            cv2.line(frame, (self.pos_centroid[0], self.pos_centroid[1] - 5),
                     (self.pos_centroid[0], self.pos_centroid[1] + 5),
                     color=(0, 255, 0), thickness=2)

            start_index = max(0, len(self.saved_nodes) - 50)
            for i in range(start_index, len(self.saved_nodes)):
                self.annotate_node(frame, point=self.node_pos[i], node=self.saved_nodes[i], t=2)

    def save_to_file(self, fname):
        savelist = []
        with open(fname, 'a+') as file:
            for k, g in groupby(self.saved_nodes):
                savelist.append(k)
            file.writelines('%s,' % items for items in savelist)
            file.write(
                '\nSummary Trial {}\nStart-Next Nodes// Time points(s) //Seconds//Lenght(cm)// Velocity(m/s)\n'.format(
                    self.trial_num))
            for i in range(0, len(self.summary_trial)):
                line = " ".join(map(str, self.summary_trial[i]))
                file.write(line + '\n')
            file.write('\n')
        file.close()

    def find_location(self, start_nodes, goal):
        nodes_dict = self.nodes_dict
        start_nodes_locations = []
        for node_name in nodes_dict:
            if node_name == str(goal):
                self.goal_location = nodes_dict[node_name]
        for node in start_nodes:
            for node_name in nodes_dict:
                if node_name == str(node):
                    start_nodes_locations.append(nodes_dict[node_name])
        return start_nodes_locations

# --- DATA LOADER ---
def parse_metadata_xlsx(xlsx_path):
    print(f"Reading configuration from: {xlsx_path}")
    try:
        df = pd.read_excel(xlsx_path, engine='openpyxl')
        row0 = df.iloc[0] # Take the first row for scalars
        
        # 1. SCALARS (Row 0)
        # Handle Trial Type safely
        trial_type = "1"
        if 'Trial_Type' in row0 and not pd.isna(row0['Trial_Type']):
             trial_type = safe_int_str(row0['Trial_Type'])
             
        start_pt = None
        s_min = float(row0.get('Start_Min', 0))
        s_sec = float(row0.get('Start_Sec', 0))
        if s_min > 0 or s_sec > 0:
            start_pt = (s_min * 60) + s_sec

        # 2. LISTS (Scan columns)
        
        # Start Nodes
        s_nodes = []
        if 'Start_Nodes' in df.columns:
            s_nodes = df['Start_Nodes'].dropna().astype(int).tolist()

        # Special Trials
        sp_trials = []
        if 'Special_Trials' in df.columns:
             sp_trials = df['Special_Trials'].dropna().astype(int).tolist()

        # Unnormal Intervals
        un_dict = {}
        if 'Unnormal_Intervals' in df.columns:
            un_list = df['Unnormal_Intervals'].dropna().astype(str).tolist()
            for item in un_list:
                # Format: "Trial:Start-End" (e.g. "1:0-5")
                item = item.strip()
                if ":" in item and "-" in item:
                    parts = item.split(":")
                    try:
                        t_num = int(float(parts[0]))
                        times = parts[1].split("-")
                        un_dict[t_num] = (float(times[0]), float(times[1]))
                    except ValueError:
                        print(f"Warning: Could not parse unnormal interval '{item}'")

        metadata = {
            'start_point': start_pt,
            'custom_trial': int(float(row0.get('Start_At_Trial_Num', 1))),
            'rat': safe_int_str(row0['Rat_ID']),
            'date': safe_int_str(row0['Date']),
            'repeat': safe_int_str(row0['Repeat']),
            'day': safe_int_str(row0['Day']),
            'session': safe_int_str(row0['Session']),
            'num_trials': safe_int_str(row0['Num_Trials']),
            'goal': safe_int_str(row0['Goal_Node']),
            'prev_goal': safe_int_str(row0.get('Prev_Goal_Node', 'N/A')),
            'trial_type': trial_type,
            'start_nodes_list': s_nodes,
            'special_trials_list': sp_trials,
            'unnormal_intervals': un_dict
        }
        
        return metadata

    except Exception as e:
        print(f"Error parsing Excel file: {e}")
        # Allow the caller to handle this or exit gracefully
        raise e

# --- MAIN ---
if __name__ == "__main__":
    try:
        node_list = Path('src/tools/node_list_new.csv').resolve()
        print('\n\nTracker version: v2.11-YOLO11 (Headless / Mass Analysis)\n\n')

        # Argument Parsing
        parser = argparse.ArgumentParser(description="Tracker Headless Mode")
        parser.add_argument('--input_folder', required=True, help="Folder containing 'stitched.mp4' and '*RecordingMeta.xlsx'")
        parser.add_argument('--output_folder', required=True, help="Path to output directory")
        parser.add_argument('--onnx_weight', required=True, help="Path to .pt model file (e.g. yolov11x.pt)")
        
        args = parser.parse_args()
        
        in_p = args.input_folder
        out_p = args.output_folder
        model_path = args.onnx_weight # Keep variable name for compatibility but it loads .pt
        print("Model path:")
        print(model_path)
        
        # 1. Define Video Path
        vid_p = os.path.join(in_p, 'stitched.mp4')
        if not os.path.exists(vid_p):
            print(f"ERROR: Video file not found at: {vid_p}")
            sys.exit(1)

        # 2. Find the meta file
        meta_files = glob.glob(os.path.join(in_p, '*RecordingMeta.xlsx'))
        if not meta_files:
            print(f"ERROR: No file found matching pattern '*RecordingMeta.xlsx' in folder: {in_p}")
            sys.exit(1)
            
        xlsx_file = meta_files[0] # Take the first matching file
        metadata = parse_metadata_xlsx(xlsx_file)

        # --- NEW CODE: Copy the Excel file to the output folder ---
        print(f"\nCopying metadata file to output folder...")
        # Ensure the output directory exists before copying
        os.makedirs(out_p, exist_ok=True)
        shutil.copy2(xlsx_file, out_p)
        print(f"Successfully copied: {os.path.basename(xlsx_file)} to {out_p}\n")
        # ----------------------------------------------------------

        # 3. Start Tracker
        tracker = Tracker(vp=vid_p, nl=node_list, out=out_p, metadata=metadata, onnx_weight=model_path)
        
        # Optional renaming
        tracker.change_name_csv(out_p)
        
        # Exit successfully
        print("Done.")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)