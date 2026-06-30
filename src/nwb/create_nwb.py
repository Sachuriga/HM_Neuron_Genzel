# RUNNING SCRIPT INFO
# For specifying rat number (if not given, default is 1):
#   --rat_nr *int* (e.g. --rat_nr 1)
# For printing options use: 
#   --sf (show input session folders) 
#   --sl (show loading and shapes of files)
#   --noprint (to disable all printing)
# For selecting the session:
#   --sess_i *int* (to specify which initial session/folder should be selected, e.g. --sess_i 1) (default is 1, meaning the first folder will be selected)
#   --sess_f *int* (to specify which final session/folder should be selected, e.g. --sess_f 3 will select session 1, 2 and 3) (default is all folders)
# For folder and current directory pathnames (if specified will override default value):
# After running script once default paths will be overriden by given paths, stored in a dictionary pickle file. Then next time the script is run, it will use the given paths from the pickle file, unless new paths are given or --usecd or --noroot is used.
#   --usecd (to use current directory instead of default) (only use this in the folder which contains the input and output folders)
#   --noroot (to set root to empty string, allowing ip and op to be completely independent)
#   --ip *folder* (to use a specified input folder) (e.g. --ip input_folder)
#   --op *folder* (to use a specified output folder) (e.g. --op output_folder)

# Example structure:
# Rat1:
#   - create_nwb.py
#   - tools
#   - input_folder
#       - session_1
#           - *.csv, .txt, .log files*
#       - session_2
#           - *.csv, .txt, .log files*
#   - output_folder
#       - session_1
#           - *.csv, .txt, .log files*
#       - session_2
#           - *.csv, .txt, .log files*


import sys
import os
import argparse
from pathlib import Path
from colorama import Fore, Style, init
import pickle

import numpy as np
import pandas as pd

from datetime import datetime
from uuid import uuid4
from dateutil import tz

from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import Position, SpatialSeries, BehavioralTimeSeries
from pynwb.file import Subject
from hdmf.common import VectorData, DynamicTable
from hdmf.container import Container

# imported from self coded tools
from tools.pathnames import find_paths, parse_folder_name, find_session_folders
from tools.process_log import process_log
from tools.process_txt import process_txt
import tools.process_dataframe as pdf

# colorama
init()

# safely read and load files from found paths based on a key (for more information read pathnames.py doc)
# every .ext is loaded using the corresponding function
# returns pd.dataframe for every .ext except for .npy it returns np.array
def safe_read(key, paths, ext):
    ext = ext.lower()
    if ext == "log":
        function = process_log
    # to be added
    #elif ext == "txt" 
    #    function = None
    elif ext == "csv":
        function = pd.read_csv
    elif ext == "npy":
        function = np.load
    elif ext == "txt":
        function = process_txt
    else:
        raise Exception("Invalid .ext!")
    try:
        # pathname is not case sensitive
        result = function(paths.get(key.lower()))
        if print_loading: print(f"Loaded {key} [Shape: {result.shape}]")
        return result
    except:
        if print_loading: print(f"{key}.{ext} not loaded.")
        return None

# parses the given date into ISO_8601 Duration format https://en.wikipedia.org/wiki/ISO_8601#Durations
# e.g.: A Rat born on jan 1 2025 and current date is jan 2 2026 will return P1Y1D
# Time can also be included but is usually unnecessary
def parse_ISO_8601(start, end=None, show_T = False):
    end = end or datetime.now()
    
    y = end.year - start.year - ((end.month, end.day) < (start.month, start.day))
    anniv = start.replace(year=start.year + y)

    delta = end - anniv
    s = int(delta.total_seconds())
    
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    
    return "P{}{}{}{}{}{}{}".format(
        f"{y}Y" if y else "",
        f"{d}D" if d else "",
        "T" if any((h, m, s)) and show_T else "",
        f"{h}H" if h and show_T else "",
        f"{m}M" if m and show_T else "",
        f"{s}S" if s and show_T else "",
        "" if any((y, d, h, m, s)) else "T0D"
    )

# takes dataframe (with column 'Time (seconds)' indicating time in seconds) and returns the element closest to 0
def find_index_time_zero(df):
    index_positive = np.where(df['Time (seconds)']>=0)[0][0] # first positive element
    if np.abs(df['Time (seconds)'][index_positive]) < np.abs(df['Time (seconds)'][index_positive-1]):
        return index_positive
    else:
        return index_positive-1

# creates a nwb file using pynwb
# uses global variables nwb_*
def create_nwb_file():
    return NWBFile(
        session_description=nwb_session_description,  # required
        identifier=str(uuid4()),  # required
        session_start_time=nwb_session_start_time,  # required
        session_id=str(nwb_session_id),  # optional
        experimenter=[
            nwb_experimenter
        ],  # optional
        lab=nwb_lab,  # optional
        institution=nwb_institution,  # optional
        experiment_description=nwb_experiment_description,  # optional
        keywords=nwb_keywords,  # optional
        related_publications=nwb_related_publications,  # optional
    )

# creates a subject to be added in nwb_file
# uses global variables subject_*
def create_subject():
    return Subject(
        subject_id= subject_id,
        age=rat_age,
        description=f"Rat {rat_nr}",
        species=species_name,
        sex=sex,
    )

# adds the created subject to nwb file
# if there is an existing subject already (e.g. in case of editing files) it will return an error
def add_subject():
    subject = create_subject()
    if nwbfile.subject is not None:
        print(f"Failed to add subject! Rat {subject.subject_id} might already exist in nwb file!")
    else:
        nwbfile.subject = subject
        print(f"Added Rat {subject.subject_id}.")
    return subject

# creates timeseries
# uses global variables lfp_*
def create_timeseries():
    return TimeSeries(
        name=lfp_name,
        description=lfp_description,
        data=lfp_data,
        unit=lfp_unit,
        timestamps=lfp_timestamps,
    )

# add timeseries to nwbfile
# if there is an existing timeseries with the same name already (e.g. in case of editing files) it will return an error
def add_timeseries(ts_name):
    ts = create_timeseries()
    try:
        nwbfile.get_acquisition(ts_name)
        print(f"Failed to add subject! {ts.name} might already exist in nwb file!")
    except:
        print(f"Added {ts.name}.")
        nwbfile.add_acquisition(ts)
    return ts

# creates a behavior module to add data such as spatial_series/position
# takes global arguments behavior_*
def create_behavior_module():
    return nwbfile.create_processing_module(
        name=behavior_name,
        description=behavior_description
    )

# splits the labels of positional data in loaded dataframe
# e.g. "Rat_X" -> X
def split_labels(df):
    return list({
        col.rsplit('_', 1)[0]
        for col in df.columns
        if col.endswith(('_X', '_Y'))
    })

# splits DLC bodypart labels (lowercase _x/_y suffix) from loaded dataframe
# e.g. "nose_x" -> nose ; ignores maze-frame columns like "Rat_X"
def split_dlc_labels(df):
    return list({
        col.rsplit('_', 1)[0]
        for col in df.columns
        if col.endswith(('_x', '_y'))
    })

# creates position object to hold positional data in the form of SpatialSeries
# takes str(pos_name) and dataframe with original data
def create_position_obj(pos_name, df, rate=float(30)):
    position_obj = Position(name=pos_name)
    labels = split_labels(df)
    for label in labels:
        if 'Time (seconds)' in df.columns:
            spatial_series_obj = SpatialSeries(
                name=label,
                description=f"(x,y) {label} position in HexMaze",
                data=np.array([df[f'{label}_X'].values, df[f'{label}_Y'].values]).T,
                timestamps=df['Time (seconds)'].values,
                unit = "pixels",
                reference_frame="(0,0) is bottom left corner",
            )
        else:
            spatial_series_obj = SpatialSeries(
                name=label,
                description=f"(x,y) {label} position in HexMaze",
                data=np.array([df[f'{label}_X'].values, df[f'{label}_Y'].values]).T,
                rate=rate,
                unit = "pixels",
                reference_frame="(0,0) is bottom left corner",
            )
        position_obj.add_spatial_series(spatial_series_obj)
    return position_obj

# creates position object to hold DLC bodypart data in the form of SpatialSeries
# reads lowercase _x/_y columns; coordinates are head-centered on mid_brain
def create_dlc_position_obj(pos_name, df, rate=float(30)):
    position_obj = Position(name=pos_name)
    labels = split_dlc_labels(df)
    for label in labels:
        if 'Time (seconds)' in df.columns:
            spatial_series_obj = SpatialSeries(
                name=label,
                description=f"(x,y) {label} DLC position, centered on mid_brain",
                data=np.array([df[f'{label}_x'].values, df[f'{label}_y'].values]).T,
                timestamps=df['Time (seconds)'].values,
                unit = "pixels",
                reference_frame="(0,0) is mid_brain (head-centered)",
            )
        else:
            spatial_series_obj = SpatialSeries(
                name=label,
                description=f"(x,y) {label} DLC position, centered on mid_brain",
                data=np.array([df[f'{label}_x'].values, df[f'{label}_y'].values]).T,
                rate=rate,
                unit = "pixels",
                reference_frame="(0,0) is mid_brain (head-centered)",
            )
        position_obj.add_spatial_series(spatial_series_obj)
    return position_obj

# creates separate TimeSeries for each column in df with timestamps
def create_metric_timeseries(df, df_filtered, descriptions, rate=float(1)):
    timeseries_list = []
    cols = df_filtered.columns
    for i, col in enumerate(cols):
        if 'Time (seconds)' in df.columns:
            ts = TimeSeries(
                name=col,
                description=descriptions[i] if i < len(descriptions) else "No description",
                data=df_filtered[col].to_numpy(),
                timestamps=df['Time (seconds)'].values,
                unit="N/A",
            )
        else:
            ts = TimeSeries(
                name=col,
                description=descriptions[i] if i < len(descriptions) else "No description",
                data=df_filtered[col].to_numpy(),
                rate=rate,
                unit="N/A",
            )
        timeseries_list.append(ts)
    return timeseries_list


# creates a BehavioralTimeSeries object to hold metrics timeseries
def create_metrics_object(timeseries_list, metrics_name):
    metrics_obj = BehavioralTimeSeries(name=metrics_name)
    for ts in timeseries_list:
        metrics_obj.add_timeseries(ts)
    return metrics_obj

# reads the per-session Goal_Node scalar from RecordingMeta.xlsx inside the session folder
# returns the goal node as an int, or None if no meta file / column is found
def read_goal_node(session_dir):
    meta_paths = list(Path(session_dir).glob("RecordingMeta.xlsx"))
    if not meta_paths:
        return None
    df = pd.read_excel(meta_paths[0], sheet_name=0)
    if "Goal_Node" not in df.columns:
        return None
    return int(df.iloc[0]["Goal_Node"])

# creates a DynamicTable from a pandas DataFrame
def create_trials_table(df, table_name, description, goal_node=None):
    """
    Converts a pandas DataFrame to a DynamicTable for NWB storage.
    
    Args:
        df: pandas DataFrame with trial data
        table_name: name for the table
        description: description of the table
    
    Returns:
        DynamicTable with columns from the DataFrame
    """
    columns = []
    
    for col in df.columns:
        vector = VectorData(
            name=col,
            description=f"Column: {col}",
            data=df[col].values,
        )
        columns.append(vector)
    
    # add per-trial Goal_node column (single session value repeated across rows, stored as bytes)
    if goal_node is not None:
        columns.append(VectorData(
            name="Goal_node",
            description="Column: Goal_node",
            data=np.array([str(goal_node).encode()] * len(df), dtype=object),
        ))
    
    table = DynamicTable(
        name=table_name,
        description=description,
        columns=columns
    )
    
    return table

# creates metadata object to hold extra information about positional data in the form of TimeSeries
# takes dataframe and outputs the start and end frames of the trials
def extract_trials(df):
    # previous and next values
    prev_trial = df["Trial_Num"].shift(1)
    next_trial = df["Trial_Num"].shift(-1)

    # start = current is a trial and previous is NaN
    start_mask = df["Trial_Num"].notna() & prev_trial.isna()

    # end = current is a trial and next is NaN
    end_mask = df["Trial_Num"].notna() & next_trial.isna()

    trial_starts = df.loc[start_mask, ["Frame_Index", "Trial_Num"]]
    trial_ends = df.loc[end_mask, ["Frame_Index", "Trial_Num"]]

    return trial_starts, trial_ends


def add_trials(nwbfile, df):
    nwbfile.add_trial_column("start_frame", "frame index of trial start")
    nwbfile.add_trial_column("stop_frame", "frame index of trial stop")
    trial_starts, trial_ends = extract_trials(df)
    for tr_start, tr_end in zip(trial_starts, trial_ends):
        nwbfile.add_trial(
            start_time=df['Time (seconds)'].values[tr_start],
            stop_time=df['Time (seconds)'].values[tr_end],
            start_frame=tr_start,
            stop_frame=tr_end,
        )
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Takes arguments"
    )
    parser.add_argument("--rat_nr", action="store", type=int, required=False, default="1", help = "Specify rat number (e.g. --rat_nr 1)")
    parser.add_argument("--sf", action="store_true", help = "Show input folders")
    parser.add_argument("--noprint", action="store_true", help = "Disable all console output/printing")
    parser.add_argument("--sl", action="store_true", help = "Show loading of files, including shapes")
    parser.add_argument("--usecd", action="store_true", help = "Use current directory as root")
    parser.add_argument("--noroot", action="store_true", help="Set root to empty string, allowing ip and op to be completely independent")
    parser.add_argument("--ip", action = "store", type=str, required=False, help = "Specify input folder (e.g. python file.py ip='input_folder'")
    parser.add_argument("--op", action = "store", type=str, required=False, help = "Specify output folder (e.g. python file.py op='output_folder' \n If not given op is same as ip")
    parser.add_argument("--sess_i", action = "store", type=int, required=False, default=1, help = "Specify what session should be selected (int)")
    parser.add_argument("--sess_f", action = "store", type=int, required=False, help = "Specify what session should be selected (int)")
    parser.add_argument("--session_folder", action="store", type=str, required=False, help = "Optional: Specify exact session folder name to process")
    args = parser.parse_args()

    if args.noprint:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    #boolean for printing loading files
    print_loading = args.sl


    # --- Filenames --- # (CHANGE TO CORRECT PATHS OR USE GIVE ARGUMENTS IN TERMINAL)

    ip_op_folders = {}
    # open dictionary of ip op folders or create new if not existing
    #try: 
    #    with open('ip_op_folders.pkl', 'rb') as f:
    #        ip_op_folders = pickle.load(f)
    #    print("Loaded IP & OP folder paths!")
    #except:
    #    ip_op_folders = {}
    #    print("IP & OP folder paths not existing. Created dictonary to store them.")

    # root to input folder
    if args.noroot: # use no root, allowing ip and op to be completely independent
        root = ""
        ip_op_folders["root"] = root
    # use current directory
    elif args.usecd:
        root = os.getcwd() + "/"
        ip_op_folders["root"] = root
    # if dictionary contains the filepath for root
    elif "root" in ip_op_folders:
        root = ip_op_folders["root"]
    # else use default value
    else:
        root = r"S:\Data\Rat1"

    # input folder after root containing folders with session data
    if args.ip is not None:
        input_folder = args.ip + "/"
        # store specified ip to dictonary
        ip_op_folders["ip"] = input_folder
    # if dictionary contains the filepath for ip
    elif "ip" in ip_op_folders:
        input_folder = ip_op_folders["ip"]
    # else use default value
    else:
        input_folder= r"nwb_data/" # Folder indicating input of a certain subject, e.g. can be ip_rat1 or ip_rat2

    # output folder/paths for created nwbfile
    if args.op is not None:
        output_folder = args.op + "/"
        # store specified op to dictionary
        ip_op_folders["op"] = output_folder
    # if dictionary contains the filepath for op
    elif "op" in ip_op_folders:
        output_folder = ip_op_folders["op"]
    # else use default value
    else:
        output_folder = input_folder


    # save given input and output folders
    #if args.ip is not None or args.op is not None:
    #    with open('ip_op_folders.pkl', 'wb') as f:
    #        pickle.dump(ip_op_folders, f)



    print(f"Input folder: {root + input_folder}")
    print(f"Output folder: {root + output_folder}")
    # folders are separated in session_folders containing the path and the date of the session
    # and notes containing suffixed notes (e.g. "session_1" and "session_1 (note about session 1)")
    session_folders, notes = find_session_folders(Path(root + input_folder)) # finds all folders in the input folder, these should be 1 folder per session
    session_paths = []


    # --- END Filenames --- #

    if args.session_folder is not None:
        session_folder, note = parse_folder_name(Path(args.session_folder).name)
        notes = [note]
        session_folders = [Path(root + input_folder + session_folder)]

    if args.sf:
        print("Found session folders: \n")

    for i, session_dir in enumerate(session_folders):
        # this appends the paths, in each session directory, to a list of all paths
        if notes[i] != "":
            session_dir = session_dir.with_name(session_dir.name + notes[i]) # reconstruct full folder name
        paths = find_paths(session_dir)
        session_paths.append(paths)
        
        # optionally prints this based on arg "--sf"
        if args.sf:
            print(f"    Session {session_dir.name}:")
            print(paths, '\n')

    if args.sess_f is None:
        args.sess_f = session_folders.__len__() # if no session end is given, set it to the total number of sessions, meaning only the session specified by --sess_i will be selected

    for session_i in range(args.sess_i-1, args.sess_f):
        # specify which session folder should be read with session_i
        paths = session_paths[session_i]
        # safe_read to corresponding variables
        df_coordinates = safe_read('Coordinates_Full', paths.csv_paths, 'csv')
        df_coordinates_with_frames = safe_read('Coordinates_Full_with_frames', paths.csv_paths, 'csv')
        # Try to load framewise files; if they don't exist, they'll be None
        df_framewise_ts = safe_read('framewise_ts', paths.csv_paths, 'csv')
        df_framewise_seconds = safe_read('stitched_framewise_seconds', paths.csv_paths, 'csv')
        df_log = safe_read(next(iter(paths.log_paths)), paths.log_paths, 'log') if paths.log_paths else None
        df_txt = safe_read(next(iter(paths.txt_paths)), paths.txt_paths, 'txt') if paths.txt_paths else None

        # Try to load LFP files; if they don't exist, they'll be None
        lfp_channels = safe_read('lfp_channels', paths.numpy_paths, 'npy')
        lfp_data = safe_read('lfp_data', paths.numpy_paths, 'npy')
        lfp_timestamps = safe_read('lfp_timestamps', paths.numpy_paths, 'npy')

        if df_framewise_seconds is not None:
            df = pdf.merge_df(df_coordinates_with_frames, df_framewise_seconds, 'Time (seconds)')
        else:
            df = df_coordinates_with_frames


        # --- METADATA --- # Can change this


        # start time of session
        # timezone
        timezone = tz.gettz('Europe/Amsterdam')
        no_timestamp_available = False

        if 'Time (seconds)' in df.columns:
            index_time_zero = find_index_time_zero(df)
            ts = df_coordinates_with_frames['Timestamp'][index_time_zero]
            from datetime import timedelta
            nwb_session_start_time = datetime(1970, 1, 1) + timedelta(seconds=float(ts))
            nwb_session_start_time = nwb_session_start_time.replace(tzinfo=timezone)

        else:
            # No timestamps available, try to get session date from txt file
            session_date = None
            no_timestamp_available = True
            session_folder_head = str(session_folders[session_i].name)
            if session_folder_head.isdigit() and len(session_folder_head) == 8: # if folder is date
                try:
                    session_date = datetime.strptime(session_folders[session_i].name, "%Y%m%d")
                    session_date = session_date.replace(tzinfo=timezone)
                except:
                    pass

            # Fallback to current date if txt failed
            if session_date is None:
                session_date = datetime.now()
            
            nwb_session_start_time = session_date.replace(hour=9, minute=0, second=0, tzinfo=timezone)
        
        # nwb metadata
        nwb_session_id = session_folders[session_i].name
        nwb_session_description = f"Rat1 Hexmaze Session {nwb_session_id}"
        nwb_experimenter = "Person"
        nwb_lab = "Genzel Lab"
        nwb_institution = "Donders Institute, Radboud University"
        nwb_experiment_description = "Rat HexMaze"
        nwb_keywords=["behavior", "ephys", "maze"]
        nwb_related_publications="N/A"

        # dropbox metadata (might not work if URL not working)
        # get the share URL from dropbox and change dl=0 suffix to dl=1 to allow direct downloading of the file
        # also needs the key in the URL, but this should be included if you share it while having access
        dropbox_URL = "https://www.dropbox.com/scl/fi/0rkk028xb77zxn182nlnr/Hexmaze_Baseline_Ephys-1_R1-2_alldata_work.xlsx?rlkey=n1egmdsb86r48yw8tq7w07ih2&st=zpbtjn3u&dl=1"
        dropbox_df = pd.read_excel(dropbox_URL, engine="openpyxl")
        # match session date to dropbox metadata to extract more information about the session
        session_date_str = f"{str(nwb_session_start_time.day).zfill(2)}.{str(nwb_session_start_time.month).zfill(2)}.{nwb_session_start_time.year}"
        current_session_mask = dropbox_df['Date'] == session_date_str
        
        if current_session_mask.any():
            nwb_experimenter = dropbox_df[current_session_mask]['Experimenter'].values[0]
            # add notes to notes list, takes notes from dropbox comment column and adds it to the notes for the session
            notes[session_i] = notes[session_i] + "\nTrial Notes:\n"+ "\n".join(f"Trial {x_i+1}: {x.strip()}" 
                                                            for x_i, x in enumerate(dropbox_df[current_session_mask]['comment'])
                                                             if pd.notna(x))
            if no_timestamp_available: # take time from dropbox if no timestamps available in data
                nwb_session_start_time = nwb_session_start_time.replace(hour=dropbox_df[current_session_mask]['Time'].values[0].hour,
                                                                        minute=dropbox_df[current_session_mask]['Time'].values[0].minute)
        # rat/subject metadata
        rat_nr = str(args.rat_nr)
        subject_id = rat_nr.zfill(3)
        rat_birthday = datetime(2019, 1, 1, 0, 0, 0, tzinfo = timezone)
        rat_age = parse_ISO_8601(rat_birthday, nwb_session_start_time)
        species_name = "Rattus norvegicus"
        sex = "M"

        # lfp/timeseries metadata
        lfp_name="lfp"
        lfp_description="lfp voltage"
        lfp_unit="uV"

        # behavior metadata
        behavior_name = "Behavior"
        behavior_description = "Positional data"

        # position metadata
        position_name = "Position"

        # dlc position metadata
        dlc_position_name = "DLC_Position"

        # metrics metadata
        metrics_cols = ['region_id', 'extracted_frame_idx']
        metrics_descriptions = ["id of camera in which rat is positioned", "Extracted Frame Index"]

        # trials table metadata (txt file)
        table_name ="Trials_Data"
        table_description = "Trial transition and speed data from txt file"

        # goal node metadata (read from RecordingMeta.xlsx in the session folder)
        goal_node = read_goal_node(session_folders[session_i])
        # --- END METADATA --- #


        # create nwb file
        nwbfile = create_nwb_file()
        print(f"Created nwb file for session {nwb_session_id}.")

        # add subject to nwb
        subject = add_subject()

        # Add a note
        if notes[session_i] != "":
            nwbfile.notes = notes[session_i].strip()
            print(f"Added note: {notes[session_i].strip()}")

        # add lfp timeseries data if available
        if lfp_data is not None:
            lfp = add_timeseries(lfp_name)

        # create behavior module
        behavior_module = create_behavior_module()
        
        # create position object
        position_obj = create_position_obj(position_name, df)

        # add position object to behavior module
        behavior_module.add(position_obj)

        # create dlc position object (head-centered bodypart coordinates)
        dlc_position_obj = create_dlc_position_obj(dlc_position_name, df)

        # add dlc position object to behavior module
        behavior_module.add(dlc_position_obj)

        # create metrics object
        df_filtered = df[[c for c in metrics_cols if c in df.columns]]
        metric_timeseries_list = create_metric_timeseries(df, df_filtered, metrics_descriptions)
        metrics_obj = create_metrics_object(metric_timeseries_list, "Metrics")

        # add metrics object to behavior module
        behavior_module.add(metrics_obj)

        # create trials table from txt data
        if df_txt is not None:
            trials_table = create_trials_table(
                df_txt,
                table_name,
                table_description,
                goal_node
            )

        # add trials table to behavior module
        if df_txt is not None:
            behavior_module.add(trials_table)

        # save nwbfile
        output_name = str(session_folders[session_i].stem) + ".nwb"
        try:
            with NWBHDF5IO(root + output_folder + output_name, "w") as io:
                io.write(nwbfile)
            print(f"Saved to {root + output_folder + output_name}!")
        except:
            print(f"{Fore.RED} ---[ERROR]--- Saving NWB file to {root + output_folder + output_name} failed! Path might not exist or is inaccessible.{Style.RESET_ALL}")