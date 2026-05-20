import os
import shutil
import traceback
from pathlib import Path
import numpy as np

import probeinterface as pi
import spikeinterface.full as si
import spikeinterface.preprocessing as spre
import argparse

# Ensure the Trodes extractor is available
try:
    from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
except ImportError:
    print("Warning: 'readTrodesExtractedDataFile3.py' not found. Please ensure it is in the directory or PYTHONPATH.")


def process_single_file(file_path, output_parent, fs=30000.0, gain=0.195, offset=0.0, n_jobs=4):
    """
    Runs the spike sorting pipeline on a single .dat file.
    Plotting is disabled, progress bars are enabled for all computations.
    All intermediate files are deleted at the end, leaving only phy_export.
    """
    file_path_obj = Path(file_path)
    file_stem = file_path_obj.stem 

    # 1. SETUP PATHS
    output_parent_obj = Path(output_parent)
    output_dir = output_parent_obj / f"{file_stem}_sorting_output"

    # Wipe any leftover artifacts from a previous (possibly failed) run so
    # nothing collides with this attempt.
    if output_dir.exists():
        print(f"Cleaning previous output: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Processing {file_stem} ---")
    print(f"Output folder: {output_dir}")

    # 2. LOAD DATA
    print("Loading data...")
    raw = readTrodesExtractedDataFile(str(file_path_obj))
    full_traces_raw = raw['data']['voltage']

    # 3. CREATE SPIKEINTERFACE RECORDING
    print("Creating recording object...")
    rec = si.NumpyRecording(traces_list=[full_traces_raw], sampling_frequency=fs)
    rec.set_channel_gains(gain)
    rec.set_channel_offsets(offset)

    # 4. CUSTOM PROBE GEOMETRY (8x4 Tetrode Grid)
    print("Configuring probe geometry...")
    rows, cols = 8, 4
    inter_tetrode_spacing = 250.0 
    diamond_offsets = np.array([[0, 10], [10, 0], [0, -10], [-10, 0]])
    
    all_positions, all_device_indices, group_ids = [], [], []
    tetrode_idx = 0
    
    for r in range(rows):
        for c in range(cols):
            x_center = c * inter_tetrode_spacing
            y_center = r * inter_tetrode_spacing
            for local_idx in range(4):
                all_positions.append([x_center + diamond_offsets[local_idx, 0], y_center + diamond_offsets[local_idx, 1]])
                all_device_indices.append(tetrode_idx * 4 + local_idx)
                group_ids.append(tetrode_idx)
            tetrode_idx += 1

    probe = pi.Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=all_positions, shapes='circle', shape_params={'radius': 5})
    probe.set_device_channel_indices(all_device_indices)

    rec = rec.set_probe(probe)
    rec.set_property("group", group_ids)
    print(f"Probe attached. Total Groups: {len(np.unique(group_ids))}")

    # 5. PREPROCESSING & SAVING
    print("Preprocessing and saving binary...")
    
    # 5.1. Bandpass filter
    rec_filtered = spre.bandpass_filter(rec, freq_min=300, freq_max=6000)
    
    # 5.2. Bad Channel Detection (Rat1-specific list; others use none)
    if file_stem.lower().startswith("rat1"):
        bad_channel_ids = [0,1,2,3,
                           4,5,6,7,
                           28,29,30,31,
                           36,37,38,39,
                           41,42,43,
                           44,45,46,47,51,
                           56,59,
                           68,69,70,71,
                           80,92,93,94,95,
                           96,97,98,99,
                           100,101,102,103,
                           108,109,110,111,
                           124,125,126,127]
    else:
        bad_channel_ids = []

    # 5.3. Bad Channel Interpolation
    if bad_channel_ids:
        print(f"Interpolating {len(bad_channel_ids)} bad channels (50 µm radius)...")
        rec_interpolated = spre.interpolate_bad_channels(
            rec_filtered,
            bad_channel_ids=bad_channel_ids
        )
    else:
        print("No bad channels configured for this rat — skipping interpolation.")
        rec_interpolated = rec_filtered

    # 5.4. Common Average Reference
    rec_cmr = spre.common_reference(rec_interpolated, reference='global', operator='median')
    
    # 5.5. Whiten (Highly recommended for MountainSort)
    rec_preprocessed = spre.whiten(rec_cmr, dtype='float32')

    processed_folder = output_dir / 'processed_binary'
    if processed_folder.exists():
        shutil.rmtree(processed_folder)

    rec_saved = rec_preprocessed.save(
        folder=processed_folder, 
        format='binary', 
        overwrite=True, 
        n_jobs=1,  # Keep this at 1 for Windows!
        chunk_duration="1s",
        progress_bar=True
    )

    # 6. SPIKE SORTING (Mountainsort4)
    win_temp = output_dir / "ms4_temp"
    win_temp.mkdir(exist_ok=True)
    os.environ['TEMPDIR'] = str(win_temp)

    sorter_name = 'mountainsort4' 
    para = si.get_default_sorter_params(sorter_name)
    para['adjacency_radius']=50
    para['filter']=False
    para['whiten']=False
    
    sorter_work_folder = output_dir / 'sorting_work_folder'
    if sorter_work_folder.exists():
        shutil.rmtree(sorter_work_folder)

    print(f"Starting sorting using {sorter_name}...")
    sorting = si.run_sorter_by_property(
        sorter_name=sorter_name,
        recording=rec_saved,
        grouping_property='group',
        folder=sorter_work_folder,
        verbose=True,
        engine="loop",
        engine_kwargs={},
        **para
    )

    final_output_folder = output_dir / 'final_sorting_result'
    if final_output_folder.exists():
        shutil.rmtree(final_output_folder)
    sorting.save(folder=final_output_folder)
    print(f"Sorting complete! Object saved to: {final_output_folder}")

    # 7. SORTING ANALYZER
    print("Initializing SortingAnalyzer (New API)...")
    analyzer_folder = output_dir / "sorting_analyzer"
    if analyzer_folder.exists():
        shutil.rmtree(analyzer_folder)

    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=rec_saved,
        format="binary_folder",
        folder=analyzer_folder, 
        overwrite=True,
        sparse=True 
    )

    # 8. COMPUTE METRICS
    print("Computing analyzer features...")
    job_kwargs = {'n_jobs': n_jobs, 'chunk_duration': '1s', 'progress_bar': True}
    
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500, **job_kwargs)
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, **job_kwargs)
    analyzer.compute("templates", **job_kwargs)
    analyzer.compute("noise_levels", **job_kwargs)
    analyzer.compute("principal_components", n_components=3, mode='by_channel_local', **job_kwargs)
    analyzer.compute("quality_metrics", metric_names=['snr', 'isi_violation', 'firing_rate'], **job_kwargs)

    # 9. PHY EXPORT
    phy_output_folder = output_dir / "phy_export"
    if phy_output_folder.exists():
        shutil.rmtree(phy_output_folder)

    print(f"Exporting results to Phy: {phy_output_folder}")
    si.export_to_phy(
        sorting_analyzer=analyzer,
        output_folder=phy_output_folder,
        compute_pc_features=True, 
        compute_amplitudes=True,
        remove_if_exists=True,
        copy_binary=True, # CHANGED TO TRUE: Necessary so we can delete the intermediate binary folder
        **job_kwargs  
    )
    
    # 10. CLEANUP INTERMEDIATE FILES
    print("\nCleaning up intermediate files...")
    for item in output_dir.iterdir():
        if item.name != "phy_export":
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                print(f"Could not delete {item.name}: {e}")
    
    print(f"Cleanup complete! Only phy_export remains.")
    print(f"Done processing {file_stem}!")
    print(f"To open Phy, run:\nphy template-gui {phy_output_folder}/params.py\n")


def run_sorting_pipeline(base_data_folder, output_data_folder, n_jobs=4):
    """
    Scans the base_data_folder for .dat files inside .raw folders 
    and processes each one, outputting strictly to output_data_folder.
    """
    base_path = Path(base_data_folder)
    output_path = Path(output_data_folder)
    
    # Ensure the main output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    dat_files = list(base_path.glob("**/*.raw/*_group0.dat"))
    
    if not dat_files:
        print(f"No .dat files found inside .raw folders under '{base_data_folder}'.")
        return

    print(f"Found {len(dat_files)} recording(s) to process.")
    
    for i, dat_file in enumerate(dat_files, 1):
        print(f"\n{'='*60}")
        print(f"File {i}/{len(dat_files)}: {dat_file.name}")
        print(f"{'='*60}")
        
        try:
            process_single_file(file_path=dat_file, output_parent=output_path, n_jobs=n_jobs)
        except Exception as e:
            print(f"Error processing {dat_file.name}:\n{e}")
            traceback.print_exc()
            print("Skipping to next file...")


# =========================================================
# HOW TO RUN
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tracker Headless Mode")

    # Input and Output arguments
    parser.add_argument('--input_folder', required=True, help="Folder containing .raw folders")
    parser.add_argument('--output_folder', required=True, help="Folder to store all outputs and Phy exports")
    
    args = parser.parse_args()

    try:
        run_sorting_pipeline(args.input_folder, args.output_folder, n_jobs=4)
    except Exception as e:
        print(f"\n[FATAL] Pipeline crashed for folder '{args.input_folder}':\n{e}")
        traceback.print_exc()
        print("[FATAL] Exiting with code 0 so the batch runner continues to the next folder.")