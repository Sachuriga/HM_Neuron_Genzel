# Troubleshooting: Electrophysiology (Spike Sorting & LFP)

## Step 7 — Spike Sorting (Mountainsort4)

**Error: `No LFP .dat files found`**

The LFP export (Step e) was not run or the output is in the wrong location.

1. Run Step e first:
   ```
   exportLFP -rec <file.rec> -outputrate 1000 -lfplowpass 500
   ```
2. After export, verify a folder named `<recording>.LFP/` exists inside the input folder, containing `_nt<N>ch<C>.dat` files.
3. If files exist but are in a different location, move the `.LFP/` folder to the input folder.

**Warning: `'readTrodesExtractedDataFile3.py' not found`**

The Trodes reader helper script is missing.

- Ensure `readTrodesExtractedDataFile3.py` is in the same folder as `sorting.py` (`src/sorter/`).
- If the file is missing from the repo, contact your supervisor to obtain the official Trodes reader script.

**Error: `Settings format not supported`**

The `.dat` file header does not match the expected Trodes format.

- The file may be corrupted or was exported by an incompatible version of Trodes.
- Re-export the recording from Trodes and try again.
- If the issue persists, open the `.dat` file in a text editor and verify the first line reads `<Start settings>`.

**Spike sorting hangs or crashes mid-run**

- Check GPU memory: `nvidia-smi`. If VRAM is full, close other GPU processes.
- Mountainsort4 can be memory-intensive for long recordings. Ensure at least 16 GB of system RAM is free.
- On Windows, multi-job sorting (`n_jobs > 1`) can cause crashes. The script sets `n_jobs=1` by default — do not change this.
- If the process freezes during preprocessing (bandpass filter step), the recording may be very long. Allow up to several hours for large datasets.

**Bad channel list produces unexpected results**

The bad channels are hardcoded in `sorting.py`. If the probe has changed or a previously bad channel is now functional, the list must be updated. Contact your supervisor before modifying `bad_channel_ids` in the sorting script.

**Output (`phy_export/`) is empty or missing units**

- Open the Phy output in Phy2 to inspect the sorting result.
- If very few or no units are found, the signal quality may be poor. Check the raw `.dat` files for noise.
- Ensure the recording did not have a power interruption mid-session (which produces a discontinuity in the signal).

---

## Step 8 — LFP Extraction

**Error: `No LFP .dat files found`**

Same as Step 7 above — run Step e first to generate the LFP export.

**Timestamps file missing — synthetic timestamps used**

If no `*.timestamps.dat` file is found in the `.LFP/` folder, the script generates timestamps using the sample rate from the file header. The resulting timestamps will be relative (starting at 0) rather than aligned to wall-clock time. This is generally acceptable for within-session analyses but will not align to neural spiking data if timestamps differ.

**EMG channel selection looks wrong**

The script selects the EMG channel by finding the one with the highest power in the 20–200 Hz band during the first 60 seconds of the recording. If the animal was very still at the start of the session, the EMG channel selected may be a noisy electrode rather than a true EMG.

- Check `channel_snr_scores.npy` to inspect the selection scores.
- If the wrong channel was selected, note the correct channel index and contact your supervisor to update the selection parameters.

**Awakeness index looks flat or unrealistic**

The awakeness index is computed as `0.6 × EMG_zscore + 0.4 × theta/delta ratio`. If either component is unusable (noise, bad reference, etc.), the index will not reflect true arousal.

- Check the raw `emg_data.npy` and `theta_delta_ratio.npy` files separately.
- If the EMG trace is flat (cable disconnected or electrode failing), the awakeness index will be dominated by the theta/delta ratio alone.
- Report the issue with the session name and rat ID in the HM_tracking channel.

**Output files not generated / folder not created**

- Verify the output folder path exists and is writable.
- Check that there is sufficient disk space (LFP output for a full session can be several GB).
