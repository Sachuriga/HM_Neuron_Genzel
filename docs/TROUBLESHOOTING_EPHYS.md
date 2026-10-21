# Troubleshooting: Electrophysiology (Spike Sorting & LFP)

## Step 7 — Spike Sorting (Mountainsort4)

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

**Error: `No LFP source found`**

Step 8 builds the all-channel LFP straight from the session's `.rec` in the input folder (an old `<recording>.raw/*_group0.dat` also works). Make sure the `.rec` is there. Without either, it falls back to an old `.LFP/` export from the retired step e, which has only one channel per nTrode.

**Warning: `No .rec / raw export — falling back to the exportLFP .dat`**

The NWB will hold 32 LFP channels (one per nTrode) instead of 128. Put the `.rec` back in the input folder and rerun Step 8 to get all channels; the stored LFP is replaced automatically.

**Timestamps file missing — synthetic timestamps used**

If no `*.timestamps.dat` file is found in the `.LFP/` folder, the script generates timestamps using the sample rate from the file header. The resulting timestamps will be relative (starting at 0) rather than aligned to wall-clock time. This is generally acceptable for within-session analyses but will not align to neural spiking data if timestamps differ.

**Old NWB still shows `awakeness`, `emg_rms` or an "EMG ch"**

Step 8 used to pick one LFP channel as "EMG" (the one with the most 20–200 Hz power) and build an awakeness index from it. That is no longer written: the only EMG is `acquisition/emg_from_lfp`, the cross-channel correlation. Re-running step 8 on the session removes the old containers from its NWB.

**Output files not generated / folder not created**

- Verify the output folder path exists and is writable.
- Check that there is sufficient disk space (LFP output for a full session can be several GB).
