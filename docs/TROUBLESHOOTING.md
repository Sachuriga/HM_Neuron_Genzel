# Troubleshooting Guide

## Synchronization Issues

**1. Blue, Red, and Init columns are empty in the log from Step 2**

Not all files were copied or the ephys files are corrupted. Stop the tracker and check if you have all the required files. If the issue persists, contact your supervisor.

**2. `stitched_framewise_seconds.csv` or `20260408_Rat1_framewise_ts.csv` is not generated or is empty**

Check the individual videos to confirm that the blue and red lights are blinking as expected.

---

## Tracker Issues

**1. After Step 4, verify the labeled video has all trials**

Check the output video to confirm all trials are present.

**2. Not all trials are present in the output**

Verify that the start node and goal node are correctly filled in for all trials, and that trial numbers are correct.

**3. Trial numbers or nodes look wrong**

Check whether the start node from the worksheet matches the actual start node in the arena. Experimenters sometimes place the rat in the wrong node.

**4. Node mismatch in the worksheet**

Fix the worksheet in Dropbox first, then fix the node in the input folder.

**5. Rat placed next to the start node instead of on it**

Use the next node that the animal travels to as the start.

**6. Error caused by something not listed above**

Note down the rat name, session name, and store the ip/op folder in the specific location. Use your phone to record a short clip of the error trial and post the message in the **HM_tracking** channel immediately.
