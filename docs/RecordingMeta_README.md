# RecordingMeta.xlsx — Filling Guide

`RecordingMeta.xlsx` is the central metadata file for the HM Tracker pipeline. It tells the tracker how to run each session (start nodes, goal nodes, trial types, etc.) and receives computed results back after processing.

**Important:** the pipeline never modifies the original file. It writes a copy with computed columns appended to the output folder (`opN/`).

---

## File structure

Each **row** represents one **trial** within a session. All rows belonging to the same session share the same `Rat_ID`, `Date`, `Repeat`, `Day`, and `Session` values.

---

## Columns — filled by the researcher

### Session-level columns
*(Same value in every row of the same session)*

| Column | Format | Example | Description |
|---|---|---|---|
| `Rat_ID` | integer | `1` | Subject identifier |
| `Date` | `YYYYMMDD` | `20200914` | Recording date — must match the folder/file name |
| `Repeat` | integer | `1` | Repeat number for this day |
| `Day` | integer | `3` | Training day number |
| `Session` | integer | `1` | Session number within the day |
| `Num_Trials` | integer | `10` | Total number of trials expected in this session |
| `Start_Min` | integer or blank | `2` | Optional — skip this many minutes at the start of the video before looking for trials |
| `Start_Sec` | integer or blank | `30` | Optional — additional seconds offset (used together with `Start_Min`) |
| `Start_At_Trial_Num` | integer or blank | `3` | Optional — resume processing from this trial number instead of from trial 1 |

---

### Trial-level columns
*(Different value per row)*

| Column | Format | Example | Description |
|---|---|---|---|
| `Start_Nodes` | node ID | `101` | The node the rat must be at to start this trial |
| `Goal_Node` | node ID | `312` | The node the rat must reach to end this trial |
| `Trial_Type` | `1`–`6` | `1` | Controls the end-condition logic — see Trial Types below |
| `Special_Trials` | integer or blank | `1` | Marks trials that need special handling (e.g. extra inter-trial lockout) |
| `Did_Not_Reach` | `0` or `1` | `0` | Set to `1` if the rat did not reach the goal for this trial |
| `Unnormal_Intervals` | `trial:start_min-end_min` | `3:2.0-4.5` | Suppresses force-end and goal-reach checks during the given window. Use commas to list multiple windows. Leave blank if not needed |

---

### Trial types

| Type | Name | End condition |
|---|---|---|
| `1` | Normal | Rat centroid within 25 px of the goal node |
| `2` | NGL | Rat visited the goal AND 10 minutes have elapsed |
| `3` | Probe | ≥ 2 minutes elapsed AND researcher near goal AND rat within 25 px of goal |
| `4`–`6` | Special NGL | Same as NGL; 10-minute inter-trial lockout begins from the start of this trial |

---

### Unnormal_Intervals format

Use this column to mark time windows during a trial where the tracker should ignore goal-reach and force-end checks (e.g. the researcher entered the maze to reposition something).

```
<trial_number>:<start_min>-<end_min>
```

Multiple windows are separated by commas:

```
3:2.0-4.5,3:7.0-7.5,5:1.0-2.0
```

Leave the cell **blank** if there are no unnormal intervals for any trial in this session.

---

## Barrier session — additional column

For sessions where a **barrier** is present in the maze, fill in the following extra column **per trial**:

| Column | Format | Example | Description |
|---|---|---|---|
| `Training_Order` | integer | `1` | The order in which this trial was presented to the rat within the barrier training sequence. Trial 1 is the first trial the rat ran in this session, trial 2 is the second, and so on. Fill in sequentially — do not skip numbers. |

**When to fill this in:** every barrier session row must have a `Training_Order` value. Leave the column blank (or omit it entirely) for non-barrier sessions.

**Example — a 5-trial barrier session:**

| Trial row | `Start_Nodes` | `Goal_Node` | `Trial_Type` | `Training_Order` |
|---|---|---|---|---|
| 1 | 101 | 312 | 1 | 1 |
| 2 | 101 | 312 | 1 | 2 |
| 3 | 101 | 312 | 1 | 3 |
| 4 | 101 | 312 | 1 | 4 |
| 5 | 101 | 312 | 1 | 5 |

---

## Columns written back by the pipeline

These columns are added automatically to the copy in the output folder. **Do not fill them in manually.**

| Column | Description |
|---|---|
| `paths` | Comma-separated node IDs visited during the trial |
| `delay` | Trial duration in seconds |
| `active_time` | Same as `delay` |
| `avg_speed` | Total path distance ÷ total path time (m/s) |
| `avg_between_node_speed` | Mean of per-segment speeds across node transitions (m/s) |
| `trial_start_time` | Synchronized timestamp (s) when the rat enters the start node |
| `trial_end_time` | Synchronized timestamp (s) when the end condition fires |

---

## Common mistakes

| Mistake | Effect |
|---|---|
| `Date` does not match the folder name | Tracker cannot find the session and skips it |
| `Num_Trials` is wrong | Tracker stops too early or waits for trials that never come |
| `Start_Nodes` / `Goal_Node` has a node ID that does not exist in the maze | Tracker never triggers a trial start / end |
| `Unnormal_Intervals` uses the wrong trial number | Immunity applied to the wrong trial |
| `Training_Order` has gaps or duplicates in a barrier session | Downstream analysis cannot sort trials correctly |
| Leaving `Training_Order` blank in a barrier session | That trial is excluded from barrier-order analysis |
