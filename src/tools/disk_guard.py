"""Keep the runner's parallel jobs from filling a disk.

Several jobs that write at the same time can fill the disk they write to. A full
disk makes every job fail half-way, and a full system disk can freeze the whole
computer. This module gives runner.py two protections:

1. BEFORE a job starts  ->  DiskBudget.room_for()
   The job's expected output must fit in the free space that is left after
   (a) what the running jobs still have to write, and
   (b) MIN_FREE_GB, which is always kept free.
   A job that does not fit waits in the queue until there is room.

2. WHILE jobs run  ->  pause_decision()
   The expected sizes are only rules of thumb, so the free space is watched too.
   Below MIN_FREE_GB the newest job is paused (one more at every check) while the
   oldest keeps running, so that it can finish and delete its temporary files.
   Below half of MIN_FREE_GB every job is paused. Paused jobs go on, oldest
   first, once 1.5 x MIN_FREE_GB is free again.

Every disk a job writes to is watched: the disk of its output folders and the
disk of the temporary folder (some steps work on local copies there).

Settings (hm_tracker_paths.txt, or the environment):
    MIN_FREE_GB   space always kept free on each of those disks (default 20).
                  The GUI's "keep free" box sets HM_MIN_FREE_GB, which wins.
    JOB_DISK_GB   expected output of ONE job in GB, used instead of the rules of
                  thumb below (default: not set = use the rules of thumb)
"""

import os
import shutil
import tempfile
import threading
from pathlib import Path

try:
    import psutil  # optional; only needed to pause and resume running jobs
except Exception:
    psutil = None

GB = 2 ** 30
DEFAULT_MIN_FREE_GB = 20

# What the small steps write together (plots, tables, logs): never much, but not zero.
SMALL_STEPS_BYTES = 1 * GB


# ------------------------------------------------------------
#                 SETTINGS
# ------------------------------------------------------------
def _number_setting(name):
    """A number from the environment, or None when it is not set / not a number.
    (runner.load_config() copies the config file into the environment.)"""
    try:
        return float(os.environ.get(name, "").strip())
    except ValueError:
        return None


def min_free_bytes():
    """The space that must always stay free on a watched disk: HM_MIN_FREE_GB
    (set by the GUI) wins over MIN_FREE_GB (the config file)."""
    gb = _number_setting("HM_MIN_FREE_GB")
    if gb is None or gb < 0:
        gb = _number_setting("MIN_FREE_GB")
    if gb is None or gb < 0:
        gb = DEFAULT_MIN_FREE_GB
    return gb * GB


def fixed_job_bytes():
    """JOB_DISK_GB in bytes when the user set it, else None (= rules of thumb)."""
    gb = _number_setting("JOB_DISK_GB")
    return gb * GB if gb is not None and gb >= 0 else None


# ------------------------------------------------------------
#                 READING A DISK
# ------------------------------------------------------------
def _existing(folder):
    """`folder`, or its nearest parent that exists. An output folder may not have
    been created yet, and a folder that does not exist has no disk to ask."""
    folder = Path(folder)
    for candidate in [folder, *folder.parents]:
        if candidate.exists():
            return candidate
    return folder


def free_bytes(folder):
    """Free bytes on the disk that holds `folder`.

    None when it cannot be read: the guard then leaves that disk alone, because a
    disk we cannot measure must not block the whole run."""
    try:
        return shutil.disk_usage(_existing(folder)).free
    except OSError:
        return None


def disk_of(folder):
    """A value that is equal for two folders on the same disk (the device number;
    on Windows the volume's serial number)."""
    try:
        return os.stat(_existing(folder)).st_dev
    except OSError:
        return str(folder)


def gb_text(n_bytes):
    """'12.3' for 12.3 GB — for the messages."""
    return f"{n_bytes / GB:.1f}"


# ------------------------------------------------------------
#                 HOW MUCH WILL A JOB WRITE?
# ------------------------------------------------------------
# Rules of thumb: bytes written per byte of input. They are meant to be a little
# too large rather than too small. They do not have to be exact: the check while
# the jobs run (pause_decision) catches what they miss.
def _files_bytes(folder, suffixes, skip_prefix=""):
    """Total size of the files directly in `folder` whose name ends in one of
    `suffixes` (compared in lower case, so .MP4 counts as .mp4)."""
    total = 0
    try:
        for f in Path(folder).iterdir():
            name = f.name.lower()
            if name.startswith("._"):                      # macOS sidecar, not data
                continue
            if skip_prefix and name.startswith(skip_prefix):
                continue
            if name.endswith(suffixes) and f.is_file():
                total += f.stat().st_size
    except OSError:
        pass                                               # unreadable folder: counts as empty
    return total


def worker_job_need(ip, op, steps, exports_raw=False):
    """Expected bytes one worker job (ip*/op* mode) writes, as {folder: bytes}.

    The inputs are the raw recording (.rec) and the camera videos in `ip`.
    `exports_raw`: step 1 also asks Trodes for the raw (wideband) signal."""
    fixed = fixed_job_bytes()
    if fixed is not None:
        return {Path(op): fixed}

    rec = _files_bytes(ip, (".rec",))
    videos = _files_bytes(ip, (".mp4",), skip_prefix="stitched")
    if videos == 0:                                        # camera videos already cleaned away
        videos = _files_bytes(ip, ("stitched.mp4",))

    into_ip, into_op = 0.0, float(SMALL_STEPS_BYTES)
    if "1" in steps:
        # Trodes analog (IMU) + DIO export: about 10 of the ~130 channels in the .rec
        into_ip += 0.10 * rec
        if exports_raw:
            # the raw signal is nearly all of the .rec: a second copy of the same size
            into_ip += 1.0 * rec
    if "3" in steps:
        # stitched.mp4: taken to be as large as the camera videos together
        into_ip += 1.0 * videos
    if "4" in steps:
        # the tracker's own video of the whole session
        into_op += 1.0 * videos
    if "8" in steps:
        # LFP at 1500 of 30000 Hz is 5 % of the .rec. It exists twice for a while
        # (the .npy the NWB is built from + the NWB), plus EMG and motion.
        keeps_npy = os.environ.get("KEEP_NPY", "").strip().lower() in ("1", "true", "yes", "on")
        into_op += (0.25 if keeps_npy else 0.15) * rec
    if "d" in steps:
        # collected_frames.mp4: only the frames DeepLabCut needs
        into_op += 0.25 * videos
    return {Path(ip): into_ip, Path(op): into_op}


def session_job_need(op, steps):
    """Expected bytes one session job (tree mode) writes, as {folder: bytes}.

    The inputs are the session NWB and, for step l, the old LFP export."""
    fixed = fixed_job_bytes()
    if fixed is not None:
        return {Path(op): fixed}

    op = Path(op)
    nwb = sum(f.stat().st_size for f in op.glob("*.nwb")
              if not f.name.startswith("._") and ".tmp." not in f.name)
    lfp_export = _files_bytes(op / "LFP_Output", ("lfp_data.npy",))

    into_op, into_temp = float(SMALL_STEPS_BYTES), 0.0
    if "l" in steps:
        # step l works on local copies in the temporary folder: the NWB, the
        # export, the resampled LFP and the repacked NWB. All deleted at its end.
        into_temp += 2.0 * nwb + 2.0 * lfp_export
        into_op += 0.5 * nwb                               # the new LFP can be larger than the old
    if "u" in steps:
        into_op += 2 * GB                                  # waveforms + recomputed metrics
    if "v" in steps:
        into_op += 2 * GB                                  # summary + one PDF per unit
    if "b" in steps:
        into_op += 6 * GB                                  # decoding figures + one video per goal trial
    return {op: into_op, Path(tempfile.gettempdir()): into_temp}


# ------------------------------------------------------------
#                 1. BEFORE A JOB STARTS
# ------------------------------------------------------------
class DiskBudget:
    """Counts the disk space promised to the running jobs.

    For every watched disk it keeps two numbers:
      promised : the expected output of all running jobs together
      used     : how far the free space has gone down since those jobs started
                 (it goes back up when a job deletes its temporary files)
    What the running jobs still have to write is  promised - used.
    A new job fits when   free - (promised - used) - MIN_FREE_GB >= its own need.
    """

    def __init__(self):
        self.disks = {}      # disk id -> {"folder", "promised", "used", "last_free"}
        self.jobs = {}       # job label -> (its promise per disk, the part of it that stays)
        self.temp_folder = Path(tempfile.gettempdir())
        # The runner's worker mode asks from two threads (launching + watching).
        self.lock = threading.RLock()

    def _per_disk(self, need):
        """{folder: bytes} -> two {disk id: bytes}: everything the job writes, and
        the part of it that stays on the disk when the job is done (what goes into
        the temporary folder is deleted again; the rest stays).
        A newly seen disk is added to the watch."""
        everything, stays = {}, {}
        for folder, n_bytes in need.items():
            disk = disk_of(folder)
            everything[disk] = everything.get(disk, 0.0) + n_bytes
            if Path(folder) != self.temp_folder:
                stays[disk] = stays.get(disk, 0.0) + n_bytes
            if disk not in self.disks:
                self.disks[disk] = {"folder": _existing(folder), "promised": 0.0,
                                    "used": 0.0, "last_free": None}
        return everything, stays

    def _measure(self, disk):
        """Read the disk's free space and update `used` with the change since the
        last reading. Returns the free bytes (None when unreadable)."""
        state = self.disks[disk]
        free = free_bytes(state["folder"])
        if free is None:
            return None
        if state["last_free"] is not None:
            state["used"] += state["last_free"] - free
        # `used` can never be below zero or above what was promised
        state["used"] = min(max(state["used"], 0.0), state["promised"])
        state["last_free"] = free
        return free

    def room_for(self, need):
        """(True, '') when a job with this `need` ({folder: bytes}) fits, else
        (False, why)."""
        with self.lock:
            reserve = min_free_bytes()
            everything, _stays = self._per_disk(need)
            for disk, wanted in everything.items():
                free = self._measure(disk)
                if free is None:
                    continue
                state = self.disks[disk]
                still_to_write = state["promised"] - state["used"]
                left = free - still_to_write - reserve
                if wanted > left:
                    why = (f"DISK {state['folder']}: job needs ~{gb_text(wanted)} GB, "
                           f"{gb_text(max(left, 0))} GB available ({gb_text(free)} free"
                           f" - {gb_text(still_to_write)} for running jobs"
                           f" - {gb_text(reserve)} kept free)")
                    if not self.jobs:
                        why += " — free up space; it goes on by itself"
                    return False, why
            return True, ""

    def forecast(self, needs):
        """For the start of a run: one line of text per disk, saying what ALL jobs
        together expect to write there and how much of the disk can be used.
        `needs` holds one {folder: bytes} per job."""
        with self.lock:
            reserve = min_free_bytes()
            total, stays_total, largest, n_jobs = {}, {}, {}, {}
            for need in needs:
                everything, stays = self._per_disk(need)
                for disk, n_bytes in everything.items():
                    if n_bytes <= 0:
                        continue
                    total[disk] = total.get(disk, 0.0) + n_bytes
                    largest[disk] = max(largest.get(disk, 0.0), n_bytes)
                    n_jobs[disk] = n_jobs.get(disk, 0) + 1
                    stays_total[disk] = stays_total.get(disk, 0.0) + stays.get(disk, 0.0)

            lines = []
            for disk, all_bytes in total.items():
                free = self._measure(disk)
                if free is None:
                    continue
                usable = max(free - reserve, 0.0)
                temporary = all_bytes - stays_total[disk]
                line = (f"{self.disks[disk]['folder']}: {n_jobs[disk]} job(s) expect to write "
                        f"~{gb_text(all_bytes)} GB"
                        + (f" ({gb_text(temporary)} GB of it temporary)" if temporary > 0 else "")
                        + f"; {gb_text(usable)} GB can be used ({gb_text(free)} free"
                          f" - {gb_text(reserve)} kept free)")
                if largest[disk] > usable:
                    line += " — NOT ENOUGH for one job: free up space"
                elif stays_total[disk] > usable:
                    line += " — not enough for all: the last jobs wait until space is freed up"
                elif all_bytes > usable:
                    line += " — not all at once: jobs take turns"
                lines.append(line)
            return lines

    def promise(self, label, need):
        """A job starts: set its expected output aside."""
        with self.lock:
            everything, stays = self._per_disk(need)
            for disk, n_bytes in everything.items():
                self._measure(disk)                        # changes before now are not this job's
                self.disks[disk]["promised"] += n_bytes
            self.jobs[label] = (everything, stays)

    def release(self, label):
        """A job finished (or failed): its promise is no longer needed."""
        with self.lock:
            everything, stays = self.jobs.pop(label, ({}, {}))
            for disk, n_bytes in everything.items():
                self._measure(disk)
                state = self.disks[disk]
                state["promised"] = max(state["promised"] - n_bytes, 0.0)
                # What the job wrote into its own folders stays on the disk, so it
                # no longer counts as "used by the running jobs". Its temporary
                # files were deleted instead: there the free space went up by itself.
                state["used"] -= stays.get(disk, 0.0)
                state["used"] = min(max(state["used"], 0.0), state["promised"])

    def fullest(self):
        """(free bytes, folder) of the watched disk with the least free space, or
        (None, None) when no disk is watched / readable."""
        with self.lock:
            worst = (None, None)
            for disk, state in self.disks.items():
                free = self._measure(disk)
                if free is not None and (worst[0] is None or free < worst[0]):
                    worst = (free, state["folder"])
            return worst


# ------------------------------------------------------------
#                 2. WHILE JOBS RUN
# ------------------------------------------------------------
def pause_decision(free, n_active, n_paused, reserve=None):
    """What to do now, given the free bytes on the fullest watched disk.

    Returns "pause" (pause the newest active job), "resume" (let the oldest paused
    job go on) or "" (nothing). One step per call, so the number of running jobs
    goes down and up gradually."""
    reserve = min_free_bytes() if reserve is None else reserve
    if free is None or reserve <= 0:
        return "resume" if n_paused else ""               # nothing to guard
    if free < reserve / 2:
        # critically low: nothing may write any more
        return "pause" if n_active else ""
    if free < reserve:
        # low: one job keeps running, so it can finish and delete its temporary files
        if n_active > 1:
            return "pause"
        return "resume" if n_active == 0 and n_paused else ""
    if free < 1.5 * reserve:
        # just enough: change nothing, but never leave every job paused
        return "resume" if n_active == 0 and n_paused else ""
    return "resume" if n_paused else ""                   # plenty: back to normal


def can_pause():
    """Pausing a running job needs psutil."""
    return psutil is not None


def _process_tree(pid):
    """The job's process and everything it started (the step script, its helpers)."""
    parent = psutil.Process(pid)
    return [parent] + parent.children(recursive=True)


def pause_job(pid):
    """Freeze a job: its process and all the processes below it. A frozen process
    keeps its memory and its open files, and writes nothing. True when it worked."""
    if psutil is None or pid is None:
        return False
    try:
        for process in _process_tree(pid):                 # parent first: it cannot start new ones
            process.suspend()
        return True
    except Exception:
        return False


def resume_job(pid):
    """Let a paused job go on. True when it worked."""
    if psutil is None or pid is None:
        return False
    try:
        for process in reversed(_process_tree(pid)):       # helpers first, the parent last
            process.resume()
        return True
    except Exception:
        return False
