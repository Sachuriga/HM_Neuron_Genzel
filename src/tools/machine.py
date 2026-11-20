"""Fit the number of worker processes to the computer the pipeline runs on.

The pipeline runs on different machines: the lab PC has 20 CPU cores and 64 GB
of RAM, a laptop 8 cores and 16 GB. A fixed "n_jobs=4" leaves most of the lab PC
idle and can be too much for a small machine. This module looks at the machine
and answers two questions:

1. step_n_jobs()         how many worker processes ONE step may use
                         (the n_jobs of SpikeInterface and joblib).
2. session_jobs_limit()  tree mode: how many sessions run side by side.

The rules for step_n_jobs(), in plain words:

  * CPU:     use the cores, but leave one or two free, so that the desktop (the
             GUI, Phy) still responds while a step runs.
  * RAM:     every worker process needs memory; together the workers of a step
             may take half of the RAM.
  * Sharing: when the runner runs several jobs side by side (one per session),
             each job gets an equal share. Together they fill the machine once,
             instead of every job trying to fill it on its own.

Settings (hm_tracker_paths.txt, or the environment):
    N_JOBS            auto (the default) = the rules above;
                      a number = every step uses exactly that many workers.
    AUTO_CURATE_JOBS  a number = the Phy auto-curation uses that many, whatever
                      N_JOBS says (see phy_autocurate.curate_jobs).
    PREPROCESS_JOBS   a number = step 7 writes processed_binary/ with that many,
                      whatever N_JOBS says (see sorting.processed_binary_jobs).
    SORT_JOBS         a number = step 7 sorts that many tetrodes side by side
                      (see sorting.sorter_jobs).
    HM_PARALLEL_JOBS  set by runner.py for the jobs it starts: how many of them
                      run side by side. Not meant for the config file.

`python src/tools/machine.py` prints what this computer gets.
"""

import os

try:
    import psutil  # optional; reads the amount of RAM
except Exception:
    psutil = None

GB = 2 ** 30

# RAM assumed when it cannot be read (Windows without psutil): a small machine,
# so that the answer is on the safe side.
UNKNOWN_RAM_GB = 16.0

# The worker processes of one step may use this share of the RAM together. The
# rest is for the step's main process, the other programs and the file cache.
RAM_SHARE_FOR_WORKERS = 0.5

# RAM one worker process needs: Python with numpy and SpikeInterface loaded, plus
# the piece of the recording it works on (1 s x 128 channels is only ~15 MB).
# A step that works on larger pieces passes its own number (see lfp_from_raw.py).
GB_PER_WORKER = 1.0

# Tree mode (see session_jobs_limit): a decode or UMAP job can take several GB,
# and more than a few jobs reading the share at once do not finish sooner.
GB_PER_SESSION_JOB = 12.0
MAX_SESSION_JOBS = 6


# ------------------------------------------------------------
#                 WHAT THE COMPUTER HAS
# ------------------------------------------------------------
def cpu_cores():
    """Logical CPU cores of this computer (at least 1)."""
    return os.cpu_count() or 1


def total_ram_gb():
    """Installed RAM in GB (UNKNOWN_RAM_GB when it cannot be read)."""
    if psutil is not None:
        try:
            return psutil.virtual_memory().total / GB
        except Exception:
            pass
    try:
        # Linux and macOS can tell without psutil; Windows has no os.sysconf.
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
    except (AttributeError, ValueError, OSError):
        return UNKNOWN_RAM_GB


def cores_for_work():
    """Cores the pipeline may keep busy: all but one or two.

    A step that takes every core makes the whole computer slow to respond while
    it runs. On a big machine two free cores cost little; a machine with fewer
    than 4 cores has none to spare.
    """
    cores = cpu_cores()
    if cores >= 16:
        return cores - 2
    if cores >= 4:
        return cores - 1
    return cores


# ------------------------------------------------------------
#                 SETTINGS
# ------------------------------------------------------------
def setting_n_jobs(name):
    """A worker count from the config file: a whole number >= 1, or None.

    None means "work it out from the machine": the setting is not there, is
    "auto", or is not a usable number.
    (runner.load_config() copies the config file into the environment.)
    """
    text = os.environ.get(name, "").strip()
    try:
        number = int(text)
    except ValueError:
        return None
    return number if number >= 1 else None


def jobs_side_by_side():
    """How many jobs the runner runs at the same time, this one included.

    runner.py sets HM_PARALLEL_JOBS for the jobs it starts. A step started by
    hand has the machine to itself: 1.
    """
    return setting_n_jobs("HM_PARALLEL_JOBS") or 1


# ------------------------------------------------------------
#                 THE ANSWERS
# ------------------------------------------------------------
def step_n_jobs(gb_per_worker=GB_PER_WORKER, side_by_side=None):
    """Worker processes for ONE step on this computer (at least 1).

    gb_per_worker  RAM one worker of this step needs.
    side_by_side   how many jobs share the machine; None = what the runner said
                   (HM_PARALLEL_JOBS), or 1 for a step started by hand.
    """
    # 1. A number in the config file always wins.
    forced = setting_n_jobs("N_JOBS")
    if forced:
        return forced

    # 2. Otherwise: this job's share of the cores and of the RAM.
    if side_by_side is None:
        side_by_side = jobs_side_by_side()
    cores_share = cores_for_work() / side_by_side
    ram_share_gb = total_ram_gb() * RAM_SHARE_FOR_WORKERS / side_by_side

    # The cores are rounded to the nearest whole worker: 18 cores between 5 jobs
    # is 3.6 each, and 4 workers each (20 in all) use the machine better than 3
    # each (15). One worker too many only makes things a little slower.
    by_cpu = int(cores_share + 0.5)
    # The RAM is rounded down: a worker that does not fit makes the job fail.
    by_ram = int(ram_share_gb / gb_per_worker)

    return max(1, min(by_cpu, by_ram))


def session_jobs_limit(n_sessions):
    """Tree mode: how many session jobs run side by side when POST_JOBS is auto.

    Half the cores (every job runs worker processes of its own), one job per
    12 GB of RAM, at most 6, and never more jobs than there are sessions.
    """
    by_cpu = cpu_cores() // 2
    by_ram = int(total_ram_gb() // GB_PER_SESSION_JOB)
    return max(1, min(by_cpu, by_ram, MAX_SESSION_JOBS, n_sessions))


def summary():
    """One line for the log: the machine, and what one step gets on it."""
    asked = os.environ.get("N_JOBS", "").strip() or "auto"
    if asked.lower() != "auto" and setting_n_jobs("N_JOBS") is None:
        asked = f"{asked} (not a number: using auto)"
    return (f"{cpu_cores()} CPU cores | {total_ram_gb():.0f} GB RAM | N_JOBS={asked} -> "
            f"{step_n_jobs(side_by_side=1)} worker process(es) for a step that runs alone")


if __name__ == "__main__":
    print(summary())
    print("A step's workers when several jobs run side by side:")
    for jobs in (2, 3, 4, 5, 6):
        print(f"    {jobs} jobs -> {step_n_jobs(side_by_side=jobs)} each")
    print(f"Tree mode, POST_JOBS=auto: up to {session_jobs_limit(MAX_SESSION_JOBS)} "
          f"session(s) at once")
