#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-process GPU encode slots.

GeForce cards cap how many NVENC sessions can be open at once — long the limit was
3, recent drivers allow 5. runner.py launches one worker per folder, so six
parallel stitches ask for six sessions in the same second. The first few get the
GPU; the rest watch nvenc fail to open, fall back to libx264, and stay there for
the entire run, because the encoder is picked once at startup and a running ffmpeg
cannot switch. That is the "a few jobs finish fast and the rest never speed up"
pattern: when the fast jobs exit their sessions go idle and nobody can take them.

A slot file turns that silent loss into an explicit queue: a job that cannot get a
session waits for one instead of committing to an hour of CPU encoding. Waiting a
few minutes for the GPU beats not using it at all.

The slot is held across BOTH the encoder probe and the real encode. Releasing in
between would let another process take the session in the gap, and the encode
would then fail rather than merely being slow.

The limit is measured, not guessed: the first job to start opens 1, 2, 3... real
encode sessions until one fails, and caches the answer for every later job. Only
one process measures at a time — concurrent measurements would consume each
other's sessions and all read low — and a measurement is only taken when nothing
is encoding, for the same reason.

    NVENC_SLOTS          override the measured limit
    NVENC_SLOT_TIMEOUT   seconds to wait before giving up and using the CPU
    NVENC_SLOT_DIR       where the slot files and the cached limit live

`python src/tools/gpuslot.py` shows the current state; `--measure` forces a fresh
measurement and rewrites the cache (do that after a driver update).
"""
import os
import sys
import json
import time
import errno
import tempfile
import contextlib
import subprocess as sp
from pathlib import Path

try:
    import psutil
except ImportError:  # optional, same as runner.py
    psutil = None

DEFAULT_SLOTS = 3
DEFAULT_TIMEOUT = 1800.0
# Poll often: a freed session should be picked up straight away, since leaving the
# encoder idle is the exact problem this module exists to fix. The check is a
# directory scan plus a few pid lookups, so once a second costs nothing next to an
# hour-long encode.
POLL_SECONDS = 1.0


def slot_dir():
    d = os.environ.get('NVENC_SLOT_DIR') or os.path.join(tempfile.gettempdir(),
                                                         'hm_tracker_gpu_slots')
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


CACHE_NAME = 'gpu_sessions.json'
LOCK_NAME = 'measure.lock'
# When there is no GPU encoder at all there is nothing to ration, and gating would
# only serialise CPU jobs that were free to run in parallel.
NO_GATE = 999
_resolved = None


def _cache_key():
    """Identifies what was measured. A different ffmpeg build or a different chosen
    encoder can mean a different session limit, so both belong in the key."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import vcodec
    enc = vcodec.select(mode='bitrate', size=(1920, 1080), verbose=False)
    return '{}|{}'.format(vcodec.ffmpeg_cmd(), enc.codec)


def _read_cache(key):
    try:
        d = json.loads((slot_dir() / CACHE_NAME).read_text())
    except (OSError, ValueError):
        return None
    if d.get('key') != key:
        return None
    v = d.get('slots')
    return v if isinstance(v, int) and v >= 1 else None


def _write_cache(key, value):
    try:
        (slot_dir() / CACHE_NAME).write_text(json.dumps(
            {'key': key, 'slots': value, 'when': int(time.time())}))
    except OSError:
        pass


def _take_lock(path):
    """Exclusive measuring lock, reclaimed if its owner died mid-measurement."""
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except OSError as e:
        if e.errno != errno.EEXIST:
            return False
    try:
        owner = int(path.read_text().strip())
    except (OSError, ValueError):
        owner = None
    if owner is not None and _alive(owner):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return _take_lock(path)


def _auto_slots(verbose=True):
    """Measured session limit, measuring once per machine and sharing the answer."""
    key = _cache_key()
    cached = _read_cache(key)
    if cached:
        return cached

    d = slot_dir()
    lock = d / LOCK_NAME
    if _take_lock(lock):
        try:
            if any(d.glob('slot*.pid')):
                # Something is already encoding, so a measurement now would count
                # its sessions against us and read low. Don't cache a bad number.
                return DEFAULT_SLOTS
            if verbose:
                _say('[INFO] Measuring how many GPU encode sessions this card allows '
                     '(once per machine)...')
            n = measure(verbose=verbose)
            n = NO_GATE if n == 0 else n      # 0 == no GPU encoder, so do not gate
            _write_cache(key, n)
            if verbose:
                _say('[INFO] GPU encode sessions available: {}.'.format(
                    'no limit to apply' if n == NO_GATE else n))
            return n
        finally:
            try:
                lock.unlink()
            except OSError:
                pass

    # Another process is measuring; wait for its answer rather than measuring too.
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(1.0)
        cached = _read_cache(key)
        if cached:
            return cached
        if not lock.exists():
            break
    return _read_cache(key) or DEFAULT_SLOTS


def n_slots(verbose=True):
    """Concurrent GPU encodes to allow: NVENC_SLOTS if set, else measured once."""
    global _resolved
    if _resolved is not None:
        return _resolved
    env = os.environ.get('NVENC_SLOTS', '').strip()
    if env:
        try:
            _resolved = max(1, int(env))
            return _resolved
        except ValueError:
            pass
    _resolved = _auto_slots(verbose=verbose)
    return _resolved


def _timeout():
    try:
        return float(os.environ.get('NVENC_SLOT_TIMEOUT', DEFAULT_TIMEOUT))
    except ValueError:
        return DEFAULT_TIMEOUT


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


def _alive(pid):
    """Is that process still running? Assume yes when we cannot tell — a false
    'dead' would hand the same session to two jobs."""
    if psutil is not None:
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    except Exception:
        return True
    return True


def _reclaim(d):
    """Free slots whose owner is gone — a crashed worker or a closed console."""
    for f in sorted(d.glob('slot*.pid')):
        try:
            pid = int(f.read_text().split()[0])
        except (OSError, ValueError, IndexError):
            pid = None
        if pid is None or not _alive(pid):
            try:
                f.unlink()
                _say('[DEBUG] Reclaimed stale GPU slot {} (owner gone).'.format(f.name))
            except OSError:
                pass


def _claim(d, i):
    """Atomically take slot `i`, or None if somebody already holds it."""
    f = d / 'slot{}.pid'.format(i)
    try:
        fd = os.open(str(f), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as e:
        if e.errno == errno.EEXIST:
            return None
        return None
    try:
        os.write(fd, '{} {}\n'.format(os.getpid(), int(time.time())).encode())
    finally:
        os.close(fd)  # closed so the file can be unlinked on Windows
    return f


@contextlib.contextmanager
def hold(timeout=None, verbose=True):
    """Hold one GPU encode slot for the duration of the block.

    Yields True when a slot was acquired, False if the wait timed out — the caller
    should carry on either way, since falling back to the CPU beats not running."""
    d = slot_dir()
    slots = n_slots(verbose=verbose)
    if slots >= NO_GATE:
        # Nothing to ration — no GPU encoder here, or no limit worth applying.
        yield False
        return

    total = _timeout() if timeout is None else timeout
    deadline = time.time() + total
    mine = None
    announced = False
    while True:
        _reclaim(d)
        for i in range(slots):
            mine = _claim(d, i)
            if mine:
                break
        if mine or time.time() >= deadline:
            break
        if verbose and not announced:
            _say('[INFO] All {} GPU encode sessions busy — waiting for one (up to '
                 '{:.0f}s) rather than falling back to the CPU.'.format(slots, total))
            announced = True
        time.sleep(POLL_SECONDS)

    if verbose:
        if mine:
            _say('[INFO] Holding GPU encode slot {}.'.format(mine.name))
        else:
            _say('[WARNING] No GPU encode slot after {:.0f}s — continuing, which will '
                 'most likely mean CPU encoding for this folder.'.format(total))
    try:
        yield bool(mine)
    finally:
        if mine:
            try:
                mine.unlink()
            except OSError:
                pass


def measure(max_try=8, verbose=True):
    """Largest number of GPU encode sessions that can be open at once.

    Run on an idle machine: anything else already encoding counts against the
    limit and will make this read low."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import vcodec
    enc = vcodec.select(mode='bitrate', size=(1920, 1080), verbose=False)
    if enc.codec == vcodec.CPU_VCODEC:
        if verbose:
            _say('[WARNING] No GPU encoder available here, so there is nothing to '
                 'measure.')
        return 0

    ok = 0
    for k in range(1, max_try + 1):
        procs = []
        for _ in range(k):
            cmd = ([vcodec.ffmpeg_cmd(), '-hide_banner', '-loglevel', 'error']
                   + list(enc.global_args)
                   + ['-f', 'lavfi', '-i', 'nullsrc=s=1280x720:d=2:r=30'])
            cmd += ['-vf', enc.filter_chain] if enc.filter_chain else ['-pix_fmt', 'yuv420p']
            cmd += ['-c:v', enc.codec] + list(enc.args) + ['-f', 'null', '-']
            procs.append(sp.Popen(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL))
        rcs = [p.wait() for p in procs]
        if all(rc == 0 for rc in rcs):
            ok = k
            if verbose:
                _say('  {} concurrent {} session(s): OK'.format(k, enc.codec))
        else:
            if verbose:
                _say('  {} concurrent {} session(s): {} failed -> limit is {}'
                     .format(k, enc.codec, sum(rc != 0 for rc in rcs), ok))
            break
    return ok


if __name__ == '__main__':
    d = slot_dir()
    if '--measure' in sys.argv:
        if any(d.glob('slot*.pid')):
            print('Jobs are encoding right now; their sessions would count against the '
                  'measurement. Re-run when idle.')
            sys.exit(1)
        n = measure()
        _write_cache(_cache_key(), NO_GATE if n == 0 else n)
        print('\n{}'.format('No GPU encoder here — slot gating disabled.' if n == 0
                            else 'Measured {} concurrent GPU encode sessions (cached).'.format(n)))
    else:
        _reclaim(d)
        held = sorted(f.name for f in d.glob('slot*.pid'))
        n = n_slots()
        print('slot dir : {}'.format(d))
        print('sessions : {}{}'.format('unlimited (no gating)' if n >= NO_GATE else n,
                                       ' (from NVENC_SLOTS)' if os.environ.get('NVENC_SLOTS') else ''))
        print('in use   : {}'.format(', '.join(held) if held else 'none'))
        print('\n--measure re-runs the measurement (do that after a driver update).')
