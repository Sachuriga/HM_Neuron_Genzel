"""All-channel LFP from the step-1 raw export (``*.raw/*_group0.dat``).

Trodes ``exportLFP`` writes ONE channel per nTrode (the workspace's LFP channel),
so step e gives 32 of the 128 channels. This builds the LFP for every channel
straight from the 30 kHz raw export instead, mirroring exportLFP's settings:

    int16 x 0.195 uV -> low-pass 700 Hz (zero-phase Butterworth) -> keep every
    20th sample (30 kHz -> 1500 Hz)

The filter runs forward-backward, so theta/gamma phases are not shifted.

Column ``k`` of the result is hardware channel ``k`` of the raw export — the
same id the sorter and ``sorting.nt_to_channel`` use, i.e. nTrode ``k//4 + 1``,
channel ``k%4 + 1``.

The result is written chunk by chunk (in parallel) into a float32 ``.npy`` on
disk and returned memory-mapped, so a multi-hour, 128-channel session (~8 GB at
1500 Hz) never has to fit in RAM.
"""

import io
import os
from pathlib import Path

import numpy as np

RAW_FS = 30000.0        # raw acquisition rate (Hz)
GAIN = 0.195            # uV per bit (as in sorting.py)
LFP_LOWPASS = 700.0     # Hz — exportLFP -lfplowpass 700 (runner step e)
FILTER_ORDER = 4
FILTER_MARGIN_MS = 50.0 # chunk-edge padding; the 700 Hz filter settles in a few ms
CH_PER_TETRODE = 4


def find_raw_sessions(input_folder):
    """``[{'name', 'file'}]`` for every ``<recording>.raw/*_group0.dat``.

    One raw folder per .rec, named like the ``<recording>.LFP`` folder exportLFP
    makes, so ``name`` (and thus session_boundaries) is the same either way.
    Sorted by name == chronological (the Trodes name embeds YYYYMMDD_HHMMSS).
    """
    base = Path(input_folder)
    sessions = []
    for d in sorted((d for d in base.glob("*.raw") if d.is_dir()),
                    key=lambda d: d.name):
        dats = sorted(f for f in d.glob("*_group0.dat")
                      if not f.name.startswith("._"))
        if dats:
            sessions.append({"name": d.stem, "file": dats[0]})
    return sessions


def _open_raw(dat_file):
    """A file-backed SpikeInterface recording of one raw .dat (never loaded)."""
    import spikeinterface.full as si
    from readTrodesExtractedDataFile3 import trodesBinaryLayout

    lay = trodesBinaryLayout(str(dat_file))
    if lay is None:
        raise RuntimeError(f"{dat_file.name}: not a plain interleaved raw export")
    return si.read_binary(str(dat_file), sampling_frequency=RAW_FS,
                          dtype=lay["dtype"], num_channels=lay["n_channels"],
                          file_offset=lay["offset"], time_axis=0)


def _npy_header(shape, dtype=np.float32):
    """The bytes of a version-1.0 .npy header for a C-ordered array."""
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buf, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
              "fortran_order": False, "shape": tuple(int(n) for n in shape)})
    return buf.getvalue()


def build_lfp_from_raw(sessions, out_npy, output_rate=1500.0, n_jobs=None,
                       log=print):
    """Low-pass + decimate every channel of every session into ``out_npy``.

    Each session is filtered on its own (no filter ringing across a recording
    boundary), then the sessions are joined sample after sample — the same
    gapless concatenation the exportLFP path does.

    Returns ``(lfp, fs, boundaries, channel_map)``: ``lfp`` is a read-only
    ``(n_samples, n_channels)`` float32 memmap of ``out_npy`` in uV;
    ``boundaries`` is ``[{'name', 'start', 'n'}]`` in LFP samples;
    ``channel_map`` is ``[{'index', 'ntrode', 'channel', 'source_file'}]``.
    ``n_jobs`` defaults to at most 8 workers — the runner may be doing several
    sessions at once.
    """
    import scipy.signal
    import spikeinterface.full as si
    import spikeinterface.preprocessing as spre

    if n_jobs is None:
        n_jobs = min(8, os.cpu_count() or 1)
    factor = RAW_FS / float(output_rate)
    if abs(factor - round(factor)) > 1e-9:
        raise ValueError(f"output rate {output_rate} Hz must divide {RAW_FS:.0f} Hz")
    factor = int(round(factor))
    fs = RAW_FS / factor

    # SpikeInterface only builds band/high-pass filters itself; a low-pass goes
    # in as coefficients.
    sos = scipy.signal.butter(FILTER_ORDER, LFP_LOWPASS, btype="lowpass",
                              fs=RAW_FS, output="sos")

    recs, boundaries, start = [], [], 0
    n_ch = None
    for s in sessions:
        raw = _open_raw(s["file"])
        if n_ch is None:
            n_ch = raw.get_num_channels()
        elif raw.get_num_channels() != n_ch:
            raise RuntimeError(f"{s['file'].name} has {raw.get_num_channels()} "
                               f"channels, the first session {n_ch}")
        lp = spre.filter(raw, coeff=sos, filter_mode="sos",
                         margin_ms=FILTER_MARGIN_MS, dtype="float32")
        rec = spre.scale(spre.decimate(lp, factor), gain=GAIN, dtype="float32")
        n = rec.get_num_samples()
        log(f"    • {s['name']}: {raw.get_num_samples()} raw samples "
            f"-> {n} @ {fs:g} Hz, {n_ch} ch")
        recs.append(rec)
        boundaries.append({"name": s["name"], "start": start, "n": n})
        start += n

    rec = recs[0] if len(recs) == 1 else si.concatenate_recordings(recs)
    shape = (rec.get_num_samples(), n_ch)

    # write_binary_recording recreates the file and leaves `byte_offset` bytes
    # of zeros in front, so the .npy header goes in after the samples.
    out_npy = Path(out_npy)
    header = _npy_header(shape)
    log(f"  Low-pass {LFP_LOWPASS:g} Hz + decimate x{factor} -> {out_npy.name} "
        f"({shape[0] * shape[1] * 4 / 2**30:.1f} GiB)")
    si.write_binary_recording(rec, file_paths=out_npy, dtype="float32",
                              add_file_extension=False, byte_offset=len(header),
                              n_jobs=n_jobs, chunk_duration="10s",
                              progress_bar=True)
    with open(out_npy, "r+b") as f:
        f.write(header)

    channel_map = [{"index": k, "ntrode": k // CH_PER_TETRODE + 1,
                    "channel": k % CH_PER_TETRODE + 1,
                    "source_file": sessions[0]["file"].name}
                   for k in range(n_ch)]
    return np.load(out_npy, mmap_mode="r"), fs, boundaries, channel_map
