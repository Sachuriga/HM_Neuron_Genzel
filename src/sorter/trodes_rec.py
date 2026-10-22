"""Read a Trodes ``.rec`` straight into SpikeInterface, identical to ``trodesexport -raw``.

Step 1 used to write ``<recording>.raw/*_group0.dat`` — a second full copy of
the wideband data (~27 GiB an hour). ``open_rec`` gives the same int16 samples,
in the same column order, lazily from the ``.rec`` itself, so that export is no
longer needed. Checked bit for bit against ``_group0.dat`` (Rat5 20260720
pre + post, and maze_merged with 97 dropped-packet gaps).

Three things ``se.read_spikegadgets`` (neo >= 0.14.5) does not do for our files:

1. Channel order. The MCU interleaves the Intan chips in each packet: byte pair
   ``i`` holds hwChan ``(i % n_chips) * 32 + i // n_chips``. neo knows this, but
   only applies it when the header's ``chanPerChip`` is valid; our HH128 headers
   say ``chanPerChip="0"`` (or garbage), so neo falls back to hwChan-ascending
   and every channel label points at the wrong data. We pick channels by slot.
2. Reference. With ``rawRefOn="1"`` the raw export is ``channel - ref channel``
   (``refNTrodeID`` / ``refChan``), so we subtract it the same way.
3. Dropped packets. neo returns one sample per packet. trodesexport fills a gap
   of up to ``-interp`` (default 100) missing packets by linear interpolation
   and splices larger gaps shut. We do the same with its exact arithmetic (see
   ``TrodesRawRecording``), so sample counts — and every ``sample / fs`` time
   axis downstream — match the export.

Columns come out in SpikeConfiguration document order (nTrode 1 ch 1..4,
nTrode 2 ch 1..4, ...) with channel ids ``0..n-1`` — the export's column ids,
i.e. nTrode ``k//4 + 1``, channel ``k%4 + 1``.
"""

import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from spikeinterface.preprocessing.basepreprocessor import (BasePreprocessor,
                                                           BasePreprocessorSegment)

__version__ = "1.0"         # SpikeInterface needs it to rebuild TrodesRawRecording in workers

INTAN_CHANS_PER_CHIP = 32   # RHD2132; used when the header's chanPerChip is unusable
TRODES_INTERP = 100         # trodesexport -interp default (max dropped packets filled)

# Gaps open_rec leaves spliced shut (> TRODES_INTERP dropped packets) or timestamp
# repeats/resets, per .rec: after each, ``sample / fs`` stops matching the Trodes
# clock that the video is synced to. Recorded so step 7 can write them next to the
# sorting output (spliced_gaps.json) and step u can warn.
SPLICED_GAPS = {}


def find_rec_files(folder, recursive=False):
    """Trodes ``.rec`` files in ``folder``, sorted by name (== chronological, the
    Trodes name embeds YYYYMMDD_HHMMSS). Skips macOS ``._`` sidecar files, and a
    ``<X>.rec`` that has a ``<X>_merged.rec`` beside it — the merged file
    supersedes it (as in tools/pathnames.py), and taking both would join the
    same data twice."""
    base = Path(folder)
    pattern = "**/*.rec" if recursive else "*.rec"
    recs = [f for f in base.glob(pattern)
            if f.is_file() and not f.name.startswith("._")]
    merged = {(f.parent, f.stem[:-len("_merged")].lower())
              for f in recs if f.stem.lower().endswith("_merged")}
    return sorted((f for f in recs if (f.parent, f.stem.lower()) not in merged),
                  key=lambda f: f.name)


def read_spike_configuration(rec_file):
    """The ``<SpikeConfiguration>`` element of the .rec's embedded XML header."""
    with open(rec_file, "rb") as f:
        head = f.read(10_000_000)
    tag = b"</Configuration>"
    end = head.find(tag)
    if end < 0:
        raise RuntimeError(f"{Path(rec_file).name}: no </Configuration> in the header")
    sconf = ET.fromstring(head[:end + len(tag)]).find("SpikeConfiguration")
    if sconf is None:
        raise RuntimeError(f"{Path(rec_file).name}: no SpikeConfiguration in the header")
    return sconf


def packet_timestamps(neo_reader, start=0, stop=None, chunk=2_000_000):
    """The uint32 Trodes timestamp of every packet in ``[start, stop)``."""
    mm, b = neo_reader._raw_memmap, neo_reader._timestamp_byte
    stop = mm.shape[0] if stop is None else stop
    ts = np.empty(stop - start, dtype=np.uint32)
    for s in range(start, stop, chunk):
        e = min(s + chunk, stop)
        ts[s - start:e - start] = np.ascontiguousarray(mm[s:e, b:b + 4]).view("<u4").ravel()
    return ts


def find_packet_gaps(neo_reader):
    """``(index, step)`` of every timestamp step != 1: dropped packets after
    packet ``index`` (step > 1), or a repeat/reset (step <= 0).

    The full scan reads the whole file, so first compare the timestamp span
    with the packet count (O(1)); scan only when they disagree.
    """
    n = neo_reader._raw_memmap.shape[0]
    if n < 2:
        return []
    first = int(packet_timestamps(neo_reader, 0, 1)[0])
    last = int(packet_timestamps(neo_reader, n - 1, n)[0])
    if last - first == n - 1:
        return []
    d = np.diff(packet_timestamps(neo_reader).astype(np.int64))
    idx = np.flatnonzero(d != 1)
    return [(int(i), int(d[i])) for i in idx]


class TrodesRawRecording(BasePreprocessor):
    """rawRef subtraction + trodesexport's gap interpolation, lazily.

    ``gap_index[j]`` is the packet after which ``gap_missing[j]`` samples are
    inserted. Matches trodesexport -raw bit for bit:

    * a raw -32768 (Trodes' "no sample") is taken as 0, with or without ref;
    * real samples: ``x - x[ref]`` in int32, saturated to int16 (a big
      channel-ref difference is clipped, not wrapped around);
    * inserted:     from the referenced (unclipped) neighbours L, R, a float32
      ``v = L; v += (R - v) / (m + 1 - k)`` per sample k, rounded half to
      even, saturated to int16. (Linear in exact arithmetic; the float32 steps
      decide the ties.)

    Checked on real gaps of 1-2 packets and on synthetic gaps of 1-100
    packets (packets cut out of a real .rec, then exported by trodesexport).
    """

    def __init__(self, recording, ref_col=None, gap_index=(), gap_missing=()):
        BasePreprocessor.__init__(self, recording, dtype="int16")
        gap_index = [int(g) for g in gap_index]
        gap_missing = [int(m) for m in gap_missing]
        for seg in recording._recording_segments:
            self.add_recording_segment(
                TrodesRawSegment(seg, ref_col, gap_index, gap_missing))
        self._kwargs = dict(recording=recording, ref_col=ref_col,
                            gap_index=gap_index, gap_missing=gap_missing)


class TrodesRawSegment(BasePreprocessorSegment):
    def __init__(self, parent_segment, ref_col, gap_index, gap_missing):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self.time_vector = None      # a parent time vector would have the wrong length
        self.ref_col = ref_col
        self.g = np.asarray(gap_index, dtype=np.int64)       # packet before each gap
        self.m = np.asarray(gap_missing, dtype=np.int64)     # samples inserted after it
        self.c = np.concatenate([[0], np.cumsum(self.m)[:-1]]).astype(np.int64)
        self.a = self.g + self.c                             # output index of that packet
        self.n = parent_segment.get_num_samples() + int(self.m.sum())

    def get_num_samples(self):
        return self.n

    def get_traces(self, start_frame, end_frame, channel_indices):
        start_frame = 0 if start_frame is None else int(start_frame)
        end_frame = self.n if end_frame is None else int(end_frame)
        o = np.arange(start_frame, end_frame, dtype=np.int64)
        if o.size == 0:
            out = np.zeros((0, self.parent_recording_segment.get_traces(0, 1, slice(None)).shape[1]),
                           dtype=np.int16)
            return out if channel_indices is None else out[:, channel_indices]
        # Map each output frame to its source packet: either a real packet, or
        # an inserted sample `off` steps after gap packet g (of m inserted).
        if self.g.size:
            j = np.searchsorted(self.a, o, side="right") - 1
            jj = np.maximum(j, 0)
            off = np.where(j >= 0, o - self.a[jj], 0)
            m = np.where(j >= 0, self.m[jj], 0)
            ins = (off >= 1) & (off <= m)
            src = o - np.where(j >= 0, self.c[jj], 0) - np.minimum(off, m)
        else:
            off = m = np.zeros(o.size, dtype=np.int64)
            ins = np.zeros(o.size, dtype=bool)
            src = o
        lo, hi = int(src[0]), int(src[-1]) + 1 + int(ins[-1])
        x = self.parent_recording_segment.get_traces(lo, hi, slice(None)).astype(np.int32)
        x[x == -32768] = 0               # Trodes' "no sample" value counts as 0
        if self.ref_col is not None:
            x -= x[:, [self.ref_col]]
        out = x[src - lo]
        if ins.any():
            # Per gap, Trodes steps a float32 value from L towards R, re-aiming
            # at R each time: v += (R - v) / (samples left + 1).
            rows = np.flatnonzero(ins)
            for gp in np.unique(src[rows]):
                sel = rows[src[rows] == gp]
                mm = int(m[sel[0]])
                L = x[gp - lo].astype(np.float32)
                R = x[gp + 1 - lo].astype(np.float32)
                seq = np.empty((mm, L.size), dtype=np.float32)
                v = L
                for k in range(mm):
                    v = (v + (R - v) / np.float32(mm + 1 - k)).astype(np.float32)
                    seq[k] = v
                out[sel] = np.round(seq[off[sel] - 1].astype(np.float64))
        out = np.clip(out, -32768, 32767).astype(np.int16)   # saturates, like Trodes
        return out if channel_indices is None else out[:, channel_indices]


def _slot_of_hwchan(n_channels, chans_per_chip):
    """{hwChan: byte slot} for the Intan chip-interleaved packet layout."""
    n_chips = n_channels // chans_per_chip
    return {(i % n_chips) * chans_per_chip + i // n_chips: i for i in range(n_channels)}


def open_rec(rec_file, apply_raw_ref=True, interp=TRODES_INTERP):
    """A lazy SpikeInterface recording of one .rec that equals its ``-raw`` export.

    int16, unscaled (as the export), channel ids ``0..n-1`` in nTrode order, with
    properties ``ntrode`` and ``hw_chan``. ``apply_raw_ref=False`` skips the
    rawRef subtraction (true single-ended data, which the export never had).
    ``interp`` is trodesexport's ``-interp``: gaps of up to that many dropped
    packets are linearly interpolated (-1 = all, 0 = none); larger gaps are
    spliced shut, with a warning — after them ``sample / fs`` runs behind the
    Trodes clock (DIO, behaviour), exactly as it did with the raw export.
    """
    import spikeinterface.extractors as se

    rec_file = Path(rec_file)
    sconf = read_spike_configuration(rec_file)
    if sconf.attrib.get("device", "intan") != "intan":
        raise NotImplementedError(f"{rec_file.name}: only Intan headstages are handled")
    trodes = list(sconf)
    order = [(t, int(c.attrib["hwChan"])) for t in trodes for c in t]
    hw = [h for _, h in order]
    n = len(hw)

    rec = se.read_spikegadgets(str(rec_file), stream_id="trodes")
    if rec.get_num_channels() != n or sorted(hw) != list(range(n)):
        raise RuntimeError(f"{rec_file.name}: expected hwChans 0..{n - 1}, neo gives "
                           f"{rec.get_num_channels()} channels")

    gaps = find_packet_gaps(rec.neo_reader)
    fill = [(g, st - 1) for g, st in gaps if st > 1 and (interp < 0 or st - 1 <= interp)]
    filled = {g for g, _ in fill}
    left = [(g, st) for g, st in gaps if g not in filled]
    if fill:
        print(f"  {rec_file.name}: interpolating {len(fill)} dropped-packet gap(s) "
              f"({sum(m for _, m in fill)} samples), as trodesexport -interp {interp}")
    if left:
        warnings.warn(
            f"{rec_file.name}: {len(left)} timestamp gap(s)/repeat(s) left as is, as "
            f"trodesexport -interp {interp} does — (packet, timestamp step): "
            f"{left[:5]}. After a gap, sample/fs runs behind the Trodes clock.",
            stacklevel=2)
    # Output sample index of each left gap = packet index + samples interpolated
    # before it; `step - 1` samples of Trodes time are missing there (negative for a
    # repeat/reset, i.e. sample/fs then runs AHEAD of the Trodes clock).
    fill_idx = np.array([g for g, _ in fill], dtype=np.int64)
    fill_cum = np.cumsum([m for _, m in fill]) if fill else np.array([], dtype=np.int64)
    table = []
    for g, st in left:
        k = int(np.searchsorted(fill_idx, g))
        ins_before = int(fill_cum[k - 1]) if k > 0 else 0
        table.append(dict(packet=int(g), output_sample=int(g + ins_before + 1),
                          timestamp_step=int(st), missing_samples=int(st - 1)))
    SPLICED_GAPS[str(rec_file)] = table

    # neo lists the channels in byte-slot order. When the header's chanPerChip is
    # valid, neo has labelled each slot with its true hwChan; otherwise (our files:
    # chanPerChip="0") the labels are wrong, so take the slot from the chip layout.
    cpc = int(sconf.attrib.get("chanPerChip", INTAN_CHANS_PER_CHIP))
    if 0 < cpc <= n and n % cpc == 0:
        ids = [str(h) for h in hw]
    else:
        if n % INTAN_CHANS_PER_CHIP:
            raise RuntimeError(f"{rec_file.name}: chanPerChip={cpc} and {n} channels "
                               f"do not fit {INTAN_CHANS_PER_CHIP}-channel chips")
        slot = _slot_of_hwchan(n, INTAN_CHANS_PER_CHIP)
        neo_ids = list(rec.channel_ids)
        ids = [neo_ids[slot[h]] for h in hw]
    rec = rec.select_channels(ids)
    rec = rec.rename_channels(new_channel_ids=np.arange(n))
    rec.set_property("ntrode", [int(t.attrib["id"]) for t, _ in order])
    rec.set_property("hw_chan", hw)

    ref_col = None
    if apply_raw_ref:
        if any(t.attrib.get("groupRefOn", "0") == "1" for t in trodes):
            raise NotImplementedError(f"{rec_file.name}: groupRefOn=1 is not handled")
        refs = {(t.attrib.get("rawRefOn", "0"), t.attrib.get("refNTrodeID"),
                 t.attrib.get("refChan")) for t in trodes}
        if len(refs) != 1:
            raise NotImplementedError(f"{rec_file.name}: nTrodes use different raw "
                                      f"references {sorted(refs)}")
        on, ref_trode, ref_chan = refs.pop()
        if on == "1":
            trode = next(t for t in trodes if t.attrib["id"] == ref_trode)
            ref_col = hw.index(int(list(trode)[int(ref_chan) - 1].attrib["hwChan"]))

    return TrodesRawRecording(rec, ref_col=ref_col,
                              gap_index=[g for g, _ in fill],
                              gap_missing=[m for _, m in fill])


if __name__ == "__main__":
    # python trodes_rec.py <file.rec> [<file.raw_group0.dat>]  — info + a full
    # packet-gap scan, and an exact comparison against an existing raw export
    # (start / middle / end and around each gap). Exits 1 on any mismatch.
    import sys

    import spikeinterface.extractors as se

    rec_file = Path(sys.argv[1])
    rec = open_rec(rec_file)
    print(f"{rec_file.name}: {rec.get_num_channels()} ch, "
          f"{rec.get_num_samples()} samples @ {rec.get_sampling_frequency():g} Hz, "
          f"{rec.get_dtype()}")
    d = np.diff(packet_timestamps(se.read_spikegadgets(str(rec_file), stream_id="trodes")
                                  .neo_reader).astype(np.int64))
    gaps = np.flatnonzero(d != 1)
    print(f"  timestamp gaps/repeats: {gaps.size}"
          + (f" ({int((d[d > 1] - 1).sum())} packets missing)" if gaps.size else ""))
    if len(sys.argv) > 2:
        from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile

        dat = readTrodesExtractedDataFile(sys.argv[2], memmap=True)["data"]["voltage"]
        n, total = 300_000, dat.shape[0]
        print(f"  frames rec/dat: {rec.get_num_samples()} / {total}")
        if rec.get_num_samples() != total:
            sys.exit(f"  MISMATCH: open_rec gives {rec.get_num_samples()} samples, the "
                     f"raw export {total} (was it exported with a non-default -interp?)")
        # Around each gap: its output position is the packet index plus the
        # samples inserted before it.
        ins_before = np.concatenate([[0], np.cumsum(np.where((d > 1) & (d <= TRODES_INTERP + 1),
                                                             d - 1, 0))])
        at_gaps = [int(g + ins_before[g]) for g in gaps]
        starts = [0, total // 2, max(total - n, 0)] + [max(a - 1000, 0) for a in at_gaps]
        ok = True
        for start in starts:
            stop = min(start + (n if start in starts[:3] else 2000), total)
            same = np.array_equal(rec.get_traces(start_frame=start, end_frame=stop),
                                  np.asarray(dat[start:stop]))
            ok &= same
            if not same or start in starts[:3]:
                print(f"  samples {start}..{stop}: identical={same}")
        if at_gaps:
            print(f"  2000 samples around each of the {len(at_gaps)} gaps checked")
        if not ok:
            sys.exit("  MISMATCH: samples differ from the raw export.")
        print("  OK: identical to the raw export")
