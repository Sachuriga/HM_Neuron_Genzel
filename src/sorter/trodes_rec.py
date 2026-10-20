"""Read a Trodes ``.rec`` straight into SpikeInterface, identical to ``trodesexport -raw``.

Step 1 used to write ``<recording>.raw/*_group0.dat`` — a second full copy of
the wideband data (~27 GiB an hour). ``open_rec`` gives the same int16 samples,
in the same column order, lazily from the ``.rec`` itself, so that export is no
longer needed. Checked bit for bit against ``_group0.dat`` (Rat5 20260720
pre + post, start/middle/end).

Two things ``se.read_spikegadgets`` (neo >= 0.14.5) does not do for our files:

1. Channel order. The MCU interleaves the Intan chips in each packet: byte pair
   ``i`` holds hwChan ``(i % n_chips) * 32 + i // n_chips``. neo knows this, but
   only applies it when the header's ``chanPerChip`` is valid; our HH128 headers
   say ``chanPerChip="0"``, so neo falls back to hwChan-ascending and every
   channel label points at the wrong data. We pick the channels by byte slot.
2. Reference. With ``rawRefOn="1"`` the raw export is ``channel - ref channel``
   (``refNTrodeID`` / ``refChan``), so we subtract it the same way.

Columns come out in SpikeConfiguration document order (nTrode 1 ch 1..4,
nTrode 2 ch 1..4, ...) with channel ids ``0..n-1`` — the export's column ids,
i.e. nTrode ``k//4 + 1``, channel ``k%4 + 1``.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

INTAN_CHANS_PER_CHIP = 32   # RHD2132; used when the header's chanPerChip is unusable


def find_rec_files(folder, recursive=False):
    """Trodes ``.rec`` files in ``folder``, sorted by name (== chronological, the
    Trodes name embeds YYYYMMDD_HHMMSS). Skips macOS ``._`` sidecar files."""
    base = Path(folder)
    pattern = "**/*.rec" if recursive else "*.rec"
    return sorted((f for f in base.glob(pattern)
                   if f.is_file() and not f.name.startswith("._")),
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


def _slot_of_hwchan(n_channels, chans_per_chip):
    """{hwChan: byte slot} for the Intan chip-interleaved packet layout."""
    n_chips = n_channels // chans_per_chip
    return {(i % n_chips) * chans_per_chip + i // n_chips: i for i in range(n_channels)}


def open_rec(rec_file, apply_raw_ref=True):
    """A lazy SpikeInterface recording of one .rec that equals its ``-raw`` export.

    int16, unscaled (as the export), channel ids ``0..n-1`` in nTrode order, with
    properties ``ntrode`` and ``hw_chan``. ``apply_raw_ref=False`` skips the
    rawRef subtraction (true single-ended data, which the export never had).
    """
    import spikeinterface.extractors as se
    import spikeinterface.preprocessing as spre

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
            ref_hw = int(list(trode)[int(ref_chan) - 1].attrib["hwChan"])
            rec = spre.common_reference(rec, reference="single",
                                        ref_channel_ids=[hw.index(ref_hw)])
    return rec


if __name__ == "__main__":
    # python trodes_rec.py <file.rec> [<file.raw_group0.dat>]  — info, and an
    # exact comparison against an existing raw export if one is given.
    import sys

    rec = open_rec(sys.argv[1])
    print(f"{Path(sys.argv[1]).name}: {rec.get_num_channels()} ch, "
          f"{rec.get_num_samples()} samples @ {rec.get_sampling_frequency():g} Hz, "
          f"{rec.get_dtype()}")
    if len(sys.argv) > 2:
        from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile

        dat = readTrodesExtractedDataFile(sys.argv[2], memmap=True)["data"]["voltage"]
        n, total = 300_000, dat.shape[0]
        print(f"  frames rec/dat: {rec.get_num_samples()} / {total}")
        for start in (0, total // 2, max(total - n, 0)):
            same = np.array_equal(rec.get_traces(start_frame=start, end_frame=start + n),
                                  np.asarray(dat[start:start + n]))
            print(f"  samples {start}..{start + n}: identical={same}")
