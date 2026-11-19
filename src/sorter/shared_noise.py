"""Remove noise that the tetrodes share with a DIFFERENT gain on each channel.

WHY
---
A common reference (single channel, or the global median) subtracts ONE waveform
with gain 1 from every channel. Interference that reaches the tetrodes with
different gains survives it: on Rat6 20260722 a ~123 Hz pulse train of 150-400 uV
came in 0.4x on some tetrodes and 2.4x on others, so the global median left
30-65 uV of it on a 13 uV noise floor, and 18 % of the sorted spikes were that
artifact (the same 0.5 ms on 8 or more tetrodes).

WHAT
----
After the global median, the data of each 1 s block is modelled as a few shared
components   y[t, c] = w[c] * z[t]   on top of the spikes and the channel noise:

  w, the spatial pattern, is an eigenvector of the BETWEEN-tetrode covariance.
     The covariances inside a tetrode are not used, so the units of a tetrode
     cannot create or bend a pattern: a spike lives on one tetrode, and shared
     noise is, by definition, what different tetrodes have in common.

  z, the time course, is at every sample the L1 fit of the channels on w: the
     weighted median of y[t, c] / w[c] with weights |w[c]|. A spike sits on the
     4 channels of one tetrode, so it cannot move the median of ~120 ratios -
     the same reason a common MEDIAN reference is safer than a common average.

A block is cleaned only when its strongest pattern is strong enough to be real
interference (TRIGGER_STRENGTH) and spread over many tetrodes
(MIN_TETRODE_FRACTION); otherwise the block comes back unchanged, sample for
sample. So a clean recording passes through (nearly) untouched, and the setting
can stay on for every session.

Do not reference a tetrode against its own channels instead (a per-tetrode
median): its 4 wires see the same neurons, so that subtracts the spike itself
(a 100/70/50/30 % spike becomes 40/10/-10/-30 %).

Checked with 2 components on 24 x 10 s of each of three Rat6 maze sessions
("artifact event" = more than 20 % of the channels over 5 SD at once, after the
global median; spikes of 80 and 300 uV injected on one tetrode):

                 artifact events/min   noise uV      blocks    injected spike kept
                 before -> after       before->after cleaned   (median / worst 1 %)
    20260722     2744 -> 158           12.6 -> 10.0   47 %     99.4-99.8 % / 90-96 %
    20260723      928 -> 154           10.6 -> 10.1   24 %     99.5-99.9 % / 93-98 %
    20260727       27 ->  22           10.2 -> 10.1    5 %     99.6 %      / 97 %
    (global median alone keeps 99.6-99.9 % / 95-99 %.)

More than 2 components removed nothing further on these sessions. The cost is in
the worst 1 %: a spike on the tetrode that carries a pattern most strongly can lose
up to ~10 uV in a cleaned block (the influence of a median is bounded, so this
does not grow with the spike).

Sorted end to end (MountainSort4, the 10 worst minutes of 20260722; a unit counts
as artifact when its spike-triggered average is also on 3 or more OTHER tetrodes):
artifact units 647 -> 91, neuron units 89 -> 128, spikes in neuron units
473 k -> 953 k; the neurons keep 99 % of their amplitude (worst unit 91 %).

It is not a complete cure. In such a stretch the average pulse goes from ~120 uV
to ~15 uV per channel, but single pulses differ in shape and timing between
tetrodes, so enough is left to form small clusters (51 of those 91 were labelled
good). Fitting each channel on the time course shifted by a few samples halves
the average again (not implemented); the real fix is at the source.
"""

import numpy as np
from spikeinterface.preprocessing.basepreprocessor import (BasePreprocessor,
                                                           BasePreprocessorSegment)

__version__ = "1.0"         # SpikeInterface needs it to rebuild the recording in workers

BLOCK_S = 1.0               # the patterns are fitted anew for every block of this length
# Strength of a pattern = its variance summed over the channels / the variance of one
# typical channel. A block is cleaned only when its strongest pattern reaches
# TRIGGER_STRENGTH; then every pattern of at least MIN_STRENGTH is removed (up to
# max_components). Measured per 1 s block on Rat6: under 8 the global median leaves no
# artifact events, 12-16 a few to ~20 per second, over 100 ~140 per second; on a clean
# day (20260727) 95 % of the blocks are under 12 and none is over 20.
TRIGGER_STRENGTH = 12.0
MIN_STRENGTH = 2.0
MIN_TETRODE_FRACTION = 0.25  # a pattern must be spread over at least this share of the tetrodes
MIN_TETRODES = 8            # with fewer tetrodes "shared by many" means nothing: do nothing
FACTOR_ITERATIONS = 3


def weighted_median_score(y, w):
    """The time course of pattern ``w`` in ``y`` (samples x channels).

    For every sample t:  z[t] = argmin_z  sum_c |y[t, c] - w[c] * z|,
    i.e. the weighted median of y[t, c] / w[c] with weights |w[c]|.
    """
    use = np.abs(w) > 1e-3 * np.abs(w).max()
    ratio = y[:, use] / w[use]
    order = np.argsort(ratio, axis=1)       # ties share one value, so their order is irrelevant
    weight_sum = np.cumsum(np.abs(w[use]).astype(np.float64)[order], axis=1)
    half = 0.5 * weight_sum[:, -1:]
    pick = (weight_sum >= half).argmax(axis=1)                 # first channel past half the weight
    column = np.take_along_axis(order, pick[:, None], axis=1)
    return np.take_along_axis(ratio, column, axis=1)[:, 0]


def shared_patterns(y, groups, max_components):
    """The shared-noise patterns of one block: ``(patterns, strengths)``.

    ``y``       samples x channels, already median-referenced
    ``groups``  the tetrode number (0..n-1) of every channel
    ``patterns``  list of unit vectors (one value per channel), strongest first
    ``strengths`` their variance, in units of one typical channel's variance

    The within-tetrode blocks of the covariance are not data here: they start at
    zero and are filled in from the low-rank fit (principal-factor iteration), so
    a tetrode with a big gain is not under-estimated.
    """
    y = np.asarray(y, dtype=np.float32)
    covariance = (y.T @ y).astype(np.float64) / max(y.shape[0], 1)
    channel_variance = float(np.median(np.diag(covariance)))
    if channel_variance <= 0:
        return [], []

    same_tetrode = groups[:, None] == groups[None, :]
    work = np.where(same_tetrode, 0.0, covariance)
    for _ in range(FACTOR_ITERATIONS + 1):
        values, vectors = np.linalg.eigh(work)
        values = values[::-1][:max_components]
        vectors = vectors[:, ::-1][:, :max_components]
        low_rank = (vectors * np.clip(values, 0, None)) @ vectors.T
        work = np.where(same_tetrode, low_rank, covariance)

    n_tetrodes = int(groups.max()) + 1
    patterns, strengths = [], []
    for k in range(vectors.shape[1]):
        strength = values[k] / channel_variance
        if strength < MIN_STRENGTH:
            break                                   # the rest is weaker still
        w = vectors[:, k]
        share = np.bincount(groups, weights=np.abs(w), minlength=n_tetrodes) / np.abs(w).sum()
        tetrodes_carrying_it = 1.0 / (share ** 2).sum()
        if tetrodes_carrying_it < MIN_TETRODE_FRACTION * n_tetrodes:
            continue                                # lives on a few tetrodes: not shared noise
        patterns.append(w.astype(np.float32))
        strengths.append(float(strength))
    if not strengths or strengths[0] < TRIGGER_STRENGTH:
        return [], []                               # no real interference in this block
    return patterns, strengths


def remove_shared_noise(y, groups, max_components=2):
    """Clean one block. Returns ``(cleaned float32, strengths of the patterns removed)``.

    ``y`` must already be median-referenced (the global median is the first,
    gain-1 component; this removes what it leaves behind).
    """
    y = np.asarray(y, dtype=np.float32)
    patterns, strengths = shared_patterns(y, groups, max_components)
    for w in patterns:
        y = y - weighted_median_score(y, w)[:, None] * w[None, :]
    return y, strengths


class SharedNoiseRemovalRecording(BasePreprocessor):
    """SpikeInterface step: remove_shared_noise on fixed 1 s blocks.

    Put it AFTER the global median reference. The tetrode of every channel is read
    from the recording's ``group`` property.

    The blocks are fixed on the recording's sample axis (block k = samples
    [k * block, (k + 1) * block); the last block also takes the remainder), and a
    request is always answered from whole blocks. So the output does not depend
    on how the recording is read: saving it in other chunk sizes, or reading any
    window of it lazily, gives the same samples.
    """

    def __init__(self, recording, max_components=2, block_s=BLOCK_S):
        BasePreprocessor.__init__(self, recording)
        groups = recording.get_property("group")
        if groups is None:
            raise ValueError("SharedNoiseRemovalRecording needs the 'group' (tetrode) property")
        _, groups = np.unique(np.asarray(groups), return_inverse=True)
        block = max(int(round(block_s * recording.get_sampling_frequency())), 1)
        enabled = max_components > 0 and (int(groups.max()) + 1) >= MIN_TETRODES
        for parent_segment in recording._recording_segments:
            self.add_recording_segment(SharedNoiseRemovalSegment(
                parent_segment, groups, int(max_components) if enabled else 0, block,
                np.dtype(recording.get_dtype())))
        self._kwargs = dict(recording=recording, max_components=int(max_components),
                            block_s=float(block_s))

    def removed_summary(self):
        """One line on what was removed in the blocks read so far (for the log)."""
        strengths = [s for seg in self._recording_segments for s in seg.removed.values()]
        if not strengths:
            return "Shared noise: no block read yet."
        n_blocks = len(strengths)
        with_noise = [s for s in strengths if s]
        if not with_noise:
            return f"Shared noise: no interference found in {n_blocks} block(s); nothing removed."
        top = np.array([s[0] for s in with_noise])
        heavy = int((top > 100).sum())
        return (f"Shared noise: interference found and removed in {len(with_noise)} of {n_blocks} "
                f"blocks ({100 * len(with_noise) / n_blocks:.0f} % of the recording). In those, the "
                f"strongest pattern was {np.median(top):.0f}x (median) / {top.max():.0f}x (max) one "
                f"channel's noise variance; {heavy} block(s) = {100 * heavy / n_blocks:.0f} % over "
                f"100x (heavy interference).")


class SharedNoiseRemovalSegment(BasePreprocessorSegment):
    def __init__(self, parent_segment, groups, max_components, block, dtype):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self.groups = groups
        self.max_components = max_components
        self.block = block
        self.dtype = dtype
        self.removed = {}            # block number -> strengths of the patterns removed

    def get_traces(self, start_frame, end_frame, channel_indices):
        parent = self.parent_recording_segment
        n_samples = parent.get_num_samples()
        start_frame = 0 if start_frame is None else int(start_frame)
        end_frame = n_samples if end_frame is None else int(end_frame)
        if self.max_components <= 0 or end_frame <= start_frame:
            return parent.get_traces(start_frame, end_frame, channel_indices)

        n_blocks = max(n_samples // self.block, 1)       # the last block takes the remainder
        first = min(start_frame // self.block, n_blocks - 1)
        last = min((end_frame - 1) // self.block, n_blocks - 1)
        pieces = []
        for k in range(first, last + 1):
            block_start = k * self.block
            block_end = n_samples if k == n_blocks - 1 else block_start + self.block
            traces = parent.get_traces(block_start, block_end, slice(None))
            cleaned, strengths = remove_shared_noise(traces, self.groups, self.max_components)
            self.removed[k] = strengths
            if not strengths:
                cleaned = traces                         # untouched: exactly the input samples
            elif self.dtype.kind in "iu":
                info = np.iinfo(self.dtype)
                cleaned = np.clip(np.rint(cleaned), info.min, info.max)
            lo = max(start_frame, block_start) - block_start
            hi = min(end_frame, block_end) - block_start
            pieces.append(np.asarray(cleaned[lo:hi]).astype(self.dtype, copy=False))
        out = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        return out if channel_indices is None else out[:, channel_indices]


def remove_shared_noise_recording(recording, max_components=2, block_s=BLOCK_S):
    """The recording with its shared noise removed (lazy); see the module text."""
    return SharedNoiseRemovalRecording(recording, max_components=max_components, block_s=block_s)
