"""EMG-from-LFP — muscle tone estimated from LFP alone (Buzsáki/Schomburg method).

No physical EMG electrode is needed: genuine high-frequency brain activity is
spatially *local*, whereas volume-conducted muscle activity (EMG) appears
*correlated* across distant electrodes. So the zero-lag correlation of the
high-frequency band across spatially separated channels tracks muscle tone.

Faithful Python port of ``anjal_sleepscore`` ``compute_emg_buzsakiMethod.m``
(itself a reimplementation of buzcode's ``bz_EMGFromLFP``), generalised to N
channels and adapted for an arbitrary sampling rate (the reference assumes
1250 Hz; our HM LFP is 1000 Hz, so the high-freq band is capped below Nyquist).

Reference: Schomburg et al. 2014; Watson et al. 2016 (Buzsáki lab).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt, decimate


# Reference band edges at high fs: [Fstop1 Fpass1 Fpass2 Fstop2] = [275 300 600 625].
# We keep Fpass1 = 300 and cap the upper edge safely below Nyquist for low fs.
_PASS_LO_HZ = 300.0
_PASS_HI_REF_HZ = 600.0


def _highfreq_passband(fs: float) -> tuple[float, float]:
    """High-frequency band-pass passband (Hz), capped below Nyquist.

    At fs >= ~1252 Hz this is the reference 300-600 Hz. At fs = 1000 Hz
    (Nyquist 500) it becomes ~300-475 Hz.
    """
    nyq = fs / 2.0
    hi = min(_PASS_HI_REF_HZ, np.floor(nyq) - 25.0)
    lo = _PASS_LO_HZ
    if hi <= lo:
        raise ValueError(f"fs={fs} Hz too low for an EMG high-freq band "
                         f"(need Nyquist well above {lo} Hz).")
    return lo, hi


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average of length ``win`` (matches MATLAB smooth(...,'moving'))."""
    win = max(1, int(win))
    if win == 1 or x.size == 0:
        return x.copy()
    # MATLAB 'moving' uses an odd span and shrinking windows at the edges.
    if win % 2 == 0:
        win -= 1
    half = win // 2
    out = np.empty_like(x, dtype=np.float64)
    csum = np.concatenate([[0.0], np.cumsum(x)])
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        out[i] = (csum[hi] - csum[lo]) / (hi - lo)
    return out


def emg_from_lfp(channels: np.ndarray, fs: float, emg_fs: float = 5.0,
                 smooth_window_s: float = 10.0, filt_order: int = 4):
    """Estimate an EMG signal from multiple spatially separated LFP channels.

    Parameters
    ----------
    channels : (n_samples, n_channels) LFP at ``fs`` Hz. Needs >= 2 channels,
        ideally spread across the probe (correlation of *distant* sites is what
        isolates the volume-conducted muscle signal).
    fs : LFP sampling rate (Hz).
    emg_fs : output EMG sampling rate (Hz). Default 5 Hz (0.2 s bins), as in the
        reference; each bin correlates a +-0.2 s window.
    smooth_window_s : moving-average smoothing window (seconds) on the EMG.
    filt_order : Butterworth order for the high-freq band-pass (applied
        zero-phase via filtfilt, i.e. forward+backward like the MATLAB).

    Returns
    -------
    dict with:
        timestamps : (n_bins,) bin-centre times (s)
        data       : (n_bins,) raw mean pairwise correlation (EMG)
        norm       : (n_bins,) data min/max-normalised to [0, 1]
        smoothed   : (n_bins,) moving-average-smoothed ``data``
        passband   : (lo, hi) high-freq band used (Hz)
    """
    x = np.asarray(channels, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n_ch = x.shape[1]
    if n_ch < 2:
        raise ValueError("EMG-from-LFP needs >= 2 channels.")

    # 1. High-frequency band-pass (zero-phase), per channel.
    lo, hi = _highfreq_passband(fs)
    b, a = butter(filt_order, [lo, hi], btype="band", fs=fs)
    filt = np.empty_like(x)
    for c in range(n_ch):
        filt[:, c] = filtfilt(b, a, x[:, c])

    # 2. Sliding-window pairwise Pearson correlation.
    out = emg_correlation(filt, fs, emg_fs=emg_fs, smooth_window_s=smooth_window_s)
    out["passband"] = (lo, hi)
    return out


def emg_correlation(filtered_channels: np.ndarray, fs: float, emg_fs: float = 5.0,
                    smooth_window_s: float = 10.0):
    """Sliding-window mean pairwise Pearson correlation of already-band-passed channels.

    Split out from :func:`emg_from_lfp` so the (expensive, memory-heavy) high-freq
    band-pass can be done once per channel up front (e.g. while decimating a raw
    30 kHz channel) and the cheap correlation run on the reduced signals.

    Parameters
    ----------
    filtered_channels : (n_samples, n_channels) high-freq-band-passed LFP at ``fs`` Hz.
    fs : sampling rate of ``filtered_channels`` (Hz).

    Returns the same dict as :func:`emg_from_lfp` (minus ``passband``).
    """
    f = np.asarray(filtered_channels, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    n_samples, n_ch = f.shape
    if n_ch < 2:
        raise ValueError("EMG-from-LFP needs >= 2 channels.")

    bin_scoot_s = 1.0 / emg_fs
    bin_scoot_samps = int(round(fs * bin_scoot_s))
    w = int(round(bin_scoot_s * fs))                 # half-window in samples
    win = np.arange(-w, w + 1)
    centres = np.arange(w, n_samples - w, bin_scoot_samps)   # bin centres (sample idx)
    n_bins = centres.size
    if n_bins == 0:
        raise ValueError("Recording too short for the chosen emg_fs/window.")

    # Mean pairwise Pearson correlation over all channel pairs, computed via the
    # identity  mean_{i<j} u_i . u_j == (||sum_c u_c||^2 - n) / (n*(n-1)),
    # where u_c are the demeaned, unit-normalised window columns. This is exact
    # and O(win * n_ch) per window instead of O(win * n_ch^2), so it scales to
    # the 32 channels selected among 128.
    emg = np.zeros(n_bins)
    for k, t in enumerate(centres):
        seg = f[t + win, :]                          # (win_len, n_ch)
        s = seg - seg.mean(axis=0)                   # demean -> Pearson
        denom = np.sqrt((s ** 2).sum(axis=0))
        valid = denom > 0
        n_eff = int(valid.sum())
        if n_eff < 2:                                # window too flat to correlate
            emg[k] = 0.0
            continue
        u = s[:, valid] / denom[valid]               # unit-norm columns
        g = float((u.sum(axis=1) ** 2).sum())        # ||sum_c u_c||^2
        emg[k] = (g - n_eff) / (n_eff * (n_eff - 1))

    rng = emg.max() - emg.min()
    return {
        "timestamps": centres / fs,
        "data": emg,
        "norm": (emg - emg.min()) / (rng + 1e-12),
        "smoothed": _moving_average(emg, round(smooth_window_s * emg_fs)),
    }


def bandpass_decimate(raw: np.ndarray, src_fs: float, work_fs: float = 1500.0,
                      band: tuple[float, float] | None = None, filt_order: int = 4):
    """Band-pass a raw wideband channel into the EMG band and decimate it.

    Built for the raw 30 kHz path: decimate to ``work_fs`` first (anti-aliased,
    in stages), then apply the high-freq band-pass on the much smaller decimated
    signal. Decimating to ``work_fs`` (Nyquist ``work_fs/2``) is safe as long as
    the band's upper edge stays below that Nyquist — 1500 Hz (Nyquist 750) keeps
    the full 300-600 Hz EMG band.

    Parameters
    ----------
    raw : 1-D raw channel at ``src_fs`` (e.g. int16 samples, any numeric dtype).
    src_fs : raw sampling rate (Hz), e.g. 30000.
    work_fs : target rate for the correlation (Hz). Must satisfy work_fs/2 > band hi.
    band : (lo, hi) EMG band; defaults to the fs-adaptive high-freq passband.

    Returns ``(sig, work_fs_actual)`` where ``sig`` is 1-D at ``work_fs_actual``.
    """
    x = np.asarray(raw, dtype=np.float32)
    q_total = src_fs / work_fs
    if q_total < 1.5:
        work_fs_actual = float(src_fs)
    else:
        # Decimate in stages (scipy recommends q <= 13 per call for FIR).
        q_total = int(round(q_total))
        factors = _factorize_decimation(q_total)
        for q in factors:
            x = decimate(x, q, ftype="fir")
        work_fs_actual = src_fs / np.prod(factors)

    lo, hi = band if band is not None else _highfreq_passband(work_fs_actual)
    sos = butter(filt_order, [lo, hi], btype="band", fs=work_fs_actual, output="sos")
    x = sosfiltfilt(sos, x).astype(np.float32)
    return x, work_fs_actual


def _factorize_decimation(q: int, max_step: int = 13):
    """Split a decimation factor into stages each <= max_step (for scipy.decimate)."""
    factors = []
    for p in (2, 3, 5, 7, 11, 13):
        while q % p == 0 and p <= max_step:
            factors.append(p)
            q //= p
    if q > 1:                       # leftover prime > max_step: take as-is
        factors.append(q)
    return factors or [1]


def pick_spread_channels(n_available: int, good_indices=None, n_pick: int = 4):
    """Choose channels spread across the probe for the EMG correlation.

    The method wants spatially *separated* channels. Given the good channels the
    tracker selected (``good_indices``, e.g. cleanest-by-SNR), keep them but, if
    too few, top up with an even spread across all channels so the pairs span
    the probe. Returns a sorted list of channel indices.
    """
    idx = set(int(i) for i in (good_indices if good_indices is not None else []))
    if len(idx) < n_pick and n_available > 0:
        spread = np.unique(np.linspace(0, n_available - 1, n_pick).round().astype(int))
        idx |= set(int(i) for i in spread)
    return sorted(i for i in idx if 0 <= i < n_available)
