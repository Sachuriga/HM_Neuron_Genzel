"""
Shared spike/waveform metrics + putative cell-type classifier.

Pure numpy/scipy (no matplotlib) so both add_units.py (step u, which PERSISTS the
metrics into the NWB units table) and visualize_nwb.py (step v, which shows them)
compute them with ONE implementation — the stored and the displayed values can
never drift apart.

Cell-type rule (CellExplorer putative classification + a firing-rate gate,
cellexplorer.org/pipeline/cell-type-classification):
    FR > 10 Hz                                          -> interneuron (FR gate)
    else troughToPeak <= 0.425 ms                       -> narrow interneuron
    else troughToPeak > 0.425 ms & acg_tau_rise > 6 ms  -> wide interneuron
    else                                                -> pyramidal
"""

import numpy as np
from scipy.optimize import curve_fit

TROUGH_PEAK_THRESH_S = 0.425e-3     # 0.425 ms trough-to-peak
ACG_TAU_RISE_THRESH_MS = 6.0        # acg_tau_rise threshold for wide interneurons
RATE_THRESH_HZ = 10.0               # firing-rate gate


def waveform_metrics(wf, fs):
    """Waveform metrics from a mean template (n_samples, n_channels) on the peak
    channel (largest peak-to-peak). Durations in SECONDS. Returns {} if unusable."""
    w = np.asarray(wf, dtype=float)
    if w.ndim == 1:
        w = w[:, None]
    if w.size == 0 or not np.isfinite(w).all():
        return {}
    ch = int(np.argmax(w.ptp(axis=0)))
    s = w[:, ch]
    n = s.size
    if n < 4 or s.ptp() == 0:
        return {}
    dt = 1.0 / fs
    trough_i = int(np.argmin(s)); trough_v = float(s[trough_i])
    peak_i = trough_i + int(np.argmax(s[trough_i:])) if trough_i < n - 1 else trough_i
    peak_v = float(s[peak_i])

    def _half_width(center_i, level, below):
        cond = (s <= level) if below else (s >= level)
        l = r = center_i
        while l - 1 >= 0 and cond[l - 1]:
            l -= 1
        while r + 1 < n and cond[r + 1]:
            r += 1
        return (r - l) * dt

    return {"peak_channel": ch,
            "peak_to_trough_s": (peak_i - trough_i) * dt,
            "trough_half_width_s": _half_width(trough_i, trough_v / 2.0, True) if trough_v < 0 else np.nan,
            "peak_half_width_s": _half_width(peak_i, peak_v / 2.0, False) if peak_v > 0 else np.nan}


def autocorrelogram(spike_times, window_s, bin_s, max_spikes=50000):
    """Spike-train autocorrelogram: counts of inter-spike lags in [-window,+window]
    (zero lag excluded). Returns (counts, bin-centres in ms)."""
    st = np.sort(np.asarray(spike_times, dtype=float))
    if st.size > max_spikes:                     # cap for runtime on fast cells
        st = np.sort(np.random.choice(st, max_spikes, replace=False))
    n = st.size
    if n < 2:
        return None, None
    lo = np.searchsorted(st, st - window_s, side="left")
    hi = np.searchsorted(st, st + window_s, side="right")
    diffs = [st[lo[i]:hi[i]] - st[i] for i in range(n)]
    d = np.concatenate(diffs)
    d = d[d != 0]
    edges = np.arange(-window_s, window_s + bin_s, bin_s)
    counts, edges = np.histogram(d, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2 * 1000.0   # ms
    return counts, centres


def _acg_triple_exp(x, tau_decay, tau_rise, c, d, asymptote, tau_burst, h, t0):
    """CellExplorer triple-exponential ACG model (fit_ACG). x in ms."""
    return np.maximum(
        c * (np.exp(-(x - t0) / tau_decay) - d * np.exp(-(x - t0) / tau_rise))
        + h * np.exp(-((x - t0) ** 2) / (tau_burst ** 2)) + asymptote, 0.0)


def acg_tau_rise(spike_times):
    """ACG rise-time constant (ms) via CellExplorer's triple-exponential fit to the
    narrow (0–50 ms, 0.5 ms bins) autocorrelogram. NaN if the fit fails."""
    counts, centres = autocorrelogram(spike_times, window_s=0.05, bin_s=0.0005)
    if counts is None:
        return np.nan
    pos = centres > 0
    x, y = centres[pos], counts[pos].astype(float)
    if y.size < 10 or y.max() <= 0:
        return np.nan
    y = y / y.max()                     # normalise amplitude
    #     tau_decay tau_rise  c    d  asymp tau_burst  h    t0
    p0 = [20.0, 1.0, 1.0, 2.0, 0.1, 1.5, 0.1, 1.0]
    lb = [1.0, 0.1, 0.0, 0.0, -1.0, 0.1, 0.0, 0.0]
    ub = [500.0, 50.0, 10.0, 15.0, 2.0, 5.0, 10.0, 20.0]
    try:
        popt, _ = curve_fit(_acg_triple_exp, x, y, p0=p0, bounds=(lb, ub), maxfev=10000)
        return float(popt[1])           # tau_rise
    except Exception:
        return np.nan


def classify_cell_type(firing_rate_hz, trough_to_peak_s, tau_rise_ms):
    """CellExplorer putative type + FR gate. Returns 'interneuron' | 'pyramidal'."""
    t2p = trough_to_peak_s
    narrow = t2p is not None and np.isfinite(t2p) and t2p <= TROUGH_PEAK_THRESH_S
    wide = (t2p is not None and np.isfinite(t2p) and t2p > TROUGH_PEAK_THRESH_S
            and tau_rise_ms is not None and np.isfinite(tau_rise_ms)
            and tau_rise_ms > ACG_TAU_RISE_THRESH_MS)
    is_int = (firing_rate_hz > RATE_THRESH_HZ) or narrow or wide
    return "interneuron" if is_int else "pyramidal"
