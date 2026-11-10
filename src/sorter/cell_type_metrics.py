"""Putative cell type of a cluster, with CellExplorer's standard classification.

The same code as the phy_genzel fork's phy/apps/cell_type.py (the live
trough_to_peak / acg_tau_rise / cell_type columns of the Template GUI), copied
here so the pipeline does not need phy installed. numpy + scipy only.

https://cellexplorer.org/pipeline/cell-type-classification/ and CellExplorer's
+celltype_classification/standard.m:

1. Every cell starts as a ``Pyramidal Cell``.
2. ``Narrow Interneuron`` if troughToPeak <= 0.425 ms.
3. ``Wide Interneuron`` if troughToPeak > 0.425 ms and acg_tau_rise > 6 ms.

troughToPeak follows calc_waveform_metrics.m: on the peak channel of the mean
waveform (band-passed 500-8000 Hz, as in getWaveformsFromDat.m), the time from
the trough, searched within +-0.25 ms of the spike time, to the largest value
after it, within +-0.8 ms of the spike time, after a spline interpolation to at
least 100 kHz.

acg_tau_rise follows calc_ACG_metrics.m and fit_ACG.m: the triple-exponential
fit of the 0.5-50 ms half of the autocorrelogram (0.5 ms bins, in spikes/s),
with the -0.5, 0 and +0.5 ms bins set to zero.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt

NARROW_INTERNEURON = "Narrow Interneuron"
WIDE_INTERNEURON = "Wide Interneuron"
PYRAMIDAL_CELL = "Pyramidal Cell"

TROUGH_TO_PEAK_BOUNDARY_MS = 0.425  # preferences.putativeCellType.troughToPeak_boundary
ACG_TAU_RISE_BOUNDARY_MS = 6.0      # preferences.putativeCellType.acg_tau_rise_boundary

# Waveforms (getWaveformsFromDat.m, preferences_ProcessCellMetrics.m).
FILTER_BAND_HZ = (500.0, 8000.0)
WAVEFORM_WINDOW_S = 0.004       # total width extracted, so filtering has no edge effect
WAVEFORM_HALF_WIDTH_S = 0.0008  # half width kept for the metrics
TROUGH_SEARCH_MS = 0.25
MIN_INTERPOLATED_RATE = 100000.0

# Autocorrelogram fit (fit_ACG.m), parameters in the order
# tau_decay, tau_rise, c, d, asymptote, refrac, tau_burst, h.
ACG_BIN_S = 0.0005
ACG_N_BINS = 100  # 0.5 .. 50 ms
ACG_FIT_START = (20.0, 1.0, 30.0, 2.0, 0.5, 5.0, 1.5, 2.0)
ACG_FIT_LOWER = (1.0, 0.1, 0.0, 0.0, -30.0, 0.0, 0.1, 0.0)
ACG_FIT_UPPER = (500.0, 50.0, 500.0, 15.0, 50.0, 20.0, 5.0, 100.0)


def bandpass_filter(waveforms, sample_rate, axis=0):
    """CellExplorer's 3rd-order Butterworth 500-8000 Hz zero-phase filter; a
    500 Hz high-pass when 8000 Hz is above the Nyquist frequency."""
    nyquist = sample_rate / 2.0
    low, high = FILTER_BAND_HZ
    if high < nyquist:
        b, a = butter(3, [low / nyquist, high / nyquist], "bandpass")
    else:
        b, a = butter(3, low / nyquist, "high")
    padlen = 3 * (max(len(a), len(b)) - 1)  # MATLAB's filtfilt padding
    if waveforms.shape[axis] <= padlen:
        return waveforms
    return filtfilt(b, a, waveforms, axis=axis, padlen=padlen)


def trough_to_peak(waveform, sample_rate, spike_index):
    """CellExplorer's troughToPeak, in ms, of a 1D mean waveform whose sample
    `spike_index` is the spike time. NaN if it cannot be measured."""
    w = np.asarray(waveform, dtype=np.float64)
    k = int(np.ceil(WAVEFORM_HALF_WIDTH_S * sample_rate))
    lo, hi = max(0, spike_index - k), min(len(w), spike_index + k)
    w = w[lo:hi]
    if len(w) < 4 or not np.all(np.isfinite(w)) or np.ptp(w) == 0:
        return np.nan
    t = (np.arange(lo, hi) - spike_index) / sample_rate * 1000.0  # ms
    oversampling = int(np.ceil(MIN_INTERPOLATED_RATE / sample_rate))
    rate = oversampling * sample_rate
    t_fine = t[0] + np.arange((len(t) - 1) * oversampling + 1) / rate * 1000.0
    wave = CubicSpline(t, w)(t_fine)
    i1 = int(np.searchsorted(t_fine, -TROUGH_SEARCH_MS))
    i2 = int(np.searchsorted(t_fine, TROUGH_SEARCH_MS))
    if i2 >= len(wave) or i1 >= i2:
        return np.nan
    inside = wave[i1:i2 + 1]
    outside = np.r_[wave[:i1 + 1], wave[i2:]]
    if inside.mean() - outside.mean() > 0:  # flip positive spikes
        wave = -wave
    trough = i1 + int(np.argmin(wave[i1:i2 + 1]))
    if trough >= len(wave) - 1:
        return np.nan
    peak = int(np.argmax(wave[trough + 1:])) + 1  # samples after the trough
    return peak / rate * 1000.0


def waveform_trough_to_peak(mean_waveform, sample_rate, spike_index=None, filter=True):
    """troughToPeak (ms) of a mean waveform `(n_samples, n_channels)`, on the
    channel with the largest peak-to-peak amplitude within +-0.8 ms of the
    spike time (sample `spike_index`)."""
    w = np.asarray(mean_waveform, dtype=np.float64)
    if w.ndim == 1:
        w = w[:, np.newaxis]
    if w.size == 0 or not np.all(np.isfinite(w)):
        return np.nan
    if filter:
        w = bandpass_filter(w, sample_rate, axis=0)
    k = int(np.ceil(WAVEFORM_HALF_WIDTH_S * sample_rate))
    if spike_index is None:
        channel = int(np.argmax(np.ptp(w, axis=0)))
        spike_index = int(np.argmax(np.abs(w[:, channel])))
    else:
        window = w[max(0, spike_index - k):spike_index + k]
        channel = int(np.argmax(np.ptp(window, axis=0)))
    return trough_to_peak(w[:, channel], sample_rate, spike_index)


def narrow_acg(spike_samples, sample_rate):
    """Autocorrelogram at the lags 0.5, 1, ..., 50 ms (0.5 ms bins centred on
    them), in spikes/s, as CellExplorer's acg.narrow, with the +0.5 ms bin set
    to zero as fit_ACG.m does."""
    st = np.sort(np.asarray(spike_samples, dtype=np.float64))
    n = len(st)
    counts = np.zeros(ACG_N_BINS + 1)
    if n < 2:
        return counts[1:]
    bin_size = max(1, round(ACG_BIN_S * sample_rate))
    max_lag = (ACG_N_BINS + 0.5) * bin_size
    for shift in range(1, n):
        lags = st[shift:] - st[:-shift]
        lags = lags[lags < max_lag]
        if not len(lags):
            break
        bins = np.floor(lags / bin_size + 0.5).astype(np.int64)
        counts += np.bincount(bins, minlength=ACG_N_BINS + 1)[:ACG_N_BINS + 1]
    acg = counts / n / (bin_size / sample_rate)
    acg[:2] = 0  # lag 0 and +0.5 ms
    return acg[1:]


def _acg_model(params, x):
    tau_decay, tau_rise, c, d, asymptote, refrac, tau_burst, h = params
    y = (c * (np.exp(-(x - refrac) / tau_decay) - d * np.exp(-(x - refrac) / tau_rise))
         + h * np.exp(-(x - refrac) / tau_burst) + asymptote)
    return np.maximum(y, 0.0)


def _acg_model_jacobian(params, x):
    tau_decay, tau_rise, c, d, asymptote, refrac, tau_burst, h = params
    u = x - refrac
    e1, e2, e3 = np.exp(-u / tau_decay), np.exp(-u / tau_rise), np.exp(-u / tau_burst)
    jac = np.stack([
        c * e1 * u / tau_decay ** 2,
        -c * d * e2 * u / tau_rise ** 2,
        e1 - d * e2,
        -c * e2,
        np.ones_like(x),
        c * (e1 / tau_decay - d * e2 / tau_rise) + h * e3 / tau_burst,
        h * e3 * u / tau_burst ** 2,
        e3,
    ], axis=1)
    jac[c * (e1 - d * e2) + h * e3 + asymptote <= 0] = 0
    return jac


def fit_acg(acg):
    """Fit fit_ACG.m's triple exponential to `narrow_acg()`; a dict of the
    parameters, or None if the autocorrelogram is empty."""
    y = np.asarray(acg, dtype=np.float64)
    if not np.any(y > 0) or not np.all(np.isfinite(y)):
        return None
    x = np.arange(1, len(y) + 1) * ACG_BIN_S * 1000.0
    res = least_squares(lambda p: _acg_model(p, x) - y, ACG_FIT_START,
                        jac=lambda p: _acg_model_jacobian(p, x),
                        bounds=(ACG_FIT_LOWER, ACG_FIT_UPPER), method="trf")
    names = ("tau_decay", "tau_rise", "c", "d", "asymptote", "refrac", "tau_burst", "h")
    return {f"acg_{name}": float(v) for name, v in zip(names, res.x)}


def acg_tau_rise(spike_samples, sample_rate):
    """CellExplorer's acg_tau_rise (ms); NaN without a spike pair 0.75-50 ms apart."""
    fit = fit_acg(narrow_acg(spike_samples, sample_rate))
    return np.nan if fit is None else fit["acg_tau_rise"]


def classify_cell_type(trough_to_peak_ms, acg_tau_rise_ms):
    """CellExplorer's standard putative cell type; '' without a waveform."""
    if trough_to_peak_ms is None or not np.isfinite(trough_to_peak_ms):
        return ""
    if trough_to_peak_ms <= TROUGH_TO_PEAK_BOUNDARY_MS:
        return NARROW_INTERNEURON
    if acg_tau_rise_ms is not None and np.isfinite(acg_tau_rise_ms) \
            and acg_tau_rise_ms > ACG_TAU_RISE_BOUNDARY_MS:
        return WIDE_INTERNEURON
    return PYRAMIDAL_CELL
