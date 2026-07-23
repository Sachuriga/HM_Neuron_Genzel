#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hardware video encoder / decoder selection for Windows and Linux.

Picks the fastest path that actually works on THIS machine, preferring the GPU and
using the CPU only when no GPU path can be made to run. macOS is supported just
well enough to develop against; the rigs are Windows, with Linux as a second
target.

Three things make naive selection fail, and all three have bitten this pipeline:

1. `ffmpeg -encoders` lists h264_nvenc even when it cannot open, because the codec
   is compiled in but the NVIDIA driver is older than the nvenc API the build
   wants ("Required: 13.1 Found: 13.0"). Only a real encode is conclusive.

2. An old GPU often has a usable encoder that merely rejects modern *flags*.
   nvenc's p1-p7 presets need an SDK 10+ card (Turing, 2018+); a Kepler or Maxwell
   card fails `-preset p6` yet encodes happily with `-preset fast`.

3. Opening successfully does not prove hardware. Windows Media Foundation silently
   returns Microsoft's software "H264 Encoder MFT" when no hardware transform
   exists — it opens, reports success, pegs the CPU, and encodes worse than
   libx264.

So each encoder carries several argument variants, newest first, and we probe with
the real arguments at the real frame size until one runs. Probes cost seconds; the
12-view stitch is ~2352x1424 for over an hour, so guessing wrong is far worse.

Environment overrides:
    FFMPEG_CMD       path to the ffmpeg binary (use this to point at an older
                     build whose nvenc matches an older driver)
    FFPROBE_CMD      path to ffprobe
    FFMPEG_VCODEC    force an encoder, e.g. h264_nvenc / libx264
    FFMPEG_HWACCEL   force decode accel ('none' disables, or cuda/d3d11va/...)
"""
import os
import sys
import glob
import platform
import collections
import subprocess as sp

# An encoder choice is more than a codec name: VAAPI needs a device set up before
# the input and the frames uploaded to a hardware surface, so the caller has to be
# told about all three pieces.
Encoder = collections.namedtuple('Encoder', 'codec args global_args filter_chain')

CPU_VCODEC = 'libx264'
CPU_ARGS = {'quality': ['-preset', 'veryfast', '-crf', '28'],
            'bitrate': ['-preset', 'veryfast', '-b:v', '{br}']}

# ---------------------------------------------------------------- encoders
# Each variant is tried in turn; the last for every encoder is bare bitrate-only,
# the most compatible thing we can ask of ancient hardware.
_NVENC = {'codec': 'h264_nvenc', 'variants': [
    # SDK 10+ (Turing and newer): p-presets with constant quality.
    {'quality': ['-preset', 'p6', '-cq', '28'],
     'bitrate': ['-preset', 'p4', '-b:v', '{br}']},
    # Older SDKs: named presets, VBR with a quality target.
    {'quality': ['-preset', 'slow', '-rc', 'vbr', '-cq', '28'],
     'bitrate': ['-preset', 'fast', '-b:v', '{br}']},
    # Kepler-era: constant QP, no rate-control selection.
    {'quality': ['-preset', 'fast', '-qp', '28'],
     'bitrate': ['-preset', 'fast', '-b:v', '{br}']},
    {'quality': ['-b:v', '4000k'], 'bitrate': ['-b:v', '{br}']}]}

_QSV = {'codec': 'h264_qsv', 'variants': [
    {'quality': ['-preset', 'faster', '-global_quality', '28'],
     'bitrate': ['-preset', 'faster', '-b:v', '{br}']},
    {'quality': ['-b:v', '4000k'], 'bitrate': ['-b:v', '{br}']}]}

_AMF = {'codec': 'h264_amf', 'variants': [
    {'quality': ['-quality', 'speed', '-rc', 'cqp', '-qp_i', '28', '-qp_p', '28'],
     'bitrate': ['-quality', 'speed', '-b:v', '{br}']},
    {'quality': ['-b:v', '4000k'], 'bitrate': ['-b:v', '{br}']}]}

# Windows Media Foundation: reaches old Intel iGPUs that the QSV path has dropped.
# -hw_encoding 1 is mandatory, not tuning — without it MF falls back to its
# software transform, which opens fine and would be scored as a GPU win.
_MF = {'codec': 'h264_mf', 'variants': [
    {'quality': ['-hw_encoding', '1', '-b:v', '4000k'],
     'bitrate': ['-hw_encoding', '1', '-b:v', '{br}']}]}

# VAAPI (Linux, Intel/AMD). Device and hwupload are filled in per render node.
_VAAPI = {'codec': 'h264_vaapi', 'vaapi': True, 'variants': [
    {'quality': ['-qp', '28'], 'bitrate': ['-b:v', '{br}']},
    {'quality': ['-b:v', '4000k'], 'bitrate': ['-b:v', '{br}']}]}

_VIDEOTOOLBOX = {'codec': 'h264_videotoolbox', 'variants': [
    {'quality': ['-q:v', '55'], 'bitrate': ['-b:v', '{br}']},
    {'quality': ['-b:v', '4000k'], 'bitrate': ['-b:v', '{br}']}]}

# NVIDIA first on both rigs: it is the fastest and it is what these machines have.
_BY_PLATFORM = {'Windows': [_NVENC, _QSV, _AMF, _MF],
                'Linux': [_NVENC, _QSV, _VAAPI, _AMF],
                'Darwin': [_VIDEOTOOLBOX]}

# Decode accelerators, best first. Saving the CPU 12 concurrent h264 decodes
# matters more here than the encoder does.
_HWACCELS = {'Windows': ['cuda', 'd3d11va', 'qsv', 'dxva2'],
             'Linux': ['cuda', 'vaapi', 'qsv'],
             'Darwin': ['videotoolbox']}

# Media Foundation transforms that are CPU implementations. A hardware MFT names
# its vendor ("NVIDIA H.264 Encoder MFT", "Intel(R) Quick Sync Video ...").
_SOFTWARE_MFT = ('h264 encoder mft', 'hevc encoder mft')

_cache = {}


def _ffmpeg():
    return os.environ.get('FFMPEG_CMD', 'ffmpeg')


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


def _fill(args, bitrate):
    return [a.replace('{br}', bitrate) for a in args]


def render_nodes():
    """Linux DRM render nodes, e.g. ['/dev/dri/renderD128', ...].

    Not always renderD128: on a box with a discrete card plus an iGPU the usable
    node may be renderD129, so every node gets probed rather than assumed."""
    return sorted(glob.glob('/dev/dri/renderD*'))


def _candidate_encoders(spec, mode, bitrate):
    """Expand one encoder spec into concrete Encoder objects, best first."""
    out = []
    for variant in spec['variants']:
        args = _fill(variant[mode], bitrate)
        if spec.get('vaapi'):
            for dev in render_nodes():
                out.append(Encoder(spec['codec'], args,
                                   ['-vaapi_device', dev], 'format=nv12,hwupload'))
        else:
            out.append(Encoder(spec['codec'], args, [], ''))
    return out


def probe_encoder(enc, size=(1920, 1080)):
    """True if ffmpeg can really encode frames as `enc` at `size`, on hardware.

    Size matters: old GPUs have low maximum encode dimensions, so a 64x64 probe
    can pass where the actual 2352x1424 stitch would fail."""
    w, h = size
    # Global options (VAAPI device) must come before the input to take effect.
    cmd = [_ffmpeg(), '-hide_banner', '-loglevel', 'verbose'] + list(enc.global_args)
    cmd += ['-f', 'lavfi', '-i', 'nullsrc=s={}x{}:d=0.07:r=30'.format(w, h)]
    cmd += ['-vf', enc.filter_chain] if enc.filter_chain else ['-pix_fmt', 'yuv420p']
    cmd += ['-c:v', enc.codec] + list(enc.args) + ['-f', 'null', '-']
    try:
        p = sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.PIPE, timeout=120)
    except (OSError, sp.SubprocessError):
        return False
    if p.returncode != 0:
        return False
    # Second line of defence behind -hw_encoding: if MF still handed us the
    # software transform, refuse it — libx264 beats that encoder anyway. Match the
    # whole name: Intel's hardware MFT is "Intel(R) Quick Sync Video H264 Encoder
    # MFT" and *contains* the software name as a substring.
    for line in (p.stderr or b'').decode('utf-8', 'replace').splitlines():
        if 'MFT name' in line:
            name = line.split('MFT name:')[-1].strip().strip("'\"")
            if name.lower() in _SOFTWARE_MFT:
                _say('[DEBUG] {} resolved to the software transform ({!r}); not hardware, '
                     'skipping.'.format(enc.codec, name))
                return False
    return True


def select(mode='quality', size=(1920, 1080), bitrate='4000k', verbose=True):
    """Return an Encoder for the best working path, GPU first.

    `mode` is 'quality' (compression, CQ/CRF-style) or 'bitrate' (stitching, fixed
    rate). FFMPEG_VCODEC overrides the search; its args are still probed, so a
    forced-but-half-broken codec degrades to simpler flags instead of dying."""
    forced = os.environ.get('FFMPEG_VCODEC', '')
    key = ('enc', mode, size, bitrate, forced)
    if key in _cache:
        return _cache[key]

    if forced:
        known = [s for group in _BY_PLATFORM.values() for s in group if s['codec'] == forced]
        specs = [known[0]] if known else [{'codec': forced, 'variants': [
            {'quality': ['-b:v', '4000k'], 'bitrate': ['-b:v', '{br}']}]}]
    else:
        specs = _BY_PLATFORM.get(platform.system(), [_NVENC])

    for spec in specs:
        tried = []
        for enc in _candidate_encoders(spec, mode, bitrate):
            sig = (enc.args, enc.global_args)
            if sig in tried:  # variants can collapse to identical flags in one mode
                continue
            tried.append(sig)
            if probe_encoder(enc, size):
                if verbose:
                    extra = ' '.join(enc.global_args)
                    _say('[INFO] GPU encoder: {} {}{}{}'.format(
                        enc.codec, ' '.join(enc.args), ' ' + extra if extra else '',
                        ' (older-style options — older GPU)' if len(tried) > 1 else ''))
                _cache[key] = enc
                return enc
            if verbose:
                _say('[DEBUG] {} {} did not open, trying next option.'
                     .format(enc.codec, ' '.join(enc.args)))

    if verbose:
        _say('[WARNING] No working GPU encoder (driver too old for this ffmpeg build, or '
             'no supported GPU). Using {} on the CPU — slower, but it will finish. '
             'Point FFMPEG_CMD at an ffmpeg whose nvenc matches your driver to get the '
             'GPU back.'.format(CPU_VCODEC))
    enc = Encoder(CPU_VCODEC, _fill(CPU_ARGS[mode], bitrate), [], '')
    _cache[key] = enc
    return enc


# ---------------------------------------------------------------- decoding
def probe_decode(videos, hwaccel):
    """True if `hwaccel` can decode ALL `videos` at once.

    Deliberately opens every input in one process: a card that decodes one stream
    may refuse twelve, and finding that out mid-stitch costs an hour."""
    cmd = [_ffmpeg(), '-hide_banner', '-loglevel', 'error', '-nostdin']
    for v in videos:
        cmd += ['-hwaccel', hwaccel, '-i', str(v)]
    for n in range(len(videos)):
        cmd += ['-map', '{}:v'.format(n)]
    cmd += ['-t', '0.2', '-f', 'null', '-']
    try:
        return sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=180).returncode == 0
    except (OSError, sp.SubprocessError):
        return False


def select_decoder(videos, verbose=True):
    """['-hwaccel', 'cuda'] to prepend to each input, or [] for software decode.

    Twelve concurrent h264 decodes are a large part of why the CPU sits at 100%
    during a stitch, and unlike the encoder this stays true even when nvenc works."""
    forced = os.environ.get('FFMPEG_HWACCEL', '')
    if forced.lower() in ('none', 'no', '0', 'off'):
        return []
    key = ('dec', tuple(str(v) for v in videos), forced)
    if key in _cache:
        return _cache[key]

    names = [forced] if forced else _HWACCELS.get(platform.system(), [])
    for name in names:
        if probe_decode(videos, name):
            if verbose:
                _say('[INFO] GPU decode: -hwaccel {} for all {} inputs.'
                     .format(name, len(videos)))
            _cache[key] = ['-hwaccel', name]
            return _cache[key]
        if verbose:
            _say('[DEBUG] -hwaccel {} cannot decode all {} inputs, trying next.'
                 .format(name, len(videos)))
    if verbose:
        _say('[INFO] Software decode (no working -hwaccel). Set FFMPEG_HWACCEL to force one.')
    _cache[key] = []
    return _cache[key]


if __name__ == '__main__':
    size = (2352, 1424)
    if len(sys.argv) >= 3:
        size = (int(sys.argv[1]), int(sys.argv[2]))
    print('platform      : {}'.format(platform.system()))
    print('ffmpeg        : {}'.format(_ffmpeg()))
    print('probe size    : {}x{}'.format(*size))
    if platform.system() == 'Linux':
        print('render nodes  : {}'.format(', '.join(render_nodes()) or 'none'))
    for mode in ('quality', 'bitrate'):
        enc = select(mode=mode, size=size)
        print('{:<14}: {} {} {}'.format(mode, enc.codec, ' '.join(enc.args),
                                        ' '.join(enc.global_args)).rstrip())
