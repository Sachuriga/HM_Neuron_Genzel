#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hardware video encoder selection.

Picks the fastest encoder that actually works on THIS machine, preferring the GPU
and only using the CPU when no GPU path can be made to run.

Two things make naive selection fail, and both show up on lab machines:

1. `ffmpeg -encoders` lists h264_nvenc even when it cannot open, because the codec
   is compiled in but the NVIDIA driver is older than the nvenc API the build
   wants ("Required: 13.1 Found: 13.0"). Only a real encode is conclusive.

2. An old GPU often has a perfectly usable encoder that merely rejects modern
   *flags*. nvenc's p1-p7 presets need an SDK 10+ card (Turing, 2018+); a Kepler
   or Maxwell card fails on `-preset p6` yet encodes happily with `-preset fast`.

So each encoder carries several argument variants, newest first, and we probe with
the real arguments at the real frame size until one runs. Losing the GPU costs far
more than a few probe encodes: the 12-view stitch is ~2352x1424 for over an hour.
"""
import os
import sys
import platform
import subprocess as sp

CPU_VCODEC = 'libx264'
CPU_ARGS = {'quality': ['-preset', 'veryfast', '-crf', '28'],
            'bitrate': ['-preset', 'veryfast', '-b:v', '{br}']}

# GPU encoders per platform, in preference order. Each variant is tried in turn;
# the last one for every encoder is bare bitrate-only, which is the most
# compatible thing we can ask of ancient hardware.
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

# Windows Media Foundation: wraps whatever the OS exposes, including old Intel
# iGPUs that the QSV path no longer supports. Bitrate-only on purpose.
_MF = {'codec': 'h264_mf', 'variants': [
    {'quality': ['-b:v', '4000k'], 'bitrate': ['-b:v', '{br}']}]}

_VAAPI = {'codec': 'h264_vaapi', 'variants': [
    {'quality': ['-qp', '28'], 'bitrate': ['-b:v', '{br}']}]}

_VIDEOTOOLBOX = {'codec': 'h264_videotoolbox', 'variants': [
    {'quality': ['-q:v', '55'], 'bitrate': ['-b:v', '{br}']},
    {'quality': ['-b:v', '4000k'], 'bitrate': ['-b:v', '{br}']}]}

_BY_PLATFORM = {'Windows': [_NVENC, _QSV, _AMF, _MF],
                'Linux': [_NVENC, _QSV, _VAAPI],
                'Darwin': [_VIDEOTOOLBOX]}

_cache = {}


def _ffmpeg():
    return os.environ.get('FFMPEG_CMD', 'ffmpeg')


def _fill(args, bitrate):
    return [a.replace('{br}', bitrate) for a in args]


def probe(codec, args, size=(1920, 1080)):
    """True if ffmpeg can encode two real frames as `codec` with `args` at `size`.

    Size matters: old GPUs have low maximum encode dimensions, so a 64x64 probe
    can pass where the actual 2352x1424 stitch would fail."""
    w, h = size
    cmd = [_ffmpeg(), '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
           '-i', 'nullsrc=s={}x{}:d=0.07:r=30'.format(w, h)]
    if codec == 'h264_vaapi':  # needs frames uploaded to the VAAPI surface
        cmd += ['-vaapi_device', '/dev/dri/renderD128', '-vf', 'format=nv12,hwupload']
    else:
        cmd += ['-pix_fmt', 'yuv420p']
    cmd += ['-c:v', codec] + list(args) + ['-f', 'null', '-']
    try:
        return sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=120).returncode == 0
    except (OSError, sp.SubprocessError):
        return False


def select(mode='quality', size=(1920, 1080), bitrate='4000k', verbose=True):
    """Return (codec, args) for the best working encoder, GPU first.

    `mode` is 'quality' (compression, CQ/CRF-style) or 'bitrate' (stitching, fixed
    rate). FFMPEG_VCODEC overrides the search; its args are still probed so a
    forced-but-broken codec degrades to a plain bitrate instead of dying."""
    key = (mode, size, bitrate, os.environ.get('FFMPEG_VCODEC', ''))
    if key in _cache:
        return _cache[key]

    forced = os.environ.get('FFMPEG_VCODEC', '')
    if forced:
        known = [p for group in _BY_PLATFORM.values() for p in group if p['codec'] == forced]
        variants = known[0]['variants'] if known else [{'quality': ['-b:v', '4000k'],
                                                        'bitrate': ['-b:v', '{br}']}]
        candidates = [{'codec': forced, 'variants': variants}]
    else:
        candidates = _BY_PLATFORM.get(platform.system(), [_NVENC])

    for cand in candidates:
        for n, variant in enumerate(cand['variants']):
            args = _fill(variant[mode], bitrate)
            if probe(cand['codec'], args, size):
                if verbose and n:
                    _say('[INFO] {} accepted only older-style options ({}) — likely an '
                         'older GPU. Using them.'.format(cand['codec'], ' '.join(args)))
                elif verbose:
                    _say('[INFO] Encoding on GPU: {} {}'.format(cand['codec'], ' '.join(args)))
                _cache[key] = (cand['codec'], args)
                return _cache[key]
            if verbose:
                _say('[DEBUG] {} {} did not open, trying next option.'
                     .format(cand['codec'], ' '.join(args)))

    if verbose:
        _say('[WARNING] No working GPU encoder found (driver too old for this ffmpeg '
             'build, or no supported GPU). Falling back to {} on the CPU — slower, but '
             'it will finish.'.format(CPU_VCODEC))
    _cache[key] = (CPU_VCODEC, _fill(CPU_ARGS[mode], bitrate))
    return _cache[key]


def _say(msg):
    print(msg, file=sys.stderr, flush=True)


if __name__ == '__main__':
    size = (2352, 1424)
    if len(sys.argv) == 3:
        size = (int(sys.argv[1]), int(sys.argv[2]))
    print('platform      : {}'.format(platform.system()))
    print('probe size    : {}x{}'.format(*size))
    for mode in ('quality', 'bitrate'):
        codec, args = select(mode=mode, size=size)
        print('{:<14}: {} {}'.format(mode, codec, ' '.join(args)))
