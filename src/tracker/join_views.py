#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 10:50:39 2021

@author: Sachuriga
Based on / modified from: https://github.com/genzellab/HM_RAT
"""
#!/usr/bin/env python
import os
import sys
import argparse
from pathlib import Path
import math
import shlex

GLOB_STR_OLD = '*_eye??.mp4'
GLOB_STR_NEW = 'eye??_*.mp4'

# This script is run standalone (`python src/tracker/join_views.py ...`), so put
# src/ on the path to reach the shared encoder picker, as TrackerYolov11 does.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
from tools import vcodec


def ffmpeg(cmd):
    return os.system(cmd)
    # p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    # p.wait()


def make_command(path, crop_x=0, crop_y=0, dur=None, quiet=True, no_stats=False, glob=GLOB_STR_NEW, n_videos=None,
                 encoder=None, hwaccel=None):
    path = Path(path).resolve()
    if not path.exists():
        print("Can't find requested path! [{}]!".format(path), file=sys.stderr)
        return

    videos = sorted(path.glob(glob))
    if not len(videos):
        print('No videos found at location [{}]!'.format(path), file=sys.stderr)
        return

    n_videos = n_videos or len(videos)
    if n_videos != len(videos):
        print('Found unexpected number of videos: {} instead of {}'.format(len(videos), n_videos), file=sys.stderr)
        return

    num_rows = 2
    num_cols = math.ceil(len(videos) / num_rows)
    frame_width = 600
    frame_height = 800

    do_crop = crop_x or crop_y
    cx = crop_x
    cy = crop_y
    cw = frame_width - 2 * crop_x
    ch = frame_height - cy

    # align to 16px block
    canvas_w = math.ceil(cw * num_cols / 16) * 16
    canvas_h = math.ceil(ch * num_rows / 16) * 16

    # Probe at the real canvas size: an old GPU can manage 1080p and still refuse
    # something this wide.
    if encoder is None:
        encoder = vcodec.select(mode='bitrate', size=(canvas_w, canvas_h))

    cmd = ''
    cmd += f'{os.environ.get("FFMPEG_CMD", "ffmpeg")} -y '
    # VAAPI needs its device opened before the inputs.
    if encoder.global_args:
        cmd += ' '.join(encoder.global_args) + ' '

    if quiet:
        cmd += '-hide_banner -loglevel info '
    if no_stats:
        cmd += '-nostats '

    # inputs — decoding 12 h264 streams is a big share of the CPU load, so offload
    # it too when the hardware can take all of them at once.
    hw = ' '.join(vcodec.select_decoder(videos)) if hwaccel is None else ' '.join(hwaccel)
    hw = hw + ' ' if hw else ''
    cmd += ' '.join([f'{hw}-r 30 -i {vp.as_posix()}' for vp in videos])

    # canvas
    cmd += ' -filter_complex "'

    cmd += f'nullsrc=size={canvas_w}x{canvas_h} [canvas0];'

    # position_assignment
    for n in range(len(videos)):
        # bottom row videos need to be flipped in both axis
        flips = ',hflip,vflip' if n >= num_cols else ''

        crop_str = f',crop=w={cw}:h={ch}:x={cx}:y={cy if n >= num_cols else 0}' if do_crop else ''
        cmd += f'[{n}:v] setpts=PTS-STARTPTS{flips}{crop_str} [r{n // num_cols}c{n % num_cols}];'

    # pasting
    for n in range(len(videos)):
        row = n // num_cols
        col = n % num_cols
        s = f'[canvas{n}][r{row}c{col}] overlay=shortest=1:x={col * cw}:y={row * ch} '
        if n < len(videos) - 1:
            s += f'[canvas{n + 1}];'
        cmd += s

    # VAAPI encodes from a hardware surface, so the finished canvas has to be
    # uploaded at the end of the graph.
    if encoder.filter_chain:
        cmd = cmd.rstrip() + ',' + encoder.filter_chain

    cmd += '" '

    cmd += f'-t {dur}' if dur else ''

    # encoding settings. -pix_fmt would fight the hwupload chain, so it is only for
    # the software-surface encoders.
    pix = '' if encoder.filter_chain else '-pix_fmt yuv420p '
    cmd += '-c:v {} {} {}-r 30 '.format(encoder.codec, ' '.join(encoder.args), pix)

    outpath = videos[0].parent / 'stitched.mp4'
    cmd += f'{outpath}'
    return cmd

def _main(cli_args):
    paths = [Path(p) for p in cli_args.paths]
    # An explicit -c skips probing entirely; otherwise make_command probes once and
    # the result is cached in tools.vcodec for the remaining paths.
    encoder = vcodec.Encoder(cli_args.vcodec, ['-b:v', '4000k'], [], '') if cli_args.vcodec else None
    for path in paths:
        print('Joining "{}"'.format(str(path)))
        command = make_command(path, crop_x=104, crop_y=91, quiet=True, glob=cli_args.glob, n_videos=cli_args.n_videos,
                               encoder=encoder)

        if not command:
            print('Command generation for video set "{}" encountered error, stopping.'.format(path), file=sys.stderr)
            return 1

        print(command)
        if cli_args.dry_run:
            print(command)
        else:
            if ffmpeg(command):  # shlex.split(command)
                print('ffmpeg failed for video set "{}", stopping.'.format(path), file=sys.stderr)
                return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RatHexMaze video view combiner')
    parser.add_argument('paths', help='Paths to video sets', nargs='+')
    parser.add_argument('-cx', type=int, default=104, help='Horizontal crop per view (default: %(default)s)')
    parser.add_argument('-cy', type=int, default=91, help='Vertical crop per view (default: %(default)s)')
    parser.add_argument('-rows', type=int, default=2, help='Number of rows to distribute views over')
    parser.add_argument('-n', '--n_videos', default=12, type=int,
                        help='Number of videos to expect. Checks glob result (default: %(default)s)')
    parser.add_argument('-D', '--dry_run', help='Do not launch process, only print the command.', action='store_true')
    parser.add_argument('-g', '--glob', help='Video file glob (default: "%(default)s")', default=GLOB_STR_NEW)
    parser.add_argument('-c', '--vcodec', default='',
                        help='Force a video encoder, e.g. h264_nvenc or libx264. Default: auto-detect, '
                             'preferring the GPU. Also settable via FFMPEG_VCODEC (which is still probed).')

    # parser.add_argument('-s', '--starttime', type=float, help='Start video from time (in seconds)')
    # parser.add_argument('-d', '--duration', type=float, help='Duration of video (in seconds)')

    cli_args = parser.parse_args()

    sys.exit(_main(cli_args))


