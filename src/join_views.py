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
import subprocess as sp

GLOB_STR_OLD = '*_eye??.mp4'
GLOB_STR_NEW = 'eye??_*.mp4'


def ffmpeg(cmd):
    os.system(cmd)
    # p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    # p.wait()


def make_command(path, crop_x=0, crop_y=0, dur=None, quiet=True, no_stats=False, glob=GLOB_STR_NEW, n_videos=None):
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

    cmd = ''
    cmd += 'ffmpeg -y ' #'ffmpeg -y ' #'ffmpeg -y -hwaccel nvdec '

    if quiet:
        cmd += '-hide_banner -loglevel info '
    if no_stats:
        cmd += '-nostats '


    # inputs
    cmd += ' '.join([f'-r 30 -i {vp.as_posix()}' for vp in videos])

    # canvas
    cmd += ' -filter_complex "'

    # align to 16px block
    canvas_w = math.ceil(cw * num_cols / 16) * 16
    canvas_h = math.ceil(ch * num_rows / 16) * 16
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

    cmd += '" '

    cmd += f'-t {dur}' if dur else ''

    # encoding settings
    # cmd += '-c:v libx264 -preset veryfast -crf 18 '
    cmd += '-c:v h264_nvenc -b:v 4000k -preset fast -pix_fmt yuv420p -r 30 '# '-c:v h264_nvenc -b:v 4000k -preset fast -pix_fmt yuv420p '

    outpath = videos[0].parent / 'stitched.mp4'
    cmd += f'{outpath}'
    return cmd

def _main(cli_args):
    paths = [Path(p) for p in cli_args.paths]
    for path in paths:
        print('Joining "{}"'.format(str(path)))
        command = make_command(path, crop_x=104, crop_y=91, quiet=True, glob=cli_args.glob, n_videos=cli_args.n_videos)

        if not command:
            print('Command generation for video set "{}" encountered error, stopping.'.format(path), file=sys.stderr)
            return 1

        print(command)
        if cli_args.dry_run:
            print(command)
        else:
            ffmpeg(command)  # shlex.split(command)


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

    # parser.add_argument('-s', '--starttime', type=float, help='Start video from time (in seconds)')
    # parser.add_argument('-d', '--duration', type=float, help='Duration of video (in seconds)')

    cli_args = parser.parse_args()

    sys.exit(_main(cli_args))


