import os
from glob import glob
import subprocess
import sys
from multiprocessing import Pool


def call(cmd):
    subprocess.call(cmd, shell=True)

def main(args):

    videos = glob(os.path.join(args.input_dir, '*.mp4'))
    videos = sorted(videos)
    print(videos[0])
    cmds = []

    for i, video in enumerate(videos):
        if 0 < args.max_videos <= len(cmds):
            break

        print(f'Video {i} of {len(videos)}')
        out = os.path.join(args.output_dir, os.path.basename(video).replace('.mp4', ''))

        if len(glob(os.path.join(out, 'uv', '*.png'))) > 0:
            print('Already processed, skipping')
            continue

        cmd = f"{sys.executable} process_video.py -i {video} -o {out}"
        if args.crop:
            cmd += ' --crop'
        cmds.append(cmd)
        #subprocess.call(cmd, shell=True)

    print(f'Running {len(cmds)} jobs in parallel with {args.num_workers} workers')
    with Pool(args.num_workers) as p:
        p.map(call, cmds)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-n', '--num_workers', type=int, default=2)
    parser.add_argument('-m', '--max_videos', type=int, default=-1)
    parser.add_argument('--crop', action='store_true')
    args = parser.parse_args()
    main(args)
