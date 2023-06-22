import os
from glob import glob
import subprocess
import sys

def main(args):

    videos = glob(os.path.join(args.data_root, '*.mp4'))

    for i, video in enumerate(videos):
        print(f'Video {i} of {len(videos)}')
        out = os.path.join(args.output_dir, os.path.basename(video).replace('.mp4', ''))
        if not os.path.exists(out):
            os.makedirs(out)
        cmd = f"{sys.executable} process_video.py -i {video} -o {out}"
        subprocess.call(cmd, shell=True)
        break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
