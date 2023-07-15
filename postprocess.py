import os
import sys
import subprocess
import shutil
import cv2
from glob import glob


def write_video(root, out_path, suffix='.png', fps=30):

    images = sorted(glob(os.path.join(root, f'*{suffix}')))
    im = cv2.imread(images[0])
    height, width, _ = im.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for im in images:
        cap.write(cv2.imread(im))
    cap.release()


def main(args):

    work_dir = args.output_dir
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
    uv_dir = os.path.join(args.output_dir, 'uv')

    #cmd = f"{sys.executable} get_uv_and_mask.py --cfg {args.cfg} --checkpoint_dir {checkpoint_dir} --output_dir {uv_dir}"
    #subprocess.call(cmd, shell=True)

    video = os.path.join(work_dir, 'video.mp4')
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    track_video_out = os.path.join(work_dir, 'track.mp4')
    track_video_in = os.path.join(work_dir, 'config', 'video')
    write_video(track_video_in, track_video_out, fps=fps, suffix='.jpg')

    # Clean up
    other_dirs = [os.path.join(work_dir, d) for d in os.listdir(work_dir) if
                  os.path.isdir(os.path.join(work_dir, d)) and d not in ['checkpoint', 'uv', 'input', 'crops', 'config']]
    for dir in other_dirs:
        shutil.rmtree(dir)

    shutil.rmtree(os.path.join(work_dir, 'config', 'depth'))
    shutil.rmtree(os.path.join(work_dir, 'config', 'video'))
    shutil.rmtree(os.path.join(work_dir, 'config', 'input'))
    shutil.rmtree(os.path.join(work_dir, 'config', 'logs'))
    shutil.rmtree(os.path.join(work_dir, 'config', 'pyramid'))
    shutil.rmtree(os.path.join(work_dir, 'config', 'mesh'))
    os.remove(os.path.join(work_dir, 'config', 'canonical.obj'))
    os.remove(os.path.join(work_dir, 'config', 'train.log'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    main(args)
