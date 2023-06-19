import os
import sys
import subprocess
import shutil

def main(args):

    work_dir = args.output_dir
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoint')
    uv_dir = os.path.join(args.output_dir, 'uv')

    cmd = f"{sys.executable} get_uv_and_mask.py --cfg {args.cfg} --checkpoint_dir {checkpoint_dir} --output_dir {uv_dir}"
    subprocess.call(cmd, shell=True)

    other_dirs = [os.path.join(work_dir, d) for d in os.listdir(work_dir) if
                  os.path.isdir(os.path.join(work_dir, d)) and d not in ['checkpoint', 'uv', 'input', 'crops']]
    for dir in other_dirs:
        shutil.rmtree(dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    main(args)