import subprocess
import os
import sys
import argparse
import shutil


def main(args):

    # Preprocess
    cmd = f"{sys.executable} preprocess.py --video {args.input} --save_root {args.output_dir}"
    if args.crop:
        cmd += ' --crop'
    subprocess.call(cmd, shell=True)

    # Run MICA
    image = os.path.join(args.output_dir, 'crops', '0000000.png')
    if not os.path.exists(image):
        print('Preprocessing failed, exiting')
        return

    MICA_dir = os.path.join(args.output_dir, 'MICA')
    if not os.path.exists(MICA_dir):
        os.mkdir(MICA_dir)
    shutil.copy(image, os.path.join(MICA_dir, '0000000.png'))
    cmd = f"{sys.executable} MICA/demo.py -i {MICA_dir} -o {args.output_dir}"
    print('------------------------------------------------------------')
    print('Running MICA')
    print(cmd)
    print('------------------------------------------------------------')
    subprocess.call(cmd, shell=True)
    shutil.copy(os.path.join(args.output_dir, '0000000', 'identity.npy'), args.output_dir)

    # Run the tracker
    cfg = os.path.join(args.output_dir, 'config.yaml')
    cmd = f"{sys.executable} tracker.py --cfg {cfg}"
    print('------------------------------------------------------------')
    print('Running tracker')
    print(cmd)
    print('------------------------------------------------------------')
    subprocess.call(cmd, shell=True)

    # Get the UVs
    checkpoint_dir = os.path.join(args.output_dir, 'config', 'checkpoint')
    uv_dir = os.path.join(args.output_dir, 'uv')
    cmd = f"{sys.executable} get_uv_and_mask.py --cfg {cfg} --checkpoint_dir {checkpoint_dir} --output_dir {uv_dir}"
    print('------------------------------------------------------------')
    print('Running UV extraction')
    print(cmd)
    print('------------------------------------------------------------')
    subprocess.call(cmd, shell=True)

    # Make the video for visualization
    cmd = f"{sys.executable} make_vis.py -i {args.output_dir} -o {os.path.join(args.output_dir, 'vis.mp4')}"
    print('------------------------------------------------------------')
    print('Running visualization')
    print(cmd)
    print('------------------------------------------------------------')
    subprocess.call(cmd, shell=True)

    # Postprocess
    cmd = f"{sys.executable} postprocess.py --cfg {cfg} --output_dir {args.output_dir}"
    print('------------------------------------------------------------')
    print('Running postprocessing')
    print(cmd)
    print('------------------------------------------------------------')
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('--crop', action='store_true')

    args = parser.parse_args()

    main(args)
