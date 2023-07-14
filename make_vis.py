import os, sys, cv2
import numpy as np


def main(args):
    in_dir = args.in_dir
    out_path = args.out_path

    n = len(os.listdir(os.path.join(in_dir, 'uv')))

    writer = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (512 * 3, 512))

    for i in range(n):
        crop = cv2.imread(os.path.join(in_dir, 'crops', f'{i:07d}.png'))
        crop = cv2.resize(crop, (512, 512))
        uv = cv2.imread(os.path.join(in_dir, 'uv', f'{i:05d}.png'))
        comb = ((uv.astype('float32') + crop.astype('float32')) / 2).astype('uint8')

        frame = np.concatenate((crop, uv, comb), axis=1)
        writer.write(frame)

    writer.release()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, required=True)
    parser.add_argument('-o', '--out_path', type=str, required=True)

    args = parser.parse_args()

    main(args)
