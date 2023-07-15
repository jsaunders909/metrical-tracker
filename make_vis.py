import os, sys, cv2
import numpy as np

def padded_crop(img, bb):
    width = bb[2] - bb[0]
    height = bb[3] - bb[1]
    pad = max(width, height)

    padded_image = cv2.copyMakeBorder(img, top=pad, left=pad, bottom=pad, right=pad,
                                      borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image[
           bb[1] + pad: bb[3] + pad,
           bb[0] + pad: bb[2] + pad
           ]

def main(args):
    in_dir = args.in_dir
    out_path = args.out_path

    n = len(os.listdir(os.path.join(in_dir, 'uv')))

    cap = cv2.VideoCapture(os.path.join(in_dir, 'video.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(args.out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (512 * 3, 512))

    for i in range(n):
        crop = cv2.imread(os.path.join(in_dir, 'config', 'input', f'{i:05d}.png'))
        #crop = cv2.resize(crop, (512, 512))
        uv = cv2.imread(os.path.join(in_dir, 'uv', f'{i:05d}.png'))
        #uv = padded_crop(uv, bb)
        comb = ((uv.astype('float32') + crop.astype('float32')) / 2).astype('uint8')

        frame = np.concatenate((crop, uv, comb), axis=1)
        writer.write(frame)

        print(f'Written video frame {i}')

    writer.release()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, required=True)
    parser.add_argument('-o', '--out_path', type=str, required=True)

    args = parser.parse_args()

    main(args)
