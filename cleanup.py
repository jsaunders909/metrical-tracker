import os
import shutil


def main(args):

    videos = os.listdir(args.input_dir)
    videos = sorted(videos)
    videos = [v for v in videos if os.path.isdir(os.path.join(args.input_dir, v))]

    for i, video in enumerate(videos):
        print(video, os.path.exists(os.path.join(args.input_dir, video, 'track.mp4')))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)