import argparse
import cv2
from glob import glob
import os
from pathlib import Path
import mediapipe as mp
import configparser
import numpy as np
from tqdm import tqdm



def main(args):

    save_root = args.save_root
    video = args.video

    if not os.path.exists(save_root):
        Path(save_root).mkdir(parents=True)

    if not os.path.exists(args.video):
        raise ValueError(f'Could not find video at {args.video}')

    # Configure face detector
    mp_face_detection = mp.solutions.face_detection

    # We look for the smallest bounding box containing all bounding boxes for each frame
    global_bb = np.array([np.inf, np.inf, -np.inf, -np.inf])
    extremes_names = [-1, -1, -1, -1]

    left_margin = 0.1
    right_margin = 0.1
    top_margin = 0.2
    bottom_margin = 0.2

    # We may need to pad the images for cropping
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

    frame_shape = ()
    with mp_face_detection.FaceDetection(model_selection=0.8) as detector:

            # Extract frames using cv2
            cap = cv2.VideoCapture(video)
            fps = cap.get(cv2.CAP_PROP_FPS)

            length = args.length
            length = length.split(':')
            total_len = 3600 * int(length[0]) * 60 + int(length[1]) + int(length[2])
            n_frames = total_len

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if n_frames > 1000:
                    break

                #if frame_idx % 10 != 0:
                #    continue
                frame_idx += 1

                frame_shape = frame.shape
                results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if not results.detections:
                    continue


                detection = results.detections[0]
                bb = detection.location_data.relative_bounding_box
                bb = np.array([int(bb.xmin * frame.shape[1]),
                               int(bb.ymin * frame.shape[0]),
                               int((bb.xmin + bb.width) * frame.shape[1]),
                               int((bb.ymin + bb.height) * frame.shape[0])])

                # Add margins
                width = bb[2] - bb[0]
                height = bb[3] - bb[1]

                bb[0] -= int(float(left_margin) * width)
                bb[1] -= int(float(top_margin) * height)
                bb[2] += int(float(right_margin) * width)
                bb[3] += int(float(bottom_margin) * height)

                global_bb[:2] = np.minimum(bb[:2], global_bb[:2])
                global_bb[2:] = np.maximum(bb[2:], global_bb[2:])

                for i in range(4):
                    if global_bb[i] == bb[i]:
                        extremes_names[i] = video

    print(f'Processed {frame_idx} frames')

    # Check for empty bounding box
    if np.any(global_bb == np.inf):
        print("No face detected in video {}".format(video))
        return

    # Make the bounding box square
    width = global_bb[2] - global_bb[0]
    height = global_bb[3] - global_bb[1]

    print(global_bb)

    if width > height:
        global_bb[1] -= (width - height) // 2
        global_bb[3] += (width - height) - ((width - height) // 2)
    elif height > width:
        global_bb[0] -= (height - width) // 2
        global_bb[2] += (height - width) - ((height - width) // 2)

    print(global_bb)

    # Shift the box to be contained within the image, if possible
    if global_bb[0] < 0:
        shift = min(-global_bb[0], frame_shape[1] - global_bb[2])
        global_bb[0] += shift
        global_bb[2] += shift
    elif global_bb[2] > frame_shape[1]:
        shift = max(global_bb[0], frame_shape[1] - global_bb[2])
        global_bb[2] += shift
        global_bb[0] += shift
    if global_bb[1] < 0:
        shift = min(-global_bb[1], frame_shape[0] - global_bb[3])
        global_bb[1] += shift
        global_bb[3] += shift
    elif global_bb[3] > frame_shape[0]:
        shift = max(-global_bb[1], frame_shape[0] - global_bb[3])
        global_bb[3] += shift
        global_bb[1] += shift

    print(global_bb)

    global_bb = global_bb.astype('int32')
    np.save(os.path.join(save_root, 'bounding_box.npy'), global_bb)

    # Create the folder for these frames
    crop_save_path = os.path.join(save_root, 'crops')

    if not os.path.exists(crop_save_path):
        Path(crop_save_path).mkdir(parents=True)

    # Extract frames using cv2
    cap = cv2.VideoCapture(video)
    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        crop = padded_crop(frame, global_bb)
        # TODO resize to max_width_det


        # Save the image
        cv2.imwrite(os.path.join(crop_save_path, f"{frame_idx:07d}.png"), crop)

    cap.release()

    # Save the video
    if args.crop:
        print('Cropping video')
        cmd = f"ffmpeg -y -i {video} -c:v copy -t {args.length} -vf crop={global_bb[2] - global_bb[0]}:{global_bb[3] - global_bb[1]}:{global_bb[0]}:{global_bb[1]} {os.path.join(save_root, 'video.mp4')}"
    else:
        print('Not cropping video')
        cmd = f"ffmpeg -y -i {video} -c:v copy -t {args.length} {os.path.join(save_root, 'video.mp4')}"
    os.system(cmd)

    cmd = f"ffmpeg -y -i {os.path.join(save_root, 'video.mp4')} {os.path.join(save_root, 'audio.wav')}"
    os.system(cmd)

    config = f"actor: {save_root}\n" \
        f"save_folder: {save_root}\n" \
        "optimize_shape: true \n" \
        "optimize_jaw: true \n" \
        "begin_frames: 1 \n" \
        "keyframes: [0, 100, 200] \n"

    with open(os.path.join(save_root, 'config.yaml'), 'w') as f:
        f.write(config)

if __name__ == '__main__':
    # Arguments from command line
    parser = argparse.ArgumentParser(description='Extract individual frames from video and crop to a fixed bounding')

    parser.add_argument("--video", type=str, required=True, help="The root directory containing the videos")
    parser.add_argument("--save_root", type=str, required=True, help="The root of the directory at which to save all "
                                                                     "results")
    parser.add_argument("--max_width_det", type=int, required=False, default='512', help="The size of the final crops")
    parser.add_argument("--crop", action='store_true', help="Whether to crop the video to the bounding box")
    parser.add_argument("--length", type=str, required=False, default='00:00:3', help="The length of the video to extract")
    args = parser.parse_args()
    main(args)
