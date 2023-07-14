import os, sys, cv2
import numpy as np

n = len(os.listdir('crops'))

writer = cv2.VideoWriter('vis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (512 * 3, 512))

for i in range(n):
    crop = cv2.imread(os.path.join('crops', f'{i:07d}.png'))
    crop = cv2.resize(crop, (512, 512))
    uv = cv2.imread(os.path.join('uv', f'{i:05d.png}'))
    comb = ((uv.astype('float32') + crop.astype('float32')) / 2).astype('uint8')

    frame = np.concatenate((crop, uv, comb), axis=1)
    writer.write(frame)

writer.release()


