import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from SubtractDominantMotion import SubtractDominantMotion
import cv2
import time

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.02, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')

frames_of_interest = [30, 60, 90, 120]

for frame in frames_of_interest:
    img1 = seq[:,:,frame-1]
    img2 = seq[:,:,frame]
    ti = time.time()
    mask = SubtractDominantMotion(img1, img2, threshold, num_iters, tolerance,1,1)
    print('Elapsed time: ', time.time()-ti)
    cur_frame = cv2.cvtColor(np.floor(255*img2).astype('uint8'),cv2.COLOR_GRAY2BGR)
    cur_frame[mask] = [0,0,255]
    plt.imshow(np.abs(seq[:,:,frame]-seq[:,:,frame-1]))
    plt.show()
    plt.imshow(cur_frame)
    plt.show()
