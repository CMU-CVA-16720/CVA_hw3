import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]
print('Number of frames: {}'.format(seq.shape[2]))
rect_array = np.zeros((seq.shape[2],4))
rect_array[0,:] = rect
p_sum = np.zeros(2)

# Compute rect_array
for i in range(1, seq.shape[2]):
    p = LucasKanade(seq[:,:,i-1], seq[:,:,i], np.transpose(rect_array[i-1,:].astype('int')), threshold, num_iters, np.zeros(2))
    p_sum += p
    rect_array[i,0] = (rect[0] + p_sum[0])
    rect_array[i,2] = (rect[2] + p_sum[0])
    rect_array[i,1] = (rect[1] + p_sum[1])
    rect_array[i,3] = (rect[3] + p_sum[1])
    print('p_sum ({}) = {}'.format(i,p_sum))
np.save('girlseqrects.npy',rect_array)

# Display rectangles
rect_array = np.load('girlseqrects.npy')
frames_of_interest = [1, 20, 40, 60, 80]
for i in frames_of_interest:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    coord = rect_array[i,:]
    box=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],ec='red',fc='None')
    img = seq[:,:,i]
    plt.imshow(img)
    ax.add_patch(box)
    plt.show()
    