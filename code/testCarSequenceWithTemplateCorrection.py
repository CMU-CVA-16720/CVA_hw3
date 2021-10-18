import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from LucasKanade import LucasKanade

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
print('Number of frames: {}'.format(seq.shape[2]))
rect_array = np.zeros((seq.shape[2],4))
rect_array[0,:] = rect[:]

# Setup for template drift correction
template_org = seq[:,:,0]   # T1
template_org_rect = [59, 116, 145, 151]
template_cur = seq[:,:,0]   # Tn
template_cur_rect = [59., 116., 145., 151.]
pn = np.zeros(2)
pn_str = np.zeros(2)

# Compute rect_array
for i in range(1, seq.shape[2]):
    pn     = LucasKanade(template_cur, seq[:,:,i], np.array(template_cur_rect), threshold, num_iters, pn)     # pn
    pn_str = LucasKanade(template_org, seq[:,:,i], np.array(template_org_rect), threshold, num_iters, pn)     # pn*
    rect_array[i,0] = (template_cur_rect[0] + pn[0])
    rect_array[i,2] = (template_cur_rect[2] + pn[0])
    rect_array[i,1] = (template_cur_rect[1] + pn[1])
    rect_array[i,3] = (template_cur_rect[3] + pn[1])
    print('Frame {}/{}; rect delta = {}'.format(i,seq.shape[2], rect_array[i,:]-rect_array[0,:]))
    # Conditionally update template
    drift = np.linalg.norm(pn_str - pn)
    if(drift <= template_threshold):
        template_cur = seq[:,:,i]
        template_cur_rect[0] = (rect[0] + pn_str[0])
        template_cur_rect[2] = (rect[2] + pn_str[0])
        template_cur_rect[1] = (rect[1] + pn_str[1])
        template_cur_rect[3] = (rect[3] + pn_str[1])
np.save('carseqrects-wcrt.npy',rect_array)

# Display rectangles
rect_array = np.load('carseqrects-wcrt.npy')
rect_array_old = np.load('carseqrects.npy')
frames_of_interest = [1, 100, 200, 300, 400]
for i in frames_of_interest:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    coord = rect_array[i,:]
    box=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],ec='red',fc='None')
    coord2 = rect_array_old[i,:]
    box2=patches.Rectangle((coord2[0],coord2[1]),coord2[2]-coord2[0],coord2[3]-coord2[1],ec='blue',fc='None')
    img = seq[:,:,i]
    plt.imshow(img)
    ax.add_patch(box)
    ax.add_patch(box2)
    plt.show()







