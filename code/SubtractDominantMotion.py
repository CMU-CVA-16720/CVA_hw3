import numpy as np

from LucasKanadeAffine import LucasKanadeAffine
from scipy.ndimage import affine_transform
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import cv2
import sys

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance, erosion=1, dilation=1):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # Get Affine transform M
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    # Compute error
    warp_img1 = affine_transform(image1, np.linalg.inv(M))
    error = np.abs(image2 - warp_img1)
    struct = np.ones((3,3))
    mask = error>tolerance
    mask = binary_erosion(mask, struct, erosion)
    mask = binary_dilation(mask, struct, dilation)

    return mask

if __name__ == "__main__":
    # Setup
    threshold = 1e-2
    num_iters = 1e3
    tolerance = 0.2
    video_ant = np.load('../data/antseq.npy') # row, col, frame
    video_car = np.load('../data/aerialseq.npy') # row, col, frame
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    # Test case 1
    # img1 = video_ant[:,:,0]
    # img2 = video_ant[:,:,1]
    # mask = SubtractDominantMotion(img1, img2, threshold, num_iters, tolerance)
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()
    # img3 = np.copy(img2)
    # img3[mask] = 1
    # plt.imshow(img3)
    # plt.show()

    # # Make video of original (ant)
    # out = cv2.VideoWriter('../vid/ant.avi', cv2.VideoWriter_fourcc(*'mp4v'), 10, (video_ant.shape[1], video_ant.shape[0]))
    # for i in range(0,video_ant.shape[2]):
    #     cur_frame = cv2.cvtColor(np.floor(255*video_ant[:,:,i]).astype('uint8'),cv2.COLOR_GRAY2BGR)
    #     out.write(cur_frame)
    # out.release()

    # # Make video of original (car)
    # out = cv2.VideoWriter('../vid/car.avi', cv2.VideoWriter_fourcc(*'mp4v'), 10, (video_car.shape[1], video_car.shape[0]))
    # for i in range(0,video_car.shape[2]):
    #     cur_frame = cv2.cvtColor(np.floor(255*video_car[:,:,i]).astype('uint8'),cv2.COLOR_GRAY2BGR)
    #     out.write(cur_frame)
    # out.release()

    # Make video of new (ant)
    # out = cv2.VideoWriter('../vid/ant_msk.avi', cv2.VideoWriter_fourcc(*'mp4v'), 10, (video_ant.shape[1], video_ant.shape[0]))
    # for i in range(1,video_ant.shape[2]):
    #     img1 = video_ant[:,:,i-1]
    #     img2 = video_ant[:,:,i]
    #     mask = SubtractDominantMotion(img1, img2, threshold, num_iters, tolerance)
    #     cur_frame = cv2.cvtColor(np.floor(255*img2).astype('uint8'),cv2.COLOR_GRAY2BGR)
    #     cur_frame[mask] = [255,0,0]
    #     out.write(cur_frame)
    # out.release()

    # # Make video of new (car)
    # out = cv2.VideoWriter('../vid/car_msk.avi', cv2.VideoWriter_fourcc(*'mp4v'), 10, (video_car.shape[1], video_car.shape[0]))
    # for i in range(1,video_car.shape[2]):
    #     img1 = video_car[:,:,i-1]
    #     img2 = video_car[:,:,i]
    #     mask = SubtractDominantMotion(img1, img2, threshold, num_iters, tolerance)
    #     cur_frame = cv2.cvtColor(np.floor(255*img2).astype('uint8'),cv2.COLOR_GRAY2BGR)
    #     cur_frame[mask] = [255,0,0]
    #     out.write(cur_frame)
    # out.release()


