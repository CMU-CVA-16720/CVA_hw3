import numpy as np
from scipy.interpolate import RectBivariateSpline

import cv2
from matplotlib import pyplot as plt

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    # Jacobian is constant
    dwdp = np.array([[1, 0],[0, 1]])
    # Get spline
    It1_spline = RectBivariateSpline(np.arange(0,It1.shape[0]),np.arange(0,It1.shape[1]),It1)

    # Gradient descent
    p = p0
    run = True
    while(run):
    # for i in range(0,1000):
        # Get I(W(x;p))
        I_warp = It1_spline(np.arange(0,It1.shape[0])+p[1], np.arange(0,It1.shape[1])+p[0])
        # T(x) - I(W(x;p))
        error = It - I_warp
        # print('Error ({}): {}'.format(i,np.sum(np.square(error))))
        print('Error: {}'.format(np.sum(np.square(error))))
        # gradient(I(W(x;p)))
        delIx = cv2.Sobel(I_warp,cv2.CV_64F,1,0,ksize=5)
        delIy = cv2.Sobel(I_warp,cv2.CV_64F,0,1,ksize=5)
        delI = np.stack((delIx, delIy), axis=2)
        # Hessian
        H = 0
        for row in range(rect[1],rect[3]+1): # y1 ~ y2
            for col in range(rect[0],rect[2]+1): # x1 ~ x2
                delI_dwdp = np.expand_dims(delI[row,col,:],axis=0) @ dwdp # (1x2)(2x2) = (1x2)
                H += np.transpose(delI_dwdp) @ delI_dwdp #(2x1)(1x2) = (2x2)
        Hinv = np.linalg.inv(H)
        # Delta P
        delta_p = np.zeros((2,1))
        for row in range(rect[1],rect[3]+1): # y1 ~ y2
            for col in range(rect[0],rect[2]+1): # x1 ~ x2
                delI_dwdp = np.expand_dims(delI[row,col,:],axis=0) @ dwdp
                delta_p += (Hinv @ np.transpose(delI_dwdp))*error[row,col]
        # Update p
        p += np.squeeze(delta_p)
        # See if can exit
        print('  p = {}'.format(np.squeeze(p)))
        if(np.sum(np.square(delta_p)) < threshold):
            print('  deltaP squared sum = {}'.format(np.sum(np.square(delta_p))))
            run = False
    
    return p



if __name__ == "__main__":
    video = np.load('../data/carseq.npy') # row, col, frame
    print('Video shape: {}'.format(video.shape))
    plt.imshow(video[:,:,0])
    plt.show()
    plt.imshow(video[:,:,12])
    plt.show()
    # Frame 0 -> 12, (137.5,125.1) -> (137.1,135.6)
    p = LucasKanade(video[:,:,0], video[:,:,12], np.transpose(np.array([59,116,145,151])), 1e-5, None)
    print('p = {}'.format(p))

