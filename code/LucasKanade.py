import numpy as np
from scipy.interpolate import RectBivariateSpline

import cv2
from matplotlib import pyplot as plt
from datetime import datetime

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
    count = 0
    while(run):
        # Get I(W(x;p))
        I_warp = It1_spline(np.arange(0,It1.shape[0])+p[1], np.arange(0,It1.shape[1])+p[0])
        # T(x) - I(W(x;p))
        error = It - I_warp
        # gradient(I(W(x;p)))
        delIx = cv2.Sobel(I_warp,cv2.CV_64F,1,0,ksize=5)
        delIy = cv2.Sobel(I_warp,cv2.CV_64F,0,1,ksize=5)
        delI = np.stack((delIx, delIy), axis=2)
        # Hessian
        H = 0
        reshp = np.reshape(delI[rect[1]:rect[3]+1,rect[0]:rect[2]+1],(-1,1,2))
        prod = reshp @ dwdp
        tp = np.transpose(prod,(0,2,1))
        prod2 = tp @ prod
        sumed = np.sum(prod2,axis=0)
        for row in range(rect[1],rect[3]+1): # y1 ~ y2
            for col in range(rect[0],rect[2]+1): # x1 ~ x2
                delI_dwdp = np.expand_dims(delI[row,col,:],axis=0) @ dwdp # (1x2)(2x2) = (1x2)
                H += np.transpose(delI_dwdp) @ delI_dwdp #(2x1)(1x2) = (2x2)
        Hinv = np.linalg.inv(H)
        # Delta P
        reshp = np.reshape(delI[rect[1]:rect[3]+1,rect[0]:rect[2]+1],(-1,1,2))
        prod = reshp @ dwdp
        tp = np.transpose(prod,(0,2,1))
        prod2 = Hinv @ tp
        error_crop = error[rect[1]:rect[3]+1,rect[0]:rect[2]+1]
        rshp = np.reshape(error_crop,(-1,1))
        error_crop2 = np.zeros((3132,2,1))
        error_crop2[:,0,:] = rshp
        error_crop2[:,1,:] = rshp
        sumed = np.sum(prod2 * error_crop2, axis=0)
        delta_p = np.zeros((2,1))
        for row in range(rect[1],rect[3]+1): # y1 ~ y2
            for col in range(rect[0],rect[2]+1): # x1 ~ x2
                delI_dwdp = np.expand_dims(delI[row,col,:],axis=0) @ dwdp
                delta_p += (Hinv @ np.transpose(delI_dwdp))*error[row,col]
        # Update p
        p += np.squeeze(delta_p)
        # See if can exit
        count += 1
        if((np.sum(np.square(delta_p)) < threshold) or (count > num_iters)):
            run = False
    
    return p


def highlight(img, rectangle):
    # Takes in image and rectangle; highlights rectangle
    img[rectangle[1],rectangle[0]:(rectangle[2]+1)] = 1.
    img[rectangle[3],rectangle[0]:(rectangle[2]+1)] = 1.
    img[rectangle[1]:(rectangle[3]+1),rectangle[0]] = 1.
    img[rectangle[1]:(rectangle[3]+1),rectangle[2]] = 1.
    return img

if __name__ == "__main__":
    video = np.load('../data/carseq.npy') # row, col, frame
    print('Video shape: {}'.format(video.shape))
    # Frame  0 -> 12, (137.5,125.1) -> (137.1,135.6); p ~ (0,10)
    # Frame 12 -> 24, (137.1,135.6) -> (142.2,147.5); p ~ (5,12)

    # 0 -> 12
    rectangle = np.array([59,116,145,151])
    now = datetime.now()
    p = LucasKanade(video[:,:,0], video[:,:,12], np.transpose(rectangle), 1e-5, 10000, np.zeros(2))
    print('Elapsed time: {}'.format(datetime.now()-now))
    print('p (0  -> 12) = {}'.format(np.squeeze(p)))
    plt.imshow(highlight(video[:,:,0],rectangle))
    plt.show()
    rectangle2 = rectangle
    rectangle2[[0,2]] = rectangle2[[0,2]] + p[0]
    rectangle2[[1,3]] = rectangle2[[1,3]] + p[1]
    plt.imshow(highlight(video[:,:,12],rectangle2))
    plt.show()

    # 12 -> 24
    now = datetime.now()
    p = LucasKanade(video[:,:,12], video[:,:,24], np.transpose(rectangle), 1e-5, 10000, np.squeeze(p))
    print('Elapsed time: {}'.format(datetime.now()-now))
    print('p (12 -> 24) = {}'.format(np.squeeze(p)))
    plt.imshow(highlight(video[:,:,12],rectangle2))
    plt.show()
    rectangle3 = rectangle2
    rectangle3[[0,2]] = rectangle3[[0,2]] + p[0]
    rectangle3[[1,3]] = rectangle3[[1,3]] + p[1]
    plt.imshow(highlight(video[:,:,24],rectangle3))
    plt.show()
    print('p = {}'.format(p))

