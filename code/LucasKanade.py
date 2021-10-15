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
    # W1 = x+p0 -> [[1 0]
    # W2 = y+p1 ->  [0 1]]
    dwdp = np.array([[1, 0],[0, 1]])
    # Get spline for image
    It1_spline = RectBivariateSpline(np.arange(0,It1.shape[0]),np.arange(0,It1.shape[1]),It1)
    # x and y vectors
    x_vect = np.arange(0,It1.shape[1])
    y_vect = np.arange(0,It1.shape[0])
    # Gradient descent
    p = np.copy(p0)
    for i in range(0,int(num_iters)):
        # W(x;p)
        X_vect = x_vect+p[0]
        Y_vect = y_vect+p[1]
        # I(W(x;p))
        I_warp = It1_spline(Y_vect, X_vect)
        # T(x) - I(W(x;p)), crop
        error = (It - I_warp)
        error = error[rect[1]:rect[3]+1,rect[0]:rect[2]+1]
        # gradient(I(W(x;p))), crop
        delIx = It1_spline(Y_vect, X_vect,dy=1)
        delIy = It1_spline(Y_vect, X_vect,dx=1)
        delI = np.stack((delIx, delIy), axis=2)
        delI = delI[rect[1]:rect[3]+1,rect[0]:rect[2]+1]
        # Hessian
        delI_array = np.reshape(delI,(-1,1,2))                      # N x 1 x 2, N = # of pixels in rect
        delI_dwdp_array = delI_array @ dwdp                         # N x 1 x 2
        delI_dwdp_T_array = np.transpose(delI_dwdp_array,(0,2,1))   # N x 2 x 1
        ATA_arary = delI_dwdp_T_array @ delI_dwdp_array             # N x 2 x 2
        H = np.sum(ATA_arary,axis=0)                                # 2 x 2
        Hinv = np.linalg.inv(H)
        # Delta P
        Hinv_delI_dwdp_array = Hinv @ delI_dwdp_T_array             # N x 2 x 1
        error_vector = np.reshape(error,(-1,1))                     # N x 1
        error_array = np.zeros((error_vector.shape[0],2,1))         # N x 2 x 1
        error_array[:,0,:] = error_vector
        error_array[:,1,:] = error_vector
        delta_p = np.sum(Hinv_delI_dwdp_array * error_array, axis=0)# 2 x 1
        # Update p
        p += np.squeeze(delta_p)                                    # (2,)
        # See if can exit
        if(np.sum(np.square(delta_p)) < threshold):
            break
    
    return p


def highlight(img, rectangle, val=1.):
    # Takes in image and rectangle; highlights rectangle
    img[rectangle[1],rectangle[0]:(rectangle[2]+1)] = val
    img[rectangle[3],rectangle[0]:(rectangle[2]+1)] = val
    img[rectangle[1]:(rectangle[3]+1),rectangle[0]] = val
    img[rectangle[1]:(rectangle[3]+1),rectangle[2]] = val
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

