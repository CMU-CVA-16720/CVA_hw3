import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform

from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_erosion, binary_dilation
import cv2
import time

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [3x3 numpy array]
    """
    # Compute Jacobian
    # [x y 1 0 0 0]
    # [0 0 0 x y 1]
    dwdp_array = np.zeros((It1.shape+(2,6))) # N x M x 2 x 6
    xv, yv = np.meshgrid(range(It1.shape[1]), range(It1.shape[0]))
    dwdp_array[:,:,[0,1],[0,3]] = np.stack((xv,xv),axis=2)
    dwdp_array[:,:,[0,1],[1,4]] = np.stack((yv,yv),axis=2)
    dwdp_array[:,:,[0,1],[2,5]] = 1
    # Get spline for image and template
    It1_spline = RectBivariateSpline(np.arange(0,It1.shape[0]),np.arange(0,It1.shape[1]),It1)
    It_spline = RectBivariateSpline(np.arange(0,It.shape[0]),np.arange(0,It.shape[1]),It)
    # Compute graident(T(x))
    grid_t = np.meshgrid(range(It.shape[1]), range(It.shape[0]))
    grid_t = np.stack((grid_t[0], grid_t[1]), axis=2)
    delTx = It_spline.ev(grid_t[:,:,1],grid_t[:,:,0],dy=1)
    delTy = It_spline.ev(grid_t[:,:,1],grid_t[:,:,0],dx=1)
    delT = np.stack((delTx, delTy), axis=2)
    # Compute Hessian matrix
    delT_array = np.reshape(delT,(-1,1,2))                      # N x 1 x 2
    dwdp_array = np.reshape(dwdp_array,(-1,2,6))                # N x 2 x 6
    delT_dwdp_array = delT_array @ dwdp_array                   # N x 1 x 6
    delT_dwdp_T_array = np.transpose(delT_dwdp_array,(0,2,1))   # N x 6 x 1
    ATA_array = delT_dwdp_T_array @ delT_dwdp_array             # N x 6 x 6

    # Initialize M as do-nothing
    # 1+p1  p2      p3
    # p4    1+p5    p6
    # 0     0       1
    M = np.array([[1, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 1.0]])
    # Gradient descent
    for i in range(int(num_iters)):
        # Check where W(x;p) is legal
        grid = np.meshgrid(range(It1.shape[1]), range(It1.shape[0]))
        grid = np.stack((grid[0], grid[1]), axis=2)
        grid_warp = np.zeros(grid.shape)
        grid_warp[:,:,0] = grid[:,:,0]*M[0,0] + grid[:,:,1]*M[0,1] + M[0,2]
        grid_warp[:,:,1] = grid[:,:,0]*M[1,0] + grid[:,:,1]*M[1,1] + M[1,2]
        illegal_x = np.logical_or(grid_warp[:,:,0] < 0, grid_warp[:,:,0] >= It1.shape[1])
        illegal_y = np.logical_or(grid_warp[:,:,1] < 0, grid_warp[:,:,1] >= It1.shape[0])
        mask = np.logical_or(illegal_x,illegal_y)
        # I(W(x;p))
        I_warp = It1_spline.ev(grid_warp[:,:,1],grid_warp[:,:,0])
        # Error = I(W)-T(x), only at overlap
        error = I_warp-It
        error[mask] = 0
        # Use mask to compute Hessian
        mask_array = mask.reshape((-1,1)) # Turn mask to N x 6 x 6
        mask_array = np.stack((mask_array,mask_array,mask_array,mask_array,mask_array,mask_array),axis=1)
        mask_array = np.stack((mask_array,mask_array,mask_array,mask_array,mask_array,mask_array),axis=2)
        mask_array = np.squeeze(mask_array)                         # N x 6 x 6
        cur_ATA_array = np.copy(ATA_array)
        cur_ATA_array[mask_array] = 0
        H = np.sum(cur_ATA_array,axis=0)                            # 6 x 6
        Hinv = np.linalg.inv(H)                                     # 6 x 6
        # Compute deltaP
        mask_array = mask.reshape((-1,1))
        mask_array = np.stack((mask_array,mask_array,mask_array,mask_array,mask_array,mask_array),axis=1)
        Hinv_delT_dwdp_array = Hinv @ delT_dwdp_T_array             # N x 6 x 1
        error_vector = np.reshape(error,(-1,1))                     # N x 1
        error_array = np.zeros((error_vector.shape[0],6,1))         # N x 6 x 1
        error_array[:,0,:] = error_vector
        error_array[:,1,:] = error_vector
        error_array[:,2,:] = error_vector
        error_array[:,3,:] = error_vector
        error_array[:,4,:] = error_vector
        error_array[:,5,:] = error_vector
        delta_p = np.sum(Hinv_delT_dwdp_array * error_array, axis=0)# 6 x 1
        # Update M
        # 1+p1  p2      p3
        # p4    1+p5    p6
        # 0     0       1
        deltaM = np.array([[1, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 1.0]])
        deltaM[0,0] += delta_p[0]
        deltaM[0,1] += delta_p[1]
        deltaM[0,2] += delta_p[2]
        deltaM[1,0] += delta_p[3]
        deltaM[1,1] += delta_p[4]
        deltaM[1,2] += delta_p[5]
        M = np.copy(M @ np.linalg.inv(deltaM))
        # print('M ({:.2f}, {:.2f})= \n{}'.format(np.linalg.norm(error),np.linalg.norm(delta_p),M))
        # See if can exit
        if(np.linalg.norm(delta_p) < threshold):
            break
    
    return M

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
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
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
    num_iter = 1e3
    video = np.load('../data/antseq.npy') # row, col, frame
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)

    # Test case - Identity
    frame = video[:,:,0]
    transf_mat = np.array([[1,0,0],[0,1,0],[0,0,1]]) # row, col, 1
    warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    M = InverseCompositionAffine(frame, warped_img, threshold, num_iter)
    print('Result 0: M = \n', M)

    # Test case - Translation 1
    frame = video[:,:,0]
    transf_mat = np.array([[1,0,5],[0,1,5],[0,0,1]]) # row, col, 1
    warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    frame = frame[50:200,50:200]
    warped_img = warped_img[50:200,50:200]
    M = InverseCompositionAffine(frame, warped_img, threshold, num_iter)
    print('Result 1: M = \n', M)

    # Test case - Translation 2
    frame = video[:,:,0]
    transf_mat = np.array([[1,0,-5],[0,1,-5],[0,0,1]]) # row, col, 1
    warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    frame = frame[50:200,50:200]
    warped_img = warped_img[50:200,50:200]
    M = InverseCompositionAffine(frame, warped_img, threshold, num_iter)
    print('Result 2: M = \n', M)

    # Test case - Skew
    frame = video[:,:,0]
    transf_mat = np.array([[1,0.25,0],[0,1,0],[0,0,1]]) # row, col, 1
    warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    frame = frame[50:200,50:200]
    warped_img = warped_img[50:200,50:200]
    M = InverseCompositionAffine(frame, warped_img, threshold, num_iter)
    print('Result 3: M = \n', M)

    # Ant seq
    threshold = 1e-2
    num_iters = 1e3
    tolerance = 0.01
    frames_of_interest = [30, 60, 90, 120]
    seq = np.load('../data/antseq.npy')
    print('Ant sequence: ')
    for frame in frames_of_interest:
        img1 = seq[:,:,frame-1]
        img2 = seq[:,:,frame]
        ti = time.time()
        mask = SubtractDominantMotion(img1, img2, threshold, num_iters, tolerance,1,1)
        print('Elapsed time: ', time.time()-ti)
        cur_frame = cv2.cvtColor(np.floor(255*img2).astype('uint8'),cv2.COLOR_GRAY2BGR)
        cur_frame[mask] = [0,0,255]
        # plt.imshow(seq[:,:,frame]-seq[:,:,frame-1])
        # plt.show()
        plt.imshow(cur_frame)
        plt.show()
    
    # Car seq
    threshold = 1e-2
    num_iters = 1e3
    tolerance = 0.04
    frames_of_interest = [30, 60, 90, 120]
    seq = np.load('../data/aerialseq.npy')
    print('Car sequence: ')
    for frame in frames_of_interest:
        img1 = seq[:,:,frame-1]
        img2 = seq[:,:,frame]
        ti = time.time()
        mask = SubtractDominantMotion(img1, img2, threshold, num_iters, tolerance,2,1)
        print('Elapsed time: ', time.time()-ti)
        cur_frame = cv2.cvtColor(np.floor(255*img2).astype('uint8'),cv2.COLOR_GRAY2BGR)
        cur_frame[mask] = [0,0,255]
        plt.imshow(cur_frame)
        plt.show()