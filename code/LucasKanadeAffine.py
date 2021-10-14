import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform

from matplotlib import pyplot as plt

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """
    # Compute Jacobian
    # [x y 1 0 0 0]
    # [0 0 0 x y 1]
    jacobian_array = np.zeros((It1.shape+(2,6)))
    xv, yv = np.meshgrid(range(It1.shape[1]), range(It1.shape[0]))
    jacobian_array[:,:,[0,1],[0,3]] = np.stack((xv,xv),axis=2)
    jacobian_array[:,:,[0,1],[1,4]] = np.stack((yv,yv),axis=2)
    jacobian_array[:,:,[0,1],[2,5]] = 1
    # Initialize M as do-nothing
    # 1+p1  p2      p3
    # p4    1+p5    p6
    # 0     0       1
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # Gradient descent
    for i in range(num_iters):
        # I(W(x;p))
        I_warp = affine_transform(It1,M)
        # Error = T(x)-I(W), only at overlap
        mask_warp = np.logical_not(affine_transform(np.ones(It1.shape),np.linalg.inv(M))) # pixels to ignore
        error = It-I_warp
        error[mask_warp] = 0
        # gradient(I(W(x;p)))
        I_warp_gradient = np.gradient(I_warp)
        I_warp_dx_spline = RectBivariateSpline(np.arange(0,I_warp.shape[0]),np.arange(0,I_warp.shape[1]),I_warp_gradient[1])
        I_warp_dy_spline = RectBivariateSpline(np.arange(0,I_warp.shape[0]),np.arange(0,I_warp.shape[1]),I_warp_gradient[0])
        delIx = I_warp_dx_spline(np.arange(0,I_warp.shape[0]), np.arange(0,I_warp.shape[1]))
        delIy = I_warp_dy_spline(np.arange(0,I_warp.shape[0]), np.arange(0,I_warp.shape[1]))
        delI = np.stack((delIx, delIy), axis=2)
        # Hessian
        H = np.zeros((6,6))
        for row in range(It1.shape[0]):
            for col in range(It1.shape[1]):
                # check if in area of interest
                if(mask_warp[row,col] == False):
                    delI_dwdp = np.expand_dims(delI[row,col],axis=0) @ jacobian_array[row,col]
                    H += np.transpose(delI_dwdp) @ delI_dwdp
        Hinv = np.linalg.inv(H)
        # Compute deltaP
        delta_p = np.zeros((6,1))
        for row in range(It1.shape[0]):
            for col in range(It1.shape[1]):
                # check if in area of interest
                if(mask_warp[row,col] == False):
                    delI_dwdp = np.expand_dims(delI[row,col],axis=0) @ jacobian_array[row,col]
                    delta_p += (Hinv @ np.transpose(delI_dwdp)) * error[row,col]
        # Update M
        # 1+p1  p2      p3
        # p4    1+p5    p6
        # 0     0       1
        M[0,0] += delta_p[4]
        M[0,1] += delta_p[3]
        M[0,2] += delta_p[5]
        M[1,0] += delta_p[1]
        M[1,1] += delta_p[0]
        M[1,2] += delta_p[2]
        print('M ({:.2f}, {:.2f})= \n{}'.format(np.linalg.norm(error),np.linalg.norm(delta_p),M))
        # See if can exit
        if(np.linalg.norm(delta_p) < threshold):
            break
    
    return M[0:2,:]


if __name__ == "__main__":
    # Setup
    threshold = 1e-2
    num_iter = int(1e3)
    video = np.load('../data/antseq.npy') # row, col, frame
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    
    # Affine testing
    # frame = video[:,:,0]
    # transf_mat = np.array([[1,0,10],[0,1,0],[0,0,1]]) # row, col, 1
    # warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    # plt.imshow(warped_img)
    # plt.show()

    # Test case - Identity
    # frame = video[:,:,0]
    # transf_mat = np.array([[1,0,0],[0,1,0],[0,0,1]]) # row, col, 1
    # warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    # plt.imshow(frame)
    # plt.show()
    # plt.imshow(warped_img)
    # plt.show()
    # M = LucasKanadeAffine(frame, warped_img, threshold, num_iter)
    # print('Result: M = \n', M)

    # Test case - Translation
    frame = video[:,:,0]
    transf_mat = np.array([[1,0,10],[0,1,10],[0,0,1]]) # row, col, 1
    warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    frame = frame[50:250,50:250]
    warped_img = warped_img[50:250,50:250]
    plt.imshow(frame)
    plt.show()
    plt.imshow(warped_img)
    plt.show()
    M = LucasKanadeAffine(frame, warped_img, threshold, num_iter)
    print('Result: M = \n', M)

    # Test case - Skew
    # frame = video[:,:,0]
    # transf_mat = np.array([[1,0.05,-10],[0.025,1,-10],[0,0,1]]) # row, col, 1
    # warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    # frame = frame[50:200,50:200]
    # warped_img = warped_img[50:200,50:200]
    # plt.imshow(frame)
    # plt.show()
    # plt.imshow(warped_img)
    # plt.show()
    # M = LucasKanadeAffine(frame, warped_img, threshold, num_iter)
    # print('Result: M = \n', M)
    




