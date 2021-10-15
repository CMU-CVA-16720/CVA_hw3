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
    dwdp_array = np.zeros((It1.shape+(2,6))) # N x M x 2 x 6
    xv, yv = np.meshgrid(range(It1.shape[1]), range(It1.shape[0]))
    dwdp_array[:,:,[0,1],[0,3]] = np.stack((xv,xv),axis=2)
    dwdp_array[:,:,[0,1],[1,4]] = np.stack((yv,yv),axis=2)
    dwdp_array[:,:,[0,1],[2,5]] = 1
    dwdp_array = np.reshape(dwdp_array,(-1,2,6))
    # Get spline for image
    It1_spline = RectBivariateSpline(np.arange(0,It1.shape[0]),np.arange(0,It1.shape[1]),It1)
    # x and y vectors
    x_vect = np.arange(0,It1.shape[1])
    y_vect = np.arange(0,It1.shape[0])
    # Initialize M as do-nothing
    # 1+p1  p2      p3
    # p4    1+p5    p6
    # 0     0       1
    M = np.array([[1, 0.0, 0.0], [0.0, 1, 0.0], [0.0, 0.0, 1.0]])
    # Gradient descent
    for i in range(num_iters):
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
        # Error = T(x)-I(W), only at overlap
        error = It-I_warp
        error[mask] = 0
        # gradient(I(W(x;p)))
        delIx = It1_spline.ev(grid_warp[:,:,1],grid_warp[:,:,0],dy=1)
        delIy = It1_spline.ev(grid_warp[:,:,1],grid_warp[:,:,0],dx=1)
        delIx[mask] = 0
        delIy[mask] = 0
        delI = np.stack((delIx, delIy), axis=2)
        # Hessian
        delI_array = np.reshape(delI,(-1,1,2))                      # N x 1 x 2
        delI_dwdp_array = delI_array @ dwdp_array                   # N x 1 x 6
        delI_dwdp_T_array = np.transpose(delI_dwdp_array,(0,2,1))   # N x 6 x 1
        ATA_arary = delI_dwdp_T_array @ delI_dwdp_array             # N x 6 x 6
        H = np.sum(ATA_arary,axis=0)                                # 6 x 6
        Hinv = np.linalg.inv(H)                                     # 6 x 6
        # Compute deltaP
        Hinv_delI_dwdp_array = Hinv @ delI_dwdp_T_array             # N x 6 x 1
        error_vector = np.reshape(error,(-1,1))                     # N x 1
        error_array = np.zeros((error_vector.shape[0],6,1))         # N x 6 x 1
        error_array[:,0,:] = error_vector
        error_array[:,1,:] = error_vector
        error_array[:,2,:] = error_vector
        error_array[:,3,:] = error_vector
        error_array[:,4,:] = error_vector
        error_array[:,5,:] = error_vector
        delta_p = np.sum(Hinv_delI_dwdp_array * error_array, axis=0)# 2 x 1
        # Update M
        # 1+p1  p2      p3
        # p4    1+p5    p6
        # 0     0       1
        M[0,0] += delta_p[0]
        M[0,1] += delta_p[1]
        M[0,2] += delta_p[2]
        M[1,0] += delta_p[3]
        M[1,1] += delta_p[4]
        M[1,2] += delta_p[5]
        # print('M ({:.2f}, {:.2f})= \n{}'.format(np.linalg.norm(error),np.linalg.norm(delta_p),M))
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

    # Test case - Translation 1
    frame = video[:,:,0]
    transf_mat = np.array([[1,0,5],[0,1,5],[0,0,1]]) # row, col, 1
    warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    frame = frame[50:200,50:200]
    warped_img = warped_img[50:200,50:200]
    plt.imshow(frame)
    plt.show()
    plt.imshow(warped_img)
    plt.show()
    M = LucasKanadeAffine(frame, warped_img, threshold, num_iter)
    print('Result 1: M = \n', M)

    # Test case - Translation 2
    frame = video[:,:,0]
    transf_mat = np.array([[1,0,-5],[0,1,-5],[0,0,1]]) # row, col, 1
    warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    frame = frame[50:200,50:200]
    warped_img = warped_img[50:200,50:200]
    plt.imshow(frame)
    plt.show()
    plt.imshow(warped_img)
    plt.show()
    M = LucasKanadeAffine(frame, warped_img, threshold, num_iter)
    print('Result 2: M = \n', M)

    # Test case - Skew
    frame = video[:,:,0]
    transf_mat = np.array([[1,0.25,0],[0,1,0],[0,0,1]]) # row, col, 1
    warped_img = affine_transform(frame,np.linalg.inv(transf_mat))
    frame = frame[50:200,50:200]
    warped_img = warped_img[50:200,50:200]
    plt.imshow(frame)
    plt.show()
    plt.imshow(warped_img)
    plt.show()
    M = LucasKanadeAffine(frame, warped_img, threshold, num_iter)
    print('Result 3: M = \n', M)
    




