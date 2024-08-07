import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF

# Insert your package here



'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to usethe normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
    num_pts = pts1.shape[0]
    A = np.zeros((num_pts,9))
    h_pts1 = toHomogenous(pts1)
    h_pts2 = toHomogenous(pts2)
    norm_pts1 = h_pts1 @ T
    norm_pts2 = h_pts2 @ T
    
    for i in range(num_pts):
        
        xd = norm_pts1[i,0]
        yd = norm_pts1[i,1]
        x = norm_pts2[i,0]
        y = norm_pts2[i,1]
        Ao = [ xd*x,xd*y,xd,yd*x,yd*y,yd,x,y,1 ]
        A[i,:] = Ao
        
    u, s,vh = np.linalg.svd(A)
    F = vh[-1, :].reshape((3,3)).T
    F_refine = refineF(F, norm_pts1[:,0:2], norm_pts2[:,0:2]) # Cant pass homogenous coords
    F_refine = F_refine / F_refine[2,2] # Divide by last element, to satisfy assert F[2,2] = 1

    F_unscale = T.transpose() @ F_refine @ T

    np.savez('q2_1.npz', F=F_unscale, M=M)

    
    return F_unscale
    




if __name__ == "__main__":
        
    # correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    # intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    # K1, K2 = intrinsics['K1'], intrinsics['K2']
    # pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    # im1 = plt.imread('data/im1.png')
    # im2 = plt.imread('data/im2.png')

    # F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    # # Q2.1
    # # Write your code here
    # displayEpipolarF(im1, im2, F)




    # # Simple Tests to verify your implementation:
    # pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    # assert(F.shape == (3, 3))
    # assert(F[2, 2] == 1)
    # assert(np.linalg.matrix_rank(F) == 2)
    # assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)


    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    # Q2.1
    # Write your code here
    displayEpipolarF(im1, im2, F)
    

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)