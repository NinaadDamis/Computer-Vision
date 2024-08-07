import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''

def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    n = pts1.shape[0]
    P = np.zeros((n,3))
    
    for i in range(n):
        
        x = pts1[i,0]
        y = pts1[i,1]
        
        xd = pts2[i,0]
        yd = pts2[i,1]
        
        A = np.zeros((4,4))
#         print("YC!", y* C1[2,:])
        A[0,:] = y* C1[2,:] - C1[1,:]
        A[1,:] = C1[0,:] - x* C1[2,:]
        
        A[2,:] = yd* C2[2,:] - C2[1,:]
        A[3,:] = C2[0,:] - xd* C2[2,:]
        
        u,s,vh = np.linalg.svd(A)
#         print("In triangulate : ")
#         print(vh)
#         print("POINT", vh[-1,:])
        p = vh[-1,:]
        p = p / p[3] # Convert to inhomogenous coord by dividing by Z
        P[i,:] = p[0:3] # Pass only X,Y,Z
        
    # error calculation
    homo_P = np.hstack((P,np.ones((n,1))))
    err = 0
    for i in range(n):
        x1_proj = C1 @ homo_P[i,:]
        x2_proj = C2 @ homo_P[i,:]
        
        x1_proj = x1_proj / x1_proj[2] # Convert to inhomogenous coord 
        x2_proj = x2_proj / x2_proj[2]
        
        err1 = np.linalg.norm(pts1[i,:] - x1_proj[0:2]) **2
        err2 = np.linalg.norm(pts2[i,:] - x2_proj[0:2]) **2
        err += err1 + err2
        
    return P, err

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def findM2(F, pts1, pts2, intrinsics, filename = 'q3_3.npz'):
    '''
    Q2.2: Function to find the camera2's projective matrix given correspondences
        Input:  F, the pre-computed fundamental matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                pts2, the Nx2 matrix with the 2D image coordinates per row
                intrinsics, the intrinsics of the cameras, load from the .npz file
                filename, the filename to store results
        Output: [M2, C2, P] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)
    
    ***
    Hints:
    (1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
        of the projection error through best_error and retain the best one. 
    (2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

    '''

    n = pts1.shape[0]
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    E = essentialMatrix(F,K1,K2)
    M2s = camera2(E)
    best_error = 100000
#     best_P = np.zeros((n,3))
    best_C2 = np.zeros((3,4))
    best_M2 = np.zeros((3,4))
    max_pos = -1000000
    for i in range(4):
        
        M2 = M2s[:,:,i]
        M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
        C1 = K1 @ M1
        C2 = K2 @ M2
        P,err = triangulate(C1, pts1, C2, pts2)
        
        # Num positive Z
        print(" M", i , "  ERROR = ", err)
        pos = np.count_nonzero(P[:,2] > 0)
        print("POSITIVE COUNT = ", pos , "OUT OF ", P.shape[0])
        
        if pos > max_pos :
#         if err < best_error :
            print("CURRENT SELECTION = M", i)
            max_pos = pos
            best_error = err
#             best_P = P
            best_C2 = C2
            best_M2 = M2
        
    C1 = K1 @ M1
    P , err = triangulate(C1, pts1, best_C2, pts2)
    print("BEST ERROR = ", err)
            
        
        
    M2, C2 = best_M2,best_C2
    
    np.savez('q3_3.npz', M2 = M2, C2 = C2, P = P)

    return M2, C2, P

    # raise NotImplementedError()

    # return M2, C2, P



if __name__ == "__main__":

    # correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    # intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    # K1, K2 = intrinsics['K1'], intrinsics['K2']
    # pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    # im1 = plt.imread('data/im1.png')
    # im2 = plt.imread('data/im2.png')

    # F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    # M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # # Simple Tests to verify your implementation:
    # M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    # C1 = K1.dot(M1)
    # C2 = K2.dot(M2)
    # P_test, err = triangulate(C1, pts1, C2, pts2)
    # assert(err < 500)
    correspondence = np.load('../data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    print(im2.shape)

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, P = findM2(F, pts1, pts2, intrinsics)

    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    
    print("CHECKING ASSERTS !")
    print("ERROR = ", err)
    assert(err < 500)