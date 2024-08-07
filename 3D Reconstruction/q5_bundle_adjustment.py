import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

import scipy
import math
import random

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=10):
    # Replace pass by your implementation
    max_iters = nIters
    n = pts1.shape[0]
    #homogenous coords
    pts1_h = np.hstack((pts1,np.ones((pts1.shape[0],1))))
    pts2_h = np.hstack((pts2,np.ones((pts2.shape[0],1))))
    c_set = np.zeros((n,1))
    best_F = np.zeros((3,3))
    max_inliers = 0
    max_idx = []
    for i in range(max_iters):
#         print("ITERATION NUMBER " , i)
        rand = random.sample(range(n), 8)        
        l1 = []
        l2 = []
        for j in rand :
            l1.append(pts1[j,0:2])
            l2.append(pts2[j,0:2])
        
        l1 = np.array(l1)
        l2 = np.array(l2)
        
        # Computing F
        F = eightpoint(l1,l2,M)

            
        error_array = calc_epi_error(pts1_h, pts2_h, F)
        print("error shape ", error_array.shape)
#         print("Error is " ,error)
        
        inliers_idx = np.where(error_array <= tol)  # Indexes where error less than tolerance
        inliers_idx = inliers_idx[0] # Getting array of inlier indexes
        num_inliers = inliers_idx.size # Number of inliers
        print("Num inliers ", num_inliers)
        
        if(num_inliers > max_inliers):
            max_inliers = num_inliers
            best_F = F
            max_idx = inliers_idx
    
    # Setting indexes of inliers to 1
    
    inliers = np.zeros((n,1))
    inliers[max_idx,0] = 1
    
    # Calculating F with the set of inliers

    newx1 = pts1[max_idx,0:2]
    newx2 = pts2[max_idx,0:2]
    F = eightpoint(newx1,newx2,M)
    
    return F, inliers



'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    theta = np.linalg.norm(r)
    if theta ==0 :
        return np.identity(3)
    u = r  / theta
    u = u.reshape((3,1))
    I = np.identity(3)
    first = I * math.cos(theta)
    second = (1 - math.cos(theta)) * u @ u.T
    
    ux = np.array([[0,-u[2,0],u[1,0]],[u[2,0],0,-u[0,0]], [-u[1,0],u[0,0],0] ])
    
    third = ux * math.sin(theta)

    R = first + second + third
    return R



'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    A = (R - R.T)/2
    ro = np.array([A[2,1],A[0,2],A[1,0]]).T
    s = np.linalg.norm(ro)
    c = (R[0,0] + R[1,1] + R[2,2] - 1)/2 
    
    if (s == 0 and c ==1 ):
        r = np.zeros((3,1))
        return r
    

    elif (s == 0 and c == -1):
        mat = R + np.identity(3)
        col = 0
        for i in range(3):
            if (mat[0,i]!= 0 or mat[1,i]!= 0 or mat[2,i] !=0):
                col = i
                break
        v = mat[:,col].reshape((3,1))
        u = v / np.linalg.norm(v)
        r = u * math.pi
        
        if(np.linalg.norm(r) == math.pi and ((r[0,0] == 0 and r[1,0] == 0 and r[2,0] < 0) or (r[0,0]
            == 0 and r[1,0] < 0) or (r[0,0] < 0))) :
            r = -r
    else :
        u = ro/s
        theta = math.atan2(s,c)
        r = u * theta
    
    return r


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    N = p1.shape[0]
    n = x.shape[0]
    P = x[0:-6].reshape((N,3))
    r2 =x[n -6:n-3].reshape((3,1))
    t2 =x[n-3:].reshape((3,1))
    R2 = rodrigues(r2)    
    M2 = np.hstack((R2,t2))
    C2 = K2 @ M2
    C1 = K1 @ M1
    
    P_homo = np.hstack((P,np.ones((N,1)))) # Homogenous
    
    proj_p1 = C1 @ P_homo.T # 3*n
    proj_p1 = proj_p1.T # n*3
    div = proj_p1[:,2].reshape((proj_p1.shape[0],1))
    proj_p1 = proj_p1 / np.hstack((div,div,div)) # Non homogenous coord
    proj_p1 = proj_p1[:,0:2]
    
    
    proj_p2 = C2 @ P_homo.T
    proj_p2 = proj_p2.T
    div2 = proj_p2[:,2].reshape((proj_p2.shape[0],1))

    proj_p2 = proj_p2 / np.hstack((div2,div2,div2)) # Non homogenous coord
    proj_p2 = proj_p2[:,0:2]
    residuals = np.concatenate([(p1-proj_p1).reshape([-1]),(p2-proj_p2).reshape([-1])])
    
    return residuals


'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    # x, the flattened concatenationg of P, r2, and t2.
    N = P_init.shape[0]
    P_flatten = P_init.flatten()
    R2 = M2_init[:,0:3]
    r2 = invRodrigues(R2).flatten()
    
    t2 = M2_init[:,-1].flatten()
    
    x = np.concatenate((P_flatten,r2,t2))
    n = x.shape[0]
#     print("X SHAPE ", n)
    
#     print("SHape of input x to residual", x.shape)    
    def obj_function(y):
        return np.sum(rodriguesResidual(K1,M1,p1,K2,p2,y)**2)
    
    obj_start = obj_function(x) #rodriguesResidual(K1, M1, p1, K2, p2, x)    
    obj_end = scipy.optimize.minimize(obj_function, x)
#     print(type(obj_end))
    obj_end = obj_end['x']
    
    P = obj_end[0:-6].reshape((N,3))
    r2 = obj_end[n-6:n-3]
    R2 = rodrigues(r2).reshape((3,3))
    t2 = obj_end[n-3:].reshape((3,1))
    M2 = np.hstack((R2,t2))
#     obj_start = obj_function(obj_start)
    obj_end = obj_function(obj_end)
#     print("OBJ START AND OBJ END ", obj_start , " , ", obj_end)
    return M2, P, obj_start, obj_end




if __name__ == "__main__":
              
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('../data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('../data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    print("FUNDAMENTAL MATRIX WITH RANSAC ", "NUM INLIERS =", np.sum(inliers))
    print(F)

    # YOUR CODE HERE


    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    

    # YOUR CODE HERE

    where = np.where(inliers == 1)   

    p1 = noisy_pts1[where[0],:]
    p2 = noisy_pts2[where[0],:]
    M2_init , C2_init ,P_init = findM2(F, p1, p2, intrinsics, filename = 'no.npz')
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))


    M2, P, obj_start, obj_end = bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init)
    print("FINISHED BUNDLE ADJUSTMENT")
    
   # Reprojection error using initial estimates
    C2 = K2 @ M2_init
    C1 = K1 @ M1 # Define M1
    n = P_init.shape[0]
    P_init_homo = np.hstack((P_init,np.ones((n,1)))) # Homogenous # DEFINE N SEPARATELY AGAIN
    proj_p1 = C1 @ P_init_homo.T # 3*n
    proj_p1 = proj_p1.T # n*3
    proj_p1 = proj_p1 / proj_p1[:,2].reshape((n,1)) # Non homogenous coord
    
    proj_p1 = proj_p1[:,0:2]
    
    proj_p2 = C2 @ P_init_homo.T
    proj_p2 = proj_p2.T
    proj_p2 = proj_p2 / proj_p2[:,2].reshape((n,1)) # Non homogenous coord
    
    proj_p2 = proj_p2[:,0:2]
    
    error = np.linalg.norm(p1 - proj_p1) **2 + np.linalg.norm(p2 - proj_p2)**2
    
    print("REPROJECTION ERROR WITH INITIAL ESTIMATES : ", error )

    ###########Remove test code ############
    F_eight_point = eightpoint(noisy_pts1,noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    print("FUNDAMENTAL MATRIX WITH EIGHTPOINT")
    print(F_eight_point)
    M2_pt , C2_pt ,P_pt = findM2(F_eight_point, p1, p2, intrinsics, filename = 'no.npz')
    C_pt = K2 @ M2_pt
    C1 = K1 @ M1 # Define M1
    
    P_homo_pt = np.hstack((P_pt,np.ones((n,1)))) # Homogenous
    
    proj_p1_pt = C1 @ P_homo_pt.T # 3*n
    proj_p1_pt = proj_p1_pt.T # n*3
    proj_p1_pt = proj_p1_pt / proj_p1_pt[:,2].reshape((n,1)) # Non homogenous coord
       
    proj_p1_pt = proj_p1_pt[:,0:2]
    
    proj_p2_pt = C2_pt @ P_homo_pt.T
    proj_p2_pt = proj_p2_pt.T
    proj_p2_pt = proj_p2_pt / proj_p2_pt[:,2].reshape((n,1)) # Non homogenous coord
    
    proj_p2_pt = proj_p2_pt[:,0:2]
    
    error_pt = np.linalg.norm(p1 - proj_p1_pt) **2 + np.linalg.norm(p2 - proj_p2_pt)**2
    print("REPROJECTION ERROR WITH EIGHTPOINT FOR COMPARISION = ", error_pt)

    ###########Remove Test Code #############
    
    
    # Reprojection  error using Final estimates 
    C2 = K2 @ M2
    C1 = K1 @ M1 # Define M1
    
    P_homo = np.hstack((P,np.ones((n,1)))) # Homogenous
    
    proj_p1 = C1 @ P_homo.T # 3*n
    proj_p1 = proj_p1.T # n*3
    proj_p1 = proj_p1 / proj_p1[:,2].reshape((n,1)) # Non homogenous coord
       
    proj_p1 = proj_p1[:,0:2]
    
    proj_p2 = C2 @ P_homo.T
    proj_p2 = proj_p2.T
    proj_p2 = proj_p2 / proj_p2[:,2].reshape((n,1)) # Non homogenous coord
    
    proj_p2 = proj_p2[:,0:2]
    
    error = np.linalg.norm(p1 - proj_p1) **2 + np.linalg.norm(p2 - proj_p2)**2
    print("REPROJECTION ERROR WITH FINAL ESTIMATES : ", error )
    
    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())

    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)

    plot_3D_dual(P_init, P)


    # YOUR CODE HERE