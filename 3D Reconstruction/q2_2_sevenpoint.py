import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here


'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two colum vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Sovling this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):

    Farray = []

    
    T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]]) # Transform
    num_pts = 7
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
    F1 = vh[-1, :].reshape((3,3)).T
    F2 = vh[-2, :].reshape((3,3)).T
    
    b1 = np.linalg.det(F2)
    b2 = np.linalg.det(F1)
    b3 = np.linalg.det(2*F2 - F1)
    b4 = np.linalg.det(2*F1 - F2)
    b = np.array([b1,b2,b3,b4])
    
    E = np.array([[0,0,0,1],[1,1,1,1],[-1,1,-1,1],[8,4,2,1]])
    coeffs = np.linalg.solve(E,b)
    
    print("Coeffiecients of cubic ", coeffs)
    
    roots = np.polynomial.polynomial.polyroots((coeffs[0],coeffs[1],coeffs[2],coeffs[3]))
    print("ROOTS", roots)

    
    for i in roots :
        if np.imag(i) == 0 :
            F = i * F1 + (1 - i) * F2


            F_refine = refineF(F, norm_pts1[:,0:2], norm_pts2[:,0:2]) # Cant pass homogenous coords
            F_refine = F_refine / F_refine[2,2] # Divide by last element, to satisfy assert F[2,2] = 1

            F_unscale = T.transpose() @ F_refine @ T
            Farray.append(F_unscale)
        
    
    
    
    return Farray







if __name__ == "__main__":
        
    # correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    # intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    # K1, K2 = intrinsics['K1'], intrinsics['K2']
    # pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    # im1 = plt.imread('data/im1.png')
    # im2 = plt.imread('data/im2.png')


    # # ----- TODO -----
    # # YOUR CODE HERE


    
    # # Simple Tests to verify your implementation:
    # # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    # np.random.seed(1) #Added for testing, can be commented out
    
    # pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    # max_iter = 500
    # pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    # pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    # ress = []
    # F_res = []
    # choices = []
    # M=np.max([*im1.shape, *im2.shape])
    # for i in range(max_iter):
    #     choice = np.random.choice(range(pts1.shape[0]), 7)
    #     pts1_choice = pts1[choice, :]
    #     pts2_choice = pts2[choice, :]
    #     Fs = sevenpoint(pts1_choice, pts2_choice, M)
    #     for F in Fs:
    #         choices.append(choice)
    #         res = calc_epi_error(pts1_homo,pts2_homo, F)
    #         F_res.append(F)
    #         ress.append(np.mean(res))
            
    # min_idx = np.argmin(np.abs(np.array(ress)))
    # F = F_res[min_idx]
    # print("Error:", ress[min_idx])

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


    # ----- TODO -----
    # YOUR CODE HERE


    
    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        # print("ITERATION NUMBER = ", i)
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
#             print("EPI_ERROR = ", res)
            F_res.append(F)
            ress.append(np.mean(res))
#             print("LENGTH RESS = ", len(ress))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    # print("Error:", ress[min_idx])
    
    # print("CHECKING ASSERTS!")
    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)
    np.savez('q2_2.npz', F= F, M = M)
    print("FUNDAMENTAL MATRIX F =")
    print(F)
    
    
    # Display
    
    displayEpipolarF(im1, im2, F)
