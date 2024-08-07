import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """
     x1 = 0
    y1 = 0
    x2 = It.shape[0]
    y2 = It.shape[1]
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.array([0,0,0,0,0,0]).astype('float64')
    
    xIt , yIt = np.arange(0,It.shape[0]), np.arange(0,It.shape[1])
    xIt1, yIt1 = np.arange(0,It1.shape[0]), np.arange(0,It1.shape[1])
    
    
    It_s = RectBivariateSpline(xIt, yIt, It)
    It1_s = RectBivariateSpline(xIt1, yIt1, It1)
    xxIt, yyIt = np.meshgrid(xIt,yIt)
    template = It_s.ev(yyIt,xxIt)
    Ix = It_s.ev(yyIt,xxIt, 0, 1)
    Iy = It_s.ev(yyIt,xxIt, 1, 0)
    
    # Compute A matrix
    A = np.zeros((xxIt.size,6))
    A[:,0] = Ix.flatten() * xxIt.flatten()
    A[:,1] = Iy.flatten() * xxIt.flatten()
    A[:,2] = Ix.flatten() * yyIt.flatten()
    A[:,3] = Iy.flatten() * yyIt.flatten()
    A[:,4] = Ix.flatten()
    A[:,5] = Iy.flatten()
    A_t = np.transpose(A)

    # Compute Hessian
    H = A_t@A
    Hinv = np.linalg.pinv(H)
    
    

    err = 1000000
    i = 0
    while(i< num_iters and err > threshold):
        
        xxt = M[0,0]*xxIt + M[0,1]*yyIt + M[0,2]
        yyt = M[1,0]*xxIt + M[1,1]*yyIt + M[1,2]
        indexs = (xxt >= 0) & (xxt < It1.shape[1]) & (yyt >= 0) & (yyt < It1.shape[0])
        yyt = yyt[indexs]
        xxt = xxt[indexs]


        I = It1_s.ev(yyt,xxt)
    
        xxt = xxt.flatten()
        yyt = yyt.flatten()
        b = -template[indexs] + I
        b = b.flatten()
        dp = Hinv @ A[indexs.flatten()].T @ b
        # Define dm
        dm = np.array([[1 + dp[0], dp[2],dp[4]],[dp[1],1 + dp[3],dp[5]],[0.0,0.0,1.0]])
        M = np.vstack((M,[0.0,0.0,1.0]))
        # Update
        M = M @ np.linalg.pinv(dm)
        M = M[0:2,:]
        err = np.linalg.norm(dp)
        i = i + 1

    return M