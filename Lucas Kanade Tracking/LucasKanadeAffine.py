import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
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
    
#     print("Image shapes : Template and Image - ", It.shape, It1.shape)
    
    It_s = RectBivariateSpline(xIt, yIt, It)
    It1_s = RectBivariateSpline(xIt1, yIt1, It1)
#     Iy , Ix = np.gradient(It1)
    xxIt, yyIt = np.meshgrid(xIt,yIt)
    template = It_s.ev(yyIt,xxIt)
    #     Iy , Ix = np.gradient(It1)

    err = 1000000
    i = 0
    while(i< num_iters and err > threshold):
        
#         xt1_t, yt1_t = np.meshgrid(xIt, yIt)
#         Minv = np.linalg.inv(M)
# #         print("Evaluated shapes : I, Ix,Iy" , I.shape,Ix.shape,Iy.shape)
# #         template = affine_transform(It,Minv)
#         I = affine_transform(It1,Minv)
#         template = affine_transform(It,Minv)
# #         print("SHapes : ", I.shape, Ix.shape, Iy.shape)

#         Ix = affine_transform(Ix,Minv)
#         Iy = affine_transform(Iy,Minv)
        
        
        xxt = M[0,0]*xxIt + M[0,1]*yyIt + M[0,2]
        yyt = M[1,0]*xxIt + M[1,1]*yyIt + M[1,2]
#         print("Warp shape ", xxt.shape,yyt.shape)
        #To prevent going out in the frame.
        indexs = (xxt >= 0) & (xxt < It1.shape[1]) & (yyt >= 0) & (yyt < It1.shape[0])
#         print("Index shape ", indexs.shape)
        yyt = yyt[indexs]
        xxt = xxt[indexs]
#         print("Size after filtering ", xt1.shape, yt1.shape)


        I = It1_s.ev(yyt,xxt)
        
        Ix = It1_s.ev(yyt,xxt, 0, 1)
        Iy = It1_s.ev(yyt,xxt, 1, 0)
#         print("Evaluated shapes : I, Ix,Iy" , I.shape,Ix.shape,Iy.shape)
        xxt = xxt.flatten()
        yyt = yyt.flatten()

        A = np.zeros((xxt.size,6))
        A[:,0] = Ix.flatten() * xxt
        A[:,1] = Iy.flatten() * xxt
        A[:,2] = Ix.flatten() * yyt
        A[:,3] = Iy.flatten() * yyt
        A[:,4] = Ix.flatten()
        A[:,5] = Iy.flatten()
        b = template[indexs] - I
        b = b.flatten()
        
        
        dp,residuals, rank,s= np.linalg.lstsq(A,b)
        p += dp
        err = np.linalg.norm(dp)
        i = i + 1
        
        M[0][0] = 1 + p[0]
        M[0][1] = p[2]
        M[0][2] = p[4]
        M[1][0] = p[1]
        M[1][1] = 1 + p[3]
        M[1][2] = p[5]
        
#     print("P returned is : ", p0)
#     print(" P shape is ", p0.shape)


    return M
