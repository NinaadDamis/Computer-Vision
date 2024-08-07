import numpy as np
from scipy.interpolate import RectBivariateSpline

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
	
 #     print("Rect is : ", rect)
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]

    xIt , yIt = np.arange(0,It.shape[0]), np.arange(0,It.shape[1])
    xIt1, yIt1 = np.arange(0,It1.shape[0]), np.arange(0,It1.shape[1])
    
#     print("Image shapes : Template and Image - ", It.shape, It1.shape)
    
    It_s = RectBivariateSpline(xIt, yIt, It)
    It1_s = RectBivariateSpline(xIt1, yIt1, It1)
    
    x2d = x2 + 0.1 # Added to include x2
    y2d = y2 + 0.1
    # Create array of x and y indices
    x = np.arange(x1, x2d)
    y = np.arange(y1, y2d)
    
    xx, yy = np.meshgrid(x,y)
    
    template = It_s.ev(yy,xx)
#     print("Rect Template shape - ", template.shape)
    err = 1000000
    i = 0
    while(i< num_iters and err > threshold):
        
        xt = np.arange(x1 + p0[0],x2 + 0.1 + p0[0])
        yt = np.arange(y1 + p0[1],y2 + 0.1 + p0[1])
        xxt, yyt = np.meshgrid(xt,yt)
        
        I = It1_s.ev(yyt,xxt)
        # Gradients
        Ix = It1_s.ev(yyt,xxt, 0, 1).flatten()
        Iy = It1_s.ev(yyt,xxt, 1, 0).flatten()
#         print("Evaluated shapes : I, Ix,Iy" , I.shape,Ix.shape,Iy.shape)
        A = np.zeros((Ix.size,2))
        A[:,0] = Ix
        A[:,1] = Iy
        b = template - I
        b = b.flatten()
        dp , residuals, rank,s= np.linalg.lstsq(A,b)
#         print("dp fromn least squares is :", dp)

        p0+= dp
        err = np.linalg.norm(dp)
        i = i + 1
    

#     print("P returned is : ", p0)
#     print(" P shape is ", p0.shape)

    return p0
        

