import numpy as np
import cv2
import skimage.io 
import skimage.color
from opts import get_opts

# Import necessary functions

# Q2.2.4

def warpImage(opts):

    cv_d = cv2.imread('../data/cv_desk.png')
    cv_c = cv2.imread('../data/cv_cover.jpg')
    hp_c = cv2.imread('../data/hp_cover.jpg')
    
    cv_d_shape = cv_d.shape
    hp_c_shape = hp_c.shape
    cv_c_shape = cv_c.shape    
    
    matches,locs1,locs2 = matchPics(cv_c,cv_d,opts)
    l1 = locs1[matches[:,0],:]
    l2 = locs2[matches[:,1],:]
    print(l1.shape,l2.shape)
    
    H2to1, inliers = computeH_ransac(l1,l2,opts)
    
    hp_c_resize =  cv2.resize(hp_c, (cv_c.shape[1], cv_c.shape[0])) # Resizing
    composite = compositeH(H2to1, hp_c_resize, cv_d)

    plt.imshow(composite)
    plt.show()



if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


