import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
#     print(x1.shape,x2.shape)
    n = x1.shape[0]
    A = []
    for i in range (n) : # QUESTION - how to choose n if n is different for each x1 and x2
        
        xd = x1[i,0]
        yd = x1[i,1]
        x = x2[i,0]
        y = x2[i,1]
        
        A1 = [-x, -y, -1, 0, 0, 0, x*xd, y*xd, xd]
        A2 = [0, 0, 0, -x, -y, -1, x*yd, y*yd, yd]
        A.append(A1)
        A.append(A2)
        
    A = np.array(A)
#     eigval, eigvec = np.linalg.eig(A)
#     ind = np.argmin(eigval)
#     H = eigvec[:,ind]
#     H2to1 = H.reshape((3,3))

    u, s, vh = np.linalg.svd(A)
    h = vh[-1,:] / vh[-1,-1]
    H2to1 = h.reshape((3,3))
    


    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
#     print(x1.shape,x2.shape)
    cx1 = np.mean(x1,axis = 0)
    cx2 = np.mean(x2,axis = 0)
    #Shift the origin of the points to the centroid
    x1 = x1 - cx1
    x2 = x2 - cx2
    

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    
    x1sq = np.max(np.linalg.norm(x1,axis = 1))
    x2sq = np.max(np.linalg.norm(x2,axis = 1)) #np.max((np.sqrt(np.sum(x2**2, axis = 1))))
    
    scale1 = np.sqrt(2)/ x1sq
    scale2 = np.sqrt(2)/ x2sq

    #Similarity transform 1

    s1 = [[scale1,0,-scale1*cx1[0]], [0,scale1,-scale1*cx1[1]], [0,0,1]]
    s1 = np.asarray(s1)
    print("S1")
    print(s1)

    #Similarity transform 2

    s2 = [[scale2,0,-scale2*cx2[0]], [0,scale2,-scale2*cx2[1]], [0,0,1]]
    s2 = np.asarray(s2)
    print("S2")
    print(s2)
    #Compute homography
    H2to1 = computeH(x1,x2)
    print(H2to1)
    #Denormalization
        
    H2to1 = np.linalg.inv(s1)@H2to1@s2

    return H2to1



def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters
    tol = opts.inlier_tol
    
    #homogenize
    locs1 = np.hstack((locs1,np.zeros((locs1.shape[0],1)))) # Stacking np.ones 
    locs2 = np.hstack((locs2,np.zeros((locs2.shape[0],1))))
    n = locs1.shape[0]

    c_set = np.zeros((n,1))
    best_H = np.zeros((3,3))
    max_inliers = 0
    max_idx = []
    for i in range(max_iters):
        rand = random.sample(range(n), 4)
        
        # Creating random sample of 4 points 
        
        l1 = []
        l2 = []
        for j in rand :
            l1.append(locs1[j,0:2])
            l2.append(locs2[j,0:2])
        
        l1 = np.array(l1)
        l2 = np.array(l2)
        
        # Computing H
        H2to1 = computeH_norm(l1,l2)
        
#         print(H2to1.shape)
        
        proj = np.matmul(H2to1,np.transpose(locs2)) # This is the estimated value of locs1
        print("Projection shape ", proj.shape)
        proj = np.transpose(proj) # Conerting from 3*N to N*3         
        error = np.sqrt(np.sum((proj - locs1)**2,axis = 1)) # Calculating error
        print("error shape ", error.shape)
        print("Error is " ,error)
        
        inliers_idx = np.where(error <= tol)  # Indexes where error less than tolerance
        print("INLIERS IDX ", inliers_idx)
        print("INLIERS IDX_0 ", inliers_idx[0])
        inliers_idx = inliers_idx[0] # Getting array of inlier indexes


#         print("Shape", inliers_idx[0].shape,type(inliers_idx[0]))
#         inliers_idx = np.array(inliers_idx[0])
        
        num_inliers = inliers_idx.size # Number of inliers
        print("Num inliers ", num_inliers)
        
        if(num_inliers > max_inliers):
            max_inliers = num_inliers
            best_H = H2to1
            max_idx = inliers_idx
    
    # Setting indexes of inliers to 1
    
    inliers = np.zeros((n,1))
    inliers[max_idx,0] = 1
    
    # Calculating H with the set of inliers

    newx1 = locs1[max_idx,0:2]
    newx2 = locs2[max_idx,0:2]
    
    H2to1 = computeH_norm(newx1,newx2)
    
    return H2to1, inliers



def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    

    #Create mask of same size as template

    #Warp mask by appropriate homography

    #Warp template by appropriate homography

    #Use mask to combine the warped template and the image
    
    return composite_img


