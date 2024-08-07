import numpy as np

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    M = LucasKanadeAffine(image1,image2,threshold,num_iters)
    M = np.vstack((M,[0.0,0.0,1.0]))
    minv = np.linalg.inv(M)
    tr = affine_transform(image1,minv)
    
    im2 = affine_transform(image2,minv)
    diff = np.absolute(im2 - tr)

    mask = diff > tolerance
#     mask = ndimage.binary_erosion(mask).astype(mask.dtype)
    mask = ndimage.binary_dilation(mask).astype(mask.dtype)
#     mask = ndimage.binary_dilation(mask).astype(mask.dtype)

    

    return mask
