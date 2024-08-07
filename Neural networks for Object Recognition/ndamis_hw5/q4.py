import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.util import img_as_float

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):

    bboxes = []
    bw = None
#     skimage.util.img_as_float(image, force_copy=False) Already input as float

    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################
    sigma_val = 1
#     noisy_img = skimage.util.random_noise(img, var=sigma**2)
#     io.imshow(noisy_img)
#     image = io.imread(i)
    gray_img = skimage.color.rgb2gray(image)

    noisy = skimage.filters.gaussian(gray_img, sigma=sigma_val)
    
    #Denoise
    denoise = denoise_tv_chambolle(noisy, weight=0.1, multichannel = True)
    
#     skimage.filters.try_all_threshold(gray, figsize=(20, 20), verbose=True)
    # Yen and otsu also show good results
    
    threshold = skimage.filters.threshold_isodata(denoise)
    mask = gray_img < threshold

    morph_open = skimage.morphology.opening(mask)
    # closed = skimage.morphology.closing(mask)

    labels = skimage.measure.label(morph_open)
    regions = skimage.measure.regionprops(labels)

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.imshow(1- morph_open, cmap = 'gray')
    avg_area = 0
    for region in regions :
        avg_area += region.area
    avg_area /= len(regions)
#     for region in regions:
    
    for region in regions:
        if region.area >= avg_area/2:
            minr, minc, maxr, maxc  = region.bbox
            arr = [minr, minc, maxr, maxc ]
            bboxes.append(arr)
    bw = 1 - morph_open


    return bboxes, bw