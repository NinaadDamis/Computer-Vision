import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4

def matchPics(I1, I2, opts):

        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

        # TODO: Convert Images to GrayScale
        if(len(I1.shape) == 3):
            img1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        else:
            img1 = I1
            
        if(len(I2.shape) == 3):
            img2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        else:
            img2 = I2

        # TODO: Detect Features in Both Images
        
        locs11 = corner_detection(img1, sigma)
        locs22 = corner_detection(img2, sigma)
        
        descs1,locs1 = computeBrief(img1, locs11)
        descs2,locs2 = computeBrief(img2, locs22)
        
        matches = briefMatch(descs1,descs2,ratio)
        return matches, locs1, locs2
