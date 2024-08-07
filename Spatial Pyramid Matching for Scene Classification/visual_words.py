import os
import multiprocessing
from os.path import join, isfile

import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
from sklearn.cluster import KMean
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def extract_filter_responses(opts, img):
    """
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    """

    h = img.shape[0]
    w = img.shape[1]
    channels = img.shape[2]
    n_scales = len(opts.filter_scales)
    f =  4*3*n_scales
    
    # If greyscale image
    
    if(channels == 1):
        cp = img[:,:,0]
        cp[:,:,1] = img[:,:,0]
        cp[:,:,2] = img[:,:,0]
        
        img = cp
    
    # If more than 3 channels
    
    if(channels > 3):
        img = img[:,:,0:3]
    

    img = img.astype(float) / 255.
    skimage.color.rgb2lab(img)

    filter_responses = np.empty((h,w,f))
    filter_scales = opts.filter_scales
    
    ind = 0
    for i in filter_scales :
        
        for j in range(3):
            
            result = scipy.ndimage.gaussian_filter(img[:,:,j],i)
            filter_responses[:,:,ind] = result
            ind = ind + 1
        
        for j in range(3):
            
            result = scipy.ndimage.gaussian_laplace(img[:,:,j],i)
            filter_responses[:,:,ind] = result
            ind = ind + 1
                    
        for j in range(3):
            
            result = scipy.ndimage.gaussian_filter(img[:,:,j],i,order = [0,1])
            filter_responses[:,:,ind] = result
            ind = ind + 1
            
        for j in range(3):
            
            result = scipy.ndimage.gaussian_filter(img[:,:,j],i,order = [1,0])
            filter_responses[:,:,ind] = result
            ind = ind + 1
        

    return filter_responses


def compute_dictionary_one_image(args):
    """
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    """
    opts, idx, img_path = args
    

    img = plt.imread(img_path)
    image_responses = extract_filter_responses(opts, img)
    filter_responses = np.empty((opts.alpha,4*3*len(opts.filter_scales)))

    h = img.shape[0]
    w = img.shape[1]
    pixel = [] # This is an array containing the samples alpha coorrdinate points
    count = h*w -1
    for i in range(opts.alpha):
        temp = np.random.randint(0,count)
        column = temp % w
        row    = int(temp / w)

        coord = [row,column]
        pixel.append(coord)
        
    pixel_count = 0
    for i in pixel:
        r = i[0]
        c = i[1]
        temp = image_responses[r,c,:]
        filter_responses[pixel_count,:] = temp
        pixel_count +=1
        
    name = "feat" + str(idx)
    np.save(os.path.join(opts.feat_dir, name),filter_responses)


def compute_dictionary(opts, n_worker=1):
    """
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel

    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    """

 
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    image_count = 1
    for i in train_files :
        print("Image count :", image_count)
        imgpath = join(data_dir, i)
        args = [opts,image_count,imgpath]
        compute_dictionary_one_image(args)
        image_count +=1

    
    filter_responses = np.zeros((opts.alpha,4*3*len(opts.filter_scales)))
    file_count = 0
    for filename in os.listdir(feat_dir):
        if filename.endswith('.npy'):
            if(file_count == 0):
                
                filter_responses[:,:] = np.load(os.path.join(feat_dir,filename))
                file_count+=1
                
            else:
                
                filter_responses = np.vstack((filter_responses,np.load(os.path.join(feat_dir,filename))))
                file_count +=1
                
    kmeans = KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    
    np.save('dictionary',dictionary)



def get_visual_words(opts, img, dictionary):
    """
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)

    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    """

    filter_responses = extract_filter_responses(opts,img)
    h = filter_responses.shape[0]
    w = filter_responses.shape[1]
    f = filter_responses.shape[2]
    filter_responses = np.reshape(filter_responses, (-1, f))
    dis = cdist( filter_responses, dictionary)
    test = dis[:2]
    minim = np.argmin(test)
    wordmap = np.argmin(dis,axis=1) # Columns = 1 - > Values between 1 and k 
    wordmap= wordmap.reshape(h,w)

    return wordmap