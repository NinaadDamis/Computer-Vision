import os
import math
import multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def get_feature_from_wordmap(opts, wordmap):
    """
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    """

    K = opts.K
    bins = np.arange(1+K)
    hist , b = np.histogram(wordmap,bins, density=True)
    hist = hist/ np.sum(hist) #Normalize the histogram
    
    return hist


def get_feature_from_wordmap_SPM(opts, wordmap):
    """
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    """

    K = opts.K
    L = opts.L
    
    weights = []
    for i in range(L+1):
        power = 0.0
        if(i ==0 or i ==1):
            power = float(1.0/2**L)
            weights.append(power)
            
        else:
            power = (2.0**(i-L-1))
            weights.append(power)
            
    weights = np.asarray(weights)
    
    max_rows = 2**L
    max_cols = 2**L
    
    h = wordmap.shape[0]
    w = wordmap.shape[1]
    
    # ----- TODO -----
    
    histogram_L = np.zeros((max_rows,max_cols,K))
    output = []
    
    for i in range(L,-1,-1):
        
        n_row = 2**i
        n_col = 2**i
        hei = int(h/n_row) #size of img height being considered
        wid = int(w/n_col)
        step = int(max_rows/n_row)
        output_new = []
        histogram_new = []
        
        if(i == L):
            
            for j in range(n_row):
                for k in range(n_col):
                    ri = j*hei
                    ro = (j+1)*hei
                    ci = k*wid
                    co = (k+1)*wid
                    sub = wordmap[ri:ro,ci:co ]
                    feat =   get_feature_from_wordmap(opts, sub)
                    histogram_L[j,k,:] = feat
                    feat = feat * weights[i]
                    output = np.hstack((output,feat))
                    
        else :
            
            for j in range(n_row):
                for k in range(n_col):
                    ri = j*step
                    ro = (j+1)*step
                    ci = k*step
                    co = (k+1)*step
                    histogram_new = histogram_L[ri:ro,ci:co,: ]
                    histogram_new = np.sum(histogram_new, axis = 0)
                    histogram_new = np.sum(histogram_new, axis = 0)
                    output_new= np.hstack([output_new, histogram_new*weights[i]])
                    
        output= np.hstack([output,output_new])      
        
    
    
    
    output = output / np.sum(output)
    hist_all = output

    return hist_all


def get_image_feature(opts, img_path, dictionary):
    """
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    """

    img = plt.imread(img_path)
    wordmap = get_visual_words(opts, img, dictionary)
    features = get_feature_from_wordmap_SPM(opts, wordmap)
    return features


def build_recognition_system(opts, n_worker=1):
    """
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, "dictionary.npy"))
    features = []
    i_count = 0
    for i in train_files :
        print("Image count is :", i_count)
        a = join(data_dir,i)
        feat = get_image_feature(opts, a, dictionary)
        if(i_count ==0):
            features = feat
            features = np.asarray(features)
            i_count+=1
        else:
            features = np.vstack((features,feat))
            i_count+=1

    print("DEBUG : SANITY CHECK BEFORE SAVING FILES :" ," LABEL SIZE = ", train_labels.shape, " feat size ", features.shape)
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )



def similarity_to_set(word_hist, histograms):
    """
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    """

    # ----- TODO -----
    pass


def evaluate_recognition_system(opts, n_worker=1):
    """
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    """

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    dictionary = trained_system["dictionary"]
  
    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]
    train_labels = trained_system["labels"]
    train_features = trained_system["features"]
    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)
    test_labels = np.asarray(test_labels)
    print("TEST LABEL SIZE ", test_labels.shape)

    pred_labels = []
    distances = []
    argminses = []
    
    
    for img in test_files:
        
        img_path = data_dir + "/" + img
        word_hist = get_image_feature(test_opts, img_path, dictionary)
        distance = distance_to_set(word_hist, train_features)
        distances.append(distance)
        argmin = np.argmin(distance)
        argminses.append(argmin)
        pred = train_labels[argmin]
        pred_labels.append(pred)
        
        
    pred_labels = np.asarray(pred_labels)
    conf=confusion_matrix(test_labels, pred_labels)
    accuracy = accuracy_score(test_labels,pred_labels)

    
#     conf, accuracy = None, None

    return conf, accuracy, pred_labels, test_labels, test_files

def compute_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass

def evaluate_recognition_System_IDF(opts, n_worker=1):
    # YOUR CODE HERE
    pass