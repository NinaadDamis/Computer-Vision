import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/antseq.npy')
num_frames = seq.shape[2]

for i in range(0,num_frames - 1):
#     print("FRAMEE ", i)
    It = seq[:,:,i]
    It1= seq[:,:,i+1]
    
    mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance)
    img = np.dstack([It1,It1,It1])
    where = np.where(mask == 1)
    img[where[0],where[1],:] = [1,0,0]
    
    if i in [30,60,90,120]:
        
#         plt.imshow(I, interpolation='nearest' )
        fig = plt.figure(1,figsize = (4,4))
        ax = fig.add_subplot(111)
        plt.imshow(img, interpolation = 'nearest')
        plt.show()
        ax.clear()
    