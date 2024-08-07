import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# write your script here, we recommend the above libraries for making your animation 1, 100, 200, 300 and 400 i

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args(args=[])
# args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = np.array([59, 116, 145, 151])
# print(seq.shape)
num_frames = seq.shape[2]
rects = []
rects.append(rect)

for i in range(num_frames-1):
#     print("FRAME : " , i)
    rect = rects[i]
    It1 = seq[:,:,i+1] # Image
    It = seq[:,:,i] # Template

    p = LucasKanade(It, It1, rect,threshold,num_iters)
    
#     print("P is :", p)

    newRect = np.array([rect[0] + p[0], rect[1] + p[1], rect[2] + p[0], rect[3] + p[1]])
#     print("NexRect shape ", newRect.shape)
    rects.append(newRect)

    if i in [1,100,200,300,400]:
        
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        patch = patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], edgecolor='blue', fill=False)
        ax.add_patch(patch)
        plt.imshow(It1, cmap='gray')
        plt.show()
        ax.clear()

# print("Shape rectangle array : ",rects.shape)
rects = np.array(rects)
print("Shape rectangle array : ",rects.shape)
np.save('carseqrects.npy', rects)


