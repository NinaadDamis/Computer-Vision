import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation


parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args(args=[])
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect_initial = np.array([59, 116, 145, 151])
# print(seq.shape)
num_frames = seq.shape[2]
rects = []
rects.append(rect_initial)

blue_rects = []
blue_rects.append(rect_initial)

p_prev = np.zeros(2)
It = seq[:,:,0]
for i in range(num_frames-1):
    
    # Blue rectangle calculation
    b_It = seq[:,:,i]
    b_It1 = seq[:,:,i+1]
    b_rect = blue_rects[i]

    b_p = LucasKanade(b_It, b_It1, b_rect,threshold,num_iters)
    newRect = np.array([b_rect[0] + b_p[0], b_rect[1] + b_p[1], b_rect[2] + b_p[0], b_rect[3] + b_p[1]])
#     print("NexRect shape ", newRect.shape)
    blue_rects.append(newRect)

    
#     print("FRAME : " , i)
#     It = seq[:,:,i]
    It1 = seq[:,:,i+1]
    rect = rects[i]
    pn = LucasKanade(It, It1, rect,threshold,num_iters,p_prev)
#     print("Pn ", pn)
    pn_cp = np.copy(pn) #copy
    pn_cp = pn_cp + [rect[0] - rect_initial[0], rect[1] - rect_initial[1]]
#     print("pn_cp ", pn_cp )
    cpp = np.copy(pn_cp)
    pn_star = LucasKanade(seq[:,:,0], It1, rect_initial,threshold,num_iters,cpp)
#     print("Pn_star ", pn_star)
#     print("pn_cp ", pn_cp)

    norm = np.linalg.norm(pn_star - pn_cp)
#     print("Norm ", norm)
    if norm <= template_threshold :
#         newRect = np.array([rect[0] + dx, rect[1] + dy, rect[0] + dx + w, rect[1] + dy + h])
        p_prev= np.zeros(2)
        pp = pn_star - [rect[0] - rect_initial[0], rect[1] - rect_initial[1]]
        rect =  [rect[0] + pp[0], rect[1] + pp[1], rect[2] + pp[0], rect[3]+ pp[1]] 
        It = seq[:,:,i+1]
        rects.append(rect)
    else:
        p_prev = pn
        rect += np.array([pn[0],pn[1],pn[0],pn[1]])
#     print("NexRect shape ", newRect.shape)
        rects.append(rect)

    # Visualize
    if i in [1,100,200,300,400]:
        
        fig = plt.figure(1,figsize=(4,4))
#         plt.figure(figsize=(3, 3))

        ax = fig.add_subplot(111)
        ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], linewidth=2, edgecolor='red', fill=False))
        ax.add_patch(patches.Rectangle((b_rect[0], b_rect[1]), b_rect[2]-b_rect[0], b_rect[3]-b_rect[1], linewidth=2, edgecolor='blue', fill=False))

        plt.imshow(It1, cmap='gray')
        plt.show()
        ax.clear()


rects = np.array(rects)
print(rects.shape)
np.save('carseqrects-wcrt.npy',rects)

