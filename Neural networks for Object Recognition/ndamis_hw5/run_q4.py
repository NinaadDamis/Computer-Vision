import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


print("INSIDE")


for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    print("OUTSIDE")
    avg_height = 0
    avg_width = 0
    for b in range(len(bboxes)) :
        box = bboxes[b]
        # print(box)
#         print("h = ", box[2] - box[0])

        avg_height += box[2] - box[0]
        avg_width += (box[1] + box[3]) / 2

    avg_height /= len(bboxes)
    avg_width /= len(bboxes)

    avg_height = int(avg_height)
    avg_width = int(avg_width)
    bboxes.sort(key = lambda x : x[0])
    sorted_bbox = bboxes

    # print(bboxes)

    threshold = avg_height 
    rows = []
    row1 = []
    for i in range(len(sorted_bbox)):
        if len(row1) != 0:
            if abs(row1[-1][0] - sorted_bbox[i][0]) < threshold:
                row1.append(sorted_bbox[i])
            else:
                # print("Row size ", len(row1))
                rows.append(row1)
                row1 = []
                row1.append(sorted_bbox[i])
        else:
            row1.append(sorted_bbox[i])

    rows.append(row1)
    new_rows = []

    # Sort according to minCol
    for i in range(len(rows)):
        newrow = rows[i]
        newrow.sort(key = lambda x : x[1]) # Sort with columns)
        new_rows.append(newrow)

    # new_rows = np.array(new_rows)
    all_letters = []
    for row in new_rows:

        for patch in row :
            
            x1,y1,x2,y2 = patch
            char = bw[x1:x2,y1:y2] 
            # print("Char shape ", char.shape)
            # skimage.io.imshow(char)
            char = np.pad(char,pad_width = 20)
            char = skimage.morphology.binary_erosion(char) 
            char = skimage.transform.resize(char, (32, 32)).T # Transpose it 
            flat = char.reshape(-1) # Reshape to 1024
            all_letters.append(flat)

    all_letters = np.array(all_letters)
    print("SHAPE LETTERS ", all_letters.shape)



    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))


    ##########################
    ##### your code here #####
    ##########################

    # Run Forward

    a = forward(all_letters,params,name='layer1',activation=sigmoid)
    out = forward(a,params,name='output',activation=softmax)

    preds = np.argmax(out,axis = 1)
    # print("PREDICTIONS : ", preds)
    pred_letters = letters[preds]


    # PRINTING LETTERS LINE BY LINE 
    count = 0
    print_row = []
    for i in range(len(new_rows)):
        curr_row = new_rows[i]
        print_row = []
        for j in range(len(curr_row)):
            print_row.append(pred_letters[count])
            count += 1  
        
        print(print_row)
    
    print("New Image")

    