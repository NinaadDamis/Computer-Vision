import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts

#Q2.1.6

def rotTest(opts):

    #Read the image and convert to grayscale, if necessary
    if(len(img.shape) ==3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    matches = []
    for i in range(36):

        #Rotate Image
        ri = scipy.ndimage.rotate(img,i*10)
        
        #Compute features, descriptors and Match features
        
        m,l1,l2 = matchPics(img,ri,opts)
        matches.append(m.shape[0])
#         print("Matches is ", m.shape[0])

    
    #Update histogram
#     print(matches)

    matches = np.array(matches)
    xaxis = 10*np.arange(36)
    plt.figure(figsize = (8,8))
    plt.bar(xaxis, matches, color='red',width = 4)
    plt.xlabel('Angle')
    plt.ylabel('Matches')
    plt.title('Matches VS Orientation')
    #Display histogram
    
    plt.show() 


if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
