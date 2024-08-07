#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models


# In[3]:





# In[18]:


# Used this link as regference 
# https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/


# In[4]:


alexnet = models.alexnet(pretrained=True)


# In[52]:


transform = transforms.Compose([            
transforms.Resize(256),   
transforms.CenterCrop(224), 
transforms.ToTensor(), 
transforms.Normalize(      
mean=[0.485, 0.456, 0.406],   
std=[0.229, 0.224, 0.225])])


# In[53]:


from PIL import Image


import cv2
images = np.zeros((316,3,640,352))
vidcap = cv2.VideoCapture('vid.mp4')
success, image = vidcap.read()
count = 1
while success:
    success, image = vidcap.read()
    if(success == False):
        break
    image = image.transpose(2,0,1)    
    images[count - 1,:,:,:] = image
        count += 1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


import cv2
import numpy as np


# In[49]:


with open('imagenet_classes.txt') as f:

    classes = [line.strip() for line in f.readlines()]
labels = classes[4:]
print(labels[0:5])


# In[50]:


from PIL import Image


import cv2
images = np.zeros((316,3,640,352))
vidcap = cv2.VideoCapture('vid.mp4')
success, image = vidcap.read()
count = 1
while success:
#     cv2.imwrite("video_data/image_%d.jpg" % count, image)    
    success, image = vidcap.read()
    if(success == False):
        break
#     print("Image ", type(image), image.shape)
    image = image.transpose(2,0,1)
#     print("New ", type(image), image.shape)
    
    images[count - 1,:,:,:] = image
    
#     print('Saved image ', count)
    count += 1
# img = Image.open("dog.jpg")
# print(img.shape)
# img = images[5,:,:,:]
# print(img.shape)
# PIL_image = Image.fromarray(np.uint8(img)).convert('RGB')


# In[62]:


for i in range(images.shape[0]):
    


    img = images[i,:,:,:]
    img = img.transpose(1,2,0)
#     print(img.shape)
    data = img.astype(np.uint8)
    img = Image.fromarray(data, 'RGB')
# img.save('my.png')
    
    
    img_t = transform(img)
#     print(img_t.shape)

    batch_t = torch.unsqueeze(img_t, 0)

    out = alexnet(batch_t)
    _, index = torch.max(out, 1)
#     print("Index = ", index.item())
    if(index.item() == 898 or index.item() == 897 or index.item() == 899 or index.item() == 720):
        print("*************************************************")
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
#     print(labels[index[0]])


# In[ ]:





# I tested a Imagenet based Alexnet on a video of a bottle I took. It was able to detect the bottle in only 5 frames out of 316, which is 1.5 %. 
# 

# In[ ]:




