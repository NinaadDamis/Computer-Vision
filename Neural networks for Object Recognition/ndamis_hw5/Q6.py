#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io
import torch
import torchvision.datasets
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim
import scipy.io
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# In[ ]:





# In[ ]:





# ## 6.1.1

# In[ ]:





# In[2]:


# Get Data from folder, load into tensors

device = torch.device('cpu')

test = scipy.io.loadmat('../data/nist36_test.mat')
train = scipy.io.loadmat('../data/nist36_train.mat')

x_test = test['test_data'].astype(np.float32)
labels_test = test['test_labels'].astype(np.int)

print("Test data and labels ", x_test.shape, labels_test.shape)

x_train = train['train_data'].astype(np.float32)
labels_train = train['train_labels'].astype(np.int)
print("Train data and labels ", x_train.shape, labels_train.shape)

x_test = torch.from_numpy(x_test)
# x_test = x_test.type(torch.float32)

labels_test = torch.from_numpy(labels_test)
# labels_test = labels_test.type(torch.int)

x_train = torch.from_numpy(x_train)
# x_train = x_train.type(torch.float32)

labels_train = torch.from_numpy(labels_train)
# labels_train = labels_train.type(torch.int)


torch.manual_seed(888)


# In[10]:


num_epochs = 50
batch_size = 64
learning_rate = 0.02
hidden_size = 64

test_loader = DataLoader(TensorDataset(x_test, labels_test), batch_size=batch_size)
train_loader = DataLoader(TensorDataset(x_train, labels_train), batch_size=batch_size, shuffle = True)


# In[11]:


class Net(nn.Module):
    def __init__(self, input_size = 1024, hidden_size = 64, output_size = 36):
        super(Net, self).__init__()

        layers = [nn.Linear(input_size,hidden_size),
        nn.Sigmoid(),
        nn.Linear(hidden_size, output_size)]
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        out = self.layers(x)

        return out


# In[ ]:





# In[12]:


model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()# TODO: What loss do you need for sequence to sequence models? 
# optimizer = torch.optim.Adam(model.parameters(), lr=0.002) # TODO: Adam works well with LSTM (use lr = 2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * num_epochs))


# In[13]:


# Train Model

model.train()

loss_arr = []
acc_arr = []

for itr in range(num_epochs):
    num_correct = 0
    total_loss = 0
    num_batches = 0
    
    for i, data in enumerate(train_loader):
#             print("Data shape ",type(data), data[0].shape, "Target shape ", data[1].shape)
        optimizer.zero_grad()
        x = data[0].to(device)
        y = data[1].to(device)
        out = model(x)
        y = torch.max(y, 1)[1]
        loss = criterion(out, y)
        
#         print("LOSS , LOSS ITEM ", loss, loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += float(loss)

        pred = torch.max(out, 1)[1] 
        num_correct += pred.eq(y).sum().item()
        num_batches += 1

    acc_arr.append(100. * num_correct / (num_batches*batch_size))
    loss_arr.append(total_loss/num_batches)
    print("Accuracy  = " , 100. * num_correct / (num_batches*batch_size), "Loss = ", total_loss/num_batches)


# In[14]:



plt.figure()
plt.plot(np.arange(num_epochs), [i*100 for i in acc_arr], label = "Train Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Train Accuracy (%)")
plt.title("Train Accuracy (%) vs Iterations")
plt.show()


# In[15]:



plt.figure()
plt.plot(np.arange(num_epochs), loss_arr, label = "Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss ")
plt.title("Loss (%) vs Iterations")
plt.show()


# In[17]:


model.eval()

loss_arr = []
acc_arr = []

num_correct = 0
total_loss = 0
num_batches = 0

with torch.no_grad() :
    for i, data in enumerate(test_loader):
    #             print("Data shape ",type(data), data[0].shape, "Target shape ", data[1].shape)

        x = data[0].to(device)
        y = data[1].to(device)
        out = model(x)
        y = torch.max(y, 1)[1]
        loss = criterion(out, y)

    #         print("LOSS , LOSS ITEM ", loss, loss.item())
        total_loss += float(loss)

        pred = torch.max(out, 1)[1] 
        num_correct += pred.eq(y).sum().item()
        num_batches += 1

    acc_arr.append(100. * num_correct / (num_batches*batch_size))
    loss_arr.append(total_loss/num_batches)
    print("Accuracy  = " , 100. * num_correct / (num_batches*batch_size), "Loss = ", total_loss/num_batches)


# In[112]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 6.1.2

# In[ ]:





# In[ ]:





# In[18]:



device = torch.device('cpu')

test = scipy.io.loadmat('../data/nist36_test.mat')
train = scipy.io.loadmat('../data/nist36_train.mat')

x_test = test['test_data'].astype(np.float32)
labels_test = test['test_labels'].astype(np.int)

print("Test data and labels ", x_test.shape, labels_test.shape)

x_train = train['train_data'].astype(np.float32)
labels_train = train['train_labels'].astype(np.int)
print("Train data and labels ", x_train.shape, labels_train.shape)

x_test = torch.from_numpy(x_test.reshape((x_test.shape[0],1,32,32)))
# x_test = x_test.type(torch.float32)

labels_test = torch.from_numpy(labels_test)
# labels_test = labels_test.type(torch.int)

x_train = torch.from_numpy(x_train.reshape((x_train.shape[0], 1, 32, 32)))
# x_train = x_train.type(torch.float32)

labels_train = torch.from_numpy(labels_train)
# labels_train = labels_train.type(torch.int)


torch.manual_seed(888)

num_epochs = 10
batch_size = 64
learning_rate = 0.02
hidden_size = 64

test_loader = DataLoader(TensorDataset(x_test, labels_test), batch_size=batch_size)
train_loader = DataLoader(TensorDataset(x_train, labels_train), batch_size=batch_size, shuffle = True)


# In[19]:


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        layers = [nn.Conv2d(1,16,3,1,1),
        nn.ReLU(),
        nn.Conv2d(16,32,3,1,1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 32 * 32, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256,36)]
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        out = self.layers(x)

        return out


# In[ ]:





# In[20]:


model = ConvNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()# TODO: What loss do you need for sequence to sequence models? 
# optimizer = torch.optim.Adam(model.parameters(), lr=0.002) # TODO: Adam works well with LSTM (use lr = 2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(train_loader) * num_epochs))


# In[21]:


# Train Model

model.train()

loss_arr = []
acc_arr = []

for itr in range(num_epochs):
    num_correct = 0
    total_loss = 0
    num_batches = 0
    
    for i, data in enumerate(train_loader):
#             print("Data shape ",type(data), data[0].shape, "Target shape ", data[1].shape)
        optimizer.zero_grad()
        x = data[0].to(device)
        n = x.shape[0]
#         x = x.reshape((n,1,32,32))
        y = data[1].to(device)
        out = model(x)
        y = torch.max(y, 1)[1]
        loss = criterion(out, y)
        
#         print("LOSS , LOSS ITEM ", loss, loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += float(loss)

        pred = torch.max(out, 1)[1] 
        num_correct += pred.eq(y).sum().item()
        num_batches += 1

    acc_arr.append(100. * num_correct / (num_batches*batch_size))
    loss_arr.append(total_loss/num_batches)
    print("Accuracy  = " , 100. * num_correct / (num_batches*batch_size), "Loss = ", total_loss/num_batches)


# In[22]:



plt.figure()
plt.plot(np.arange(num_epochs), [i*100 for i in acc_arr], label = "Train Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Train Accuracy (%)")
plt.title("Train Accuracy (%) vs Iterations")
plt.show()


# In[23]:



plt.figure()
plt.plot(np.arange(num_epochs), loss_arr, label = "Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss ")
plt.title("Loss (%) vs Iterations")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 6.1.3

# In[ ]:





# In[ ]:





# In[3]:


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 10

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)


# In[4]:


class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()

        layers = [nn.Conv2d(3,16,3,1,1),
        nn.ReLU(),
        nn.Conv2d(16,32,3,1,1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 32 * 32, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256,10)]
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        out = self.layers(x)

        return out


# In[6]:


num_epochs = 10
# batch_size = 16
learning_rate = 0.02
hidden_size = 64
model = CifarNet()
optimizer = torch.optim.SGD(model.parameters(), lr=0.025, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()# TODO: What loss do you need for sequence to sequence models? 
# optimizer = torch.optim.Adam(model.parameters(), lr=0.002) # TODO: Adam works well with LSTM (use lr = 2e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(len(trainloader) * num_epochs))


# In[8]:


model.train()

loss_arr = []
acc_arr = []

for itr in range(num_epochs):
    num_correct = 0
    total_loss = 0
    num_batches = 0
    print("Epoch", itr, total_loss)
    for i, data in enumerate(trainloader):
        #print("Data shape ",type(data), data[0].shape, "Target shape ", data[1].shape)
        optimizer.zero_grad()
        x = data[0].to(device)
        n = x.shape[0]
#         x = x.reshape((n,1,32,32))
        y = data[1].to(device)
        #print("Y SHAPE ", y.shape, y.data)
        out = model(x)
        #out = torch.max(out.data, 1)[1]
        #print("Type ", y.shape, out.shape)
        loss = criterion(out, y)
        
#         print("LOSS , LOSS ITEM ", loss, loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += float(loss)

        pred = torch.max(out, 1)[1] 
        num_correct += pred.eq(y).sum().item()
        num_batches += 1

    acc_arr.append(100. * num_correct / (num_batches*batch_size))
    loss_arr.append(total_loss/num_batches)
    print("Accuracy  = " , 100. * num_correct / (num_batches*batch_size), "Loss = ", total_loss/num_batches)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 6.3

# In[ ]:




