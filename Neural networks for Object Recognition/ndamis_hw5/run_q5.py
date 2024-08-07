import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
print("TYPE", type(params))
initialize_weights(1024,32,params,'layer1') 
print("COUNTER",params.keys())
initialize_weights(32,32,params,'layer2') 
initialize_weights(32,32,params,'layer3') 
initialize_weights(32,1024,params,'output') 

print("COUNTER",params.keys())

print("Param shape",params['Woutput'].shape )
# Add zero initialzion momentum variables
params['m_Woutput'] = np.zeros(params['Woutput'].shape)
params['m_boutput'] = np.zeros(params['boutput'].shape)

 
params['m_Wlayer3'] = np.zeros(params['Wlayer3'].shape)
params['m_blayer3'] = np.zeros(params['blayer3'].shape)

params['m_Wlayer2'] = np.zeros(params['Wlayer2'].shape)
params['m_blayer2'] = np.zeros(params['blayer2'].shape)

params['m_Wlayer1'] = np.zeros(params['Wlayer1'].shape)
params['m_blayer1'] = np.zeros(params['blayer1'].shape)

# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:

        ##########################
        ##### your code here #####
        ##########################
        a = forward(xb,params,name='layer1',activation=relu)
        h1 = forward(a,params,name='layer2',activation=relu)
        h2 = forward(h1,params,name='layer3',activation=relu)
        out = forward(h2,params,name='output',activation=sigmoid)
        # loss, accuracy = compute_loss_and_acc(yb,out)
        loss = np.sum(np.power(xb - out,2)) # square of difference
        # print("L1 ", loss)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss
        # avg_acc +=accuracy

        # backward
        delta = (out - xb) * 2
        d1 = backwards(delta,params,name='output',activation_deriv=sigmoid_deriv)
        d2 = backwards(d1,params,name='layer3',activation_deriv=relu_deriv)
        d3 = backwards(d2,params,name='layer2',activation_deriv=relu_deriv)
        d4 = backwards(d3,params,name='layer1',activation_deriv=relu_deriv)

        # Add m_ and create momentum variables
        # Npot += for momentum vars, overflow encountered error
        params['m_' +'Woutput' ] = 0.9 * params['m_' +'Woutput' ] - learning_rate *  params['grad_Woutput']  
        params['m_' +'boutput' ] = 0.9 * params['m_' +'boutput' ] - learning_rate *  params['grad_boutput']  
        params['Woutput'] += params['m_Woutput']
        params['boutput'] += params['m_boutput']

        #Layer 3,2,1

        params['m_' +'Wlayer3' ] = 0.9 * params['m_' +'Wlayer3' ] - learning_rate *  params['grad_Wlayer3']  
        params['m_' +'blayer3' ] = 0.9 * params['m_' +'blayer3' ] - learning_rate *  params['grad_blayer3']  
        params['Wlayer3'] += params['m_Wlayer3']
        params['blayer3'] += params['m_blayer3']

        params['m_' +'Wlayer2' ] = 0.9 * params['m_' +'Wlayer2' ] - learning_rate *  params['grad_Wlayer2']  
        params['m_' +'blayer2' ] = 0.9 * params['m_' +'blayer2' ] - learning_rate *  params['grad_blayer2']  
        params['Wlayer2'] += params['m_Wlayer2']
        params['blayer2'] += params['m_blayer2']

        params['m_' +'Wlayer1' ] = 0.9 * params['m_' +'Wlayer1' ] - learning_rate *  params['grad_Wlayer1']  
        params['m_' +'blayer1' ] = 0.9 * params['m_' +'blayer1' ] - learning_rate *  params['grad_blayer1']  
        params['Wlayer1'] += params['m_Wlayer1']
        params['blayer1'] += params['m_blayer1']


        
    
    losses.append(total_loss/train_x.shape[0])
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

# plot loss curve
plt.plot(range(len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("average loss")
plt.xlim(0, len(losses)-1)
plt.ylim(0, None)
plt.grid()
plt.show()

        
# Q5.3.1
# choose 5 labels (change if you want)
visualize_labels = ["A", "B", "C", "1", "2"]

# Running forward for valid dataset

a = forward(valid_x,params,name='layer1',activation=relu)
h1 = forward(a,params,name='layer2',activation=relu)
h2 = forward(h1,params,name='layer3',activation=relu)
out = forward(h2,params,name='output',activation=sigmoid)
print("Out shape and valid_x shape ", out.shape, valid_x.shape)
print("One out shape ", out[0].shape)
print("one valid shape", valid_x[0].shape)
# get 2 validation images from each label to visualize
visualize_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))
reconstructed_x = np.zeros((2*len(visualize_labels), valid_x.shape[1]))

for i, label in enumerate(visualize_labels):
    idx = 26+int(label) if label.isnumeric() else string.ascii_lowercase.index(label.lower())
    choices = np.random.choice(np.arange(100*idx, 100*(idx+1)), 2, replace=False)
    visualize_x[2*i:2*i+2] = valid_x[choices]
    reconstructed_x[2*i:2*i+2] = out[choices]

# run visualize_x through your network
# name the output reconstructed_x
##########################
##### your code here #####
##########################



# visualize
fig = plt.figure()
plt.axis("off")
grid = ImageGrid(fig, 111, nrows_ncols=(len(visualize_labels), 4), axes_pad=0.05)
for i, ax in enumerate(grid):
    if i % 2 == 0:
        ax.imshow(visualize_x[i//2].reshape((32, 32)).T, cmap="Greys")
    else:
        ax.imshow(reconstructed_x[i//2].reshape((32, 32)).T, cmap="Greys")
    ax.set_axis_off()
plt.show()


# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################
p = 0
for i in range(valid_x.shape[0]):

    p += peak_signal_noise_ratio(valid_x[i],out[i])

avg_p = p / valid_x.shape[0]

print("AVERAGE PSNR IS ", avg_p)
