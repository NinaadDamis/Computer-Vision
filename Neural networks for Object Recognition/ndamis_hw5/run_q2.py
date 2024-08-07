import numpy as np
import time
# you should write your functions in nn.py
from nn import *
from util import *

#print("EHAAAAT  ")
# time.sleep(10)
# fake data
# feel free to plot it in 2D
# what do you think these 4 classes are?
g0 = np.random.multivariate_normal([3.6,40],[[0.05,0],[0,10]],10)
g1 = np.random.multivariate_normal([3.9,10],[[0.01,0],[0,5]],10)
g2 = np.random.multivariate_normal([3.4,30],[[0.25,0],[0,5]],10)
g3 = np.random.multivariate_normal([2.0,10],[[0.5,0],[0,10]],10)
x = np.vstack([g0,g1,g2,g3])
# we will do XW + B
# that implies that the data is N x D
#print("HERE0!!!")
# create labels
y_idx = np.array([0 for _ in range(10)] + [1 for _ in range(10)] + [2 for _ in range(10)] + [3 for _ in range(10)])
# turn to one_hot
y = np.zeros((y_idx.shape[0],y_idx.max()+1))
y[np.arange(y_idx.shape[0]),y_idx] = 1

# parameters in a dictionary
params = {}

#print("HERE!!!")
# Q 2.1
# initialize a layer
initialize_weights(2,25,params,'layer1')
initialize_weights(25,4,params,'output')
assert(params['Wlayer1'].shape == (2,25))
assert(params['blayer1'].shape == (25,))


print("TETSING INITIALIZATION - expect 0, [0.05 to 0.12]")
print("{}, {:.2f}".format(params['blayer1'].mean(),params['Wlayer1'].std()**2))
print("{}, {:.2f}".format(params['boutput'].mean(),params['Woutput'].std()**2))

# Q 2.2.1
# implement sigmoid

print("********************TESTIMG SIGMOID***************************** ****************")

test = sigmoid(np.array([-1000,1000]))
print('should be zero and one\t',test.min(),test.max())
# implement forward
h1 = forward(x,params,'layer1')
print(h1.shape)
# Q 2.2.2
# implement softmax
probs = forward(h1,params,'output',softmax)
# make sure you understand these values!
print("TESTING SOFTMAX : VALUES SHOULD BE positive, ~1, ~1, (40,4))")
print(probs.min(),min(probs.sum(1)),max(probs.sum(1)),probs.shape)

# Q 2.2.3
# implement compute_loss_and_acc
print("***************************TESTING LOSS AND ACCURACY****************************")
loss, acc = compute_loss_and_acc(y, probs)
print("should be around -np.log(0.25)*40 [~55] or higher, and 0.25")
# if it is not, check softmax!
print("{}, {:.2f}".format(loss,acc))

# here we cheat for you
# the derivative of cross-entropy(softmax(x)) is probs - 1[correct actions]
delta1 = probs - y

# we already did derivative through softmax
# so we pass in a linear_deriv, which is just a vector of ones
# to make this a no-op
delta2 = backwards(delta1,params,'output',linear_deriv)
# Implement backwards!
backwards(delta2,params,'layer1',sigmoid_deriv)

# W and b should match their gradients sizes

print("******************TESTING gradient and parameter shapes ************************ ")

for k,v in sorted(list(params.items())):
    if 'grad' in k:
        name = k.split('_')[1]
        print(name,v.shape, params[name].shape)

# Q 2.4
print("******************Random batch creation ************************ ")

batches = get_random_batches(x,y,5)
# print batch sizes
print([_[0].shape[0] for _ in batches])
batch_num = len(batches)

# WRITE A TRAINING LOOP HERE
print("***************TESTING TRAIN LOOP : should get loss < 35 and accuracy > 75% ************************ ")

max_iters = 500
learning_rate = 1e-3
# with default settings, you should get loss < 35 and accuracy > 75%
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    for xb,yb in batches:
        ##########################
        ##### your code here #####
        ##########################
        
        # forward
        a = forward(xb,params,name='layer1',activation=sigmoid)
        out = forward(a,params,name='output',activation=softmax)
        loss, accuracy = compute_loss_and_acc(yb,out)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss
        avg_acc +=accuracy

        # backward
        delta = out - yb
        d1 = backwards(delta,params,name='output',activation_deriv=linear_deriv)
        d2 = backwards(d1,params,name='layer1',activation_deriv=sigmoid_deriv)

        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']

        # apply gradient 
        # gradients should be summed over batch samples

    avg_acc /= len(batches)  

        
    if (itr % 100)== 0 or itr == 499:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,avg_acc))


# Q 2.5 should be implemented in this file
# you can do this before or after training the network. 

# compute gradients using forward and backward
h1 = forward(x,params,'layer1')
probs = forward(h1,params,'output',softmax)
loss, acc = compute_loss_and_acc(y, probs)
delta1 = probs - y
delta2 = backwards(delta1,params,'output',linear_deriv)
backwards(delta2,params,'layer1',sigmoid_deriv)

# save the old params
import copy
params_orig = copy.deepcopy(params)
print("***************TESTING GRADIENTS USING FINITE DIFFERENCE************************ ")

print("PARAMS ARE ")
for k,v in params.items():
    print(k)
    # print(v.shape)
# compute gradients using finite difference
eps = 1e-6
for k,v in params.items():
    if '_' in k: 
        continue
    # we have a real parameter!
    # for each value inside the parameter
    print("KEY BEING CONSIDERED IS ", k)
    if v.ndim < 2 : # If its one d array
        for i in range(v.shape[0]):
            vi_orig = v[i]
            v[i] += eps
            a1 = forward(x, params, 'layer1')
            out1 = forward(a1, params, 'output', softmax)
            print("shape y, out1 ", y.shape,out1.shape)
            loss1, acc1 = compute_loss_and_acc(y, out1)

            v[i] -= 2*eps
            a2 = forward(x, params, 'layer1')
            out2 = forward(a2, params, 'output', softmax)
            loss2, acc2 = compute_loss_and_acc(y, out2)

            v[i] = vi_orig # Restorre original

            diff = (loss1 - loss2) / (2 * eps)
            params['grad_' + k][i] = diff

    else : # If its one d array
        for i in range(v.shape[0]):
            for j in range (v.shape[1]):
                vi_orig = v[i,j]
                v[i,j] += eps
                a1 = forward(x, params, 'layer1')
                out1 = forward(a1, params, 'output', softmax)
                print("shape y, out1 ", y.shape,out1.shape)
                loss1, acc1 = compute_loss_and_acc(y, out1)

                v[i,j] -= 2*eps
                a2 = forward(x, params, 'layer1')
                out2 = forward(a2, params, 'output', softmax)
                loss2, acc2 = compute_loss_and_acc(y, out2)

                v[i,j] = vi_orig # Restorre original

                diff = (loss1 - loss2) / (2 * eps)
                params['grad_' + k][i,j] = diff

    #   add epsilon
    #   run the network
    #   get the loss
    #   subtract 2*epsilon
    #   run the network
    #   get the loss
    #   restore the original parameter value
    #   compute derivative with central diffs
    
    ##########################
    ##### your code here #####
    ##########################

total_error = 0
for k in params.keys():
    if 'grad_' in k:
        # relative error
        err = np.abs(params[k] - params_orig[k])/np.maximum(np.abs(params[k]),np.abs(params_orig[k]))
        err = err.sum()
        print('{} {:.2e}'.format(k, err))
        total_error += err
# should be less than 1e-4
print('Total Error{:.2e}'.format(total_error))
print("Above error should be less than 1e-04")
