import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    b = np.zeros(out_size)
    num_weights = in_size * out_size
    interval = np.sqrt(6)/ np.sqrt(in_size + out_size)

    W = np.random.uniform(-interval,interval,num_weights)
    W = W.reshape((in_size,out_size)) 


    ##########################
    ##### your code here #####
    ##########################

    params['W' + name] = W
    params['b' + name] = b


############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    # print("In sigmoid()")
    res = 1.0 / ( 1.0 + np.exp(-x))

    ##########################
    ##### your code here #####
    ##########################

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # print("In forward")
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    pre_act = X@W + b
    post_act = activation(pre_act)

    ##########################
    ##### your code here #####
    ##########################


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act



############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):

    max_c = np.max(x,axis = 1).reshape((-1,1))
    # print("Shape max_c in softmax()", max_c.shape)
    x = x - max_c
    denominator = np.sum(np.exp(x), axis = 1).reshape((-1,1))
    res = np.exp(x) / denominator

    

    ##########################
    ##### your code here #####
    ##########################

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):

    loss, acc = None, None
    argmx = np.argmax(probs,axis = 1)
    pred = np.zeros((y.shape[0],y.shape[1]))
    for i in range(len(argmx)) :
        pred[i,argmx[i]] = 1
    
    pred = np.equal(y,pred)
    pred = np.all(pred,axis = 1)
    acc = np.sum(pred) / y.shape[0]

    loss = - np.sum(y * np.log(probs))
    ##########################
    ##### your code here #####
    ##########################

    return loss, acc 


############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    
    # print( "DEBUG : backwards ()")
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # (don't forget activation_deriv is a function of post_act)
    # then compute the derivative W, b, and X
    ##########################
    ##### your code here #####
    ##########################
    dldz = delta * activation_deriv(post_act)
    # print("Shape dldz, X, W , b: ", dldz.shape,X.shape,W.shape, b.shape)

    grad_W = (dldz.T @ X).T # Last transpoae to convert to W.shape
    grad_b = np.sum(dldz, axis = 0) # np.sum ? --> To convert to b.shape
    grad_X = dldz @ W.T
    # print("Shape of derivatives calculated : grad_W, grad_b, grad_X", grad_W.shape, grad_b.shape, grad_X.shape)

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b

    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################
    n = y.shape[0]
    num_batches = int (n / batch_size)

    for i in range(num_batches):
        rand_inds = np.random.randint(0, n +0.1, size = batch_size)
        x_batch = x[rand_inds,:]
        y_batch = y[rand_inds,:]

        batches.append((x_batch,y_batch))


    return batches
