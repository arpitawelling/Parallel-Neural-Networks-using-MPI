# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:24:31 2021

@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:58:12 2021

@author: lenovo
"""


from mpi4py import MPI
import numpy as np
import pandas as pd
import timeit

#Implementing backward propagation for ReLu function
def relu_backward(dA, cache):
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    assert (dZ.shape == Z.shape)

    return dZ

#Implementing backward function for sigmoid function
def sigmoid_backward(dA, cache):
    
    Z = cache

    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ

#linear backpropagation
def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

#Forward propagation main function for prediction
def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)

    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))

    return AL, caches

#Back Propagation
def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

#forward sigmoid
def sigmoid(Z):

    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache

#Forward ReLU
def relu(Z):

    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)

    cache = Z
    return A, cache

#linear forward propagation
def linear_forward(A, W, b):

    Z = W.dot(A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache

#Forward propagation
def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

#Initialize all layers and nodes in layers
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    print("W1 shape",W1.shape)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

#update all parameters after all epochs
def update_parameters(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

#Predict labels
def predict(X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)


    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0


    accuracy=np.sum((p == y)/m)

    return accuracy

#Compute cost
def compute_cost(AL, Y):

    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))

    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())

    return cost

#Reading data
def load_data():
    train_dataset = np.array(pd.read_csv("diabetes_new.csv",header=None,skiprows=1))
    train_set_x_orig=train_dataset[:110000][:]
    train_set_x_orig=np.delete(train_set_x_orig,[8,9],axis=1)
    train_set_y_orig=train_dataset[:110000,8]
    #'train_set_y_orig=train_set_y_orig.reshape(1,-1)
    #print(train_set_y_orig)
    #print(train_dataset[1000])
    test_set_x_orig=train_dataset[110000:][:]
    test_set_x_orig=np.delete(test_set_x_orig,[8,9],axis=1)
    test_set_y_orig=train_dataset[110000:,8]

    #print(len(test_set_y_orig))
    #print(train_dataset[1000])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


#main program
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


send_data=None

#reading data at process0
if rank == 0:
    print("Rank is",rank)
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig=load_data()

    split = np.array_split(train_set_x_orig,size,axis = 0)
    split2=np.array_split(train_set_y_orig,size,axis=0)
    raveled = [np.ravel(arr) for arr in split]
    raveled2=[np.ravel(arr) for arr in split2]
    print("Shape of y",train_set_y_orig.shape)

    X = np.concatenate(raveled)
    Y=np.concatenate(raveled2)

    split_sizes = []

    for i in range(0,len(split),1):
        split_sizes = np.append(split_sizes, len(split[i]))

    print("split sizes",split_sizes)

    split_sizes2 = []

    for i in range(0,len(split2),1):
        split_sizes2 = np.append(split_sizes2, len(split2[i]))

    print("split sizes 2",split_sizes2)

    split_sizes_input = split_sizes*8
    displacements_input = np.insert(np.cumsum(split_sizes_input),0,0)[0:-1]

    split_sizes_input2 = split_sizes
    displacements_input2 = np.insert(np.cumsum(split_sizes_input2),0,0)[0:-1]
else:
#Create variables on other cores
    split_sizes_input = None
    displacements_input = None
    split_sizes_output = None
    displacements_output = None
    split = None
    test = None
    outputData = None
    X=None
    split_sizes_input2 = None
    displacements_input2 = None
    split2=None
    Y=None


start=timeit.default_timer()
split = comm.bcast(split, root=0) 
split_sizes = comm.bcast(split_sizes_input, root = 0)
displacements = comm.bcast(displacements_input, root = 0)

split2 = comm.bcast(split2, root=0) 
split_sizes2 = comm.bcast(split_sizes_input2, root = 0)
displacements2 = comm.bcast(displacements_input2, root = 0)



output_chunk = np.zeros(np.shape(split[rank])) 
label_chunk=np.zeros(np.shape(split2[rank]))

comm.Scatterv([X,split_sizes_input, displacements_input,MPI.DOUBLE],output_chunk,root=0)
comm.Scatterv([Y,split_sizes_input2, displacements_input2,MPI.DOUBLE],label_chunk,root=0)


#print("Rank and scattered array",rank,output_chunk.shape)

grads_local = {}
cost_local=np.zeros(1)
acc_local=np.zeros(1)

X_local=output_chunk.T
Y_local=label_chunk.reshape(1,-1)
#print("Rank with test_chunk shape",rank,label_chunk.shape)
#print("Rank and scattered test array",rank,label_chunk)
n_x = X_local.shape[0]
n_h = 7
n_y = 1
#print("Rank and shape Y_local",rank,Y_local.shape)
local_parameters=initialize_parameters(n_x,n_h,n_y)
#print("Rank and local_parameters",rank,local_parameters)

W1_local = local_parameters["W1"]
b1_local = local_parameters["b1"]
W2_local = local_parameters["W2"]
b2_local = local_parameters["b2"]


#comm.Barrier()
#Start epochs
for i in range(0, 10000):
    #print("rank and epoch",rank,i)

    #print("i is",i)
    A1_local, cache1_local = linear_activation_forward(X_local, W1_local, b1_local, activation='relu')
    A2_local, cache2_local = linear_activation_forward(A1_local, W2_local, b2_local, activation='sigmoid')

    cost_local[0] = compute_cost(A2_local, Y_local)
    #print("cost_local, rank",cost_local,rank)

    cost_send=np.zeros(1)
    comm.Reduce(cost_local,cost_send,MPI.SUM,root=0)
    if rank==0:
        cost_global=np.zeros(1)
        cost_global[0]=cost_send[0]/size
    #print("Cost reduced is:",cost_send)
        #print("Global cost is",cost_global)
    else:
        cost_global=None

    #comm.Barrier()

    #cost_global=comm.bcast(cost_global,root=0)
    #print("rank and global cost",rank,cost_global)
    dA2_local = -(np.divide(Y_local, A2_local) - np.divide(1-Y_local, 1-A2_local))
    #print("dA2 local and shape",dA2_local,dA2_local.shape)
    dA1_local, dW2_local, db2_local = linear_activation_backward(dA2_local, cache2_local, activation='sigmoid')
    dA0_local, dW1_local, db1_local = linear_activation_backward(dA1_local, cache1_local, activation='relu')



    dW1_update=np.zeros(dW1_local.shape)
    comm.Allreduce(dW1_local,dW1_update,MPI.SUM)
    dW1_local=dW1_update/size

    dW2_update=np.zeros(dW2_local.shape)
    comm.Allreduce(dW2_local,dW2_update,MPI.SUM)
    dW2_local=dW2_update/size

    #print("W1_local after allreduce",rank,W1_local)

    db1_update=np.zeros(db1_local.shape)
    comm.Allreduce(db1_local,db1_update,MPI.SUM)
    db1_local=db1_update/size

    db2_update=np.zeros(db2_local.shape)
    comm.Allreduce(db2_local,db2_update,MPI.SUM)
    db2_local=db2_update/size

    grads_local["dW1"] = dW1_local
    grads_local["db1"] = db1_local
    grads_local["dW2"] = dW2_local
    grads_local["db2"] = db2_local

    local_parameters2 = update_parameters(local_parameters, grads_local, learning_rate=0.00991)
    comm.Barrier()

    W1_local = local_parameters2['W1']
    b1_local = local_parameters2['b1']
    W2_local = local_parameters2['W2']
    b2_local = local_parameters2['b2']

#end of epochs
#comm.Barrier()


#
predictions_train_local = predict(X_local, Y_local, local_parameters2)
acc_local[0]=predictions_train_local
#print("Rank and accuracy",rank,predictions_train_local)

acc_send=np.zeros(1)
comm.Reduce(acc_local,acc_send,MPI.SUM,root=0)
if rank==0:
    acc_global=np.zeros(1)
    acc_global[0]=acc_send[0]/size
    #print("Cost reduced is:",cost_send)
    print("Accuracy is",acc_global[0])
else:
    acc_global=None

end=timeit.default_timer()
comm.Barrier()

total_time=np.zeros(1)
total_time[0]=end-start
time_send=np.zeros(1)
comm.Reduce(total_time,time_send,MPI.SUM,root=0)
if rank==0:
    time_global=np.zeros(1)
    time_global[0]=time_send[0]/size
    #print("Cost reduced is:",cost_send)
    print("Time is",time_global[0])
else:
    time_global=None











