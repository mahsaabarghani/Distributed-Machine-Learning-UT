import time
from mpi4py import MPI
import numpy as np
import os


def train_test_split_manual(X, Y, test_size=0.2):
    data_size = X.shape[0]
    test_size = int(data_size * test_size)
    indices = np.random.permutation(data_size)  
    i_test = indices[:test_size]
    i_train = indices[test_size:]
    X_train, X_test = X[i_train], X[i_test]
    Y_train, Y_test = Y[i_train], Y[i_test]
    return X_train, X_test, Y_train, Y_test

def compute_gradient(X, Y, weights):
    predicted = 1 / (1 + np.exp(-np.dot(X, weights)))
    errors = Y - predicted
    gradient = -np.dot(X.T, errors) / len(Y)  
    return gradient

def update_weights(weights, gradient, lr):
    weights -= lr * gradient
    return weights

def evaluate_model(X, Y, weights):
    predicted = 1 / (1 + np.exp(-np.dot(X, weights)))
    output = (predicted >= 0.5).astype(int)
    accuracy = np.mean(output == Y)
    return accuracy

def distributed_logistic_regression(X, Y, lr, epochs):
    comm = MPI.COMM_WORLD  
    rank = comm.Get_rank() 
    size = comm.Get_size() 

    total_features = X.shape[1]
    weights = np.zeros(total_features) 

    data_size = X.shape[0]
    sub_size = data_size // size

    local_X = X[rank * sub_size:(rank + 1) * sub_size]
    local_Y = Y[rank * sub_size:(rank + 1) * sub_size]

    for epoch in range(epochs):
        sub_gradient = compute_gradient(local_X, local_Y, weights)
        total_gradient = np.zeros_like(sub_gradient)
        comm.Allreduce(sub_gradient, total_gradient, op=MPI.SUM)
        weights = update_weights(weights, total_gradient / size, lr)

    return weights

lr = 0.01
epochs = 100

startTime = time.time()

data = np.load('/home/dmls/aqdam/tmrine1/2/a/data.npy')
labels = np.load('/home/dmls/aqdam/tmrine1/2/a/labels.npy') 

X_train, X_test, Y_train, Y_test = train_test_split_manual(data, labels)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    final_weights = distributed_logistic_regression(X_train, Y_train, lr, epochs)
    test_accuracy = evaluate_model(X_test, Y_test, final_weights)
    print("Accuracy: ", test_accuracy)
    endTime = time.time()
    print('\n')
    print(f"Total time: {endTime - startTime} seconds")
else:
    distributed_logistic_regression(X_train, Y_train, lr, epochs)
