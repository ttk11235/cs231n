# -*- coding: utf-8 -*-
import numpy as np
import pickle

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_CIFAR10(path):
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    train_batch1 = unpickle("./cifar-10-batches-py/data_batch_1")
    train_batch2 = unpickle("./cifar-10-batches-py/data_batch_2")
    train_batch3 = unpickle("./cifar-10-batches-py/data_batch_3")
    train_batch4 = unpickle("./cifar-10-batches-py/data_batch_4")
    train_batch5 = unpickle("./cifar-10-batches-py/data_batch_5")
    test_batch1 = unpickle("./cifar-10-batches-py/test_batch")
    
    X_tr = np.concatenate((train_batch1[b'data'], 
                        train_batch2[b'data'], 
                        train_batch3[b'data'], 
                        train_batch4[b'data'], 
                        train_batch5[b'data']), axis=0)
    Y_tr = np.concatenate((train_batch1[b'labels'],
                        train_batch2[b'labels'],
                        train_batch3[b'labels'],
                        train_batch4[b'labels'],
                        train_batch5[b'labels']))

    X_te = np.array(test_batch1[b'data'])
    Y_te = np.array(test_batch1[b'labels'])

    return X_tr, Y_tr, X_tr, Y_te

class NearestNeighbor(object):
    def __init__(self, X, y):
        self.Xtr = X
        self.Ytr = y
    
    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.Ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.Ytr[min_index]

        return Ypred


X_tr, Y_tr, X_te, Y_te = load_CIFAR10('./cifar-10-batches-py/')

nn = NearestNeighbor(X_tr, Y_tr)
acc = np.median(nn.predict(X_te) == Y_te)
print (acc)