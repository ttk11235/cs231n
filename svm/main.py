import numpy as np

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

    return X_tr, Y_tr, X_te, Y_te

class Loss:
    def __init__(self):
        # 10 class, 32*32*3
        self.weight = np.random.rand(10, 3072)
    def svm_loss(self, pic_tr, correct_class):
        # pic_tr    :3072*1 matrix
        # perform loss_i = max(0, si-sj+delta)
        delta = 1.0
        bias = np.random.rand(10,1)
        scores = self.weight.dot(pic_tr) + bias
        correct_class_score = scores.item(correct_class)
        D = self.weight.shape[0]

        loss_i = 0.0
        for i in range(D):
            if i == correct_class:
                continue

            loss_i += max(0, scores.item(i) - correct_class_score + delta)

        return loss_i
        
        


X_tr, Y_tr, X_te, Y_te = load_CIFAR10('./cifar-10-batches-py/')

loss = Loss()
num_tr = X_tr.shape[0]
loss_total = 0.0

for i in range(num_tr):
    loss_total += loss.svm_loss(np.atleast_2d(X_tr[i]).T, Y_tr[i])

print('===== svm loss for unvectorized version =====')
print('average loss is ', loss_total/num_tr)
