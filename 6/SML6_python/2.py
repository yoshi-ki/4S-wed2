import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

data = loadmat('digit.mat')
train = data['X']
test = data['T']


def knn(train_x, train_y, test_x, k_list):
    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    dist_matrix = np.sqrt(np.sum((train_x[None] - test_x[:, None]) ** 2,
                                 axis=2))
    sorted_index_matrix = np.argsort(dist_matrix, axis=1)
    ret_matrix = None
    for k in k_list:
        knn_label = train_y[sorted_index_matrix[:, :k]]
        label_sum_matrix = None
        for i in range(10):
            predict = np.sum(np.where(knn_label == i, 1, 0), axis=1)[:, None]
            if label_sum_matrix is None:
                label_sum_matrix = predict
            else:
                label_sum_matrix = np.concatenate([label_sum_matrix,
                                                   predict], axis=1)
        if ret_matrix is None:
            ret_matrix = np.argmax(label_sum_matrix, axis=1)[None]
        else:
            ret_matrix = np.concatenate([ret_matrix, np.argmax(
                label_sum_matrix, axis=1)[None]], axis=0)
    return ret_matrix  # ret_matrix.shape == (len(k_list), len(test_x))

if __name__ == "__main__":
    ks = [1]
    #cross validation with 5 subsets
    train_x = np.array([])
    train_y = np.array([])
    for i in range(10):
        for j in range(500):
            train_x = np.append(train_x,train[:,j,i])
            train_y = np.append(train_y,i)
    train_x = train_x.reshape(5000,256)

    for i in range(10):
        test_x = np.array([])
        for j in range(200):
            test_x = np.append(test_x,test[:,j,i])
        test_x = test_x.reshape(200,256)
        ret_mat = knn(train_x,train_y,test_x,ks)
        print(ret_mat)

