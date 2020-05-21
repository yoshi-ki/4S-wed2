import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat



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

def shuffle_two_array(x,y):
    shuffled_x = np.array([])
    shuffled_y = np.array([])
    permutation = np.random.permutation(len(x))
    for i in permutation:
        shuffled_x = np.append(shuffled_x,x[i])
        shuffled_y = np.append(shuffled_y,y[i])
    return (shuffled_x,shuffled_y)

if __name__ == "__main__":
    ks = [1,2,3,4,5]
    kscore = np.array([])
    
    #first, take data from the dataset
    data = loadmat('digit.mat')
    train = data['X']
    test = data['T']
    train_x = np.array([])
    train_y = np.array([])
    for i in range(10):
        for j in range(500):
            train_x = np.append(train_x,train[:,j,i])
            train_y = np.append(train_y,i)
    train_x = train_x.reshape(5000,256)


    #second, shuffle the data for dividing to sets of subsets
    shuffled_x, shuffled_y = shuffle_two_array(train_x, train_y)
    shuffled_x = shuffled_x.reshape(5000,256)
    
    n = 20
    subset_size = int(5000/n)
    #cross validation with n subsets
    k_score = np.zeros(len(ks))
    for i in range(n):
        #create ith data
        i_train_x = np.append(shuffled_x[0:(i * subset_size)],shuffled_x[((i+1) * subset_size):])
        i_train_y = np.append(shuffled_y[0:(i * subset_size)],shuffled_y[((i+1) * subset_size):])
        i_test_x = shuffled_x[(i * subset_size) : ((i+1) * subset_size)]
        i_test_y = shuffled_y[(i * subset_size) : ((i+1) * subset_size)]

        i_train_x = i_train_x.reshape(len(shuffled_x) - subset_size,256)
        i_test_x = i_test_x.reshape(subset_size,256)


        i_result = knn(i_train_x,i_train_y,i_test_x,ks)
        print(i_result)
        




    for i in range(10):
        test_x = np.array([])
        for j in range(200):
            test_x = np.append(test_x,test[:,j,i])
        test_x = test_x.reshape(200,256)
        ret_mat = knn(train_x,train_y,test_x,ks)
        print(ret_mat)













