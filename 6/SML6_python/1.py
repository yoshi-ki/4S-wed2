import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)

def data_generate(n=3000):
    x = np.zeros(n)
    u = np.random.rand(n)
    index1 = np.where((0 <= u) & (u < 1 / 8))
    x[index1] = np.sqrt(8 * u[index1])
    index2 = np.where((1 / 8 <= u) & (u < 1 / 4))
    x[index2] = 2 - np.sqrt(2 - 8 * u[index2])
    index3 = np.where((1 / 4 <= u) & (u < 1 / 2))
    x[index3] = 1 + 4 * u[index3]
    index4 = np.where((1 / 2 <= u) & (u < 3 / 4))
    x[index4] = 3 + np.sqrt(4 * u[index4] - 2)
    index5 = np.where((3 / 4 <= u) & (u <= 1))
    x[index5] = 5 - np.sqrt(4 - 4 * u[index5])

    return x

def kernel_function(x):
    #compute kernel function
    return np.exp(- (x * x) / 2) / (np.sqrt(2 * np.pi))

def p_hat(x,train_data,h):
    tempsum = 0
    for xi in train_data:
        tempsum = tempsum + kernel_function((x - xi)/h)
    return tempsum / (len(train_data) * h)

def kernel_density_method(data,h):
    #exec kernel density method with band width h and draw the result

    #draw the function with the help of p_hat
    plt.figure()
    X = np.arange(0,5,h)
    Y = p_hat(X,data,h)
    plt.hist(data,bins = 30,density = True)
    plt.plot(X,Y)
    plt.savefig("h" + str(int(h*100)) + ".png")

    return


def cross_validation(data,n):
    #exec cross validation with n sunsets of data to decide band width
    hs = [0.01,0.1,0.5]
    hscore = np.array([])

    #first, shuffle the data
    ndata = np.array([])
    permutation = np.random.permutation(len(data))
    for i in permutation:
        ndata = np.append(ndata,data[i])

    subset_size = int(len(ndata) / n)
    for h in hs:
        score = 0
        for i in range(n):
            #select test data as ith subset
            #and create test and train data
            train_data = np.append(ndata[0:(i*subset_size)],ndata[((i+1)*subset_size):])
            test_data = ndata[(i*subset_size):((i+1)*subset_size)]
            
            # compute the score
            tempscore = 0
            for x in test_data:
                tempscore = tempscore + np.log(p_hat(x, train_data, h))
            tempscore = tempscore / len(test_data)
            score = score + tempscore

        score = score / n
        hscore = np.append(hscore, score)
    
    # finally plot the score for each h
    plt.figure()
    plt.plot(hs,hscore)
    plt.savefig("cross_validation.png")

    return


if __name__ == "__main__":
    n = 3000
    data = data_generate(n)
    
    kernel_density_method(data,0.01)
    kernel_density_method(data,0.1)
    kernel_density_method(data,0.5)

    cross_validation(data,5)


