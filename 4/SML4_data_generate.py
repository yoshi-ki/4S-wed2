import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm


def generate_data(n=1000):
    x = np.concatenate([np.random.rand(n, 1), np.random.randn(n, 1)], axis=1)
    x[0, 1] = 6   # outlier
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)  # Standardization
    M = np.array([[1, 3], [5, 3]])
    x = x.dot(M.T)
    x = np.linalg.inv(sqrtm(np.cov(x, rowvar=False))).dot(x.T).T

    return x


def centering(data):
    center = np.array([0,0])
    for i in range(len(data)):
        center = center + data[i]
    center = center / len(data)
    newdata = np.array([])
    for i in range(len(data)):
        newdata = np.append(newdata,data[i]-center)
    newdata = np.reshape(newdata,(len(data),-1))
    return newdata

def sphering(data):
    newmatrix = np.array([[0,0],[0,0]])
    for i in range(len(data)):
        newmatrix = newmatrix + np.dot(data[i].transpose(),data[i])
    newmatrix = newmatrix / len(data)
    newdata = np.array([])
    for i in range(len(data)):
        newdata = np.append(newdata,(np.dot(newmatrix ** (-1/2), data[i].transpose())).transpose())
    newdata = np.reshape(newdata,(len(data),-1))
    return data



def g1(s):
    return s * s * s

def g1_prime(s):
    return 3 * s * s

def g2(s):
    return np.tanh(s)

def g2_prime(s):
    return 1 - np.tanh(s) * np.tanh(s)

#dataを受け取って最適なbを返す関数
def newton_method(data,g,g_prime):
    #最初の値は適当に定めておく
    b = np.array([1,0])

    #収束するまでループする
    while(1):
        oldb = b
        n = len(data)

        """
        b = np.array([0,0])
        for i in range(n):
            #sigmaの中身の計算
            xi = data[i][:]
            b = b + (1/n) * oldb * g_prime(np.dot(oldb,xi.transpose())) - (1/n) * xi * g(np.dot(oldb,xi.transpose()))
        """
        sigma1 = 0
        sigma2 = np.array([0,0])
        for i in range(n):
            xi = data[i][:]
            sigma1 = sigma1 + g_prime(np.dot(oldb,xi.transpose()))
            sigma2 = sigma2 + xi * g(np.dot(oldb,xi.transpose()))
        b = (1/n) * oldb * sigma1 - (1/n) * sigma2




        #正規化
        b = b / np.sqrt(np.dot(b,b.transpose()))

        print(b)
        
        #収束判定
        if(np.sqrt(np.dot((b-oldb),(b-oldb).transpose())) < 0.0001):
            break
        #向きが反対になっている場合
        if(np.sqrt(np.dot((b+oldb),(b+oldb).transpose())) < 0.0001):
            break
    return b


if __name__ == "__main__":
    # generate data, centering shpering
    data = generate_data()
    data = centering(data)
    data = sphering(data)

    # calculate b
    b1 = newton_method(data,g1,g1_prime)
    b2 = newton_method(data,g2,g2_prime)

    #plot
    plt.figure()
    plt.xlim(-4,6)
    plt.ylim(-5,5)
    x = np.linspace(-4,6,100)
    y1 = b1[1] * x / b1[0]
    y2 = b2[1] * x / b2[0]
    plt.plot(x, y1, '-r', label='g(s) = s^3', color = "blue")
    plt.plot(x, y2, '-r', label='g(s) = tanh(s)', color = "green")
    plt.scatter(data.transpose()[0][:],data.transpose()[1][:],color = "red", marker = "o")
    plt.legend(loc='upper left')



    plt.savefig("test.png")
    
