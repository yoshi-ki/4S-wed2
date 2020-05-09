import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import sqrtm


def generate_data(n=1000):
    np.random.seed(4)
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

def g3(s):
    return s * np.exp(- s * s / 2)

def g3_prime(s):
    return (1-s*s) * np.exp(- s * s / 2)

#dataを受け取って最適なbを返す関数
def newton_method(data,g,g_prime):
    #最初の値は適当に定めておく
    b = np.array([1,0])
    count = 0

    #収束するまで最大1000回ループする
    while(count < 1000):
        count = count + 1
        oldb = b
        n = len(data)

        sigma1 = 0
        sigma2 = np.array([0,0])
        for i in range(n):
            xi = data[i][:]
            sigma1 = sigma1 + g_prime(np.dot(oldb,xi.transpose()))
            sigma2 = sigma2 + xi * g(np.dot(oldb,xi.transpose()))
        b = (1/n) * oldb * sigma1 - (1/n) * sigma2




        #正規化
        b = b / np.sqrt(np.dot(b,b.transpose()))

        #収束判定
        if(np.sqrt(np.dot((b-oldb),(b-oldb).transpose())) < 0.001):
            break
        #向きが反対になっている場合
        if(np.sqrt(np.dot((b+oldb),(b+oldb).transpose())) < 0.001):
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
    b3 = newton_method(data,g3,g3_prime)

    #plot
    plt.figure()
    plt.xlim(-4,6)
    plt.ylim(-5,5)
    x = np.linspace(-4,6,100)
    y1 = b1[1] * x / b1[0]
    plt.plot(x, y1, '-r', label='g(s) = s^3', color = "blue")
    plt.scatter(data.transpose()[0][:],data.transpose()[1][:],color = "red", marker = "o")
    plt.legend(loc='upper left')
    plt.savefig("test1.png")

    plt.figure()
    plt.xlim(-4,6)
    plt.ylim(-5,5)
    x = np.linspace(-4,6,100)
    y2 = b2[1] * x / b2[0]
    plt.plot(x, y2, '-r', label='g(s) = tanh(s)', color = "black")
    plt.scatter(data.transpose()[0][:],data.transpose()[1][:],color = "red", marker = "o")
    plt.legend(loc='upper left')
    plt.savefig("test2.png")

    plt.figure()
    plt.xlim(-4,6)
    plt.ylim(-5,5)
    x = np.linspace(-4,6,100)
    y3 = b3[1] * x / b3[0]
    plt.plot(x, y3, '-r', label='g(s) = s*exp(-s*s/2)', color = "green")
    plt.scatter(data.transpose()[0][:],data.transpose()[1][:],color = "red", marker = "o")
    plt.legend(loc='upper left')
    plt.savefig("test3.png")
    
