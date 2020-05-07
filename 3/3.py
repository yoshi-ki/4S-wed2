import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


#平均0,標準偏差1の正規乱数をn件
def generate_rand(n):
    x = np.random.normal(0,1,n)
    return x

def maximum_likelihood(x):
    sigma_ml = 0
    n = len(x)
    for i in range(n):
        sigma_ml = sigma_ml + (x[i] - np.sum(x)/n) * (x[i] - np.sum(x)/n) / n
    return sigma_ml


if __name__ == "__main__":
    #m回標本を発生させる
    #n個の標本を発生させる
    m = 10000
    n = 100
    sigmas = np.array([])
    for i in range(m):
        x = generate_rand(n)
        sigmas = np.append(sigmas,maximum_likelihood(x))
    plt.figure()
    plt.hist(sigmas,range = (0,2), bins=100, color = 'blue',alpha = 0.5, density = True)
    X = np.arange(0,2,0.01)
    Y = norm.pdf(X,1,np.sqrt(2/n))
    plt.plot(X,Y,color='r')
    plt.savefig(str(m) + ".png")
    










