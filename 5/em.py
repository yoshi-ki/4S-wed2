import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

data_num = 1000
ndist = 3
nrepeat = 3


np.random.seed(0)
def data_generate(n):
    return (np.random.randn(n) + np.where(np.random.rand(n) > 0.3, 2., -2.))

def initialize():
    ws = np.array([0.2,0.5,0.3])
    mus = np.array([-3,0,3])
    sigmas = np.array([1,1,1])
    return ws,mus,sigmas

def phi(x,mu,sigma):
    return np.exp(- ((x-mu) * (x-mu) / ( 2 * sigma * sigma))) / np.sqrt(2 * np.pi * sigma * sigma)

def em_step(ws,mus,sigmas,data):
    #calculate eta
    eta = np.zeros((data_num,ndist))
    for i in range(data_num):
        #calculate denominator
        denom = 0
        for j in range(ndist):
            denom = denom + ws[j] * phi(data[i],mus[j],sigmas[j])
        for j in range(ndist):
            eta[i][j] = ws[j] * phi(data[i],mus[j],sigmas[j]) / denom     
    
    #update hyper-parameters
    newws = np.array([])
    newmus = np.array([])
    newsigmas = np.array([])
    for j in range(ndist):
        #update ws
        neww = 0
        for i in range(data_num):
            neww = neww + eta[i][j]
        neww = neww / data_num
        newws = np.append(newws,neww)

        #update mus
        newmu = 0
        denom = 0
        for i in range(data_num):
            denom = denom + eta[i][j]
        for i in range(data_num):
            newmu = newmu + eta[i][j] * data[i]
        newmu = newmu / denom
        newmus = np.append(newmus,newmu)

        #update sigmas
        newsigma = 0
        for i in range(data_num):
            newsigma = newsigma + eta[i][j] * (data[i] - mus[j]) * (data[i] - mus[j])
        newsigma = np.sqrt(newsigma / denom)
        newsigmas = np.append(newsigmas,newsigma)

    return newws, newmus, newsigmas

if __name__ == "__main__":

    #initialize hyper-parameters and data
    data = data_generate(data_num)
    ws,mus,sigmas = initialize()

    #calculate distribution
    X = np.arange(-6,6,0.1)
    Y = ws[0] * norm.pdf(X,mus[0],sigmas[0])
    for i in range(ndist-1):
        Y = Y + ws[i+1] * norm.pdf(X,mus[i+1],sigmas[i+1])
    # draw
    plt.figure()
    plt.plot(X,Y)
    plt.hist(data,bins=30,density=True)
    plt.savefig("test" + "0" + ".png")

    
    #EM algorithm
    for i in range(nrepeat):
        ws,mus,sigmas = em_step(ws,mus,sigmas,data)
    
        #calculate distribution
        X = np.arange(-6,6,0.1)
        Y = ws[0] * norm.pdf(X,mus[0],sigmas[0])
        for j in range(ndist-1):
            Y = Y + ws[j+1] * norm.pdf(X,mus[j+1],sigmas[j+1])
    
        # draw
        plt.figure()
        plt.plot(X,Y)
        plt.hist(data,bins=30,density=True)
        plt.savefig("test" + str(i+1) + ".png")


