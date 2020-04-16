import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('digit.mat')
train = data['X']
test = data['T']

#calculate mu
mus = np.array([])
for i in range(10):
    mus = np.append(mus,np.mean(train[:,:,i],axis=1))
mus = mus.reshape(10,256)

#calculate sigma
sigmas = np.array([])
for i in range(10):
    sigmas = np.append(sigmas,np.cov(train[:,:,i]))
sigmas = sigmas.reshape(10,256,256)

#calculate invsigma
invsigmas = np.array([])
for i in range(10):
    invsigmas = np.append(invsigmas,np.linalg.inv(sigmas[i] + 0.000001 * np.identity(256)))
invsigmas = invsigmas.reshape(10,256,256)

#fissher
S = np.cov(train[:,:,0])
for i in range(1,10):
    S = S + np.cov(train[:, :, i])
S = S / 10
invS = np.linalg.inv(S + 0.000001 * np.identity(256))

# model estimates it as j but the truth is i
results = [[0 for _ in range(10)] for _ in range(10)]

count = 0

for i in range(10):
    #ts = test[:,:,i]
    for k in range(200):
        t = test[:,k,i]
        tempans = 0
        tempp = 0
        for j in range(10):
            pj = mus[j].transpose().dot(invsigmas[j]).dot(t) - mus[j].transpose().dot(invsigmas[j]).dot(mus[j])/2
            #pj = mus[j].transpose().dot(invS).dot(t) - mus[j].transpose().dot(invS).dot(mus[j])/2
            if(tempp < pj):
                tempans = j
                tempp = pj
        count = count + 1
        print(i,tempans)
        results[i][tempans] = results[i][tempans] + 1

        

if __name__ == "__main__":
    print(count)
    print(results)
