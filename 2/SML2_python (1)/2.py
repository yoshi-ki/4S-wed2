import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(0)


def generate_sample(n, alpha):
    n1 = sum(np.random.rand(n) < alpha)
    n2 = n - n1
    mean1, mean2 = np.array([2, 0]), np.array([-2, 0])
    cov = np.array([[1, 0], [0, 9]])
    x1 = np.random.multivariate_normal(mean1, cov, n1).transpose()
    x2 = np.random.multivariate_normal(mean2, cov, n2).transpose()
    return x1, x2



if __name__ == "__main__":
    x1,x2 = generate_sample(600,0.1)
    x1 = x1.transpose()
    x2 = x2.transpose()
    n1 = len(x1)
    n2 = len(x2)
    mu1 = np.mean(x1,axis=0)
    mu2 = np.mean(x2,axis=0)
    sigma = (n1 * np.cov(x1, rowvar=0, bias=1) + n2 * np.cov(x2, rowvar=0, bias=1))/(n1+n2)
    sigma_inv = np.linalg.inv(sigma)
    a = np.dot(sigma_inv,(mu1-mu2))
    b = -(1/2) * (np.dot(np.dot(mu1.transpose(),sigma_inv),mu1) - np.dot(np.dot(mu2.transpose(),sigma_inv),mu2)) + np.log(n1/n2)
    print(a)
    print(b)
    plt.figure()
    plt.scatter(x1.transpose()[0],x1.transpose()[1],color = "red",marker = "o")
    plt.scatter(x2.transpose()[0],x2.transpose()[1],color = "blue",marker = "x")
    y = np.arange(-10,10)
    x = -(a[1]*y+b)/a[0]
    plt.plot(x,y,color = "blue")
    plt.savefig("02-01.png")

