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

if __name__ == "__main__":
    data = generate_data()
    plt.figure()
    plt.scatter(data.transpose()[0][:],data.transpose()[1][:],color = "red", marker = "o")


    plt.savefig("test.png")
    
