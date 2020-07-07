import numpy as np
from scipy import stats


R = np.array([[3,3,0,1],[3,0,3,0],[1,0,0,3],[0,3,3,0],[0,0,1,3]])

U0, S0, V0 = np.linalg.svd(R, full_matrices=True)

#特異値分解の結果を今回使う形に変更する
U = U0.T[:2].T
V = V0[:2]


print(U)
print(V)


# initialize some variables
I = 4
J = 5
K = 2
mu_u = np.random.rand(
mu_v = np.random.rand(
V_u = 
V_v = 
raw_u = 
raw_v = 
sigma = 1

# iterationを回す
while(condition):
    # userに関するiteration
    for j in range(J):
        tempsum = np.zeros(mu_v.shape)
        for i in range(I):
            tempsum= tempsum + R[j][i] * mu_v[i]
        mu_u[j] = np.dot(V_u[j], tempsum) / (sigma ** 2)
        tempsum = np.zeros(V_v.shape)
        for i in range(I):
            tempsum = tempsum + (V_v - np.dot(mu_v.T, mu_v))
        V_u[j] = (sigma ** 2) * np.linalg.inv(tempsum + (sigma ** 2) * raw_u)

    # hyper parameterの更新
    for k in range(K):
        tempsum = 0
        for j in range(J):
            tempsum = tempsum + (V_u[j][j] + mu_u[j][k])
        raw_u[k] = tempsum / J


    # itemに関するiteration
    for i in range(I):
        tempsum = np.zeros(mu_u.shape)
        for j in range(J):
            tempsum = tempsum + R[j][i] * mu_u[j
        mu_v[j] = np.dot(V_v[j], tempsum) / (sigma ** 2)
        tempsum = np.zeros(V_u.shape)
        for i in range(J):
            tempsum = tempsum + (V_u - np.dot(mu_u.T, mu_u))
        V_v[j] = (sigma ** 2) * np.linalg.inv(tempsum + (sigma ** 2) * raw_v)


    # hyper parameterの更新
    for k in range(K):
        tempsum = 0
        for i in range(I):
            tempsum = tempsum + (V_v[i][i] + mu_v[i][k])
        raw_v[k] = tempsum / I

    # hyper parameterの更新
    tempsum = 0
    for i in range(I):
        for j in range(J):
            tempsum = tempsum + r[i][j] * r[i][j] - 2 * r[i][j] * np.dot(mu_u[i].T, mu_v[i]) + np.trace( np.dot( V_u[j] + np.dot(mu_u[j], mu_u[j].T), V_v[i] + np.dot(mu_v[i], mu_v[i].T)) )
    sigma = tempsum / (I * J)

        




