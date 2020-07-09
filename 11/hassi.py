import numpy as np
from scipy import stats

np.random.seed(0)

R = np.array([[3.,3.,0.,1.],[3.,0.,3.,0.],[1.,0.,0.,3.],[0.,3.,3.,0.],[0.,0.,1.,3.]])


# initialize some variables
I = 4
J = 5
K = 2
mu_u = np.random.rand(J,K)
mu_u = np.array([[1.0 if k == j else 0.0 for k in range(K)] for j in range(J)], dtype=float)
mu_v = np.random.rand(I,K)
mu_v = np.array([[1.0 if k == i else 0.0 for k in range(K)] for i in range(I)], dtype=float)
V_u = np.array(
        [ [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]] ])
V_v = np.array([ [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]], [[1.,0.],[0.,1.]] ])
rho_u = np.zeros((K,K))
for i in range(K):
    rho_u[i][i] = np.random.rand()
    rho_u[i][i] = 1.0/2.0
rho_v = np.zeros((K,K))
for i in range(K):
    rho_v[i][i] = np.random.rand()
    rho_v[i][i] = 1.0
sigma = 1.0



# iterationを回す
count = 0
while(count < 100):
    count = count + 1


    for cc in range(20):
    # userに関するiteration
        for j in range(J):
            tempsum = np.zeros((K,))
            for i in range(I):
                tempsum = tempsum + R[j][i] * mu_v[i].T
            mu_u[j] = np.dot(V_u[j], tempsum) / (sigma ** 2)
            tempsum = np.zeros((K,K))
            for i in range(I):
                if((j==0 and i==2) or (j==1 and i==1) or (j==1 and i==3) or (j==2 and i==1) or (j==2 and i==2)  or (j==3 and i==0) or (j==3 and i==3) or (j==4 and i==0) or (j==4 and i==1)):
                    continue
                muv = mu_v[i].reshape(2,1)
                tempsum = tempsum + (V_v[i] + np.dot(muv, muv.T))
            V_u[j] = (sigma ** 2) * np.linalg.inv(tempsum + (sigma ** 2) * rho_u)

    for cc in range(20):
        # itemに関するiteration
        for i in range(I):
            tempsum = np.zeros((K,))
            for j in range(J):
                tempsum = tempsum + R[j][i] * mu_u[j]
            mu_v[i] = np.dot(V_v[i], tempsum) / (sigma ** 2)
            tempsum = np.zeros((K,K))
            for j in range(J):
                muu = mu_u[j].reshape(2,1)
                tempsum = tempsum + (V_u[j] + np.dot(muu, muu.T))
            V_v[i] = (sigma ** 2) * np.linalg.inv(tempsum + (sigma ** 2) * rho_v)

    # hyper parameterの更新
    # for k in range(K):
    #     tempsum = 0
    #     for j in range(J):
    #         tempsum = tempsum + (V_u[j][k][k] + mu_u[j][k] * mu_u[j][k])
    #     rho_u[k][k] = tempsum / J

    # hyper parameterの更新
    for k in range(K):
        tempsum = 0
        for i in range(I):
            tempsum = tempsum + (V_v[i][k][k] + mu_v[i][k] *  mu_v[i][k])
        rho_v[k][k] = tempsum / I

    # hyper parameterの更新
    tempsum = 0
    for i in range(I):
        for j in range(J):
            if((j==0 and i==2) or (j==1 and i==1) or (j==1 and i==3) or (j==2 and i==1) or (j==2 and i==2)  or (j==3 and i==0) or (j==3 and i==3) or (j==4 and i==0) or (j==4 and i==1)):
                continue
            muu = mu_u[j].reshape(2,1)
            muv = mu_v[i].reshape(2,1)
            rr = R[j][i] ** 2
            rmumu = R[j][i] * (np.dot(muu.T, muv)[0][0])
            tr = np.trace( np.dot( V_u[j] + np.dot(muu, muu.T), V_v[i] + np.dot(muv, muv.T)) )
            tempsum = tempsum + (rr - 2. * rmumu + tr)
    sigma = 1./(tempsum / 11.)

np.set_printoptions(precision=3,suppress=True)
print("")
print("---parameter---")
print("rho_u")
print(rho_u)
print("rho_v")
print(rho_v)
print("sigma")
print(sigma)
print("V_u")
print(V_u)
print("mu_u")
print(mu_u)
print("V_v")
print(V_v)
print("mu_v")
print(mu_v)



# 現在のhyper parameterに基づいて、UとVを生成する
result = np.zeros((J,I))
for j in range(J):
    # u_jの生成をする
    u_j = np.random.multivariate_normal( mu_u[j], V_u[j], 1 )
    for i in range(I):
        # v_iの生成をする
        v_i = np.random.multivariate_normal( mu_v[i], V_v[i], 1 )
        uj = u_j.reshape(2,)
        vi = v_i.reshape(2,)
        result[j][i] = np.dot(uj.T, vi)

np.set_printoptions(precision=3,suppress=True)
print("")
print("---result---")
print(result)
"""
for j in range(J):
    for i in range(I):
        if(R[j][i] != 0):
            result[j][i] = R[j][i]
print("")
print("---result---")
print(result)
"""
