import numpy as np


R = np.array([[3,3,0,1],[3,0,3,0],[1,0,0,3],[0,3,3,0],[0,0,1,3]])

U0, S0, V0 = np.linalg.svd(R, full_matrices=True)

#特異値分解の結果を今回使う形に変更する
U = U0.T[:2].T
V = V0[:2]

print(U)
print(V)

