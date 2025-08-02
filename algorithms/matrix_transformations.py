import numpy as np

a = 2
b = 3
c = 4
T = np.zeros((6,6))
# print(T)
T[0,0] = a
T[0,1] = b
for i in range(1,5):
    T[i,i-1] = c
    T[i,i] = a
    T[i,i+1] = b
T[4,5] = c
T[5,5] = a
print(T)
I = np.eye(6)

T_old = T
for i in range(5):
    v = T_old[:, i]
    w = np.linalg.norm(v)
    x = np.zeros(6)
    x[i] = w
    x += v
    mu = -2 / ( ( np.linalg.norm(x) ) ** 2 )
    print(mu)
    print(x)
    H = I + (mu * np.outer(x,x))
    T_new = np.multiply(H.T,T_old)
    T_old = T_new
    print(T_new)
