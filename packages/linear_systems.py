import numpy as np

# Algorithms
def col_sweep(L , b): # L lower triangular
    n = len(b)
    x = np.zeros(n)
    for j in range(n-1):
        x[j] = b[j] / L[j,j]
        for i in range(j+1, n):
            b[i] -= L[i,j] * x[j]
    x[-1] = b[-1] / L[-1, -1]
    return x

def row_sweep(L , b): # L lower triangular
    n = len(b)
    x = np.zeros(n)
    x[0] = b[0] / L[0,0]
    for i in range(1, n):
        for j in range(i+1):
            x[i] -= L[i,j] * x[j]
        x[i] = b[i] / L[i,i]
    return x



# Testing
L = np.array([[1.,0.],[3.,4.]])
b = np.array([2.,5.])
print(b)

x = col_sweep(L,b)
print(L @ x)

y = row_sweep(L,b)
print(L @ y)
