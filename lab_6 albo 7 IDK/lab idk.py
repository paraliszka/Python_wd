import numpy as np

def pot(n: int):
    m = np.identity(n, int)
    for i in range(n):
        for j in range(n):
            m[i, j] = 2 ** (i + j)
    return m


print(pot(5))