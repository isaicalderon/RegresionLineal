# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import numpy as np
import matplotlib.pyplot as plt

def J(A, Y, h):
    N = np.shape(A)[0]
    B = np.zeros((N, 1))
    for iter in range(0, N):
        B[iter] = ((h[0]+h[1] * A[iter, 0]) - Y[iter])**2
    return np.sum(B)/(2*N)

def getHipotesis(X, Y, alpha):
    theta0 = 0
    theta1 = 0
    M = np.shape(X)[0]
    for inf in range(0, 1000000):
        acum0 = 0
        acum1 = 0
        for iter in range(0, M):
            acum0 = acum0 + (theta0 + theta1*X[iter] - Y[iter])
            acum1 = acum1 + (theta0 + theta1*X[iter] - Y[iter]) * X[iter]
        acum0 = (acum0 / M) * alpha
        acum1 = (acum1 / M) * alpha
        theta0 = theta0 - acum0
        theta1 = theta1 - acum1
    return (theta0, theta1)

datos = np.genfromtxt("datos.txt")
M = np.shape(datos)[0]
N = np.shape(datos)[1]-1
X = datos[:, 0:N]
Y = datos[:, N:N+1]

theta = np.array([0, 1.1])
error = J(X, Y, theta)

theta = []

for iter in range(0, 23):
    theta.append(iter/10.0)

error = []
for iter in theta:
    error.append(J(X, Y, [0, iter]))

# plt.plot(theta[:], error[:], "k*")
# plt.plot(theta[:], error[:], "b-")
# plt.show()
    

hipotesis = getHipotesis(X, Y, 0.16)

Y2 = []
for iter in X:
    Y2.append(hipotesis[0] + hipotesis[1] * iter)


print("Hipotesis T1: ",hipotesis[0])
print("Hipotesis T2: ",hipotesis[1])

plt.plot(X, Y, "k*")
plt.plot(X, Y2, "r-")

plt.show()
