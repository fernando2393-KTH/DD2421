import numpy as np
from matplotlib import pyplot as plt
import math
import random
from scipy import optimize as opt

random.seed(100)

def kernelScalar(x, y):
    return np.dot(x,y)

def kernelPoly(x, y):
    p = 4
    return (np.dot(x,y) + 1)**p

def zerofun(alphas):
    global targets
    return sum(np.multiply(alphas, targets))

def objective(alphas):
    global N, targets, inputs
    res = 0
    for i in range(N):
        for j in range(N):
            res += alphas[i] * alphas[j] * targets[i] * targets[j] * kernel_matrix[i][j]
    res /= 2
    res -= sum(alphas)
    return res

def nonzero_alphas(alphas):
    # vector of pairs (alpha, index) only for alphas > 10^-5
    nonzero_alphas = [(a, i) for i, a in enumerate(alphas) if a > 10**-5]
    return nonzero_alphas

# Returns the first support vector, which is the first one with 0 < alpha < C
def valid_support_vector(valid_alphas, C):
    global inputs
    for alpha in valid_alphas:
        if alpha[0] < C:
            index = alpha[1]
            vector = inputs[index]
            return index, vector

def computeB(valid_alphas, targets, kernel_matrix, C):
    # Getting a support vector
    support_index, support_vector = valid_support_vector(valid_alphas, C)
    b = 0
    # alpha[0] is the value, alpha[1] its index
    for alpha in valid_alphas:
        i = alpha[1]
        b += alpha[0] * targets[i] * kernel_matrix[support_index][i]
    b -= targets[support_index]
    return b

def indicator(s):
    global valid_alphas, targets, kernel_matrix, b
    ind = 0
    # alpha[0] is the value, alpha[1] its index
    for alpha in valid_alphas:
        i = alpha[1]
        ind += alpha[0] * targets[i] * kernelPoly(s, inputs[i])
    ind -= b
    return ind
    
C = 5
# Creating dataset
classA = np.concatenate((np.random.randn(10,2) * 0.2 + [0.0, -0.5], np.random.randn(10, 2) * 0.2 + [0.0, -0.5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
N = inputs.shape[0] # Number of vector in our dataset
permute = list(range(N))
np.random.shuffle(permute)
inputs = inputs [permute, :] # Nx2 matrix with vectors in the rows
targets = targets[permute] # Array with classes of each vector

kernel_matrix = np.zeros([N, N]) # Creating kernel matrix
for i in range(N):
    for j in range(N):
        kernel_matrix[i][j] = kernelPoly(inputs[i], inputs[j])

# Calling minimize
alphas = opt.minimize(objective, np.zeros(N),
        bounds=[(0, C) for b in range(N)],
        constraints={'type':'eq', 'fun':zerofun})['x']

# valid_alphas contains pairs (alpha, index)
valid_alphas = nonzero_alphas(alphas)


b = computeB(valid_alphas, targets, kernel_matrix, C)

# Plotting everything
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')
plt.axis('equal')

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[indicator(np.array([x, y])) for x in xgrid] for y in ygrid])

plt.contour(xgrid, ygrid, grid,
        (-1.0, 0.0, 1.0),
        colors=('red', 'black', 'blue'),
        linewidths=(1, 3, 1))
plt.savefig('svmplot.pdf')
plt.show()
