import numpy, random , math
from scipy . optimize import minimize
import matplotlib . pyplot as plt

# CONSTANTS

dimension = 2
C = 1
dataset = [] # List of dataset elements
N = len(dataset)

kernel_matrix = numpy.zeros([N,N])

for i in range(N):
    for j in range (N):
        kernel[i][j] = kernel(dataset[i].x, dataset[j].x)

# FUNCTIONS

class Pair:
    x = numpy.zeros(dimension)
    y = 1

def zeroFun(alphas):

    result = 0

    for i, alpha in enumerate(alphas):
        result += alpha * dataset[i].y

    return result

def kernel(x, y):
    p = 2
    return (numpy.multiply(x,y)+1)**p

def objective(vector):

    result = 0

    for i, alphai in enumerate(vector):
        for j, alphaj in enumerate(vector):
            result += alphai * alphaj * dataset[i].y * dataset[j].y * kernel_matrix[i][j]

    result = result/2
    result -= sum(vector)

    return result

def nonZeroAlpha(alphas):

    nonzeroalphas = []
    
    for i, alpha in enumerate(alphas):
        if (alpha > (10**(-5))):
            nonzeroalphas.append([alpha, i])

    return nonzeroalphas

def bComputation(alphas):

    b = 0
    support_vector = []

    for i, alpha in enumerate(alphas):
        if (alpha < C):
            support_vector.append(dataset[i], i)
            break

    for i, alpha in enumerate(alphas):
        b += alpha * dataset[i].y * kernel_matrix[support_vector.y][i] - support_vector.y

    return b

def indicatorFunction(alphas, s):

    result = 0
    b = bComputation(alphas)
    
    for i, alpha in enumerate(alphas):
        result += alpha * dataset[i].y * kernel_matrix[s.y][i] - b

    return result
        
# MAIN CODE

ret = minimize(objective, numpy.zeros(N), bounds=[(0, C) for b in range(N)], constraints={'type':'eq', 'fun':zeroFun})

alpha = nonZeroAlpha(ret['x'])
    